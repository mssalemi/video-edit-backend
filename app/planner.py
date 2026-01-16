"""
AI Planning Layer for Edit Plans

Builds edit plans (chunks + keep_ms) from transcript segments.
Supports three modes:
- stub: trivial one-clip plan for wiring
- heuristic: deterministic multi-clip plan without AI
- ai: builds prompt and returns 501 (no external API call yet)
"""

import re
import uuid
from typing import Literal, Optional

from app.selection import select_clips, is_marker_segment


# --- Type Aliases ---

PlannerMode = Literal["stub", "heuristic", "ai"]
CleanLevel = Literal["none", "light", "aggressive"]


# --- Constants ---

# Phrases that indicate a reset/restart (for topic splitting)
RESET_PHRASES = [
    "restart",
    "take two",
    "scratch that",
    "let me start over",
    "start over",
    "actually",
    "wait",
    "hold on",
    "never mind",
    "let me try again",
    "one more time",
]

# Gap threshold for detecting topic shifts (ms)
TOPIC_SHIFT_GAP_MS = 3000


# --- Validation Functions ---


def validate_segments(segments: list[dict]) -> None:
    """
    Validate segment list structure.

    Args:
        segments: List of segment dicts

    Raises:
        ValueError: If segments are invalid
    """
    if not segments:
        raise ValueError("segments list cannot be empty")

    for i, seg in enumerate(segments):
        if "start" not in seg or "end" not in seg:
            raise ValueError(f"Segment {i}: missing 'start' or 'end' field")

        if "text" not in seg:
            raise ValueError(f"Segment {i}: missing 'text' field")

        if seg["end"] <= seg["start"]:
            raise ValueError(f"Segment {i}: end ({seg['end']}) must be greater than start ({seg['start']})")

        if seg["start"] < 0:
            raise ValueError(f"Segment {i}: start cannot be negative ({seg['start']})")


def snap_ms_to_segment_boundaries(
    ms: int,
    segments: list[dict],
    which: Literal["start", "end"] = "start",
) -> int:
    """
    Snap a millisecond value to the nearest segment boundary.

    Args:
        ms: Millisecond value to snap
        segments: List of segments with start_ms/end_ms
        which: "start" snaps to segment starts, "end" snaps to segment ends

    Returns:
        Snapped millisecond value
    """
    if not segments:
        return ms

    # Build list of boundary values
    if which == "start":
        boundaries = [seg.get("start_ms", int(seg["start"] * 1000)) for seg in segments]
    else:
        boundaries = [seg.get("end_ms", int(seg["end"] * 1000)) for seg in segments]

    # Find nearest boundary
    nearest = min(boundaries, key=lambda b: abs(b - ms))
    return nearest


def normalize_keep_ranges(
    keep_ms: list[list[int]],
    merge_gap_ms: int = 250,
    min_segment_ms: int = 600,
) -> list[list[int]]:
    """
    Normalize keep ranges: sort, merge close ranges, filter short segments.

    Args:
        keep_ms: List of [start_ms, end_ms] pairs
        merge_gap_ms: Merge ranges within this gap
        min_segment_ms: Filter out segments shorter than this

    Returns:
        Normalized list of [start_ms, end_ms] pairs
    """
    if not keep_ms:
        return []

    # Convert to tuples and sort
    ranges = [(r[0], r[1]) for r in keep_ms]
    ranges.sort(key=lambda x: x[0])

    # Merge overlapping/close ranges
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_ms:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Filter short segments
    filtered = [(s, e) for s, e in merged if (e - s) >= min_segment_ms]

    # Convert back to lists
    return [[s, e] for s, e in filtered]


def validate_keep_ranges(
    keep_ms: list[list[int]],
    min_ms: int,
    max_ms: int,
    duration_ms: int,
) -> None:
    """
    Validate keep ranges against constraints.

    Args:
        keep_ms: List of [start_ms, end_ms] pairs
        min_ms: Minimum total kept duration
        max_ms: Maximum total kept duration
        duration_ms: Total transcript duration

    Raises:
        ValueError: If ranges are invalid
    """
    if not keep_ms:
        raise ValueError("keep_ms cannot be empty")

    total_kept = 0
    prev_end = -1

    for i, pair in enumerate(keep_ms):
        if len(pair) != 2:
            raise ValueError(f"Range {i}: must have exactly 2 elements [start_ms, end_ms]")

        start, end = pair

        if start < 0:
            raise ValueError(f"Range {i}: start_ms cannot be negative ({start})")

        if end <= start:
            raise ValueError(f"Range {i}: end_ms ({end}) must be greater than start_ms ({start})")

        if end > duration_ms:
            raise ValueError(f"Range {i}: end_ms ({end}) exceeds transcript duration ({duration_ms})")

        if start < prev_end:
            raise ValueError(f"Range {i}: overlaps with previous range (start {start} < prev_end {prev_end})")

        if start <= prev_end:
            raise ValueError(f"Range {i}: ranges must be non-overlapping and sorted")

        total_kept += (end - start)
        prev_end = end

    if total_kept < min_ms:
        raise ValueError(f"Total kept duration ({total_kept}ms) is less than minimum ({min_ms}ms)")

    if total_kept > max_ms:
        raise ValueError(f"Total kept duration ({total_kept}ms) exceeds maximum ({max_ms}ms)")


# --- Prompt Builder ---


def build_ai_plan_prompt(
    segments: list[dict],
    max_clips: int,
    clip_types: list[str],
    preferred_clip_type: str,
    markers: list[str],
    clean_level: str,
    min_clip_ms: int,
    max_clip_ms: int,
    max_keep_ranges: int,
    enforce_segment_boundaries: bool,
) -> dict:
    """
    Build the AI prompt payload for edit planning.

    Args:
        segments: Transcript segments
        max_clips: Maximum clips to generate
        clip_types: Allowed clip types
        preferred_clip_type: Preferred clip type hint
        markers: Marker words for mess-up detection
        clean_level: Cleanup aggressiveness
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration
        max_keep_ranges: Maximum keep ranges per clip
        enforce_segment_boundaries: Whether to snap to segment boundaries

    Returns:
        Dict with system_prompt, user_prompt, json_schema, segments_compact
    """
    # Build compact segment representation
    segments_compact = []
    for i, seg in enumerate(segments):
        start_ms = seg.get("start_ms", int(seg["start"] * 1000))
        end_ms = seg.get("end_ms", int(seg["end"] * 1000))
        segments_compact.append({
            "idx": i,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": seg["text"],
        })

    # Calculate duration
    if segments:
        last_seg = segments[-1]
        duration_ms = last_seg.get("end_ms", int(last_seg["end"] * 1000))
    else:
        duration_ms = 0

    # System prompt
    system_prompt = """You are an expert video editor AI. Your task is to analyze transcript segments and create an optimal edit plan that identifies the best clips from the content.

CRITICAL RULES:
1. Output ONLY valid JSON matching the exact schema provided
2. No explanations, comments, or text outside the JSON
3. All keep_ms ranges must be sorted by start time
4. Ranges must not overlap
5. Each range must have start_ms < end_ms
6. Identify and EXCLUDE mess-ups (false starts, filler bursts, "restart", corrections) by creating gaps in keep_ms
7. Prefer coherent, standalone clips that don't cut mid-thought
8. When possible, separate different topics into different clips
9. Use marker words as hints for mess-up detection"""

    # User prompt
    marker_hint = ""
    if markers:
        marker_hint = f"\n\nMARKER WORDS (indicate mess-ups/resets): {markers}"

    boundary_hint = ""
    if enforce_segment_boundaries:
        boundary_hint = "\n\nIMPORTANT: All keep_ms values must exactly match segment boundaries (start_ms or end_ms from the segments list)."

    user_prompt = f"""Analyze the following transcript and create an edit plan.

CONSTRAINTS:
- Maximum clips: {max_clips}
- Allowed clip types: {clip_types}
- Preferred clip type: {preferred_clip_type}
- Clean level: {clean_level}
- Minimum clip duration: {min_clip_ms}ms
- Maximum clip duration: {max_clip_ms}ms
- Maximum keep ranges per clip: {max_keep_ranges}
- Total transcript duration: {duration_ms}ms{marker_hint}{boundary_hint}

TRANSCRIPT SEGMENTS:
{_format_segments_for_prompt(segments_compact)}

Create an edit plan with up to {max_clips} clips. For each clip:
1. Identify a coherent topic or moment
2. Set keep_ms to include good content and exclude mess-ups
3. If there are mess-ups within a clip's content, use multiple keep_ms ranges to skip them
4. Provide a short title and reason for each clip
5. Assign confidence (0-1) based on content quality

Output your response as JSON matching the schema exactly."""

    # JSON schema
    json_schema = {
        "type": "object",
        "required": ["clips"],
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["clip_id", "clip_type", "title", "keep_ms", "reason", "confidence"],
                    "properties": {
                        "clip_id": {"type": "string", "description": "Unique identifier (UUID)"},
                        "clip_type": {"type": "string", "enum": clip_types},
                        "title": {"type": "string", "description": "Short descriptive title"},
                        "keep_ms": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "description": "List of [start_ms, end_ms] ranges to keep",
                        },
                        "reason": {"type": "string", "description": "Why this clip was selected"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
    }

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "json_schema": json_schema,
        "segments_compact": segments_compact,
    }


def _format_segments_for_prompt(segments_compact: list[dict]) -> str:
    """Format segments for inclusion in prompt."""
    lines = []
    for seg in segments_compact:
        lines.append(f"[{seg['idx']}] {seg['start_ms']}ms-{seg['end_ms']}ms: \"{seg['text']}\"")
    return "\n".join(lines)


# --- Heuristic Planner ---


def plan_edits_stub(
    segments: list[dict],
    preferred_clip_type: str,
) -> dict:
    """
    Create a trivial one-clip plan (for wiring/testing).

    Args:
        segments: Transcript segments
        preferred_clip_type: Clip type to use

    Returns:
        PlanEditsResponse dict
    """
    if not segments:
        return {
            "clips": [],
            "meta": {
                "planner": "stub",
                "segments_in": 0,
                "max_clips": 1,
            },
        }

    # Get duration from segments
    first_seg = segments[0]
    last_seg = segments[-1]
    start_ms = first_seg.get("start_ms") or int(first_seg["start"] * 1000)
    end_ms = last_seg.get("end_ms") or int(last_seg["end"] * 1000)

    return {
        "clips": [
            {
                "clip_id": str(uuid.uuid4()),
                "clip_type": preferred_clip_type,
                "title": "Full transcript",
                "keep_ms": [[start_ms, end_ms]],
                "total_ms": end_ms - start_ms,
                "reason": "Stub planner: keeping entire transcript as single clip",
                "confidence": 0.5,
            }
        ],
        "meta": {
            "planner": "stub",
            "segments_in": len(segments),
            "max_clips": 1,
        },
    }


def plan_edits_heuristic(
    segments: list[dict],
    max_clips: int,
    clip_types: list[str],
    preferred_clip_type: str,
    markers: list[str],
    clean_level: str,
    min_clip_ms: int,
    max_clip_ms: int,
    max_keep_ranges: int,
    enforce_segment_boundaries: bool,
) -> dict:
    """
    Create a multi-clip plan using heuristics (no AI).

    Approach:
    1. Use markers to split segments into chunks if markers found
    2. Otherwise, detect topic shifts via gaps or reset phrases
    3. Fall back to equal time buckets
    4. For each chunk, use select_clips to find best window
    5. Convert to keep_ms format

    Args:
        segments: Transcript segments
        max_clips: Maximum clips to generate
        clip_types: Allowed clip types
        preferred_clip_type: Preferred clip type
        markers: Marker words for splitting
        clean_level: Cleanup level
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration
        max_keep_ranges: Maximum keep ranges per clip
        enforce_segment_boundaries: Snap to boundaries

    Returns:
        PlanEditsResponse dict
    """
    if not segments:
        return {
            "clips": [],
            "meta": {
                "planner": "heuristic",
                "segments_in": 0,
                "max_clips": max_clips,
            },
        }

    # Normalize segments (ensure ms fields)
    normalized_segments = _normalize_segments(segments)

    # Split segments into chunks
    chunks = _split_into_chunks(normalized_segments, markers, max_clips)

    # Generate clips from chunks
    clips = []
    for i, chunk_segments in enumerate(chunks):
        if not chunk_segments:
            continue

        if len(clips) >= max_clips:
            break

        clip = _generate_clip_from_chunk(
            chunk_segments=chunk_segments,
            clip_index=i,
            preferred_clip_type=preferred_clip_type,
            clean_level=clean_level,
            min_clip_ms=min_clip_ms,
            max_clip_ms=max_clip_ms,
            enforce_segment_boundaries=enforce_segment_boundaries,
        )

        if clip:
            clips.append(clip)

    return {
        "clips": clips,
        "meta": {
            "planner": "heuristic",
            "segments_in": len(segments),
            "max_clips": max_clips,
        },
    }


def _normalize_segments(segments: list[dict]) -> list[dict]:
    """Ensure segments have start_ms and end_ms fields."""
    normalized = []
    for seg in segments:
        norm_seg = dict(seg)
        if norm_seg.get("start_ms") is None:
            norm_seg["start_ms"] = int(seg["start"] * 1000)
        if norm_seg.get("end_ms") is None:
            norm_seg["end_ms"] = int(seg["end"] * 1000)
        normalized.append(norm_seg)
    return normalized


def _split_into_chunks(
    segments: list[dict],
    markers: list[str],
    max_chunks: int,
) -> list[list[dict]]:
    """
    Split segments into chunks for separate clips.

    Strategy:
    1. If markers provided and found, split at marker segments
    2. Otherwise, split at topic shifts (large gaps or reset phrases)
    3. Fall back to equal time buckets
    """
    if not segments:
        return []

    # Try marker-based splitting first
    if markers:
        chunks = _split_by_markers(segments, markers)
        if len(chunks) > 1:
            return chunks[:max_chunks]

    # Try topic-shift splitting
    chunks = _split_by_topic_shifts(segments)
    if len(chunks) > 1:
        return chunks[:max_chunks]

    # Fall back to time-based splitting
    return _split_by_time(segments, max_chunks)


def _split_by_markers(
    segments: list[dict],
    markers: list[str],
) -> list[list[dict]]:
    """Split segments at marker boundaries."""
    chunks = []
    current_chunk = []

    for seg in segments:
        if is_marker_segment(seg, markers):
            # End current chunk before the marker
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            # Skip the marker segment itself
        else:
            current_chunk.append(seg)

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_by_topic_shifts(segments: list[dict]) -> list[list[dict]]:
    """Split segments at topic shifts (gaps or reset phrases)."""
    if len(segments) <= 1:
        return [segments] if segments else []

    chunks = []
    current_chunk = [segments[0]]

    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]

        # Check for large gap
        gap_ms = curr_seg["start_ms"] - prev_seg["end_ms"]
        is_gap_split = gap_ms > TOPIC_SHIFT_GAP_MS

        # Check for reset phrase
        text_lower = curr_seg["text"].lower()
        is_reset_split = any(phrase in text_lower for phrase in RESET_PHRASES)

        if is_gap_split or is_reset_split:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []

            # If reset phrase, skip that segment
            if is_reset_split:
                continue

        current_chunk.append(curr_seg)

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_by_time(
    segments: list[dict],
    max_chunks: int,
) -> list[list[dict]]:
    """Split segments into roughly equal time buckets."""
    if not segments or max_chunks <= 1:
        return [segments] if segments else []

    total_duration = segments[-1]["end_ms"] - segments[0]["start_ms"]
    chunk_duration = total_duration / max_chunks

    chunks = []
    current_chunk = []
    chunk_start = segments[0]["start_ms"]

    for seg in segments:
        # Check if this segment belongs to next chunk
        seg_midpoint = (seg["start_ms"] + seg["end_ms"]) / 2
        chunk_threshold = chunk_start + chunk_duration

        if seg_midpoint > chunk_threshold and current_chunk and len(chunks) < max_chunks - 1:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start = seg["start_ms"]

        current_chunk.append(seg)

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _generate_clip_from_chunk(
    chunk_segments: list[dict],
    clip_index: int,
    preferred_clip_type: str,
    clean_level: str,
    min_clip_ms: int,
    max_clip_ms: int,
    enforce_segment_boundaries: bool,
) -> Optional[dict]:
    """Generate a clip from a chunk of segments."""
    if not chunk_segments:
        return None

    # Use select_clips to find best window
    min_s = min_clip_ms / 1000.0
    max_s = max_clip_ms / 1000.0
    target_s = (min_s + max_s) / 2

    selected = select_clips(
        segments=chunk_segments,
        target_s=target_s,
        min_s=min_s,
        max_s=max_s,
        max_clips=1,
        max_gap_s=2.0,
        clip_type=preferred_clip_type,
        clean_level=clean_level,
        markers=[],  # Already handled in splitting
    )

    if not selected:
        # Fallback: use the whole chunk
        start_ms = chunk_segments[0]["start_ms"]
        end_ms = chunk_segments[-1]["end_ms"]

        # Enforce duration limits
        duration_ms = end_ms - start_ms
        if duration_ms < min_clip_ms:
            return None
        if duration_ms > max_clip_ms:
            end_ms = start_ms + max_clip_ms

        return {
            "clip_id": str(uuid.uuid4()),
            "clip_type": preferred_clip_type,
            "title": f"Clip {clip_index + 1}",
            "keep_ms": [[start_ms, end_ms]],
            "total_ms": end_ms - start_ms,
            "reason": f"heuristic fallback: chunk {clip_index + 1}",
            "confidence": 0.4,
        }

    # Convert selected clip to keep_ms format
    clip = selected[0]
    start_ms = clip["start_ms"]
    end_ms = clip["end_ms"]

    # Snap to boundaries if required
    if enforce_segment_boundaries:
        start_ms = snap_ms_to_segment_boundaries(start_ms, chunk_segments, "start")
        end_ms = snap_ms_to_segment_boundaries(end_ms, chunk_segments, "end")

    # Generate title from first segment text
    first_text = chunk_segments[0]["text"][:50]
    title = first_text.split(".")[0] if "." in first_text else first_text
    title = title.strip()
    if len(title) > 40:
        title = title[:37] + "..."

    return {
        "clip_id": str(uuid.uuid4()),
        "clip_type": preferred_clip_type,
        "title": title or f"Clip {clip_index + 1}",
        "keep_ms": [[start_ms, end_ms]],
        "total_ms": end_ms - start_ms,
        "reason": clip.get("reason", f"heuristic: {preferred_clip_type} selection"),
        "confidence": min(clip.get("score", 0.5) + 0.2, 1.0),
    }
