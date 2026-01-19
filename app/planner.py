"""
AI Planning Layer for Edit Plans

Builds edit plans (chunks + keep_ms) from transcript segments.
Supports four modes:
- stub: trivial one-clip plan for wiring
- heuristic: deterministic multi-clip plan without AI
- ai: calls Claude API to generate intelligent edit plans with timestamps
- ai_labels: calls Claude to label segments (keep/cut/unsure), then deterministic code converts to clips
"""

import json
import logging
import re
import uuid
from typing import Literal, Optional

import anthropic

from app.selection import select_clips, is_marker_segment
from app.settings import settings

logger = logging.getLogger(__name__)


# --- Type Aliases ---

PlannerMode = Literal["stub", "heuristic", "ai", "ai_labels"]
CleanLevel = Literal["none", "light", "aggressive"]
SegmentAction = Literal["keep", "cut", "unsure"]
UnsurePolicy = Literal["keep", "cut", "adjacent"]


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


def validate_planned_clips(
    clips: list[dict],
    segments: list[dict],
    min_clip_ms: int,
    max_clip_ms: int,
) -> tuple[bool, str]:
    """
    Validate AI-generated clips to detect junk output.

    Checks for:
    - Empty clips list
    - All clips too short (< min_clip_ms)
    - All clips too long (> max_clip_ms)
    - Clips that don't overlap with any segments
    - keep_ms ranges outside transcript bounds

    Args:
        clips: List of clip dicts with keep_ms ranges
        segments: Original segments list
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration

    Returns:
        Tuple of (is_valid, reason). If is_valid is False, reason explains why.
    """
    if not clips:
        return False, "no clips generated"

    if not segments:
        return False, "no segments provided"

    # Get transcript bounds
    normalized = _normalize_segments(segments)
    transcript_start = normalized[0]["start_ms"]
    transcript_end = normalized[-1]["end_ms"]

    valid_clip_count = 0
    total_clips = len(clips)

    for clip in clips:
        keep_ms = clip.get("keep_ms", [])
        if not keep_ms:
            continue

        # Calculate total duration
        total_ms = sum(r[1] - r[0] for r in keep_ms if len(r) == 2 and r[1] > r[0])

        # Check if within bounds
        if total_ms < min_clip_ms:
            continue  # Too short

        if total_ms > max_clip_ms * 1.5:  # Allow some tolerance
            continue  # Way too long

        # Check if ranges are within transcript bounds
        ranges_valid = True
        for r in keep_ms:
            if len(r) != 2:
                ranges_valid = False
                break
            start, end = r
            if start < transcript_start - 1000:  # 1s tolerance
                ranges_valid = False
                break
            if end > transcript_end + 1000:  # 1s tolerance
                ranges_valid = False
                break

        if ranges_valid:
            valid_clip_count += 1

    if valid_clip_count == 0:
        return False, "all clips failed validation (out of bounds or wrong duration)"

    # At least one valid clip
    return True, "ok"


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
    system_prompt = """You are a video edit planning assistant.

Your job is to analyze a transcript that has been segmented with timestamps and produce a high-level edit plan.
You do not edit video files. You do not trim frames.
You only decide which time ranges should be kept together as meaningful clips.

## What You're Looking For

1. **Story Segments**: Identify coherent sections where the speaker is making a complete point, telling a story, or demonstrating something useful. These become clips.

2. **Mess-Ups to Exclude**: Detect and skip:
   - False starts ("Actually, let me...", "Wait, no...")
   - Reset phrases ("Take two", "Let me start over", "Scratch that")
   - Filler bursts (excessive "um", "uh", stammering)
   - Corrections where the speaker repeats themselves better
   - Long pauses or silence gaps

3. **Splitting Content**: If the transcript contains multiple distinct topics or natural breaks (gaps > 3 seconds, topic changes), consider splitting them into separate clips rather than one long clip.

## Constraints

- **Do not invent timestamps**: Only use start_ms and end_ms values from the provided segments
- **All keep_ms ranges must align exactly to provided segment boundaries**: Use the exact start_ms or end_ms values from segments
- **Do not micro-trim silence or filler words**: Keep whole segments or skip them entirely
- **Prefer fewer, clearer clips over many small ones**: A 30-second coherent clip is better than three 10-second fragments
- **Ranges must be sorted by start time and must not overlap**
- **Each range must have start_ms < end_ms**

## Output Format

Return ONLY valid JSON matching the schema. No explanations or text outside the JSON."""

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

Create an edit plan with up to {max_clips} clips. For each clip, you MUST include:
- clip_type: one of {clip_types}
- title: short 2-6 word title
- keep_ms: array of [start_ms, end_ms] pairs from the segment timestamps above
- total_ms: sum of all kept ranges
- reason: why this clip was selected
- confidence: 0-1 score

EXAMPLE OUTPUT FORMAT:
{{
  "clips": [
    {{
      "clip_type": "document",
      "title": "Python Coding Tip",
      "keep_ms": [[0, 5000], [12000, 20000]],
      "total_ms": 13000,
      "reason": "Coherent explanation after skipping the restart",
      "confidence": 0.8
    }}
  ]
}}

Now analyze the transcript and output ONLY valid JSON:"""

    # JSON schema
    json_schema = {
        "type": "object",
        "required": ["clips"],
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["clip_id", "clip_type", "title", "keep_ms", "total_ms", "reason", "confidence"],
                    "properties": {
                        "clip_id": {"type": "string", "description": "Unique identifier (UUID)"},
                        "clip_type": {"type": "string", "enum": clip_types},
                        "title": {"type": "string", "description": "Short descriptive title (2-6 words)"},
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
                        "total_ms": {"type": "integer", "description": "Sum of all kept ranges in milliseconds"},
                        "reason": {"type": "string", "description": "Why this clip was selected"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "0-1 confidence score"},
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


# --- AI Planner ---


def plan_edits_ai(
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
    Create an edit plan using Claude AI.

    Args:
        segments: Transcript segments
        max_clips: Maximum clips to generate
        clip_types: Allowed clip types
        preferred_clip_type: Preferred clip type
        markers: Marker words for mess-up detection
        clean_level: Cleanup level
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration
        max_keep_ranges: Maximum keep ranges per clip
        enforce_segment_boundaries: Snap to boundaries

    Returns:
        PlanEditsResponse dict

    Raises:
        ValueError: If API key is not configured or API call fails
    """
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    # Build the prompt
    prompt_payload = build_ai_plan_prompt(
        segments=segments,
        max_clips=max_clips,
        clip_types=clip_types,
        preferred_clip_type=preferred_clip_type,
        markers=markers,
        clean_level=clean_level,
        min_clip_ms=min_clip_ms,
        max_clip_ms=max_clip_ms,
        max_keep_ranges=max_keep_ranges,
        enforce_segment_boundaries=enforce_segment_boundaries,
    )

    # Call Claude API
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    logger.info(f"Calling Claude API with {len(segments)} segments")

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4096,
        system=prompt_payload["system_prompt"],
        messages=[
            {"role": "user", "content": prompt_payload["user_prompt"]}
        ],
    )

    # Extract response text
    response_text = message.content[0].text
    logger.info(f"Claude response: {response_text[:200]}...")

    # Parse JSON response
    try:
        # Try to extract JSON from response (in case there's extra text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.error(f"Response was: {response_text}")
        raise ValueError(f"Claude returned invalid JSON: {e}")

    # Validate and normalize the response
    if "clips" not in result:
        raise ValueError("Claude response missing 'clips' field")

    clips = result["clips"]

    # Ensure each clip has required fields and valid structure
    normalized_segments = _normalize_segments(segments)
    validated_clips = []

    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            logger.warning(f"Skipping invalid clip {i}: not a dict")
            continue

        # Ensure required fields
        if "keep_ms" not in clip or not clip["keep_ms"]:
            logger.warning(f"Skipping clip {i}: missing or empty keep_ms")
            continue

        # Normalize keep_ms format (Claude may return dicts or lists)
        raw_ranges = clip["keep_ms"]
        normalized_ranges = []
        for r in raw_ranges:
            if isinstance(r, dict):
                # Handle {"start": X, "end": Y} or {"start_ms": X, "end_ms": Y}
                start = r.get("start_ms") or r.get("start") or 0
                end = r.get("end_ms") or r.get("end") or 0
                if end > start:
                    normalized_ranges.append([int(start), int(end)])
            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                # Handle [start, end] format
                normalized_ranges.append([int(r[0]), int(r[1])])
        clip["keep_ms"] = normalized_ranges

        if not clip["keep_ms"]:
            logger.warning(f"Skipping clip {i}: no valid keep_ms ranges after normalization")
            continue

        # Generate clip_id if missing
        if "clip_id" not in clip:
            clip["clip_id"] = str(uuid.uuid4())

        # Handle "type" vs "clip_type" (Claude may use either)
        if "clip_type" not in clip:
            clip["clip_type"] = clip.get("type", preferred_clip_type)

        # Calculate total_ms if missing
        if "total_ms" not in clip:
            clip["total_ms"] = sum(
                (r[1] - r[0]) for r in clip["keep_ms"]
            )

        # Default other fields
        if "title" not in clip:
            clip["title"] = f"AI Clip {i + 1}"
        if "reason" not in clip:
            clip["reason"] = "AI-generated clip"
        if "confidence" not in clip:
            clip["confidence"] = 0.8

        # Snap to segment boundaries if required
        if enforce_segment_boundaries:
            snapped_ranges = []
            for r in clip["keep_ms"]:
                if len(r) == 2:
                    start = snap_ms_to_segment_boundaries(r[0], normalized_segments, "start")
                    end = snap_ms_to_segment_boundaries(r[1], normalized_segments, "end")
                    if end > start:
                        snapped_ranges.append([start, end])
            clip["keep_ms"] = snapped_ranges
            clip["total_ms"] = sum((r[1] - r[0]) for r in snapped_ranges)

        if clip["keep_ms"]:  # Only add if we have valid ranges
            validated_clips.append(clip)

    return {
        "clips": validated_clips[:max_clips],
        "meta": {
            "planner": "ai",
            "segments_in": len(segments),
            "max_clips": max_clips,
        },
    }


# --- AI Labels Planner ---


# Valid tags for segment labeling
VALID_TAGS = [
    "clean_story",      # Good content, flows well
    "retake_repeat",    # Earlier version of something said better later
    "filler",           # "Um", "uh", excessive pauses
    "false_start",      # "Actually let me...", incomplete thought
    "outro",            # Sign-off, "thanks for watching"
    "intro",            # Opening greeting, setup
    "topic_shift",      # Marks transition to new subject
    "question",         # Speaker asking a question
    "tangent",          # Off-topic digression
    "correction",       # Speaker correcting themselves
    # Extended tags (added for normalization completeness)
    "restart_phrase",   # "Take two", "let me start over"
    "garbled",          # Unclear speech, mumbling
    "non_story",        # Off-topic, non-content segments
    "meta_commentary",  # "Is this recording?", meta discussion
]

# Tags that force action="cut" regardless of AI label
CUT_FORCING_TAGS = [
    "false_start",
    "retake_repeat",
    "filler",
    "restart_phrase",
    "garbled",
    "non_story",
    "meta_commentary",
    "outro",
]

# Common outro/wrap-up phrases to auto-cut
OUTRO_PHRASES = [
    "let's see",
    "lets see",
    "that's it",
    "thats it",
    "anyway",
    "okay bye",
    "ok bye",
    "cool",
    "alright so",
    "alright then",
    "so yeah",
    "yeah so",
]

# Minimum duration for first range to be kept (ms)
MIN_BRIDGE_RANGE_MS = 4000

# Minimum duration for lead-in range (first range exception threshold)
MIN_LEADIN_RANGE_MS = 2500

# Default padding for clip expansion
DEFAULT_LEAD_IN_MS = 300
DEFAULT_TAIL_OUT_MS = 300


def expand_keep_ranges(
    clips: list[dict],
    segments: list[dict],
    lead_in_ms: int = DEFAULT_LEAD_IN_MS,
    tail_out_ms: int = DEFAULT_TAIL_OUT_MS,
) -> list[dict]:
    """
    Expand keep ranges with lead-in and tail-out padding.

    Expands each keep range:
    - Start is moved earlier by lead_in_ms
    - End is moved later by tail_out_ms
    - Clamped to transcript bounds
    - Snapped to nearest segment boundaries

    This creates more natural cuts that don't start/end abruptly.

    Args:
        clips: List of clip dicts with keep_ms
        segments: Original segment list for boundary snapping
        lead_in_ms: Amount to expand start (subtract from start_ms)
        tail_out_ms: Amount to expand end (add to end_ms)

    Returns:
        Clips with expanded keep_ms ranges
    """
    if not clips or not segments:
        return clips

    # Normalize segments for timing
    normalized = _normalize_segments(segments)
    if not normalized:
        return clips

    # Get transcript bounds
    transcript_start = normalized[0]["start_ms"]
    transcript_end = normalized[-1]["end_ms"]

    processed = []
    for clip in clips:
        keep_ms = clip.get("keep_ms", [])
        if not keep_ms:
            processed.append(clip)
            continue

        new_ranges = []
        for r in keep_ms:
            if len(r) != 2:
                new_ranges.append(r)
                continue

            start, end = r

            # Expand
            new_start = start - lead_in_ms
            new_end = end + tail_out_ms

            # Clamp to transcript bounds
            new_start = max(new_start, transcript_start)
            new_end = min(new_end, transcript_end)

            # Snap to segment boundaries
            new_start = snap_ms_to_segment_boundaries(new_start, normalized, "start")
            new_end = snap_ms_to_segment_boundaries(new_end, normalized, "end")

            # Ensure valid range
            if new_end > new_start:
                new_ranges.append([new_start, new_end])
            else:
                # Keep original if expansion failed
                new_ranges.append([start, end])

        if new_ranges:
            # Normalize ranges (merge overlapping after expansion)
            new_ranges = normalize_keep_ranges(new_ranges, merge_gap_ms=100, min_segment_ms=0)

            new_clip = clip.copy()
            new_clip["keep_ms"] = new_ranges
            new_clip["total_ms"] = sum(r[1] - r[0] for r in new_ranges)
            processed.append(new_clip)
        else:
            processed.append(clip)

    return processed


def is_outro_text(text: str) -> bool:
    """
    Detect if text contains common outro/wrap-up phrases.

    Args:
        text: Segment text to check

    Returns:
        True if text appears to be an outro segment
    """
    text_lower = text.lower().strip()

    # Check if text starts with or is primarily an outro phrase
    for phrase in OUTRO_PHRASES:
        if text_lower.startswith(phrase) or text_lower == phrase:
            return True

    # Also check for short segments that are just the phrase
    if len(text_lower) < 30:
        for phrase in OUTRO_PHRASES:
            if phrase in text_lower:
                return True

    return False


def normalize_labels(
    labels: list[dict],
    debug: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Normalize AI-generated labels before converting to clips.

    This function applies deterministic guardrails:
    1. If tags contain any CUT_FORCING_TAGS -> force action="cut"
    2. Validate tags against VALID_TAGS, drop unknown tags
    3. Track normalization changes for debugging

    Args:
        labels: List of label dicts with idx, action, tags, story_id
        debug: If True, collect debug notes about changes

    Returns:
        Tuple of (normalized_labels, debug_notes).
        debug_notes is a list of strings describing changes made.
    """
    normalized = []
    debug_notes = []

    for lbl in labels:
        idx = lbl.get("idx")
        action = lbl.get("action", "unsure")
        tags = lbl.get("tags", [])
        story_id = lbl.get("story_id", 1)

        if not isinstance(tags, list):
            tags = []

        # Step 1: Validate tags against VALID_TAGS
        valid_tags = []
        for tag in tags:
            if tag in VALID_TAGS:
                valid_tags.append(tag)
            elif debug:
                debug_notes.append(f"idx={idx}: dropped unknown tag '{tag}'")

        # Step 2: Force cut if any CUT_FORCING_TAGS present
        original_action = action
        has_cut_forcing_tag = any(tag in CUT_FORCING_TAGS for tag in valid_tags)

        if has_cut_forcing_tag and action != "cut":
            action = "cut"
            forcing_tags = [t for t in valid_tags if t in CUT_FORCING_TAGS]
            if debug:
                debug_notes.append(
                    f"idx={idx}: normalized {original_action}->{action} due to tags {forcing_tags}"
                )

        normalized.append({
            "idx": idx,
            "action": action,
            "tags": valid_tags,
            "story_id": story_id,
        })

    return normalized, debug_notes


def drop_short_bridge_ranges(clips: list[dict]) -> list[dict]:
    """
    Post-process clips to remove short "bridge" ranges at the start.

    If a clip has 2+ keep_ms ranges and the first range is < MIN_BRIDGE_RANGE_MS,
    drop that first range. This removes tiny intro bits before the main content.

    Args:
        clips: List of clip dicts with keep_ms

    Returns:
        Processed clips with short bridge ranges removed
    """
    processed = []

    for clip in clips:
        keep_ms = clip.get("keep_ms", [])

        # Only process if there are 2+ ranges
        if len(keep_ms) >= 2:
            first_range = keep_ms[0]
            first_duration = first_range[1] - first_range[0]

            if first_duration < MIN_BRIDGE_RANGE_MS:
                # Drop the first range
                new_keep_ms = keep_ms[1:]
                new_total_ms = sum(r[1] - r[0] for r in new_keep_ms)

                clip = clip.copy()
                clip["keep_ms"] = new_keep_ms
                clip["total_ms"] = new_total_ms

        processed.append(clip)

    return processed


def drop_short_leadin_ranges(
    clips: list[dict],
    labels: list[dict],
    clip_sources: list[dict],
    segments: list[dict],
) -> list[dict]:
    """
    Post-process clips to remove short lead-in ranges at the start.

    More sophisticated than drop_short_bridge_ranges - uses label context.

    Rules for dropping first range:
    1. Must have 2+ keep_ms ranges
    2. First range duration < MIN_LEADIN_RANGE_MS (2500ms)
    3. UNLESS:
       - >= 2 kept segments in that first range, OR
       - Any segment in range has "clean_story" tag

    Args:
        clips: List of clip dicts with keep_ms
        labels: Normalized labels with idx, action, tags
        clip_sources: List of clip source info with kept_segment_indexes
        segments: Original segment list

    Returns:
        Processed clips with short lead-in ranges removed
    """
    if not clips or not labels or not clip_sources:
        return clips

    # Build label map: idx -> label
    label_map = {}
    for lbl in labels:
        idx = lbl.get("idx")
        if idx is not None:
            label_map[idx] = lbl

    # Normalize segments for timing
    normalized = _normalize_segments(segments)

    processed = []
    for i, clip in enumerate(clips):
        keep_ms = clip.get("keep_ms", [])

        # Only process if there are 2+ ranges
        if len(keep_ms) < 2:
            processed.append(clip)
            continue

        first_range = keep_ms[0]
        first_duration = first_range[1] - first_range[0]

        # If first range is long enough, keep it
        if first_duration >= MIN_LEADIN_RANGE_MS:
            processed.append(clip)
            continue

        # Find clip_source for this clip
        clip_src = None
        for src in clip_sources:
            if src.get("clip_index") == i:
                clip_src = src
                break

        if not clip_src:
            processed.append(clip)
            continue

        # Find segments that fall within the first range
        kept_indices = clip_src.get("kept_segment_indexes", [])
        first_range_start = first_range[0]
        first_range_end = first_range[1]

        segments_in_first_range = []
        for idx in kept_indices:
            if idx < len(normalized):
                seg = normalized[idx]
                # Segment is in first range if it overlaps
                if seg["start_ms"] >= first_range_start and seg["start_ms"] < first_range_end:
                    segments_in_first_range.append(idx)

        # Exception 1: >= 2 kept segments in first range
        if len(segments_in_first_range) >= 2:
            processed.append(clip)
            continue

        # Exception 2: Any segment has "clean_story" tag
        has_clean_story = False
        for idx in segments_in_first_range:
            lbl = label_map.get(idx, {})
            tags = lbl.get("tags", [])
            if "clean_story" in tags:
                has_clean_story = True
                break

        if has_clean_story:
            processed.append(clip)
            continue

        # Drop the first range
        new_keep_ms = keep_ms[1:]
        new_total_ms = sum(r[1] - r[0] for r in new_keep_ms)

        new_clip = clip.copy()
        new_clip["keep_ms"] = new_keep_ms
        new_clip["total_ms"] = new_total_ms
        processed.append(new_clip)

    return processed


def trim_trailing_unsure(
    clips: list[dict],
    labels: list[dict],
    clip_sources: list[dict],
    segments: list[dict],
    min_clip_ms: int,
) -> list[dict]:
    """
    Remove trailing "unsure" segments from clips.

    For document-style clips, trailing unsure segments are often mumbling/outro
    and should be trimmed from the end to create cleaner clips.

    Args:
        clips: List of clip dicts with keep_ms
        labels: Original labels (before resolution) with action field
        clip_sources: List of clip source info with kept_segment_indexes
        segments: Original segment list
        min_clip_ms: Minimum clip duration (won't trim below this)

    Returns:
        Processed clips with trailing unsure segments removed
    """
    if not clips or not labels or not clip_sources:
        return clips

    # Build label map: idx -> original action
    label_map = {}
    for lbl in labels:
        idx = lbl.get("idx")
        if idx is not None:
            label_map[idx] = lbl.get("action", "unsure")

    # Normalize segments for timing
    normalized = _normalize_segments(segments)

    processed = []
    for i, clip in enumerate(clips):
        # Find clip_sources for this clip
        clip_src = None
        for src in clip_sources:
            if src.get("clip_index") == i:
                clip_src = src
                break

        if not clip_src:
            processed.append(clip)
            continue

        kept_indices = clip_src.get("kept_segment_indexes", [])
        if not kept_indices:
            processed.append(clip)
            continue

        # Find trailing unsure segments (from the end)
        trailing_unsure = []
        for idx in reversed(sorted(kept_indices)):
            if label_map.get(idx) == "unsure":
                trailing_unsure.append(idx)
            else:
                break  # Stop at first non-unsure segment

        if not trailing_unsure:
            processed.append(clip)
            continue

        # Remove trailing unsure from kept_indices
        trailing_set = set(trailing_unsure)
        new_kept = [idx for idx in kept_indices if idx not in trailing_set]

        if not new_kept:
            # Would remove entire clip, keep original
            processed.append(clip)
            continue

        # Rebuild keep_ms from new_kept
        sorted_new_kept = sorted(new_kept)
        keep_ranges = []
        current_start = None
        current_end = None
        prev_idx = None

        for idx in sorted_new_kept:
            if idx >= len(normalized):
                continue
            seg = normalized[idx]

            if prev_idx is None or idx != prev_idx + 1:
                # Start new range
                if current_start is not None:
                    keep_ranges.append([current_start, current_end])
                current_start = seg["start_ms"]
                current_end = seg["end_ms"]
            else:
                # Extend current range
                current_end = seg["end_ms"]

            prev_idx = idx

        # Add final range
        if current_start is not None:
            keep_ranges.append([current_start, current_end])

        if not keep_ranges:
            processed.append(clip)
            continue

        # Calculate new total
        new_total = sum(r[1] - r[0] for r in keep_ranges)

        # Skip trimming if would make clip too short
        if new_total < min_clip_ms:
            processed.append(clip)  # Keep original
            continue

        # Update clip
        new_clip = clip.copy()
        new_clip["keep_ms"] = keep_ranges
        new_clip["total_ms"] = new_total
        processed.append(new_clip)

    return processed


def build_ai_labels_prompt(
    segments: list[dict],
    markers: list[str],
) -> dict:
    """
    Build the AI prompt for segment labeling.

    Unlike build_ai_plan_prompt, this asks the LLM to label each segment
    by index rather than outputting timestamps directly.

    Args:
        segments: Transcript segments
        markers: Marker words that indicate mess-ups

    Returns:
        Dict with system_prompt, user_prompt
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

    # System prompt
    system_prompt = """You are a video editing assistant that labels transcript segments.

Your job is to classify each segment so that deterministic code can assemble clips from your labels.

## Your Task

For each segment, output:
1. **action**: "keep" | "cut" | "unsure"
2. **tags**: Array of relevant tags (can be empty)
3. **story_id**: Integer grouping segments into stories (1, 2, 3, etc.)

## Actions

- **keep**: This segment contains good content that should appear in the final video
- **cut**: This segment should be removed (mess-up, filler, repeated content, etc.)
- **unsure**: You're not confident; let the heuristic decide

## Tags (use any that apply)

- **clean_story**: Good, coherent content
- **retake_repeat**: An EARLIER attempt at something said BETTER later (cut the earlier one, keep the later)
- **filler**: Excessive "um", "uh", stammering, or dead air
- **false_start**: Incomplete thought, "Actually let me...", abandoned sentence
- **outro**: Sign-off, "thanks for watching", closing
- **intro**: Opening greeting, setup context
- **topic_shift**: This segment starts a new topic (bump story_id)
- **question**: Speaker asking something
- **tangent**: Off-topic digression
- **correction**: Speaker fixing a mistake

## Story Grouping

- Segments with the same story_id will be merged into one clip
- Increment story_id when the topic changes significantly
- Most short recordings have just 1 story
- Use story_id to split multi-topic content

## Critical Rules

1. **DO NOT output timestamps** - only segment indices and labels
2. **Retakes**: If content is repeated, mark the EARLIER version as "cut" with tag "retake_repeat", keep the LATER version
3. **Be conservative**: When unsure, use "unsure" and let heuristics decide
4. **Output valid JSON only** - no explanations outside the JSON

## Output Format

Return ONLY a JSON object like this:
{
  "labels": [
    {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
    {"idx": 1, "action": "cut", "tags": ["filler"], "story_id": 1},
    {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1}
  ]
}"""

    # User prompt
    marker_hint = ""
    if markers:
        marker_hint = f"\n\nMARKER WORDS (these indicate mess-ups/resets - segments containing these should usually be cut): {markers}"

    user_prompt = f"""Label each segment in this transcript.{marker_hint}

TRANSCRIPT SEGMENTS:
{_format_segments_for_prompt(segments_compact)}

Analyze each segment and output ONLY valid JSON with your labels:"""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "segments_compact": segments_compact,
    }


def labels_to_clips(
    labels: list[dict],
    segments: list[dict],
    max_clips: int,
    preferred_clip_type: str,
    min_clip_ms: int,
    max_clip_ms: int,
    unsure_policy: UnsurePolicy = "keep",
    debug: bool = False,
) -> tuple[list[dict], Optional[dict]]:
    """
    Convert segment labels to clips with keep_ms ranges.

    Merges consecutive "keep" segments with the same story_id into clips.

    Args:
        labels: List of {"idx": int, "action": str, "tags": list, "story_id": int}
        segments: Original segment list (normalized with start_ms/end_ms)
        max_clips: Maximum clips to generate
        preferred_clip_type: Clip type for output
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration
        unsure_policy: How to handle "unsure" segments:
            - "keep": Treat unsure as keep (conservative, default for "document")
            - "cut": Treat unsure as cut (aggressive, default for "fun")
            - "adjacent": Keep if neighbors are keep, cut if neighbors are cut
        debug: If True, include debug info in second return value

    Returns:
        Tuple of (clips, debug_info). debug_info is None if debug=False.
        debug_info contains: labels, clip_sources (kept/cut segment indices per clip)
    """
    if not labels or not segments:
        return [], None if not debug else {"labels": [], "clip_sources": []}

    # Normalize segments
    normalized = _normalize_segments(segments)

    # Build a map of idx -> label
    label_map = {}
    for lbl in labels:
        idx = lbl.get("idx")
        if idx is not None and 0 <= idx < len(normalized):
            label_map[idx] = lbl

    # Resolve "unsure" actions based on policy
    resolved_actions = {}  # idx -> resolved action ("keep" or "cut")

    for i in range(len(normalized)):
        lbl = label_map.get(i, {"action": "unsure", "tags": [], "story_id": 1})
        action = lbl.get("action", "unsure")

        if action != "unsure":
            resolved_actions[i] = action
        elif unsure_policy == "keep":
            resolved_actions[i] = "keep"
        elif unsure_policy == "cut":
            resolved_actions[i] = "cut"
        else:  # "adjacent"
            # Look at neighbors
            prev_action = None
            next_action = None

            if i > 0:
                prev_lbl = label_map.get(i - 1, {})
                prev_action = prev_lbl.get("action")
                if prev_action == "unsure":
                    prev_action = None

            if i < len(normalized) - 1:
                next_lbl = label_map.get(i + 1, {})
                next_action = next_lbl.get("action")
                if next_action == "unsure":
                    next_action = None

            # Decide based on neighbors
            if prev_action == "keep" or next_action == "keep":
                resolved_actions[i] = "keep"
            elif prev_action == "cut" or next_action == "cut":
                resolved_actions[i] = "cut"
            else:
                # No clear neighbors, default to keep (conservative)
                resolved_actions[i] = "keep"

    # Group consecutive "keep" segments by story_id
    stories = {}  # story_id -> list of segment indices
    cut_indices_all = set()  # Track all cut indices

    for i, seg in enumerate(normalized):
        lbl = label_map.get(i, {"action": "unsure", "tags": [], "story_id": 1})
        story_id = lbl.get("story_id", 1)

        if resolved_actions.get(i) == "keep":
            if story_id not in stories:
                stories[story_id] = []
            stories[story_id].append(i)
        else:
            cut_indices_all.add(i)

    # Convert each story to a clip
    clips = []
    clip_sources_list = []  # For debug output

    for story_id in sorted(stories.keys()):
        if len(clips) >= max_clips:
            break

        indices = stories[story_id]
        if not indices:
            continue

        # Track which segments were cut within this story's range
        sorted_indices = sorted(indices)
        story_start_idx = sorted_indices[0]
        story_end_idx = sorted_indices[-1]
        cut_in_story = [i for i in range(story_start_idx, story_end_idx + 1) if i not in indices]

        # Build keep_ms ranges by merging consecutive segments
        keep_ranges = []
        current_start = None
        current_end = None
        prev_idx = None

        for idx in sorted_indices:
            seg = normalized[idx]

            if prev_idx is None or idx != prev_idx + 1:
                # Start new range
                if current_start is not None:
                    keep_ranges.append([current_start, current_end])
                current_start = seg["start_ms"]
                current_end = seg["end_ms"]
            else:
                # Extend current range
                current_end = seg["end_ms"]

            prev_idx = idx

        # Add final range
        if current_start is not None:
            keep_ranges.append([current_start, current_end])

        if not keep_ranges:
            continue

        # Calculate total duration
        total_ms = sum(r[1] - r[0] for r in keep_ranges)

        # Skip if too short
        if total_ms < min_clip_ms:
            continue

        # Truncate if too long (take from the start to preserve beginning)
        if total_ms > max_clip_ms:
            keep_ranges = _truncate_ranges_to_max(keep_ranges, max_clip_ms)
            total_ms = sum(r[1] - r[0] for r in keep_ranges)

        # Generate title from first kept segment
        first_kept_idx = sorted_indices[0]
        first_text = normalized[first_kept_idx]["text"][:50]
        title = first_text.split(".")[0] if "." in first_text else first_text
        title = title.strip()
        if len(title) > 40:
            title = title[:37] + "..."

        # Collect tags from all segments in this story
        all_tags = set()
        for idx in sorted_indices:
            lbl = label_map.get(idx, {})
            for tag in lbl.get("tags", []):
                all_tags.add(tag)

        clips.append({
            "clip_id": str(uuid.uuid4()),
            "clip_type": preferred_clip_type,
            "title": title or f"Story {story_id}",
            "keep_ms": keep_ranges,
            "total_ms": total_ms,
            "reason": f"ai_labels: story {story_id}, tags: {sorted(all_tags) if all_tags else ['none']}",
            "confidence": 0.85,
        })

        # Track debug info
        if debug:
            clip_sources_list.append({
                "clip_index": len(clips) - 1,
                "story_id": story_id,
                "kept_segment_indexes": sorted_indices,
                "cut_segment_indexes": cut_in_story,
            })

    # Build debug info if requested
    debug_info = None
    if debug:
        debug_info = {
            "labels": labels,
            "clip_sources": clip_sources_list,
        }

    return clips, debug_info


def _truncate_ranges_to_max(ranges: list[list[int]], max_ms: int) -> list[list[int]]:
    """Truncate keep ranges to fit within max_ms, keeping from the start."""
    result = []
    remaining = max_ms

    for start, end in ranges:
        duration = end - start
        if duration <= remaining:
            result.append([start, end])
            remaining -= duration
        else:
            # Partial range
            if remaining > 0:
                result.append([start, start + remaining])
            break

    return result


def plan_edits_ai_labels(
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
    unsure_policy: Optional[UnsurePolicy] = None,
    debug: bool = False,
    lead_in_ms: int = DEFAULT_LEAD_IN_MS,
    tail_out_ms: int = DEFAULT_TAIL_OUT_MS,
) -> dict:
    """
    Create an edit plan using AI segment labels.

    This mode asks Claude to label each segment (keep/cut/unsure + tags + story_id),
    then deterministic code converts labels into clips.

    If the AI returns invalid JSON or labels, falls back to heuristic mode.

    Args:
        segments: Transcript segments
        max_clips: Maximum clips to generate
        clip_types: Allowed clip types
        preferred_clip_type: Preferred clip type
        markers: Marker words for mess-up detection
        clean_level: Cleanup level
        min_clip_ms: Minimum clip duration
        max_clip_ms: Maximum clip duration
        max_keep_ranges: Maximum keep ranges per clip
        enforce_segment_boundaries: Snap to boundaries
        unsure_policy: How to handle "unsure" segments. If None, defaults based on
            preferred_clip_type: "keep" for "document", "cut" for "fun", "adjacent" for "mixed"
        debug: If True, include labels and clip_sources in response meta

    Returns:
        PlanEditsResponse dict

    Raises:
        ValueError: If API key is not configured or API call fails
    """
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    # Determine default unsure_policy based on clip type if not specified
    if unsure_policy is None:
        if preferred_clip_type == "document":
            unsure_policy = "keep"
        elif preferred_clip_type == "fun":
            unsure_policy = "cut"
        else:  # "mixed" or other
            unsure_policy = "adjacent"

    if not segments:
        meta = {
            "planner": "ai_labels",
            "segments_in": 0,
            "max_clips": max_clips,
            "unsure_policy": unsure_policy,
        }
        if debug:
            meta["labels"] = []
            meta["clip_sources"] = []
        return {
            "clips": [],
            "meta": meta,
        }

    # Build the prompt
    prompt_payload = build_ai_labels_prompt(
        segments=segments,
        markers=markers,
    )

    # Call Claude API
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    logger.info(f"Calling Claude API (ai_labels mode) with {len(segments)} segments, unsure_policy={unsure_policy}")

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4096,
            system=prompt_payload["system_prompt"],
            messages=[
                {"role": "user", "content": prompt_payload["user_prompt"]}
            ],
        )

        # Extract response text
        response_text = message.content[0].text
        logger.info(f"Claude response (ai_labels): {response_text[:300]}...")

        # Parse JSON response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        # Validate response structure
        if "labels" not in result or not isinstance(result["labels"], list):
            raise ValueError("Response missing 'labels' array")

        labels = result["labels"]

        # Validate each label
        validated_labels = []
        for lbl in labels:
            if not isinstance(lbl, dict):
                continue
            if "idx" not in lbl:
                continue

            idx = lbl["idx"]
            if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
                continue

            action = lbl.get("action", "unsure")
            if action not in ("keep", "cut", "unsure"):
                action = "unsure"

            tags = lbl.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            # Filter to valid tags
            tags = [t for t in tags if t in VALID_TAGS]

            story_id = lbl.get("story_id", 1)
            if not isinstance(story_id, int) or story_id < 1:
                story_id = 1

            validated_labels.append({
                "idx": idx,
                "action": action,
                "tags": tags,
                "story_id": story_id,
            })

        if not validated_labels:
            raise ValueError("No valid labels in response")

        # Normalize segments for text lookup
        normalized_segments = _normalize_segments(segments)

        # Apply outro auto-cut: force segments with outro text to be cut
        for lbl in validated_labels:
            idx = lbl["idx"]
            if idx < len(normalized_segments):
                seg_text = normalized_segments[idx].get("text", "")
                if is_outro_text(seg_text):
                    lbl["action"] = "cut"
                    if "outro" not in lbl["tags"]:
                        lbl["tags"].append("outro")

        # Normalize labels: force cut for CUT_FORCING_TAGS, drop unknown tags
        validated_labels, normalization_notes = normalize_labels(
            labels=validated_labels,
            debug=debug,
        )

        if debug and normalization_notes:
            logger.info(f"Label normalization notes: {normalization_notes}")

        # Convert labels to clips
        # Always use debug=True internally to get clip_sources for trimming
        clips, internal_debug_info = labels_to_clips(
            labels=validated_labels,
            segments=segments,
            max_clips=max_clips,
            preferred_clip_type=preferred_clip_type,
            min_clip_ms=min_clip_ms,
            max_clip_ms=max_clip_ms,
            unsure_policy=unsure_policy,
            debug=True,  # Always true internally for post-processing
        )

        # Post-process: trim trailing unsure segments (document mode only)
        # This removes mumbling/outro at the end of clips
        if internal_debug_info and (preferred_clip_type == "document" or unsure_policy == "keep"):
            clips = trim_trailing_unsure(
                clips=clips,
                labels=validated_labels,
                clip_sources=internal_debug_info.get("clip_sources", []),
                segments=segments,
                min_clip_ms=min_clip_ms,
            )

        # Post-process: drop short lead-in ranges at start of multi-range clips
        # Uses smarter logic with label context
        if internal_debug_info:
            clips = drop_short_leadin_ranges(
                clips=clips,
                labels=validated_labels,
                clip_sources=internal_debug_info.get("clip_sources", []),
                segments=segments,
            )
        else:
            # Fallback to simple version without context
            clips = drop_short_bridge_ranges(clips)

        # Post-process: expand keep ranges with lead-in and tail-out padding
        if lead_in_ms > 0 or tail_out_ms > 0:
            clips = expand_keep_ranges(
                clips=clips,
                segments=segments,
                lead_in_ms=lead_in_ms,
                tail_out_ms=tail_out_ms,
            )

        # Set debug_info for response (only if original debug param was True)
        debug_info = internal_debug_info if debug else None

        # Validate the generated clips
        is_valid, validation_reason = validate_planned_clips(
            clips=clips,
            segments=segments,
            min_clip_ms=min_clip_ms,
            max_clip_ms=max_clip_ms,
        )

        if not is_valid:
            logger.warning(f"ai_labels clips failed validation ({validation_reason}), falling back to heuristic")
            raise ValueError(f"Clip validation failed: {validation_reason}")

        # Build response
        meta = {
            "planner": "ai_labels",
            "segments_in": len(segments),
            "max_clips": max_clips,
            "labels_count": len(validated_labels),
            "unsure_policy": unsure_policy,
        }

        # Include debug info if requested
        if debug and debug_info:
            meta["labels"] = debug_info["labels"]
            meta["clip_sources"] = debug_info["clip_sources"]
            if normalization_notes:
                meta["normalization_notes"] = normalization_notes

        return {
            "clips": clips,
            "meta": meta,
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback to heuristic mode
        logger.warning(f"ai_labels mode failed ({e}), falling back to heuristic")
        result = plan_edits_heuristic(
            segments=segments,
            max_clips=max_clips,
            clip_types=clip_types,
            preferred_clip_type=preferred_clip_type,
            markers=markers,
            clean_level=clean_level,
            min_clip_ms=min_clip_ms,
            max_clip_ms=max_clip_ms,
            max_keep_ranges=max_keep_ranges,
            enforce_segment_boundaries=enforce_segment_boundaries,
        )
        result["meta"]["planner"] = "ai_labels_fallback"
        result["meta"]["fallback_reason"] = str(e)
        result["meta"]["unsure_policy"] = unsure_policy
        return result

    except anthropic.APIError as e:
        logger.error(f"Claude API error in ai_labels mode: {e}")
        raise ValueError(f"Claude API error: {e}")
