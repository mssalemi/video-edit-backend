"""
Clip selection logic for identifying good video segments.

Uses deterministic heuristics to select clips from transcript segments.
Supports different clip profiles: document, fun, mixed.
Supports content-aware cleanup with clean_level parameter.
Supports marker-based hard boundaries for creator control.
"""

import logging
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Type aliases
ClipType = Literal["document", "fun", "mixed"]
CleanLevel = Literal["none", "light", "aggressive"]

# Profile defaults (used when user doesn't specify values)
PROFILE_DEFAULTS = {
    "document": {
        "target_s": 30.0,
        "min_s": 15.0,
        "max_s": 60.0,
    },
    "fun": {
        "target_s": 12.0,
        "min_s": 6.0,
        "max_s": 20.0,
    },
    "mixed": {
        "target_s": 15.0,
        "min_s": 10.0,
        "max_s": 25.0,
    },
}

# Interesting words for "fun" scoring
EXPRESSIVE_WORDS = {
    "amazing", "awesome", "incredible", "unbelievable", "crazy", "insane",
    "wow", "omg", "hilarious", "epic", "legendary", "mind-blowing",
    "shocking", "surprising", "exciting", "fantastic", "brilliant",
}

# Filler words for cleanup detection (lowercase)
FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally", "honestly",
    "okay", "so", "anyway", "alright", "right", "i mean", "sort of",
    "kind of", "actually", "just", "really", "very", "pretty much",
}

# Words that indicate a "fluffy" start (lowercase)
FLUFFY_START_WORDS = {
    "okay", "so", "anyway", "alright", "um", "uh", "like", "well",
    "right", "yeah", "yes", "no", "hey", "hi", "hello",
}

# Punch words for fun punchline detection (lowercase)
PUNCH_WORDS = {
    "wow", "no way", "insane", "crazy", "bro", "wild", "unreal",
    "incredible", "amazing", "awesome", "epic", "legendary", "whoa",
    "damn", "dude", "omg", "oh my god", "what the",
}


def normalize_segment(segment: dict) -> dict:
    """Ensure segment has start_ms and end_ms fields."""
    seg = segment.copy()
    if "start_ms" not in seg:
        seg["start_ms"] = int(seg["start"] * 1000)
    if "end_ms" not in seg:
        seg["end_ms"] = int(seg["end"] * 1000)
    return seg


def compute_segment_quality(segment: dict) -> dict:
    """
    Compute quality signals for a segment.

    Returns dict with:
    - word_count: total words
    - filler_count: count of filler words/phrases
    - filler_ratio: filler_count / word_count
    - starts_fluffy: bool, starts with fluffy word
    - ends_clean: bool, ends with .!?
    - duration_s: segment duration
    """
    text = segment.get("text", "")
    text_lower = text.lower().strip()
    duration = segment["end"] - segment["start"]

    # Word count
    words = text.split()
    word_count = len(words)

    # Filler count - check for multi-word fillers first, then single words
    filler_count = 0
    temp_text = text_lower
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        if " " in filler:  # Multi-word filler
            count = temp_text.count(filler)
            filler_count += count
            temp_text = temp_text.replace(filler, "")

    # Now count single-word fillers
    remaining_words = temp_text.split()
    for word in remaining_words:
        # Strip punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in FILLER_WORDS:
            filler_count += 1

    # Filler ratio
    filler_ratio = filler_count / word_count if word_count > 0 else 0.0

    # Starts fluffy check
    first_word = ""
    if words:
        first_word = re.sub(r'[^\w]', '', words[0].lower())
    starts_fluffy = first_word in FLUFFY_START_WORDS

    # Ends clean check
    text_stripped = text.rstrip()
    ends_clean = bool(text_stripped and text_stripped[-1] in ".!?")

    return {
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 3),
        "starts_fluffy": starts_fluffy,
        "ends_clean": ends_clean,
        "duration_s": round(duration, 3),
    }


def is_bad_segment(segment: dict, quality: dict = None) -> bool:
    """
    Determine if a segment is "bad" (filler-heavy, short fluff, etc.).

    Rules:
    - filler_ratio > 0.25
    - duration < 2s AND starts_fluffy
    - word_count <= 3 AND starts_fluffy (acknowledgement-only)
    """
    if quality is None:
        quality = compute_segment_quality(segment)

    # High filler density
    if quality["filler_ratio"] > 0.25:
        return True

    # Short fluff segment
    if quality["duration_s"] < 2.0 and quality["starts_fluffy"]:
        return True

    # Acknowledgement-only segment
    if quality["word_count"] <= 3 and quality["starts_fluffy"]:
        return True

    return False


def is_marker_segment(segment: dict, markers: list[str]) -> bool:
    """Check if segment contains any marker (case-insensitive)."""
    if not markers:
        return False

    text_lower = segment.get("text", "").lower()
    for marker in markers:
        if marker.lower() in text_lower:
            return True
    return False


def is_punchy_segment(segment: dict, quality: dict = None) -> bool:
    """
    Check if segment is a "punchy" single segment good for fun clips.

    Criteria:
    - Duration between 5-12 seconds
    - Ends with ! OR contains punch words
    """
    if quality is None:
        quality = compute_segment_quality(segment)

    duration = quality["duration_s"]
    if not (5.0 <= duration <= 12.0):
        return False

    text = segment.get("text", "")
    text_lower = text.lower()

    # Ends with exclamation
    if text.rstrip().endswith("!"):
        return True

    # Contains punch words
    for punch in PUNCH_WORDS:
        if punch in text_lower:
            return True

    return False


def build_candidate_windows(
    segments: list[dict],
    max_s: float,
    max_gap_s: float,
    markers: list[str] = None,
) -> list[dict]:
    """
    Build candidate clip windows by merging adjacent segments.

    Args:
        segments: List of transcript segments (must have start, end, text)
        max_s: Maximum clip duration in seconds
        max_gap_s: Maximum gap between segments to allow merging
        markers: List of marker strings that act as hard boundaries

    Returns:
        List of candidate windows with merged segment info
    """
    if not segments:
        return []

    markers = markers or []

    # Normalize segments and compute quality
    segments = [normalize_segment(s) for s in segments]
    for seg in segments:
        seg["_quality"] = compute_segment_quality(seg)
        seg["_is_marker"] = is_marker_segment(seg, markers)

    # Sort by start time
    segments = sorted(segments, key=lambda s: s["start"])

    candidates = []

    # Try building windows starting from each segment
    for i in range(len(segments)):
        # Skip if this segment is a marker (don't start clips on markers)
        if segments[i]["_is_marker"]:
            continue

        # Start a new window from segment i
        window_start = segments[i]["start"]
        window_end = segments[i]["end"]
        window_texts = [segments[i]["text"]]
        window_segments = [segments[i]]
        total_gap = 0.0

        # Try extending the window with subsequent segments
        for j in range(i + 1, len(segments)):
            # Stop if we hit a marker segment
            if segments[j]["_is_marker"]:
                break

            gap = segments[j]["start"] - window_end

            # Check if gap is acceptable
            if gap > max_gap_s:
                break

            # Check if adding this segment exceeds max duration
            new_end = segments[j]["end"]
            if new_end - window_start > max_s:
                break

            # Extend the window
            if gap > 0:
                total_gap += gap
            window_end = new_end
            window_texts.append(segments[j]["text"])
            window_segments.append(segments[j])

        # Create candidate
        duration = window_end - window_start
        combined_text = " ".join(window_texts)

        candidates.append({
            "start": round(window_start, 3),
            "end": round(window_end, 3),
            "duration_s": round(duration, 3),
            "start_ms": int(window_start * 1000),
            "end_ms": int(window_end * 1000),
            "text": combined_text,
            "segment_count": len(window_segments),
            "total_gap_s": round(total_gap, 3),
            "_segments": window_segments,  # Keep for cleanup processing
        })

    return candidates


def compute_cleanup_penalty(
    candidate: dict,
    clean_level: CleanLevel,
) -> tuple[float, list[str], dict]:
    """
    Compute cleanup penalty and gather info for a candidate.

    Returns:
        (penalty, reasons, shift_info)
        - penalty: float to subtract from score (0-0.5)
        - reasons: list of cleanup-related reason strings
        - shift_info: dict with potential start shift info
    """
    if clean_level == "none":
        return 0.0, [], {}

    segments = candidate.get("_segments", [])
    if not segments:
        return 0.0, [], {}

    reasons = []
    penalty = 0.0

    # Count bad segments
    bad_count = 0
    bad_segments = []
    for seg in segments:
        quality = seg.get("_quality", compute_segment_quality(seg))
        if is_bad_segment(seg, quality):
            bad_count += 1
            bad_segments.append(seg)

    bad_ratio = bad_count / len(segments) if segments else 0

    # Compute total filler ratio for candidate
    total_fillers = sum(seg.get("_quality", {}).get("filler_count", 0) for seg in segments)
    total_words = sum(seg.get("_quality", {}).get("word_count", 0) for seg in segments)
    overall_filler_ratio = total_fillers / total_words if total_words > 0 else 0

    # Check if first segment is fluffy
    first_quality = segments[0].get("_quality", compute_segment_quality(segments[0]))
    starts_fluffy = first_quality["starts_fluffy"] or is_bad_segment(segments[0], first_quality)

    # Shift info: find first good segment
    shift_info = {}
    if starts_fluffy and len(segments) > 1:
        for idx, seg in enumerate(segments[1:], start=1):
            seg_quality = seg.get("_quality", compute_segment_quality(seg))
            if not is_bad_segment(seg, seg_quality) and not seg_quality["starts_fluffy"]:
                shift_info = {
                    "can_shift": True,
                    "shift_to_idx": idx,
                    "new_start": seg["start"],
                    "new_start_ms": seg["start_ms"],
                    "skipped_count": idx,
                }
                break

    # Apply penalties based on clean_level
    if clean_level == "light":
        # Mild penalties
        if starts_fluffy:
            penalty += 0.1
            reasons.append("fluffy start")
        if overall_filler_ratio > 0.15:
            penalty += min(0.15, overall_filler_ratio * 0.5)
            reasons.append("filler-heavy")
        if bad_ratio > 0.3:
            penalty += 0.1
            reasons.append("some bad segments")

    elif clean_level == "aggressive":
        # Stronger penalties
        if starts_fluffy:
            penalty += 0.2
            reasons.append("fluffy start")
        if overall_filler_ratio > 0.1:
            penalty += min(0.25, overall_filler_ratio * 0.8)
            reasons.append("filler-heavy")
        if bad_ratio > 0.2:
            penalty += 0.15
            reasons.append("bad segments")

        # Check for excessive dead air
        total_gap = candidate.get("total_gap_s", 0)
        duration = candidate.get("duration_s", 1)
        if total_gap / duration > 0.15:
            penalty += 0.1
            reasons.append("dead air")

    return min(penalty, 0.5), reasons, shift_info


def apply_start_shift(
    candidate: dict,
    shift_info: dict,
    min_s: float,
) -> dict:
    """
    Apply start shifting to skip initial fluffy segments.

    Returns new candidate with adjusted start, or original if shift not viable.
    """
    if not shift_info.get("can_shift"):
        return candidate

    new_start = shift_info["new_start"]
    new_start_ms = shift_info["new_start_ms"]
    end = candidate["end"]
    new_duration = end - new_start

    # Only shift if resulting duration is still acceptable
    if new_duration < min_s:
        return candidate

    # Create shifted candidate
    shifted = candidate.copy()
    shifted["start"] = round(new_start, 3)
    shifted["start_ms"] = new_start_ms
    shifted["duration_s"] = round(new_duration, 3)
    shifted["_shifted"] = True
    shifted["_skipped_count"] = shift_info["skipped_count"]

    # Update segments list
    shift_idx = shift_info["shift_to_idx"]
    if "_segments" in shifted:
        shifted["_segments"] = shifted["_segments"][shift_idx:]
        shifted["segment_count"] = len(shifted["_segments"])
        shifted["text"] = " ".join(s["text"] for s in shifted["_segments"])

    return shifted


def score_candidate_document(
    candidate: dict,
    target_s: float,
    clean_level: CleanLevel = "none",
) -> tuple[float, str]:
    """
    Score a candidate for 'document' profile.

    Preferences:
    - Longer durations closer to target (default 30s)
    - Penalize gaps heavily
    - Strong bonus for clean endings (.!?)
    - Prefer more words/content
    """
    duration = candidate["duration_s"]
    text = candidate["text"]
    total_gap = candidate["total_gap_s"]
    segment_count = candidate["segment_count"]

    reasons = []
    score = 0.0

    # Duration score: penalize distance from target (max 0.25 points)
    duration_diff = abs(duration - target_s)
    duration_score = max(0, 0.25 - (duration_diff / target_s) * 0.25)
    score += duration_score
    if duration_diff < 5:
        reasons.append("good duration")

    # Text density/word count score (max 0.25 points)
    word_count = len(text.split())
    words_per_second = word_count / duration if duration > 0 else 0
    text_score = min(0.25, words_per_second * 0.08)
    score += text_score
    if words_per_second > 2:
        reasons.append("dense speech")

    # Gap penalty: penalize gaps heavily (max 0.25 points)
    gap_ratio = total_gap / duration if duration > 0 else 0
    gap_score = max(0, 0.25 - gap_ratio * 0.6)
    score += gap_score
    if gap_ratio < 0.05:
        reasons.append("low gaps")

    # Segment count bonus: more content (max 0.1 points)
    segment_score = min(0.1, segment_count * 0.015)
    score += segment_score

    # Punctuation bonus: ends cleanly (max 0.15 points - stronger for document)
    text_stripped = text.rstrip()
    if text_stripped and text_stripped[-1] in ".!?":
        score += 0.15
        reasons.append("clean ending")

    # Apply cleanup penalty
    cleanup_penalty, cleanup_reasons, shift_info = compute_cleanup_penalty(candidate, clean_level)
    score -= cleanup_penalty
    reasons.extend(cleanup_reasons)

    # Check if shifted
    if candidate.get("_shifted"):
        reasons.append("shifted start")

    # Normalize score to 0-1
    score = max(0, min(1.0, score))

    # Build reason string
    clean_suffix = f" + {clean_level} cleanup" if clean_level != "none" else ""
    reason = f"document{clean_suffix}: {', '.join(reasons)}" if reasons else f"document{clean_suffix}: acceptable"

    return round(score, 2), reason


def score_candidate_fun(
    candidate: dict,
    target_s: float,
    clean_level: CleanLevel = "none",
) -> tuple[float, str]:
    """
    Score a candidate for 'fun' profile.

    Preferences:
    - Shorter durations closer to target (default 12s)
    - Penalize gaps less
    - Bonus for expressive punctuation (!)
    - Bonus for "interesting" words
    - Punchline boost for punchy single segments
    """
    duration = candidate["duration_s"]
    text = candidate["text"]
    total_gap = candidate["total_gap_s"]
    segment_count = candidate["segment_count"]

    reasons = []
    score = 0.0

    # Duration score: prefer shorter clips (max 0.25 points)
    duration_diff = abs(duration - target_s)
    duration_score = max(0, 0.25 - (duration_diff / target_s) * 0.25)
    score += duration_score
    if duration_diff < 3:
        reasons.append("snappy duration")

    # Text density (max 0.2 points)
    word_count = len(text.split())
    words_per_second = word_count / duration if duration > 0 else 0
    text_score = min(0.2, words_per_second * 0.07)
    score += text_score

    # Gap penalty: less strict (max 0.15 points)
    gap_ratio = total_gap / duration if duration > 0 else 0
    gap_score = max(0, 0.15 - gap_ratio * 0.3)
    score += gap_score

    # Expressive punctuation bonus (max 0.2 points)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    if exclamation_count > 0:
        score += min(0.15, exclamation_count * 0.05)
        reasons.append("expressive")
    if question_count > 0:
        score += min(0.05, question_count * 0.02)

    # Interesting words bonus (max 0.15 points)
    text_lower = text.lower()
    interesting_count = sum(1 for word in EXPRESSIVE_WORDS if word in text_lower)
    if interesting_count > 0:
        score += min(0.15, interesting_count * 0.05)
        reasons.append("engaging words")

    # Punchline boost: single punchy segments get significant bonus
    segments = candidate.get("_segments", [])
    if segment_count == 1 and segments:
        seg = segments[0]
        seg_quality = seg.get("_quality", compute_segment_quality(seg))
        if is_punchy_segment(seg, seg_quality):
            score += 0.25
            reasons.append("punchline")

    # Long words bonus: words > 8 chars suggest emphasis (max 0.05 points)
    words = text.split()
    long_words = sum(1 for w in words if len(w) > 8)
    if long_words > 2:
        score += min(0.05, long_words * 0.01)

    # Apply cleanup penalty
    cleanup_penalty, cleanup_reasons, shift_info = compute_cleanup_penalty(candidate, clean_level)
    score -= cleanup_penalty
    reasons.extend(cleanup_reasons)

    # Check if shifted
    if candidate.get("_shifted"):
        reasons.append("shifted start")

    # Normalize score to 0-1
    score = max(0, min(1.0, score))

    # Build reason string
    clean_suffix = f" + {clean_level} cleanup" if clean_level != "none" else ""
    reason = f"fun{clean_suffix}: {', '.join(reasons)}" if reasons else f"fun{clean_suffix}: acceptable"

    return round(score, 2), reason


def score_candidate_mixed(
    candidate: dict,
    target_s: float,
    clean_level: CleanLevel = "none",
) -> tuple[float, str]:
    """
    Score a candidate for 'mixed' profile (balanced scoring).

    Uses the original balanced scoring approach.
    """
    duration = candidate["duration_s"]
    text = candidate["text"]
    total_gap = candidate["total_gap_s"]
    segment_count = candidate["segment_count"]

    reasons = []
    score = 0.0

    # Duration score: penalize distance from target (max 0.3 points)
    duration_diff = abs(duration - target_s)
    duration_score = max(0, 0.3 - (duration_diff / target_s) * 0.3)
    score += duration_score
    if duration_diff < 2:
        reasons.append("good duration")

    # Text density score: more words = better (max 0.3 points)
    word_count = len(text.split())
    words_per_second = word_count / duration if duration > 0 else 0
    text_score = min(0.3, words_per_second * 0.1)
    score += text_score
    if words_per_second > 2:
        reasons.append("dense speech")

    # Gap penalty: less gaps = better (max 0.2 points)
    gap_ratio = total_gap / duration if duration > 0 else 0
    gap_score = max(0, 0.2 - gap_ratio * 0.4)
    score += gap_score
    if gap_ratio < 0.1:
        reasons.append("low gaps")

    # Segment count bonus: more segments merged = more content (max 0.1 points)
    segment_score = min(0.1, segment_count * 0.02)
    score += segment_score

    # Punctuation bonus: ends cleanly (max 0.1 points)
    text_stripped = text.rstrip()
    if text_stripped and text_stripped[-1] in ".!?":
        score += 0.1
        reasons.append("clean ending")

    # Apply cleanup penalty
    cleanup_penalty, cleanup_reasons, shift_info = compute_cleanup_penalty(candidate, clean_level)
    score -= cleanup_penalty
    reasons.extend(cleanup_reasons)

    # Check if shifted
    if candidate.get("_shifted"):
        reasons.append("shifted start")

    # Normalize score to 0-1
    score = max(0, min(1.0, score))

    # Build reason string
    clean_suffix = f" + {clean_level} cleanup" if clean_level != "none" else ""
    reason = f"mixed{clean_suffix}: {', '.join(reasons)}" if reasons else f"mixed{clean_suffix}: acceptable"

    return round(score, 2), reason


def score_candidate(
    candidate: dict,
    target_s: float,
    clip_type: ClipType = "mixed",
    clean_level: CleanLevel = "none",
) -> tuple[float, str]:
    """
    Score a candidate clip window based on clip type and cleanup level.

    Args:
        candidate: Candidate window dict
        target_s: Target duration in seconds
        clip_type: Profile to use for scoring
        clean_level: Cleanup level for content-aware filtering

    Returns:
        Tuple of (score, reason_string)
    """
    if clip_type == "document":
        return score_candidate_document(candidate, target_s, clean_level)
    elif clip_type == "fun":
        return score_candidate_fun(candidate, target_s, clean_level)
    else:
        return score_candidate_mixed(candidate, target_s, clean_level)


def _select_clips_single_profile(
    segments: list[dict],
    target_s: float,
    min_s: float,
    max_s: float,
    max_clips: int,
    max_gap_s: float,
    clip_type: ClipType,
    clean_level: CleanLevel = "none",
    markers: list[str] = None,
) -> list[dict]:
    """
    Internal: Select clips using a single profile.
    """
    if not segments:
        return []

    markers = markers or []

    # Build all candidate windows (respecting markers)
    candidates = build_candidate_windows(segments, max_s, max_gap_s, markers)

    # Filter by min/max duration
    valid_candidates = [
        c for c in candidates
        if min_s <= c["duration_s"] <= max_s
    ]

    # For aggressive cleanup, filter out candidates with too many bad segments
    if clean_level == "aggressive" and valid_candidates:
        filtered = []
        for c in valid_candidates:
            segs = c.get("_segments", [])
            if segs:
                bad_count = sum(1 for s in segs if is_bad_segment(s, s.get("_quality")))
                bad_ratio = bad_count / len(segs)
                # Reject if > 50% bad segments
                if bad_ratio <= 0.5:
                    filtered.append(c)
        # Only use filtered if we have results
        if filtered:
            valid_candidates = filtered

    # If no valid candidates, use best-effort fallback
    if not valid_candidates:
        valid_candidates = _best_effort_fallback(candidates, segments, max_s)

    # Apply start shifting for light/aggressive cleanup
    if clean_level in ("light", "aggressive"):
        shifted_candidates = []
        for c in valid_candidates:
            _, _, shift_info = compute_cleanup_penalty(c, clean_level)
            if shift_info.get("can_shift"):
                shifted = apply_start_shift(c, shift_info, min_s)
                shifted_candidates.append(shifted)
            else:
                shifted_candidates.append(c)
        valid_candidates = shifted_candidates

    # Score candidates with the specified profile
    scored = []
    for c in valid_candidates:
        score, reason = score_candidate(c, target_s, clip_type, clean_level)
        clip = {
            "start": c["start"],
            "end": c["end"],
            "duration_s": c["duration_s"],
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "score": score,
            "reason": reason,
        }
        scored.append(clip)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Remove overlapping clips (greedy: keep higher scored)
    selected = _remove_overlaps(scored, max_clips)

    return selected


def select_clips(
    segments: list[dict],
    target_s: Optional[float] = None,
    min_s: Optional[float] = None,
    max_s: Optional[float] = None,
    max_clips: int = 3,
    max_gap_s: float = 1.2,
    clip_type: ClipType = "mixed",
    clean_level: CleanLevel = "light",
    markers: list[str] = None,
) -> list[dict]:
    """
    Select the best clip windows from transcript segments.

    Args:
        segments: List of transcript segments
        target_s: Target clip duration (uses profile default if None)
        min_s: Minimum clip duration (uses profile default if None)
        max_s: Maximum clip duration (uses profile default if None)
        max_clips: Maximum number of clips to return
        max_gap_s: Maximum gap between segments for merging
        clip_type: Selection profile ("document", "fun", or "mixed")
        clean_level: Cleanup level ("none", "light", or "aggressive")
        markers: List of marker strings that act as hard boundaries

    Returns:
        List of selected clips with scores
    """
    # Apply profile defaults for any unspecified values
    defaults = PROFILE_DEFAULTS[clip_type]
    target_s = target_s if target_s is not None else defaults["target_s"]
    min_s = min_s if min_s is not None else defaults["min_s"]
    max_s = max_s if max_s is not None else defaults["max_s"]
    markers = markers or []

    logger.info(
        f"Selecting clips: clip_type={clip_type}, clean_level={clean_level}, "
        f"target={target_s}s, min={min_s}s, max={max_s}s, max_clips={max_clips}, "
        f"markers={len(markers)}"
    )

    if not segments:
        logger.warning("No segments provided")
        return []

    # For "mixed" with max_clips >= 2, try to get variety
    if clip_type == "mixed" and max_clips >= 2:
        return _select_mixed_variety(
            segments, target_s, min_s, max_s, max_clips, max_gap_s,
            clean_level, markers
        )

    # Single profile selection
    candidates = build_candidate_windows(segments, max_s, max_gap_s, markers)
    logger.info(f"Built {len(candidates)} candidate windows")

    valid_candidates = [
        c for c in candidates
        if min_s <= c["duration_s"] <= max_s
    ]
    logger.info(f"Found {len(valid_candidates)} candidates within duration bounds")

    # For aggressive cleanup, filter out high-bad-ratio candidates
    if clean_level == "aggressive" and valid_candidates:
        filtered = []
        for c in valid_candidates:
            segs = c.get("_segments", [])
            if segs:
                bad_count = sum(1 for s in segs if is_bad_segment(s, s.get("_quality")))
                bad_ratio = bad_count / len(segs)
                if bad_ratio <= 0.5:
                    filtered.append(c)
        if filtered:
            valid_candidates = filtered
            logger.info(f"Aggressive filter: {len(valid_candidates)} candidates remain")

    if not valid_candidates:
        logger.warning("No candidates within bounds, using best-effort fallback")
        valid_candidates = _best_effort_fallback(candidates, segments, max_s)

    # Apply start shifting for light/aggressive cleanup
    if clean_level in ("light", "aggressive"):
        shifted_candidates = []
        for c in valid_candidates:
            _, _, shift_info = compute_cleanup_penalty(c, clean_level)
            if shift_info.get("can_shift"):
                shifted = apply_start_shift(c, shift_info, min_s)
                shifted_candidates.append(shifted)
            else:
                shifted_candidates.append(c)
        valid_candidates = shifted_candidates

    # Score candidates
    scored = []
    for c in valid_candidates:
        score, reason = score_candidate(c, target_s, clip_type, clean_level)
        clip = {
            "start": c["start"],
            "end": c["end"],
            "duration_s": c["duration_s"],
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "score": score,
            "reason": reason,
        }
        scored.append(clip)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Remove overlapping clips
    selected = _remove_overlaps(scored, max_clips)

    logger.info(f"Selected {len(selected)} clips")
    return selected


def _select_mixed_variety(
    segments: list[dict],
    target_s: float,
    min_s: float,
    max_s: float,
    max_clips: int,
    max_gap_s: float,
    clean_level: CleanLevel = "none",
    markers: list[str] = None,
) -> list[dict]:
    """
    Select clips with variety: try to include both document-ish and fun-ish clips.

    Strategy:
    1. Get top clip from document profile
    2. Get top clip from fun profile
    3. Dedupe overlaps
    4. If more clips needed, fill with mixed profile
    """
    logger.info("Using mixed variety selection")
    markers = markers or []

    # Get document-style clips (use document defaults for this sub-selection)
    doc_defaults = PROFILE_DEFAULTS["document"]
    doc_clips = _select_clips_single_profile(
        segments,
        target_s=doc_defaults["target_s"],
        min_s=min_s,  # Use user's min if specified
        max_s=max(max_s, doc_defaults["max_s"]),  # Allow longer for document
        max_clips=1,
        max_gap_s=max_gap_s,
        clip_type="document",
        clean_level=clean_level,
        markers=markers,
    )

    # Get fun-style clips (use fun defaults for this sub-selection)
    fun_defaults = PROFILE_DEFAULTS["fun"]
    fun_clips = _select_clips_single_profile(
        segments,
        target_s=fun_defaults["target_s"],
        min_s=min(min_s, fun_defaults["min_s"]),  # Allow shorter for fun
        max_s=max_s,
        max_clips=1,
        max_gap_s=max_gap_s,
        clip_type="fun",
        clean_level=clean_level,
        markers=markers,
    )

    # Combine and dedupe
    combined = doc_clips + fun_clips
    selected = _remove_overlaps(combined, max_clips)

    # If we need more clips, get additional with mixed profile
    if len(selected) < max_clips:
        remaining = max_clips - len(selected)
        mixed_clips = _select_clips_single_profile(
            segments,
            target_s=target_s,
            min_s=min_s,
            max_s=max_s,
            max_clips=remaining + 2,  # Get extras to account for overlaps
            max_gap_s=max_gap_s,
            clip_type="mixed",
            clean_level=clean_level,
            markers=markers,
        )

        # Add non-overlapping mixed clips
        for clip in mixed_clips:
            if len(selected) >= max_clips:
                break
            overlaps = any(
                clip["start"] < s["end"] and s["start"] < clip["end"]
                for s in selected
            )
            if not overlaps:
                selected.append(clip)

    # Re-sort by score
    selected.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"Mixed variety selected {len(selected)} clips")
    return selected


def _best_effort_fallback(
    candidates: list[dict],
    segments: list[dict],
    max_s: float,
) -> list[dict]:
    """
    Fallback when no candidates meet min_s/max_s bounds.

    Returns the longest merged chunk <= max_s, or the single longest segment.
    """
    # Try longest candidate <= max_s
    valid = [c for c in candidates if c["duration_s"] <= max_s]
    if valid:
        valid.sort(key=lambda x: x["duration_s"], reverse=True)
        return [valid[0]]

    # Fallback to single longest segment
    if segments:
        segments = [normalize_segment(s) for s in segments]
        longest = max(segments, key=lambda s: s["end"] - s["start"])
        duration = longest["end"] - longest["start"]
        return [{
            "start": round(longest["start"], 3),
            "end": round(longest["end"], 3),
            "duration_s": round(duration, 3),
            "start_ms": longest["start_ms"],
            "end_ms": longest["end_ms"],
            "text": longest["text"],
            "segment_count": 1,
            "total_gap_s": 0.0,
            "_segments": [longest],
        }]

    return []


def _remove_overlaps(clips: list[dict], max_clips: int) -> list[dict]:
    """
    Remove overlapping clips, keeping higher-scored ones.
    Greedy algorithm: iterate through scored clips, skip if overlaps with selected.
    """
    selected = []

    for clip in clips:
        if len(selected) >= max_clips:
            break

        # Check for overlap with already selected clips
        overlaps = False
        for s in selected:
            # Two intervals overlap if: start1 < end2 AND start2 < end1
            if clip["start"] < s["end"] and s["start"] < clip["end"]:
                overlaps = True
                break

        if not overlaps:
            selected.append(clip)

    return selected
