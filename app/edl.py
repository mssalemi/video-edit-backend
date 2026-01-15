"""
EDL (Edit Decision List) Renderer

Renders a final MP4 from multiple keep ranges by stitching together segments.
Used to cut out mess-ups by keeping only the good parts.
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

from app.settings import settings

logger = logging.getLogger(__name__)


def validate_ranges(keep_ms: list[list[int]]) -> list[tuple[int, int]]:
    """
    Validate and normalize keep ranges.

    Args:
        keep_ms: List of [start_ms, end_ms] pairs

    Returns:
        List of validated (start_ms, end_ms) tuples

    Raises:
        ValueError: If ranges are invalid
    """
    validated = []
    for i, pair in enumerate(keep_ms):
        if len(pair) != 2:
            raise ValueError(f"Range {i} must have exactly 2 elements [start_ms, end_ms]")

        start, end = pair

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"Range {i}: start and end must be numbers")

        start = int(start)
        end = int(end)

        if start < 0:
            raise ValueError(f"Range {i}: start_ms cannot be negative ({start})")

        if end <= start:
            raise ValueError(f"Range {i}: end_ms ({end}) must be greater than start_ms ({start})")

        validated.append((start, end))

    return validated


def sort_and_merge_ranges(
    ranges: list[tuple[int, int]],
    merge_gap_ms: int = 0,
) -> list[tuple[int, int]]:
    """
    Sort ranges by start time and merge overlapping/adjacent ranges.

    Args:
        ranges: List of (start_ms, end_ms) tuples
        merge_gap_ms: Merge ranges that are within this gap (ms)

    Returns:
        Sorted and merged list of (start_ms, end_ms) tuples
    """
    if not ranges:
        return []

    # Sort by start time
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]

        # Check if current range overlaps with or is close enough to merge with previous
        if start <= prev_end + merge_gap_ms:
            # Extend the previous range
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def filter_short_segments(
    ranges: list[tuple[int, int]],
    min_segment_ms: int,
) -> list[tuple[int, int]]:
    """
    Filter out segments shorter than the minimum duration.

    Args:
        ranges: List of (start_ms, end_ms) tuples
        min_segment_ms: Minimum segment duration in milliseconds

    Returns:
        Filtered list of ranges
    """
    return [
        (start, end) for start, end in ranges
        if (end - start) >= min_segment_ms
    ]


def limit_segments(
    ranges: list[tuple[int, int]],
    max_segments: int,
) -> list[tuple[int, int]]:
    """
    Limit the number of segments (keep the longest ones).

    Args:
        ranges: List of (start_ms, end_ms) tuples
        max_segments: Maximum number of segments to keep

    Returns:
        Limited list of ranges (sorted by start time)
    """
    if len(ranges) <= max_segments:
        return ranges

    # Sort by duration (longest first)
    by_duration = sorted(ranges, key=lambda x: x[1] - x[0], reverse=True)

    # Keep the longest segments
    kept = by_duration[:max_segments]

    # Re-sort by start time
    return sorted(kept, key=lambda x: x[0])


def process_ranges(
    keep_ms: list[list[int]],
    min_segment_ms: int = 500,
    merge_gap_ms: int = 100,
    max_segments: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    Process and prepare ranges for rendering.

    Validates, sorts, merges, filters, and limits ranges.

    Args:
        keep_ms: Raw list of [start_ms, end_ms] pairs
        min_segment_ms: Minimum segment duration (filter shorter)
        merge_gap_ms: Merge ranges within this gap
        max_segments: Maximum segments to keep (None = unlimited)

    Returns:
        Processed list of (start_ms, end_ms) tuples
    """
    # Validate input ranges
    validated = validate_ranges(keep_ms)

    # Sort and merge overlapping/close ranges
    merged = sort_and_merge_ranges(validated, merge_gap_ms)

    # Filter out too-short segments
    filtered = filter_short_segments(merged, min_segment_ms)

    # Limit number of segments if specified
    if max_segments is not None and max_segments > 0:
        filtered = limit_segments(filtered, max_segments)

    return filtered


def render_edl(
    input_path: str,
    keep_ms: list[list[int]],
    output_path: str,
    min_segment_ms: int = 500,
    merge_gap_ms: int = 100,
    max_segments: Optional[int] = None,
    reencode: bool = False,
) -> dict:
    """
    Render an EDL by trimming and concatenating kept ranges.

    Uses ffmpeg concat demuxer with stream copy when possible.

    Args:
        input_path: Path to input video
        keep_ms: List of [start_ms, end_ms] pairs to keep
        output_path: Path for output file
        min_segment_ms: Minimum segment duration (filter shorter)
        merge_gap_ms: Merge ranges within this gap
        max_segments: Maximum segments to keep
        reencode: Force re-encoding (default: try stream copy)

    Returns:
        Response dict with input, output, kept ranges, duration, segment count
    """
    logger.info(f"Rendering EDL: {input_path} -> {output_path}")
    logger.info(f"Input ranges: {len(keep_ms)}, min_segment_ms={min_segment_ms}, merge_gap_ms={merge_gap_ms}")

    # Process ranges
    processed = process_ranges(
        keep_ms=keep_ms,
        min_segment_ms=min_segment_ms,
        merge_gap_ms=merge_gap_ms,
        max_segments=max_segments,
    )

    if not processed:
        raise ValueError("No valid segments to render after processing")

    logger.info(f"Processed ranges: {len(processed)} segments")

    # Ensure temp directory exists
    os.makedirs(settings.TMP_DIR, exist_ok=True)

    # Create temp directory for segment files
    with tempfile.TemporaryDirectory(dir=settings.TMP_DIR) as tmp_dir:
        segment_paths = []

        # Trim each segment
        for i, (start_ms, end_ms) in enumerate(processed):
            start_s = start_ms / 1000.0
            duration_s = (end_ms - start_ms) / 1000.0
            segment_path = os.path.join(tmp_dir, f"segment_{i:03d}.ts")

            logger.info(f"Trimming segment {i}: {start_s}s for {duration_s}s -> {segment_path}")

            # Use MPEG-TS container for reliable concatenation
            if reencode:
                cmd = [
                    "ffmpeg",
                    "-ss", str(start_s),
                    "-i", input_path,
                    "-t", str(duration_s),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-f", "mpegts",
                    "-y",
                    segment_path,
                ]
            else:
                # Try stream copy first
                cmd = [
                    "ffmpeg",
                    "-ss", str(start_s),
                    "-i", input_path,
                    "-t", str(duration_s),
                    "-c", "copy",
                    "-f", "mpegts",
                    "-avoid_negative_ts", "make_zero",
                    "-y",
                    segment_path,
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                if not reencode:
                    # Fallback to re-encode
                    logger.warning(f"Stream copy failed for segment {i}, re-encoding")
                    cmd = [
                        "ffmpeg",
                        "-ss", str(start_s),
                        "-i", input_path,
                        "-t", str(duration_s),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-c:a", "aac",
                        "-f", "mpegts",
                        "-y",
                        segment_path,
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg segment trim failed: {result.stderr}")
                else:
                    raise RuntimeError(f"ffmpeg segment trim failed: {result.stderr}")

            segment_paths.append(segment_path)

        # Concatenate segments using concat protocol
        concat_input = "|".join(segment_paths)

        logger.info(f"Concatenating {len(segment_paths)} segments")

        # Use concat demuxer for reliable stitching
        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for segment_path in segment_paths:
                f.write(f"file '{segment_path}'\n")

        concat_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            "-y",
            output_path,
        ]

        result = subprocess.run(concat_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg concat failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    # Calculate total duration
    total_duration_ms = sum(end - start for start, end in processed)
    total_duration_s = round(total_duration_ms / 1000.0, 2)

    # Convert to response format
    kept_ms_response = [[start, end] for start, end in processed]

    logger.info(f"EDL render complete: {output_path}, {len(processed)} segments, {total_duration_s}s")

    return {
        "input": input_path,
        "output": output_path,
        "kept_ms": kept_ms_response,
        "duration_s": total_duration_s,
        "segments_rendered": len(processed),
    }
