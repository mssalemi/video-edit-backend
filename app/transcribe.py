import logging
import os
import subprocess
import tempfile
from typing import Optional

from faster_whisper import WhisperModel

from app.settings import settings

logger = logging.getLogger(__name__)

# Module-level cache for loaded models
_model_cache: dict[str, WhisperModel] = {}


def get_model(model_name: str) -> WhisperModel:
    """Get or load a whisper model (lazy loading with caching)."""
    if model_name not in _model_cache:
        logger.info(f"Loading model '{model_name}' (first use, will be cached)")
        _model_cache[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
        logger.info(f"Model '{model_name}' loaded successfully")
    return _model_cache[model_name]


def extract_audio(input_path: str, output_wav_path: str) -> None:
    """Extract mono 16kHz WAV audio from input media using ffmpeg."""
    logger.info(f"Extracting audio from '{input_path}' to '{output_wav_path}'")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        "-y",                     # Overwrite output
        output_wav_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr}")
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

    logger.info("Audio extraction complete")


def transcribe_audio(
    wav_path: str,
    model_name: str,
    language: Optional[str] = None,
) -> tuple[str, list[dict], Optional[str], Optional[float]]:
    """
    Transcribe audio using faster-whisper.

    Returns:
        tuple of (full_text, segments, detected_language, duration_seconds)
    """
    logger.info(f"Transcribing '{wav_path}' with model '{model_name}', language={language or 'auto'}")

    model = get_model(model_name)

    segments_iter, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=5,
    )

    segments = []
    full_text_parts = []

    for segment in segments_iter:
        start_sec = round(segment.start, 3)
        end_sec = round(segment.end, 3)
        segments.append({
            "start": start_sec,
            "end": end_sec,
            "start_ms": int(start_sec * 1000),
            "end_ms": int(end_sec * 1000),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

    full_text = " ".join(full_text_parts)
    detected_language = info.language if info else None
    duration = info.duration if info else None

    logger.info(f"Transcription complete: {len(segments)} segments, {len(full_text)} chars")

    return full_text, segments, detected_language, duration


def chunk_segments(
    segments: list[dict],
    max_segment_ms: int,
) -> list[dict]:
    """
    Split long segments into smaller chunks.

    Any segment with duration_ms > max_segment_ms is split by words
    into subsegments that respect the max duration.

    Args:
        segments: List of segment dicts with start, end, start_ms, end_ms, text
        max_segment_ms: Maximum segment duration in milliseconds

    Returns:
        New list of segments with long segments chunked
    """
    chunked = []

    for seg in segments:
        duration_ms = seg["end_ms"] - seg["start_ms"]

        # If segment is short enough, keep as-is
        if duration_ms <= max_segment_ms:
            chunked.append(seg.copy())
            continue

        # Split long segment by words
        text = seg["text"]
        words = text.split()

        if len(words) <= 1:
            # Can't split single word, keep as-is
            chunked.append(seg.copy())
            continue

        # Calculate approximate ms per word
        ms_per_word = duration_ms / len(words)

        # Determine how many words fit in max_segment_ms
        words_per_chunk = max(1, int(max_segment_ms / ms_per_word))

        # Split into chunks
        start_ms = seg["start_ms"]
        word_idx = 0

        while word_idx < len(words):
            chunk_words = words[word_idx : word_idx + words_per_chunk]
            chunk_word_count = len(chunk_words)

            # Calculate chunk timing
            chunk_duration_ms = int(chunk_word_count * ms_per_word)
            end_ms = start_ms + chunk_duration_ms

            # Clamp end_ms to segment end
            if end_ms > seg["end_ms"] or word_idx + chunk_word_count >= len(words):
                end_ms = seg["end_ms"]

            chunk_text = " ".join(chunk_words)

            chunked.append({
                "start": round(start_ms / 1000, 3),
                "end": round(end_ms / 1000, 3),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": chunk_text,
            })

            start_ms = end_ms
            word_idx += words_per_chunk

    return chunked


def transcribe_media(
    input_path: str,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    granularity: Optional[str] = None,
) -> dict:
    """
    Full transcription pipeline: extract audio and transcribe.

    Args:
        input_path: Path to media file
        model_name: Whisper model name (default from settings)
        language: Language code or None for auto-detect
        granularity: "default" or "chunked" - if chunked, splits long segments

    Returns:
        Response dict with text, segments, and metadata
    """
    model_name = model_name or settings.DEFAULT_MODEL
    granularity = granularity or "default"

    # Ensure temp directory exists
    os.makedirs(settings.TMP_DIR, exist_ok=True)

    # Create temp file for extracted audio
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=settings.TMP_DIR,
        delete=False,
    ) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        # Extract audio
        extract_audio(input_path, wav_path)

        # Transcribe
        full_text, segments, detected_language, duration = transcribe_audio(
            wav_path,
            model_name,
            language,
        )

        # Apply chunking if requested
        if granularity == "chunked":
            segments = chunk_segments(segments, settings.MAX_SEGMENT_MS)

        return {
            "text": full_text,
            "segments": segments,
            "meta": {
                "language": detected_language,
                "duration_s": round(duration, 2) if duration else None,
                "engine": "faster-whisper",
                "model": model_name,
                "granularity": granularity,
            },
        }
    finally:
        # Cleanup temp file
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logger.debug(f"Cleaned up temp file: {wav_path}")


def trim_video(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
) -> dict:
    """
    Trim a video using ffmpeg.

    Attempts fast stream copy first, falls back to re-encoding if needed.

    Args:
        input_path: Path to input video
        start: Start time in seconds
        end: End time in seconds
        output_path: Path for output file

    Returns:
        Response dict with input, output, timing info
    """
    duration = round(end - start, 3)
    logger.info(f"Trimming video: {input_path} [{start}s - {end}s] -> {output_path}")

    # Try fast stream copy first
    cmd_copy = [
        "ffmpeg",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",             # Stream copy (fast, no re-encoding)
        "-avoid_negative_ts", "make_zero",
        "-y",                     # Overwrite output
        output_path,
    ]

    logger.info(f"Attempting fast trim (stream copy): {' '.join(cmd_copy)}")
    result = subprocess.run(cmd_copy, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning(f"Stream copy failed, falling back to re-encode: {result.stderr}")

        # Fallback: re-encode
        cmd_reencode = [
            "ffmpeg",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-y",
            output_path,
        ]

        logger.info(f"Re-encoding trim: {' '.join(cmd_reencode)}")
        result = subprocess.run(cmd_reencode, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg trim failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg trim failed: {result.stderr}")

    logger.info(f"Trim complete: {output_path}")

    return {
        "input": input_path,
        "output": output_path,
        "start": start,
        "end": end,
        "duration_s": round(duration, 2),
    }
