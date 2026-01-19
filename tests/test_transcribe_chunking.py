"""Unit tests for segment chunking logic."""

import pytest

from app.transcribe import chunk_segments


# Sample segments - mix of short and long durations
SAMPLE_SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "start_ms": 0,
        "end_ms": 2500,
        "text": "Hello world.",
    },
    {
        "start": 3.0,
        "end": 10.0,
        "start_ms": 3000,
        "end_ms": 10000,
        "text": "This is a longer segment that has many words and should be split into multiple chunks when using chunked granularity.",
    },
    {
        "start": 11.0,
        "end": 13.5,
        "start_ms": 11000,
        "end_ms": 13500,
        "text": "Short segment.",
    },
]


class TestChunkSegments:
    def test_short_segments_unchanged(self):
        """Segments shorter than max_segment_ms should remain unchanged."""
        segments = [
            {"start": 0.0, "end": 2.0, "start_ms": 0, "end_ms": 2000, "text": "Short one."},
            {"start": 2.5, "end": 4.0, "start_ms": 2500, "end_ms": 4000, "text": "Short two."},
        ]
        result = chunk_segments(segments, max_segment_ms=3000)

        assert len(result) == 2
        assert result[0]["text"] == "Short one."
        assert result[1]["text"] == "Short two."

    def test_long_segment_is_split(self):
        """Segments longer than max_segment_ms should be split."""
        segments = [
            {
                "start": 0.0,
                "end": 9.0,
                "start_ms": 0,
                "end_ms": 9000,
                "text": "Word one two three four five six seven eight nine.",
            }
        ]
        result = chunk_segments(segments, max_segment_ms=3000)

        # 9000ms with 10 words = 900ms per word
        # 3000ms max = ~3 words per chunk
        assert len(result) > 1
        # Original segment should be split
        total_words = sum(len(r["text"].split()) for r in result)
        assert total_words == 10

    def test_chunked_timing_is_consistent(self):
        """Chunked segments should have proper start/end times."""
        segments = [
            {
                "start": 0.0,
                "end": 6.0,
                "start_ms": 0,
                "end_ms": 6000,
                "text": "One two three four five six.",
            }
        ]
        result = chunk_segments(segments, max_segment_ms=2000)

        # Check that chunks are contiguous
        for i in range(len(result) - 1):
            assert result[i]["end_ms"] == result[i + 1]["start_ms"]

        # First chunk starts at original start
        assert result[0]["start_ms"] == 0

        # Last chunk ends at original end
        assert result[-1]["end_ms"] == 6000

    def test_ms_and_seconds_match(self):
        """start/end should match start_ms/end_ms."""
        segments = [
            {
                "start": 0.0,
                "end": 6.0,
                "start_ms": 0,
                "end_ms": 6000,
                "text": "One two three four five six.",
            }
        ]
        result = chunk_segments(segments, max_segment_ms=2000)

        for chunk in result:
            assert chunk["start"] == round(chunk["start_ms"] / 1000, 3)
            assert chunk["end"] == round(chunk["end_ms"] / 1000, 3)

    def test_single_word_segment_unchanged(self):
        """A single-word segment cannot be split and should remain unchanged."""
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Hello"},
        ]
        result = chunk_segments(segments, max_segment_ms=2000)

        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_empty_segments_returns_empty(self):
        """Empty input returns empty output."""
        result = chunk_segments([], max_segment_ms=3000)
        assert result == []

    def test_mixed_short_and_long_segments(self):
        """Mix of short and long segments processes correctly."""
        result = chunk_segments(SAMPLE_SEGMENTS, max_segment_ms=3000)

        # First segment (2500ms) should be unchanged
        assert result[0]["text"] == "Hello world."
        assert result[0]["start_ms"] == 0
        assert result[0]["end_ms"] == 2500

        # Second segment (7000ms with ~20 words) should be split
        # Find chunks that came from the second segment (starting at 3000ms)
        second_seg_chunks = [r for r in result if 3000 <= r["start_ms"] < 10000]
        assert len(second_seg_chunks) > 1

        # Last segment (2500ms) should be unchanged
        assert result[-1]["text"] == "Short segment."
        assert result[-1]["start_ms"] == 11000
        assert result[-1]["end_ms"] == 13500

    def test_does_not_modify_original(self):
        """chunk_segments should not modify original segments."""
        segments = [
            {"start": 0.0, "end": 2.0, "start_ms": 0, "end_ms": 2000, "text": "Test."},
        ]
        original_text = segments[0]["text"]
        chunk_segments(segments, max_segment_ms=3000)
        assert segments[0]["text"] == original_text

    def test_all_text_preserved(self):
        """All text from original segments should be in chunked output."""
        segments = [
            {
                "start": 0.0,
                "end": 9.0,
                "start_ms": 0,
                "end_ms": 9000,
                "text": "The quick brown fox jumps over the lazy dog.",
            }
        ]
        result = chunk_segments(segments, max_segment_ms=3000)

        # Reconstruct text from chunks
        reconstructed = " ".join(r["text"] for r in result)
        assert reconstructed == segments[0]["text"]

    def test_exact_boundary_segment(self):
        """Segment exactly at max_segment_ms should not be split."""
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "start_ms": 0,
                "end_ms": 3000,
                "text": "Exactly at boundary.",
            }
        ]
        result = chunk_segments(segments, max_segment_ms=3000)

        assert len(result) == 1
        assert result[0]["text"] == "Exactly at boundary."

    def test_small_max_segment_ms(self):
        """Very small max_segment_ms should create many small chunks."""
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "start_ms": 0,
                "end_ms": 5000,
                "text": "One two three four five.",
            }
        ]
        # 5000ms with 5 words = 1000ms per word
        # 1000ms max = 1 word per chunk
        result = chunk_segments(segments, max_segment_ms=1000)

        assert len(result) == 5
        assert result[0]["text"] == "One"
        assert result[1]["text"] == "two"
        assert result[2]["text"] == "three"
        assert result[3]["text"] == "four"
        assert result[4]["text"] == "five."
