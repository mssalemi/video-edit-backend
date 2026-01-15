"""Tests for EDL (Edit Decision List) rendering logic."""

import pytest

from app.edl import (
    validate_ranges,
    sort_and_merge_ranges,
    filter_short_segments,
    limit_segments,
    process_ranges,
)


# --- Test validate_ranges ---


class TestValidateRanges:
    """Tests for range validation."""

    def test_valid_single_range(self):
        """Single valid range passes."""
        result = validate_ranges([[0, 1000]])
        assert result == [(0, 1000)]

    def test_valid_multiple_ranges(self):
        """Multiple valid ranges pass."""
        result = validate_ranges([[0, 1000], [2000, 3000], [5000, 6000]])
        assert result == [(0, 1000), (2000, 3000), (5000, 6000)]

    def test_converts_floats_to_ints(self):
        """Float values are converted to integers."""
        result = validate_ranges([[0.5, 1000.7]])
        assert result == [(0, 1000)]

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = validate_ranges([])
        assert result == []

    def test_rejects_wrong_element_count(self):
        """Raises error for range with wrong element count."""
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            validate_ranges([[0, 1000, 2000]])

    def test_rejects_single_element(self):
        """Raises error for range with single element."""
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            validate_ranges([[1000]])

    def test_rejects_negative_start(self):
        """Raises error for negative start time."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_ranges([[-100, 1000]])

    def test_rejects_end_equal_to_start(self):
        """Raises error when end equals start."""
        with pytest.raises(ValueError, match="must be greater than"):
            validate_ranges([[1000, 1000]])

    def test_rejects_end_less_than_start(self):
        """Raises error when end is less than start."""
        with pytest.raises(ValueError, match="must be greater than"):
            validate_ranges([[2000, 1000]])

    def test_rejects_non_numeric_values(self):
        """Raises error for non-numeric values."""
        with pytest.raises(ValueError, match="must be numbers"):
            validate_ranges([["foo", 1000]])


# --- Test sort_and_merge_ranges ---


class TestSortAndMergeRanges:
    """Tests for sorting and merging ranges."""

    def test_already_sorted_no_overlap(self):
        """Non-overlapping sorted ranges pass through."""
        ranges = [(0, 1000), (2000, 3000), (5000, 6000)]
        result = sort_and_merge_ranges(ranges)
        assert result == [(0, 1000), (2000, 3000), (5000, 6000)]

    def test_unsorted_ranges_get_sorted(self):
        """Unsorted ranges are sorted by start time."""
        ranges = [(5000, 6000), (0, 1000), (2000, 3000)]
        result = sort_and_merge_ranges(ranges)
        assert result == [(0, 1000), (2000, 3000), (5000, 6000)]

    def test_overlapping_ranges_merge(self):
        """Overlapping ranges are merged."""
        ranges = [(0, 2000), (1500, 3000)]
        result = sort_and_merge_ranges(ranges)
        assert result == [(0, 3000)]

    def test_adjacent_ranges_do_not_merge_by_default(self):
        """Adjacent ranges (end == start) don't merge with gap=0."""
        ranges = [(0, 1000), (1000, 2000)]
        result = sort_and_merge_ranges(ranges, merge_gap_ms=0)
        assert result == [(0, 2000)]  # Actually they touch, so they merge

    def test_close_ranges_merge_with_gap(self):
        """Ranges within merge_gap_ms are merged."""
        ranges = [(0, 1000), (1050, 2000)]
        result = sort_and_merge_ranges(ranges, merge_gap_ms=100)
        assert result == [(0, 2000)]

    def test_distant_ranges_do_not_merge(self):
        """Ranges farther than merge_gap_ms stay separate."""
        ranges = [(0, 1000), (2000, 3000)]
        result = sort_and_merge_ranges(ranges, merge_gap_ms=100)
        assert result == [(0, 1000), (2000, 3000)]

    def test_multiple_overlapping_ranges(self):
        """Multiple overlapping ranges merge into one."""
        ranges = [(0, 1000), (500, 1500), (1200, 2000), (1800, 2500)]
        result = sort_and_merge_ranges(ranges)
        assert result == [(0, 2500)]

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = sort_and_merge_ranges([])
        assert result == []

    def test_single_range_unchanged(self):
        """Single range passes through unchanged."""
        result = sort_and_merge_ranges([(1000, 2000)])
        assert result == [(1000, 2000)]

    def test_contained_range_merges(self):
        """Range contained within another merges."""
        ranges = [(0, 5000), (1000, 2000)]
        result = sort_and_merge_ranges(ranges)
        assert result == [(0, 5000)]


# --- Test filter_short_segments ---


class TestFilterShortSegments:
    """Tests for filtering short segments."""

    def test_filters_short_segments(self):
        """Segments shorter than min are filtered."""
        ranges = [(0, 1000), (2000, 2300), (3000, 4000)]
        result = filter_short_segments(ranges, min_segment_ms=500)
        assert result == [(0, 1000), (3000, 4000)]

    def test_keeps_segments_at_minimum(self):
        """Segments exactly at minimum are kept."""
        ranges = [(0, 500), (1000, 2000)]
        result = filter_short_segments(ranges, min_segment_ms=500)
        assert result == [(0, 500), (1000, 2000)]

    def test_keeps_segments_above_minimum(self):
        """Segments above minimum are kept."""
        ranges = [(0, 1000), (2000, 4000)]
        result = filter_short_segments(ranges, min_segment_ms=500)
        assert result == [(0, 1000), (2000, 4000)]

    def test_all_filtered_returns_empty(self):
        """Returns empty if all segments are too short."""
        ranges = [(0, 100), (200, 300)]
        result = filter_short_segments(ranges, min_segment_ms=500)
        assert result == []

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = filter_short_segments([], min_segment_ms=500)
        assert result == []


# --- Test limit_segments ---


class TestLimitSegments:
    """Tests for limiting number of segments."""

    def test_no_limit_needed(self):
        """When segments <= max, all are kept."""
        ranges = [(0, 1000), (2000, 3000)]
        result = limit_segments(ranges, max_segments=5)
        assert result == [(0, 1000), (2000, 3000)]

    def test_limit_keeps_longest(self):
        """When limiting, keeps longest segments."""
        ranges = [(0, 1000), (2000, 5000), (6000, 6500)]  # durations: 1000, 3000, 500
        result = limit_segments(ranges, max_segments=2)
        # Should keep (2000, 5000) and (0, 1000), sorted by start
        assert result == [(0, 1000), (2000, 5000)]

    def test_limit_maintains_order(self):
        """Result is sorted by start time, not duration."""
        ranges = [(5000, 6000), (0, 4000), (7000, 8000)]  # durations: 1000, 4000, 1000
        result = limit_segments(ranges, max_segments=2)
        # Keeps longest two: (0, 4000) and one of the 1000ms ones
        assert result[0] == (0, 4000)
        assert len(result) == 2

    def test_limit_to_one(self):
        """Can limit to single segment."""
        ranges = [(0, 1000), (2000, 5000), (6000, 7000)]
        result = limit_segments(ranges, max_segments=1)
        assert result == [(2000, 5000)]  # longest

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = limit_segments([], max_segments=5)
        assert result == []


# --- Test process_ranges (integration) ---


class TestProcessRanges:
    """Tests for the full range processing pipeline."""

    def test_basic_processing(self):
        """Basic processing works end-to-end."""
        keep_ms = [[0, 1000], [2000, 3000]]
        result = process_ranges(keep_ms)
        assert result == [(0, 1000), (2000, 3000)]

    def test_sorts_and_merges(self):
        """Unsorted overlapping ranges are processed correctly."""
        keep_ms = [[2000, 3000], [0, 1500], [1000, 2500]]
        result = process_ranges(keep_ms)
        assert result == [(0, 3000)]

    def test_filters_short_after_merge(self):
        """Short segments are filtered after merging."""
        keep_ms = [[0, 1000], [5000, 5200]]  # Second is too short
        result = process_ranges(keep_ms, min_segment_ms=500)
        assert result == [(0, 1000)]

    def test_limits_segments(self):
        """Segments are limited to max_segments."""
        keep_ms = [[0, 1000], [2000, 4000], [5000, 5500]]
        result = process_ranges(keep_ms, max_segments=2, min_segment_ms=100)
        assert len(result) == 2

    def test_merge_gap_parameter(self):
        """merge_gap_ms parameter is respected."""
        keep_ms = [[0, 1000], [1100, 2000]]
        # Without enough gap, they stay separate
        result = process_ranges(keep_ms, merge_gap_ms=50)
        assert result == [(0, 1000), (1100, 2000)]
        # With enough gap, they merge
        result = process_ranges(keep_ms, merge_gap_ms=150)
        assert result == [(0, 2000)]

    def test_full_pipeline(self):
        """Full pipeline with all options."""
        keep_ms = [
            [10000, 12000],  # 2000ms
            [0, 500],        # 500ms (at min)
            [5000, 5100],    # 100ms (too short, will be filtered)
            [11500, 13000],  # Overlaps with first, merges to (10000, 13000) = 3000ms
            [3000, 4500],    # 1500ms
        ]
        result = process_ranges(
            keep_ms,
            min_segment_ms=500,
            merge_gap_ms=100,
            max_segments=2,
        )
        # After sort: [(0, 500), (3000, 4500), (5000, 5100), (10000, 12000), (11500, 13000)]
        # After merge with gap=100: [(0, 500), (3000, 4500), (5000, 5100), (10000, 13000)]
        # After filter min=500: [(0, 500), (3000, 4500), (10000, 13000)]
        # After limit to 2 (keep longest): [(3000, 4500), (10000, 13000)]
        assert len(result) == 2
        assert (10000, 13000) in result  # longest at 3000ms
        assert (3000, 4500) in result    # second longest at 1500ms


# --- Test edge cases ---


class TestEdgeCases:
    """Edge case tests."""

    def test_single_range_processing(self):
        """Single range processes correctly."""
        result = process_ranges([[1000, 5000]])
        assert result == [(1000, 5000)]

    def test_zero_start_time(self):
        """Range starting at 0 is valid."""
        result = process_ranges([[0, 1000]])
        assert result == [(0, 1000)]

    def test_large_timestamps(self):
        """Large timestamps (e.g., 1 hour) work."""
        # 1 hour = 3600000 ms
        result = process_ranges([[0, 1000], [3600000, 3601000]])
        assert result == [(0, 1000), (3600000, 3601000)]

    def test_tiny_gap_merging(self):
        """Very small gaps can be merged."""
        keep_ms = [[0, 1000], [1001, 2000]]
        result = process_ranges(keep_ms, merge_gap_ms=5)
        assert result == [(0, 2000)]

    def test_exact_boundary_overlap(self):
        """Ranges that touch exactly at boundary merge."""
        keep_ms = [[0, 1000], [1000, 2000]]
        result = process_ranges(keep_ms, merge_gap_ms=0)
        assert result == [(0, 2000)]

    def test_no_max_segments_unlimited(self):
        """None max_segments means unlimited."""
        keep_ms = [[i * 2000, i * 2000 + 1000] for i in range(10)]
        result = process_ranges(keep_ms, max_segments=None)
        assert len(result) == 10

    def test_max_segments_zero_treated_as_unlimited(self):
        """max_segments=0 is treated as unlimited by limit_segments."""
        # Note: process_ranges only calls limit_segments if max_segments > 0
        keep_ms = [[0, 1000], [2000, 3000]]
        result = process_ranges(keep_ms, max_segments=0)
        assert len(result) == 2  # No limiting applied


# --- Test validation errors ---


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_process_ranges_with_invalid_input(self):
        """process_ranges raises on invalid input."""
        with pytest.raises(ValueError):
            process_ranges([[1000, 500]])  # end < start

    def test_empty_after_filtering(self):
        """Empty result after filtering is valid."""
        result = process_ranges([[0, 100]], min_segment_ms=500)
        assert result == []
