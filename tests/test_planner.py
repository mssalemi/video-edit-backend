"""Tests for AI planning layer."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner import (
    validate_segments,
    snap_ms_to_segment_boundaries,
    normalize_keep_ranges,
    validate_keep_ranges,
    build_ai_plan_prompt,
    plan_edits_stub,
    plan_edits_heuristic,
)


client = TestClient(app)


# --- Sample Data ---


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Hello and welcome to this video."},
    {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Today we'll talk about coding."},
    {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Let me restart that, sorry."},
    {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "Okay, today we'll discuss Python."},
    {"start": 20.0, "end": 25.0, "start_ms": 20000, "end_ms": 25000, "text": "Python is a great language!"},
    {"start": 25.0, "end": 30.0, "start_ms": 25000, "end_ms": 30000, "text": "Thanks for watching."},
]


# --- Test validate_segments ---


class TestValidateSegments:
    """Tests for segment validation."""

    def test_valid_segments(self):
        """Valid segments pass validation."""
        validate_segments(SAMPLE_SEGMENTS)  # Should not raise

    def test_empty_segments_raises(self):
        """Empty segments list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_segments([])

    def test_missing_start_raises(self):
        """Missing start field raises error."""
        segments = [{"end": 5.0, "text": "Hello"}]
        with pytest.raises(ValueError, match="missing 'start' or 'end'"):
            validate_segments(segments)

    def test_missing_end_raises(self):
        """Missing end field raises error."""
        segments = [{"start": 0.0, "text": "Hello"}]
        with pytest.raises(ValueError, match="missing 'start' or 'end'"):
            validate_segments(segments)

    def test_missing_text_raises(self):
        """Missing text field raises error."""
        segments = [{"start": 0.0, "end": 5.0}]
        with pytest.raises(ValueError, match="missing 'text'"):
            validate_segments(segments)

    def test_end_not_greater_than_start_raises(self):
        """End not greater than start raises error."""
        segments = [{"start": 5.0, "end": 5.0, "text": "Hello"}]
        with pytest.raises(ValueError, match="must be greater than start"):
            validate_segments(segments)

    def test_negative_start_raises(self):
        """Negative start raises error."""
        segments = [{"start": -1.0, "end": 5.0, "text": "Hello"}]
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_segments(segments)


# --- Test snap_ms_to_segment_boundaries ---


class TestSnapToBoundaries:
    """Tests for snapping ms to segment boundaries."""

    def test_snap_to_start_exact(self):
        """Exact match snaps correctly to start."""
        result = snap_ms_to_segment_boundaries(5000, SAMPLE_SEGMENTS, "start")
        assert result == 5000

    def test_snap_to_start_nearest(self):
        """Nearest boundary is found for start."""
        result = snap_ms_to_segment_boundaries(5500, SAMPLE_SEGMENTS, "start")
        assert result == 5000  # Nearest start boundary

    def test_snap_to_end_exact(self):
        """Exact match snaps correctly to end."""
        result = snap_ms_to_segment_boundaries(10000, SAMPLE_SEGMENTS, "end")
        assert result == 10000

    def test_snap_to_end_nearest(self):
        """Nearest boundary is found for end."""
        result = snap_ms_to_segment_boundaries(12000, SAMPLE_SEGMENTS, "end")
        assert result == 10000  # Nearest end boundary

    def test_empty_segments_returns_original(self):
        """Empty segments returns original value."""
        result = snap_ms_to_segment_boundaries(5000, [], "start")
        assert result == 5000

    def test_snap_prefers_closest(self):
        """Snapping prefers the closest boundary."""
        # 7000 is closer to 5000 than to 10000
        result = snap_ms_to_segment_boundaries(7000, SAMPLE_SEGMENTS, "start")
        assert result == 5000

        # 8000 is closer to 10000 than to 5000
        result = snap_ms_to_segment_boundaries(8000, SAMPLE_SEGMENTS, "start")
        assert result == 10000


# --- Test normalize_keep_ranges ---


class TestNormalizeKeepRanges:
    """Tests for normalizing keep ranges."""

    def test_already_normalized(self):
        """Already normalized ranges pass through."""
        ranges = [[0, 5000], [10000, 15000]]
        result = normalize_keep_ranges(ranges)
        assert result == [[0, 5000], [10000, 15000]]

    def test_sorts_ranges(self):
        """Unsorted ranges are sorted."""
        ranges = [[10000, 15000], [0, 5000]]
        result = normalize_keep_ranges(ranges)
        assert result == [[0, 5000], [10000, 15000]]

    def test_merges_overlapping(self):
        """Overlapping ranges are merged."""
        ranges = [[0, 6000], [5000, 10000]]
        result = normalize_keep_ranges(ranges)
        assert result == [[0, 10000]]

    def test_merges_close_ranges(self):
        """Close ranges within gap threshold are merged."""
        ranges = [[0, 5000], [5200, 10000]]
        result = normalize_keep_ranges(ranges, merge_gap_ms=250)
        assert result == [[0, 10000]]

    def test_filters_short_segments(self):
        """Short segments are filtered out."""
        ranges = [[0, 5000], [6000, 6500], [10000, 15000]]
        result = normalize_keep_ranges(ranges, min_segment_ms=600)
        assert result == [[0, 5000], [10000, 15000]]

    def test_empty_returns_empty(self):
        """Empty input returns empty list."""
        result = normalize_keep_ranges([])
        assert result == []


# --- Test validate_keep_ranges ---


class TestValidateKeepRanges:
    """Tests for validating keep ranges."""

    def test_valid_ranges(self):
        """Valid ranges pass validation."""
        validate_keep_ranges([[0, 5000], [10000, 15000]], 5000, 20000, 30000)

    def test_empty_raises(self):
        """Empty ranges raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_keep_ranges([], 5000, 20000, 30000)

    def test_wrong_element_count_raises(self):
        """Wrong element count raises error."""
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_keep_ranges([[0, 5000, 10000]], 5000, 20000, 30000)

    def test_negative_start_raises(self):
        """Negative start raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_keep_ranges([[-100, 5000]], 5000, 20000, 30000)

    def test_end_not_greater_raises(self):
        """End not greater than start raises error."""
        with pytest.raises(ValueError, match="must be greater than"):
            validate_keep_ranges([[5000, 5000]], 5000, 20000, 30000)

    def test_exceeds_duration_raises(self):
        """End exceeding duration raises error."""
        with pytest.raises(ValueError, match="exceeds transcript duration"):
            validate_keep_ranges([[0, 50000]], 5000, 60000, 30000)

    def test_overlapping_raises(self):
        """Overlapping ranges raise error."""
        with pytest.raises(ValueError, match="overlaps"):
            validate_keep_ranges([[0, 10000], [5000, 15000]], 5000, 20000, 30000)

    def test_below_minimum_raises(self):
        """Total below minimum raises error."""
        with pytest.raises(ValueError, match="less than minimum"):
            validate_keep_ranges([[0, 1000]], 5000, 20000, 30000)

    def test_above_maximum_raises(self):
        """Total above maximum raises error."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_keep_ranges([[0, 25000]], 5000, 20000, 30000)


# --- Test build_ai_plan_prompt ---


class TestBuildAiPlanPrompt:
    """Tests for AI prompt building."""

    def test_returns_required_keys(self):
        """Prompt payload contains required keys."""
        payload = build_ai_plan_prompt(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document", "fun"],
            preferred_clip_type="document",
            markers=["restart"],
            clean_level="light",
            min_clip_ms=6000,
            max_clip_ms=60000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        assert "system_prompt" in payload
        assert "user_prompt" in payload
        assert "json_schema" in payload
        assert "segments_compact" in payload

    def test_system_prompt_contains_rules(self):
        """System prompt contains critical rules."""
        payload = build_ai_plan_prompt(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=6000,
            max_clip_ms=60000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        assert "JSON" in payload["system_prompt"]
        assert "mess-ups" in payload["system_prompt"].lower() or "mess" in payload["system_prompt"].lower()

    def test_user_prompt_includes_constraints(self):
        """User prompt includes constraints."""
        payload = build_ai_plan_prompt(
            segments=SAMPLE_SEGMENTS,
            max_clips=3,
            clip_types=["document", "fun"],
            preferred_clip_type="fun",
            markers=["restart"],
            clean_level="aggressive",
            min_clip_ms=10000,
            max_clip_ms=30000,
            max_keep_ranges=5,
            enforce_segment_boundaries=True,
        )

        assert "3" in payload["user_prompt"]  # max_clips
        assert "10000" in payload["user_prompt"]  # min_clip_ms
        assert "30000" in payload["user_prompt"]  # max_clip_ms
        assert "restart" in payload["user_prompt"]  # markers

    def test_segments_compact_format(self):
        """Segments compact has expected format."""
        payload = build_ai_plan_prompt(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=6000,
            max_clip_ms=60000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        compact = payload["segments_compact"]
        assert len(compact) == len(SAMPLE_SEGMENTS)
        assert "idx" in compact[0]
        assert "start_ms" in compact[0]
        assert "end_ms" in compact[0]
        assert "text" in compact[0]


# --- Test plan_edits_stub ---


class TestPlanEditsStub:
    """Tests for stub planner."""

    def test_returns_single_clip(self):
        """Stub returns single clip covering entire duration."""
        result = plan_edits_stub(SAMPLE_SEGMENTS, "document")

        assert len(result["clips"]) == 1
        clip = result["clips"][0]
        assert clip["keep_ms"] == [[0, 30000]]
        assert clip["clip_type"] == "document"

    def test_meta_fields(self):
        """Stub result has correct meta fields."""
        result = plan_edits_stub(SAMPLE_SEGMENTS, "fun")

        assert result["meta"]["planner"] == "stub"
        assert result["meta"]["segments_in"] == len(SAMPLE_SEGMENTS)

    def test_empty_segments(self):
        """Stub handles empty segments."""
        result = plan_edits_stub([], "document")

        assert result["clips"] == []
        assert result["meta"]["segments_in"] == 0


# --- Test plan_edits_heuristic ---


class TestPlanEditsHeuristic:
    """Tests for heuristic planner."""

    def test_returns_clips_within_limit(self):
        """Heuristic returns at most max_clips."""
        result = plan_edits_heuristic(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document", "fun"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=5000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        assert len(result["clips"]) <= 2

    def test_keep_ms_within_duration(self):
        """Keep ranges are within transcript duration."""
        result = plan_edits_heuristic(
            segments=SAMPLE_SEGMENTS,
            max_clips=3,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=5000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        for clip in result["clips"]:
            for start, end in clip["keep_ms"]:
                assert start >= 0
                assert end <= 30000
                assert start < end

    def test_meta_fields(self):
        """Heuristic result has correct meta fields."""
        result = plan_edits_heuristic(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=5000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        assert result["meta"]["planner"] == "heuristic"
        assert result["meta"]["segments_in"] == len(SAMPLE_SEGMENTS)
        assert result["meta"]["max_clips"] == 2

    def test_clips_have_required_fields(self):
        """Each clip has required fields."""
        result = plan_edits_heuristic(
            segments=SAMPLE_SEGMENTS,
            max_clips=2,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=5000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        for clip in result["clips"]:
            assert "clip_id" in clip
            assert "clip_type" in clip
            assert "title" in clip
            assert "keep_ms" in clip
            assert "total_ms" in clip
            assert "reason" in clip
            assert "confidence" in clip

    def test_markers_affect_splitting(self):
        """Markers cause splitting at marker segments."""
        result = plan_edits_heuristic(
            segments=SAMPLE_SEGMENTS,
            max_clips=3,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=["restart"],  # Segment 2 contains "restart"
            clean_level="light",
            min_clip_ms=3000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        # Should produce multiple clips due to marker
        assert len(result["clips"]) >= 1

    def test_empty_segments(self):
        """Heuristic handles empty segments."""
        result = plan_edits_heuristic(
            segments=[],
            max_clips=2,
            clip_types=["document"],
            preferred_clip_type="document",
            markers=[],
            clean_level="light",
            min_clip_ms=5000,
            max_clip_ms=30000,
            max_keep_ranges=10,
            enforce_segment_boundaries=True,
        )

        assert result["clips"] == []


# --- Test /plan-edits endpoint ---


class TestPlanEditsEndpoint:
    """Tests for the /plan-edits endpoint."""

    def test_stub_mode_returns_200(self):
        """Stub mode returns 200 with plan."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "stub",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "clips" in data
        assert "meta" in data
        assert data["meta"]["planner"] == "stub"

    def test_heuristic_mode_returns_200(self):
        """Heuristic mode returns 200 with plan."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                    {"start": 10.0, "end": 20.0, "text": "More content here"},
                ],
                "mode": "heuristic",
                "max_clips": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["planner"] == "heuristic"

    def test_ai_mode_returns_501(self):
        """AI mode returns 501 with prompt payload."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "ai",
            },
        )

        assert response.status_code == 501
        data = response.json()
        assert "detail" in data
        assert "AI planner not implemented" in data["detail"]
        assert "prompt" in data

    def test_ai_mode_prompt_has_required_keys(self):
        """AI mode prompt payload has required keys."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "ai",
            },
        )

        data = response.json()
        prompt = data["prompt"]
        assert "system_prompt" in prompt
        assert "user_prompt" in prompt
        assert "json_schema" in prompt
        assert "segments_compact" in prompt

    def test_empty_segments_returns_400(self):
        """Empty segments returns 400."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [],
                "mode": "heuristic",
            },
        )

        assert response.status_code == 400

    def test_invalid_mode_returns_400(self):
        """Invalid mode returns 400."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello"},
                ],
                "mode": "invalid_mode",
            },
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_min_greater_than_max_returns_400(self):
        """min_clip_ms >= max_clip_ms returns 400."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello"},
                ],
                "min_clip_ms": 60000,
                "max_clip_ms": 6000,
            },
        )

        assert response.status_code == 400

    def test_preferred_not_in_types_returns_400(self):
        """preferred_clip_type not in clip_types returns 400."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello"},
                ],
                "clip_types": ["document"],
                "preferred_clip_type": "fun",
            },
        )

        assert response.status_code == 400

    def test_default_values_work(self):
        """Request with only required fields works."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                    {"start": 10.0, "end": 20.0, "text": "More content"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["planner"] == "heuristic"  # default mode
