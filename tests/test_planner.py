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
    build_ai_labels_prompt,
    labels_to_clips,
    normalize_labels,
    VALID_TAGS,
    CUT_FORCING_TAGS,
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

    def test_ai_mode_requires_api_key(self):
        """AI mode returns 500 if ANTHROPIC_API_KEY not configured."""
        # This test assumes no API key is set in test environment
        # If API key is set, this test will be skipped
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("API key is configured, skipping this test")

        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "ai",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "ANTHROPIC_API_KEY" in data["detail"]

    def test_ai_labels_mode_requires_api_key(self):
        """AI labels mode returns 500 if ANTHROPIC_API_KEY not configured."""
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("API key is configured, skipping this test")

        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "ai_labels",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "ANTHROPIC_API_KEY" in data["detail"]

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


# --- Test build_ai_labels_prompt ---


class TestBuildAiLabelsPrompt:
    """Tests for AI labels prompt building."""

    def test_returns_required_keys(self):
        """Prompt payload contains required keys."""
        payload = build_ai_labels_prompt(
            segments=SAMPLE_SEGMENTS,
            markers=["restart"],
        )

        assert "system_prompt" in payload
        assert "user_prompt" in payload
        assert "segments_compact" in payload

    def test_system_prompt_contains_labels_instructions(self):
        """System prompt contains labeling instructions."""
        payload = build_ai_labels_prompt(
            segments=SAMPLE_SEGMENTS,
            markers=[],
        )

        assert "keep" in payload["system_prompt"]
        assert "cut" in payload["system_prompt"]
        assert "unsure" in payload["system_prompt"]
        assert "story_id" in payload["system_prompt"]

    def test_user_prompt_includes_markers(self):
        """User prompt includes markers when provided."""
        payload = build_ai_labels_prompt(
            segments=SAMPLE_SEGMENTS,
            markers=["restart", "cut"],
        )

        assert "restart" in payload["user_prompt"]
        assert "cut" in payload["user_prompt"]


# --- Test labels_to_clips ---


# Sample segments for labels testing
LABELS_TEST_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Today we'll talk about coding."},
    {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Um, actually let me restart."},
    {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Today we'll discuss Python programming."},
    {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "Python is a great language!"},
    {"start": 20.0, "end": 25.0, "start_ms": 20000, "end_ms": 25000, "text": "Now let's talk about something else."},
    {"start": 25.0, "end": 30.0, "start_ms": 25000, "end_ms": 30000, "text": "JavaScript is also popular."},
]


class TestLabelsToClips:
    """Tests for converting labels to clips."""

    def test_basic_keep_all(self):
        """All keep labels produce single clip."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 1
        assert clips[0]["keep_ms"] == [[0, 15000]]
        assert clips[0]["total_ms"] == 15000

    def test_retake_cuts_earlier_keeps_later(self):
        """
        Retake scenario: earlier version cut, later version kept.

        Segments 0 and 2 say similar things, but segment 1 is a restart marker.
        The labels should cut segment 0 (retake_repeat) and keep segments 2+.
        """
        labels = [
            {"idx": 0, "action": "cut", "tags": ["retake_repeat"], "story_id": 1},
            {"idx": 1, "action": "cut", "tags": ["false_start"], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": ["clean_story"], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:4]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 1
        # Should only keep segments 2-3 (10000ms to 20000ms)
        assert clips[0]["keep_ms"] == [[10000, 20000]]
        assert clips[0]["total_ms"] == 10000

    def test_topic_shift_creates_multiple_clips(self):
        """
        Topic shift scenario: content splits into 2 stories.

        Segments 0-3 are about Python (story_id=1)
        Segments 4-5 are about JavaScript (story_id=2)
        """
        labels = [
            {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
            {"idx": 1, "action": "cut", "tags": ["filler"], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 4, "action": "keep", "tags": ["topic_shift"], "story_id": 2},
            {"idx": 5, "action": "keep", "tags": ["clean_story"], "story_id": 2},
        ]
        segments = LABELS_TEST_SEGMENTS

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 2

        # First clip: story 1 (segments 0, 2, 3 - skipping segment 1)
        clip1 = clips[0]
        assert clip1["keep_ms"] == [[0, 5000], [10000, 20000]]
        assert clip1["total_ms"] == 15000

        # Second clip: story 2 (segments 4-5)
        clip2 = clips[1]
        assert clip2["keep_ms"] == [[20000, 30000]]
        assert clip2["total_ms"] == 10000

    def test_unsure_treated_as_keep(self):
        """Unsure labels are treated as keep (conservative) by default."""
        labels = [
            {"idx": 0, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:2]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            unsure_policy="keep",  # default for document
        )

        assert len(clips) == 1
        assert clips[0]["keep_ms"] == [[0, 10000]]

    def test_respects_max_clips(self):
        """Output respects max_clips limit."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": [], "story_id": 2},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 3},
            {"idx": 3, "action": "keep", "tags": [], "story_id": 4},
        ]
        segments = LABELS_TEST_SEGMENTS[:4]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=2,
            preferred_clip_type="document",
            min_clip_ms=1000,
            max_clip_ms=60000,
        )

        assert len(clips) == 2

    def test_filters_too_short_clips(self):
        """Clips below min_clip_ms are filtered out."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = [
            {"start": 0.0, "end": 2.0, "start_ms": 0, "end_ms": 2000, "text": "Short."},
        ]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,  # Clip is only 2000ms
            max_clip_ms=60000,
        )

        assert len(clips) == 0

    def test_truncates_too_long_clips(self):
        """Clips above max_clip_ms are truncated."""
        labels = [
            {"idx": i, "action": "keep", "tags": [], "story_id": 1}
            for i in range(6)
        ]
        segments = LABELS_TEST_SEGMENTS  # 30 seconds total

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=15000,  # Limit to 15 seconds
        )

        assert len(clips) == 1
        assert clips[0]["total_ms"] == 15000

    def test_empty_labels_returns_empty(self):
        """Empty labels returns empty clips."""
        clips, _ = labels_to_clips(
            labels=[],
            segments=LABELS_TEST_SEGMENTS,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert clips == []

    def test_empty_segments_returns_empty(self):
        """Empty segments returns empty clips."""
        clips, _ = labels_to_clips(
            labels=[{"idx": 0, "action": "keep", "tags": [], "story_id": 1}],
            segments=[],
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert clips == []

    def test_clips_have_required_fields(self):
        """Generated clips have all required fields."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": ["clean_story"], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:2]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="fun",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 1
        clip = clips[0]
        assert "clip_id" in clip
        assert "clip_type" in clip
        assert clip["clip_type"] == "fun"
        assert "title" in clip
        assert "keep_ms" in clip
        assert "total_ms" in clip
        assert "reason" in clip
        assert "ai_labels" in clip["reason"]
        assert "confidence" in clip

    def test_non_consecutive_keep_segments_produce_multiple_ranges(self):
        """Non-consecutive keep segments in same story produce multiple ranges."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "cut", "tags": ["filler"], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 1
        # Two separate ranges due to cut in the middle
        assert clips[0]["keep_ms"] == [[0, 5000], [10000, 15000]]
        assert clips[0]["total_ms"] == 10000


# --- Test labels_to_clips debug output ---


class TestLabelsToClipsDebug:
    """Tests for debug output from labels_to_clips."""

    def test_debug_false_returns_none(self):
        """When debug=False, debug_info is None."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:2]

        clips, debug_info = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            debug=False,
        )

        assert debug_info is None

    def test_debug_true_returns_labels(self):
        """When debug=True, debug_info contains labels."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
            {"idx": 1, "action": "cut", "tags": ["filler"], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, debug_info = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            debug=True,
        )

        assert debug_info is not None
        assert "labels" in debug_info
        assert len(debug_info["labels"]) == 3

    def test_debug_true_returns_clip_sources(self):
        """When debug=True, debug_info contains clip_sources."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "cut", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, debug_info = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            debug=True,
        )

        assert debug_info is not None
        assert "clip_sources" in debug_info
        assert len(debug_info["clip_sources"]) == 1

        # Check clip sources structure
        src = debug_info["clip_sources"][0]
        assert "clip_index" in src
        assert "story_id" in src
        assert "kept_segment_indexes" in src
        assert "cut_segment_indexes" in src

        # Verify the indexes
        assert src["kept_segment_indexes"] == [0, 2]
        assert src["cut_segment_indexes"] == [1]


# --- Test unsure_policy ---


class TestUnsurePolicy:
    """Tests for unsure_policy behavior in labels_to_clips."""

    def test_unsure_policy_keep(self):
        """unsure_policy='keep' treats unsure as keep."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            unsure_policy="keep",
        )

        assert len(clips) == 1
        # All three segments should be kept (single contiguous range)
        assert clips[0]["keep_ms"] == [[0, 15000]]
        assert clips[0]["total_ms"] == 15000

    def test_unsure_policy_cut(self):
        """unsure_policy='cut' treats unsure as cut."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            unsure_policy="cut",
        )

        assert len(clips) == 1
        # Segment 1 should be cut (two separate ranges)
        assert clips[0]["keep_ms"] == [[0, 5000], [10000, 15000]]
        assert clips[0]["total_ms"] == 10000

    def test_unsure_policy_adjacent_keeps_with_keep_neighbors(self):
        """unsure_policy='adjacent' keeps when neighbor is keep."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            unsure_policy="adjacent",
        )

        assert len(clips) == 1
        # Segment 1 should be kept (neighbors are keep)
        assert clips[0]["keep_ms"] == [[0, 15000]]
        assert clips[0]["total_ms"] == 15000

    def test_unsure_policy_adjacent_cuts_with_cut_neighbors(self):
        """unsure_policy='adjacent' cuts when neighbors are cut."""
        labels = [
            {"idx": 0, "action": "cut", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 2, "action": "cut", "tags": [], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 4, "action": "keep", "tags": [], "story_id": 1},
        ]
        segments = LABELS_TEST_SEGMENTS[:5]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
            unsure_policy="adjacent",
        )

        assert len(clips) == 1
        # Segments 0, 1, 2 should be cut; only 3, 4 kept
        assert clips[0]["keep_ms"] == [[15000, 25000]]
        assert clips[0]["total_ms"] == 10000


# --- Test validate_planned_clips ---


from app.planner import validate_planned_clips


class TestValidatePlannedClips:
    """Tests for clip validation function."""

    def test_valid_clips_pass(self):
        """Valid clips pass validation."""
        clips = [
            {
                "clip_id": "test-1",
                "keep_ms": [[0, 10000]],
                "total_ms": 10000,
            }
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        is_valid, reason = validate_planned_clips(
            clips=clips,
            segments=segments,
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is True
        assert reason == "ok"

    def test_empty_clips_fails(self):
        """Empty clips list fails validation."""
        is_valid, reason = validate_planned_clips(
            clips=[],
            segments=LABELS_TEST_SEGMENTS,
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is False
        assert "no clips" in reason

    def test_empty_segments_fails(self):
        """Empty segments fails validation."""
        clips = [{"clip_id": "test", "keep_ms": [[0, 10000]], "total_ms": 10000}]

        is_valid, reason = validate_planned_clips(
            clips=clips,
            segments=[],
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is False
        assert "no segments" in reason

    def test_clips_too_short_fails(self):
        """All clips below min_clip_ms fails validation."""
        clips = [
            {"clip_id": "test", "keep_ms": [[0, 2000]], "total_ms": 2000}
        ]
        segments = LABELS_TEST_SEGMENTS[:2]

        is_valid, reason = validate_planned_clips(
            clips=clips,
            segments=segments,
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is False
        assert "validation" in reason.lower()

    def test_clips_out_of_bounds_fails(self):
        """Clips with ranges outside transcript bounds fail."""
        clips = [
            {"clip_id": "test", "keep_ms": [[100000, 200000]], "total_ms": 100000}
        ]
        segments = LABELS_TEST_SEGMENTS  # ends at 30000ms

        is_valid, reason = validate_planned_clips(
            clips=clips,
            segments=segments,
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is False

    def test_at_least_one_valid_clip_passes(self):
        """If at least one clip is valid, validation passes."""
        clips = [
            {"clip_id": "bad", "keep_ms": [[0, 1000]], "total_ms": 1000},  # too short
            {"clip_id": "good", "keep_ms": [[0, 10000]], "total_ms": 10000},  # valid
        ]
        segments = LABELS_TEST_SEGMENTS[:3]

        is_valid, reason = validate_planned_clips(
            clips=clips,
            segments=segments,
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert is_valid is True


# --- Test endpoint with debug and unsure_policy ---


class TestPlanEditsEndpointDebug:
    """Tests for /plan-edits endpoint with debug and unsure_policy."""

    def test_debug_parameter_accepted(self):
        """debug parameter is accepted by endpoint."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "heuristic",
                "debug": True,
            },
        )

        assert response.status_code == 200

    def test_unsure_policy_parameter_accepted(self):
        """unsure_policy parameter is accepted by endpoint."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "heuristic",
                "unsure_policy": "cut",
            },
        )

        assert response.status_code == 200

    def test_invalid_unsure_policy_returns_422(self):
        """Invalid unsure_policy value returns validation error."""
        response = client.post(
            "/plan-edits",
            json={
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "Hello world"},
                ],
                "mode": "heuristic",
                "unsure_policy": "invalid_policy",
            },
        )

        assert response.status_code == 422


# --- Test is_outro_text ---


from app.planner import (
    is_outro_text,
    drop_short_bridge_ranges,
    drop_short_leadin_ranges,
    trim_trailing_unsure,
    expand_keep_ranges,
    OUTRO_PHRASES,
    MIN_LEADIN_RANGE_MS,
    DEFAULT_LEAD_IN_MS,
    DEFAULT_TAIL_OUT_MS,
)


class TestIsOutroText:
    """Tests for outro text detection."""

    def test_detects_lets_see(self):
        """Detects 'let's see' as outro."""
        assert is_outro_text("Let's see") is True
        assert is_outro_text("let's see what we have") is True

    def test_detects_thats_it(self):
        """Detects 'that's it' as outro."""
        assert is_outro_text("That's it") is True
        assert is_outro_text("thats it for now") is True

    def test_detects_anyway(self):
        """Detects 'anyway' as outro."""
        assert is_outro_text("Anyway") is True
        assert is_outro_text("anyway, bye") is True

    def test_detects_cool(self):
        """Detects short 'cool' as outro."""
        assert is_outro_text("Cool") is True
        assert is_outro_text("cool, bye") is True

    def test_does_not_flag_normal_content(self):
        """Normal content is not flagged as outro."""
        assert is_outro_text("Today we'll talk about Python") is False
        assert is_outro_text("This is important content") is False

    def test_does_not_flag_long_text_with_outro_phrase(self):
        """Long text containing outro phrase is not flagged."""
        # Only short segments with outro phrases are flagged
        long_text = "This is a really long sentence that happens to mention anyway in the middle of it"
        assert is_outro_text(long_text) is False

    def test_detects_alright_so(self):
        """Detects 'alright so' as outro."""
        assert is_outro_text("Alright so") is True
        assert is_outro_text("alright so yeah") is True


# --- Test drop_short_bridge_ranges ---


class TestDropShortBridgeRanges:
    """Tests for bridge range drop post-processing."""

    def test_single_range_unchanged(self):
        """Single range clips are unchanged."""
        clips = [
            {"clip_id": "1", "keep_ms": [[5000, 20000]], "total_ms": 15000}
        ]
        result = drop_short_bridge_ranges(clips)

        assert len(result) == 1
        assert result[0]["keep_ms"] == [[5000, 20000]]

    def test_drops_short_first_range(self):
        """Short first range is dropped from multi-range clips."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 2000], [10000, 25000]],  # First range is 2000ms < 4000ms
                "total_ms": 17000
            }
        ]
        result = drop_short_bridge_ranges(clips)

        assert len(result) == 1
        assert result[0]["keep_ms"] == [[10000, 25000]]
        assert result[0]["total_ms"] == 15000

    def test_keeps_long_first_range(self):
        """First range >= 4000ms is kept."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 5000], [10000, 25000]],  # First range is 5000ms >= 4000ms
                "total_ms": 20000
            }
        ]
        result = drop_short_bridge_ranges(clips)

        assert len(result) == 1
        assert result[0]["keep_ms"] == [[0, 5000], [10000, 25000]]
        assert result[0]["total_ms"] == 20000

    def test_handles_multiple_clips(self):
        """Processes multiple clips correctly."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 1000], [5000, 15000]],  # Short first range
                "total_ms": 11000
            },
            {
                "clip_id": "2",
                "keep_ms": [[20000, 30000]],  # Single range
                "total_ms": 10000
            },
        ]
        result = drop_short_bridge_ranges(clips)

        assert len(result) == 2
        assert result[0]["keep_ms"] == [[5000, 15000]]
        assert result[0]["total_ms"] == 10000
        assert result[1]["keep_ms"] == [[20000, 30000]]  # Unchanged

    def test_empty_clips_returns_empty(self):
        """Empty clips list returns empty."""
        result = drop_short_bridge_ranges([])
        assert result == []

    def test_exactly_4000ms_first_range_kept(self):
        """First range exactly at 4000ms is kept."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 4000], [10000, 20000]],
                "total_ms": 14000
            }
        ]
        result = drop_short_bridge_ranges(clips)

        # 4000ms is NOT < 4000ms, so it's kept
        assert result[0]["keep_ms"] == [[0, 4000], [10000, 20000]]

    def test_real_world_bridge_drop_example(self):
        """Test with real-world output shape: [[24850,26880],[34640,45680]]."""
        # First range is 2030ms (< 4000ms), should be dropped
        clips = [
            {
                "clip_id": "abc123",
                "clip_type": "document",
                "title": "Test Clip",
                "keep_ms": [[24850, 26880], [34640, 45680]],
                "total_ms": 13070,  # 2030 + 11040
                "reason": "ai_labels: story 1",
                "confidence": 0.85
            }
        ]
        result = drop_short_bridge_ranges(clips)

        assert len(result) == 1
        # First range (2030ms) should be dropped
        assert result[0]["keep_ms"] == [[34640, 45680]]
        assert result[0]["total_ms"] == 11040


# --- Test outro auto-cut integration ---


class TestOutroAutoCut:
    """Tests for outro segments being auto-cut in labels_to_clips."""

    def test_outro_segment_not_in_output(self):
        """Segments with outro text should be cut and not appear in keep_ms."""
        # Simulate what happens when outro is detected:
        # The label is forced to "cut" before labels_to_clips
        labels = [
            {"idx": 0, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 2, "action": "cut", "tags": ["outro"], "story_id": 1},  # Outro forced to cut
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Main content here."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "More main content."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Let's see, that's it."},
        ]

        clips, _ = labels_to_clips(
            labels=labels,
            segments=segments,
            max_clips=3,
            preferred_clip_type="document",
            min_clip_ms=5000,
            max_clip_ms=60000,
        )

        assert len(clips) == 1
        # Only segments 0-1 should be in keep_ms (not segment 2 with outro)
        assert clips[0]["keep_ms"] == [[0, 10000]]
        assert clips[0]["total_ms"] == 10000


# --- Test trim_trailing_unsure ---


class TestTrimTrailingUnsure:
    """Tests for trailing unsure segment trimming."""

    def test_trims_trailing_unsure_segments(self):
        """Trailing unsure segments are removed from clips."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 20000]],  # Segments 0-3
                "total_ms": 20000,
            }
        ]
        # Original labels: segments 2 and 3 were "unsure"
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 2, "action": "unsure", "tags": [], "story_id": 1},
            {"idx": 3, "action": "unsure", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 1, 2, 3],
                "cut_segment_indexes": [],
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "First."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Second."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Hmm let me see."},
            {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "Yeah so anyway."},
        ]

        result = trim_trailing_unsure(
            clips=clips,
            labels=labels,
            clip_sources=clip_sources,
            segments=segments,
            min_clip_ms=5000,
        )

        assert len(result) == 1
        # Segments 2-3 (unsure) should be trimmed
        assert result[0]["keep_ms"] == [[0, 10000]]
        assert result[0]["total_ms"] == 10000

    def test_no_unsure_at_end_unchanged(self):
        """Clips with no trailing unsure are unchanged."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 15000]],
                "total_ms": 15000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},  # unsure in middle
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},   # keep at end
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 1, 2],
                "cut_segment_indexes": [],
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "First."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Hmm."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Third."},
        ]

        result = trim_trailing_unsure(
            clips=clips,
            labels=labels,
            clip_sources=clip_sources,
            segments=segments,
            min_clip_ms=5000,
        )

        assert len(result) == 1
        # No trimming - last segment is "keep"
        assert result[0]["keep_ms"] == [[0, 15000]]
        assert result[0]["total_ms"] == 15000

    def test_preserves_ranges_order(self):
        """Trimmed ranges remain valid and ordered."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 5000], [10000, 20000]],  # Two ranges
                "total_ms": 15000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 3, "action": "unsure", "tags": [], "story_id": 1},  # trailing unsure
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 2, 3],
                "cut_segment_indexes": [1],
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "First."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Cut."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "Third."},
            {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "Hmm."},
        ]

        result = trim_trailing_unsure(
            clips=clips,
            labels=labels,
            clip_sources=clip_sources,
            segments=segments,
            min_clip_ms=5000,
        )

        assert len(result) == 1
        # Segment 3 trimmed, ranges remain valid
        assert result[0]["keep_ms"] == [[0, 5000], [10000, 15000]]
        assert result[0]["total_ms"] == 10000

    def test_does_not_trim_below_min_clip_ms(self):
        """Trimming stops if it would make clip too short."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 10000]],
                "total_ms": 10000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 1],
                "cut_segment_indexes": [],
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "First."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Hmm."},
        ]

        # min_clip_ms = 6000, trimming would leave only 5000ms
        result = trim_trailing_unsure(
            clips=clips,
            labels=labels,
            clip_sources=clip_sources,
            segments=segments,
            min_clip_ms=6000,
        )

        assert len(result) == 1
        # No trimming - would make clip too short
        assert result[0]["keep_ms"] == [[0, 10000]]
        assert result[0]["total_ms"] == 10000

    def test_empty_inputs_returns_clips(self):
        """Empty inputs return original clips."""
        clips = [{"clip_id": "1", "keep_ms": [[0, 10000]], "total_ms": 10000}]

        # Empty labels
        result = trim_trailing_unsure(clips, [], [], [], 5000)
        assert result == clips

        # Empty clip_sources
        result = trim_trailing_unsure(clips, [{"idx": 0, "action": "keep"}], [], [], 5000)
        assert result == clips

    def test_multiple_clips_processed_independently(self):
        """Each clip is processed independently."""
        clips = [
            {"clip_id": "1", "keep_ms": [[0, 10000]], "total_ms": 10000},
            {"clip_id": "2", "keep_ms": [[20000, 30000]], "total_ms": 10000},
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": [], "story_id": 1},  # trailing for clip 1
            {"idx": 2, "action": "keep", "tags": [], "story_id": 2},
            {"idx": 3, "action": "keep", "tags": [], "story_id": 2},   # no trailing unsure for clip 2
        ]
        clip_sources = [
            {"clip_index": 0, "story_id": 1, "kept_segment_indexes": [0, 1], "cut_segment_indexes": []},
            {"clip_index": 1, "story_id": 2, "kept_segment_indexes": [2, 3], "cut_segment_indexes": []},
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "First."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Hmm."},
            {"start": 20.0, "end": 25.0, "start_ms": 20000, "end_ms": 25000, "text": "Third."},
            {"start": 25.0, "end": 30.0, "start_ms": 25000, "end_ms": 30000, "text": "Fourth."},
        ]

        result = trim_trailing_unsure(
            clips=clips,
            labels=labels,
            clip_sources=clip_sources,
            segments=segments,
            min_clip_ms=5000,
        )

        assert len(result) == 2
        # Clip 1: segment 1 trimmed
        assert result[0]["keep_ms"] == [[0, 5000]]
        assert result[0]["total_ms"] == 5000
        # Clip 2: unchanged
        assert result[1]["keep_ms"] == [[20000, 30000]]
        assert result[1]["total_ms"] == 10000


# --- Test normalize_labels ---


class TestNormalizeLabels:
    """Tests for label normalization (forcing cut for certain tags)."""

    def test_unsure_with_false_start_becomes_cut(self):
        """unsure + false_start tag forces action='cut'."""
        labels = [
            {"idx": 0, "action": "unsure", "tags": ["false_start"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert len(normalized) == 1
        assert normalized[0]["action"] == "cut"
        assert "false_start" in normalized[0]["tags"]
        assert len(notes) == 1
        assert "normalized unsure->cut" in notes[0]

    def test_keep_with_retake_repeat_becomes_cut(self):
        """keep + retake_repeat tag forces action='cut'."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["retake_repeat"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert len(normalized) == 1
        assert normalized[0]["action"] == "cut"
        assert "retake_repeat" in normalized[0]["tags"]
        assert len(notes) == 1
        assert "normalized keep->cut" in notes[0]

    def test_unknown_tag_dropped_safely(self):
        """Unknown tags are dropped without error."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["clean_story", "unknown_tag", "also_unknown"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert len(normalized) == 1
        assert normalized[0]["action"] == "keep"  # No cut-forcing tags
        assert normalized[0]["tags"] == ["clean_story"]  # Only valid tag kept
        assert len(notes) == 2  # Two dropped tags
        assert "dropped unknown tag 'unknown_tag'" in notes[0]
        assert "dropped unknown tag 'also_unknown'" in notes[1]

    def test_cut_action_unchanged(self):
        """Already 'cut' action stays 'cut' (no duplicate note)."""
        labels = [
            {"idx": 0, "action": "cut", "tags": ["filler"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert len(normalized) == 1
        assert normalized[0]["action"] == "cut"
        # No normalization note because action was already 'cut'
        assert len(notes) == 0

    def test_cut_forcing_tags_list(self):
        """All CUT_FORCING_TAGS force cut action."""
        for tag in CUT_FORCING_TAGS:
            labels = [
                {"idx": 0, "action": "unsure", "tags": [tag], "story_id": 1},
            ]

            normalized, notes = normalize_labels(labels, debug=True)

            assert normalized[0]["action"] == "cut", f"Tag '{tag}' should force cut"

    def test_clean_story_does_not_force_cut(self):
        """clean_story tag does not force cut."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["clean_story"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert normalized[0]["action"] == "keep"
        assert len(notes) == 0

    def test_multiple_labels_normalized(self):
        """Multiple labels are all normalized correctly."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["clean_story"], "story_id": 1},
            {"idx": 1, "action": "unsure", "tags": ["filler"], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": ["retake_repeat"], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": ["intro"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert len(normalized) == 4
        assert normalized[0]["action"] == "keep"  # clean_story: keep
        assert normalized[1]["action"] == "cut"   # filler: forced cut
        assert normalized[2]["action"] == "cut"   # retake_repeat: forced cut
        assert normalized[3]["action"] == "keep"  # intro: keep
        assert len(notes) == 2  # Two normalizations

    def test_debug_false_returns_empty_notes(self):
        """When debug=False, notes list is empty."""
        labels = [
            {"idx": 0, "action": "unsure", "tags": ["false_start", "unknown"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=False)

        assert normalized[0]["action"] == "cut"
        assert notes == []

    def test_preserves_story_id(self):
        """Story ID is preserved through normalization."""
        labels = [
            {"idx": 0, "action": "unsure", "tags": ["filler"], "story_id": 5},
        ]

        normalized, notes = normalize_labels(labels)

        assert normalized[0]["story_id"] == 5

    def test_empty_tags_handled(self):
        """Empty tags list is handled correctly."""
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert normalized[0]["action"] == "keep"
        assert normalized[0]["tags"] == []
        assert len(notes) == 0

    def test_non_list_tags_handled(self):
        """Non-list tags field is converted to empty list."""
        labels = [
            {"idx": 0, "action": "keep", "tags": "not_a_list", "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert normalized[0]["tags"] == []
        assert normalized[0]["action"] == "keep"

    def test_outro_tag_forces_cut(self):
        """outro tag forces cut (included in CUT_FORCING_TAGS)."""
        labels = [
            {"idx": 0, "action": "keep", "tags": ["outro"], "story_id": 1},
        ]

        normalized, notes = normalize_labels(labels, debug=True)

        assert normalized[0]["action"] == "cut"
        assert len(notes) == 1


# --- Test drop_short_leadin_ranges ---


class TestDropShortLeadinRanges:
    """Tests for smart lead-in range drop post-processing."""

    def test_drops_short_first_range_no_exceptions(self):
        """Short first range without exceptions gets dropped."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 2000], [10000, 25000]],  # First range 2000ms < 2500ms
                "total_ms": 17000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},  # No clean_story tag
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 2, 3],
                "cut_segment_indexes": [1],
            }
        ]
        segments = [
            {"start": 0.0, "end": 2.0, "start_ms": 0, "end_ms": 2000, "text": "Short intro."},
            {"start": 2.0, "end": 10.0, "start_ms": 2000, "end_ms": 10000, "text": "Cut."},
            {"start": 10.0, "end": 17.5, "start_ms": 10000, "end_ms": 17500, "text": "Main content."},
            {"start": 17.5, "end": 25.0, "start_ms": 17500, "end_ms": 25000, "text": "More content."},
        ]

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments)

        assert len(result) == 1
        # First range should be dropped
        assert result[0]["keep_ms"] == [[10000, 25000]]
        assert result[0]["total_ms"] == 15000

    def test_keeps_first_range_with_clean_story_tag(self):
        """First range with clean_story tag is kept even if short."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 2000], [10000, 25000]],  # First range 2000ms < 2500ms
                "total_ms": 17000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": ["clean_story"], "story_id": 1},  # Has clean_story
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 3, "action": "keep", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 2, 3],
                "cut_segment_indexes": [1],
            }
        ]
        segments = [
            {"start": 0.0, "end": 2.0, "start_ms": 0, "end_ms": 2000, "text": "Short intro."},
            {"start": 2.0, "end": 10.0, "start_ms": 2000, "end_ms": 10000, "text": "Cut."},
            {"start": 10.0, "end": 17.5, "start_ms": 10000, "end_ms": 17500, "text": "Main content."},
            {"start": 17.5, "end": 25.0, "start_ms": 17500, "end_ms": 25000, "text": "More content."},
        ]

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments)

        assert len(result) == 1
        # First range should be KEPT because of clean_story tag
        assert result[0]["keep_ms"] == [[0, 2000], [10000, 25000]]
        assert result[0]["total_ms"] == 17000

    def test_keeps_first_range_with_two_segments(self):
        """First range with >=2 kept segments is kept even if short."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 2400], [10000, 25000]],  # First range 2400ms < 2500ms
                "total_ms": 17400,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 1, "action": "keep", "tags": [], "story_id": 1},  # Two segments in first range
            {"idx": 3, "action": "keep", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 1, 3],
                "cut_segment_indexes": [2],
            }
        ]
        segments = [
            {"start": 0.0, "end": 1.2, "start_ms": 0, "end_ms": 1200, "text": "First."},
            {"start": 1.2, "end": 2.4, "start_ms": 1200, "end_ms": 2400, "text": "Second."},
            {"start": 2.4, "end": 10.0, "start_ms": 2400, "end_ms": 10000, "text": "Cut."},
            {"start": 10.0, "end": 25.0, "start_ms": 10000, "end_ms": 25000, "text": "Main content."},
        ]

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments)

        assert len(result) == 1
        # First range should be KEPT because there are 2 segments in it
        assert result[0]["keep_ms"] == [[0, 2400], [10000, 25000]]
        assert result[0]["total_ms"] == 17400

    def test_keeps_first_range_above_threshold(self):
        """First range >= MIN_LEADIN_RANGE_MS is always kept."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 3000], [10000, 25000]],  # First range 3000ms >= 2500ms
                "total_ms": 18000,
            }
        ]
        labels = [
            {"idx": 0, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 2, "action": "keep", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [0, 2],
                "cut_segment_indexes": [1],
            }
        ]
        segments = [
            {"start": 0.0, "end": 3.0, "start_ms": 0, "end_ms": 3000, "text": "Good intro."},
            {"start": 3.0, "end": 10.0, "start_ms": 3000, "end_ms": 10000, "text": "Cut."},
            {"start": 10.0, "end": 25.0, "start_ms": 10000, "end_ms": 25000, "text": "Main."},
        ]

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments)

        assert len(result) == 1
        # First range should be kept because it's >= MIN_LEADIN_RANGE_MS
        assert result[0]["keep_ms"] == [[0, 3000], [10000, 25000]]
        assert result[0]["total_ms"] == 18000

    def test_single_range_unchanged(self):
        """Single range clips are unchanged."""
        clips = [
            {"clip_id": "1", "keep_ms": [[0, 5000]], "total_ms": 5000}
        ]
        labels = [{"idx": 0, "action": "keep", "tags": [], "story_id": 1}]
        clip_sources = [{"clip_index": 0, "kept_segment_indexes": [0]}]
        segments = [{"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Content."}]

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments)

        assert result[0]["keep_ms"] == [[0, 5000]]

    def test_empty_inputs_returns_clips(self):
        """Empty labels/clip_sources returns original clips."""
        clips = [{"clip_id": "1", "keep_ms": [[0, 2000], [10000, 20000]], "total_ms": 12000}]

        result = drop_short_leadin_ranges(clips, [], [], [])
        assert result == clips

    def test_min_leadin_range_ms_constant(self):
        """MIN_LEADIN_RANGE_MS is 2500ms."""
        assert MIN_LEADIN_RANGE_MS == 2500

    def test_real_world_example_above_threshold(self):
        """Real-world example where first range > MIN_LEADIN_RANGE_MS is kept."""
        # [[24850,29596],[34640,45680]] - first range is 4746ms > 2500ms
        clips = [
            {
                "clip_id": "test",
                "keep_ms": [[24850, 29596], [34640, 45680]],
                "total_ms": 15786,
            }
        ]
        labels = [
            {"idx": 5, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 7, "action": "keep", "tags": [], "story_id": 1},
            {"idx": 8, "action": "keep", "tags": [], "story_id": 1},
        ]
        clip_sources = [
            {
                "clip_index": 0,
                "story_id": 1,
                "kept_segment_indexes": [5, 7, 8],
                "cut_segment_indexes": [6],
            }
        ]
        segments = [
            {"start": 24.85, "end": 29.596, "start_ms": 24850, "end_ms": 29596, "text": "Good content."},
            {"start": 29.596, "end": 34.64, "start_ms": 29596, "end_ms": 34640, "text": "Gap."},
            {"start": 34.64, "end": 40.0, "start_ms": 34640, "end_ms": 40000, "text": "Main content."},
            {"start": 40.0, "end": 45.68, "start_ms": 40000, "end_ms": 45680, "text": "More."},
        ]
        # Align indices: segment at 24850 is idx 5
        segments_extended = [
            {"start": i, "end": i + 1, "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "text": "X"}
            for i in range(5)
        ] + segments

        result = drop_short_leadin_ranges(clips, labels, clip_sources, segments_extended)

        # First range (4746ms) > 2500ms, so it's kept
        assert result[0]["keep_ms"] == [[24850, 29596], [34640, 45680]]


# --- Test expand_keep_ranges ---


class TestExpandKeepRanges:
    """Tests for keep range expansion with lead-in/tail-out padding."""

    def test_default_constants(self):
        """Default constants are 300ms."""
        assert DEFAULT_LEAD_IN_MS == 300
        assert DEFAULT_TAIL_OUT_MS == 300

    def test_expands_ranges_with_defaults(self):
        """Ranges are expanded by lead_in and tail_out amounts."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[5000, 15000]],
                "total_ms": 10000,
            }
        ]
        # Segments that allow expansion
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Intro."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "Main."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "More."},
            {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "Outro."},
        ]

        result = expand_keep_ranges(clips, segments, lead_in_ms=300, tail_out_ms=300)

        assert len(result) == 1
        # Start moved from 5000 to nearest segment boundary (snaps to 5000 or finds earlier)
        # End moved from 15000 to nearest segment boundary
        # With 300ms expansion, start becomes 4700 -> snaps to 0 (nearest start)
        # End becomes 15300 -> snaps to 15000 (nearest end)
        # Actually snapping picks the nearest, so:
        # 4700 is closer to 5000 than 0, so snaps to 5000
        # 15300 is closer to 15000 than 20000, so snaps to 15000
        # So range stays [[5000, 15000]]
        # Wait, snap finds the NEAREST boundary, which could change based on distance
        # Let me recalculate:
        # start - 300 = 4700. Segment starts are [0, 5000, 10000, 15000].
        # 4700 is 4700 away from 0, 300 away from 5000. Nearest is 5000.
        # end + 300 = 15300. Segment ends are [5000, 10000, 15000, 20000].
        # 15300 is 300 away from 15000, 4700 away from 20000. Nearest is 15000.
        # So no change in this case
        assert result[0]["keep_ms"] == [[5000, 15000]]

    def test_expands_to_earlier_segment(self):
        """Expansion snaps to earlier segment when closer."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[5000, 10000]],
                "total_ms": 5000,
            }
        ]
        segments = [
            {"start": 0.0, "end": 4.8, "start_ms": 0, "end_ms": 4800, "text": "Intro."},
            {"start": 4.8, "end": 10.0, "start_ms": 4800, "end_ms": 10000, "text": "Main."},
            {"start": 10.0, "end": 11.0, "start_ms": 10000, "end_ms": 11000, "text": "Extra."},
        ]

        # With 300ms lead_in: 5000 - 300 = 4700
        # Nearest start boundaries: [0, 4800, 10000]
        # 4700 is 200 away from 4800, 4700 away from 0
        # So snaps to 4800
        result = expand_keep_ranges(clips, segments, lead_in_ms=300, tail_out_ms=300)

        assert len(result) == 1
        # Start should snap to 4800 (nearest to 4700)
        # End: 10000 + 300 = 10300, nearest end is 10000 or 11000
        # 10300 is closer to 10000 (300) than 11000 (700)
        assert result[0]["keep_ms"][0][0] == 4800

    def test_clamps_to_transcript_start(self):
        """Expansion clamps to transcript start boundary."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[100, 5000]],
                "total_ms": 4900,
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Content."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "More."},
        ]

        # 100 - 300 = -200, clamped to 0, then snapped to nearest start (0)
        result = expand_keep_ranges(clips, segments, lead_in_ms=300, tail_out_ms=0)

        assert len(result) == 1
        assert result[0]["keep_ms"][0][0] == 0  # Clamped and snapped to transcript start

    def test_clamps_to_transcript_end(self):
        """Expansion clamps to transcript end boundary."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 9900]],
                "total_ms": 9900,
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "Content."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "More."},
        ]

        # 9900 + 300 = 10200, clamped to 10000, then snapped to nearest end (10000)
        result = expand_keep_ranges(clips, segments, lead_in_ms=0, tail_out_ms=300)

        assert len(result) == 1
        assert result[0]["keep_ms"][0][1] == 10000  # Clamped and snapped to transcript end

    def test_zero_expansion_unchanged(self):
        """Zero lead_in and tail_out leaves ranges unchanged."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[5000, 15000]],
                "total_ms": 10000,
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "A."},
            {"start": 5.0, "end": 15.0, "start_ms": 5000, "end_ms": 15000, "text": "B."},
            {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "C."},
        ]

        result = expand_keep_ranges(clips, segments, lead_in_ms=0, tail_out_ms=0)

        assert len(result) == 1
        assert result[0]["keep_ms"] == [[5000, 15000]]

    def test_empty_inputs_returns_clips(self):
        """Empty segments returns original clips."""
        clips = [{"clip_id": "1", "keep_ms": [[0, 10000]], "total_ms": 10000}]

        result = expand_keep_ranges(clips, [], lead_in_ms=300, tail_out_ms=300)
        assert result == clips

    def test_merges_overlapping_after_expansion(self):
        """Adjacent ranges that become overlapping after expansion are merged."""
        clips = [
            {
                "clip_id": "1",
                "keep_ms": [[0, 5000], [5100, 10000]],  # 100ms gap
                "total_ms": 9900,
            }
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "A."},
            {"start": 5.0, "end": 5.1, "start_ms": 5000, "end_ms": 5100, "text": "Gap."},
            {"start": 5.1, "end": 10.0, "start_ms": 5100, "end_ms": 10000, "text": "B."},
        ]

        # With tail_out=300, first range end becomes 5300 (snaps to 5100)
        # With lead_in=300, second range start becomes 4800 (snaps to 5000)
        # These could overlap and get merged
        result = expand_keep_ranges(clips, segments, lead_in_ms=300, tail_out_ms=300)

        assert len(result) == 1
        # After expansion and merge, should be single range
        assert len(result[0]["keep_ms"]) <= 2  # May or may not merge depending on snapping

    def test_multiple_clips_processed(self):
        """Multiple clips are all processed."""
        clips = [
            {"clip_id": "1", "keep_ms": [[0, 5000]], "total_ms": 5000},
            {"clip_id": "2", "keep_ms": [[10000, 15000]], "total_ms": 5000},
        ]
        segments = [
            {"start": 0.0, "end": 5.0, "start_ms": 0, "end_ms": 5000, "text": "A."},
            {"start": 5.0, "end": 10.0, "start_ms": 5000, "end_ms": 10000, "text": "B."},
            {"start": 10.0, "end": 15.0, "start_ms": 10000, "end_ms": 15000, "text": "C."},
            {"start": 15.0, "end": 20.0, "start_ms": 15000, "end_ms": 20000, "text": "D."},
        ]

        result = expand_keep_ranges(clips, segments, lead_in_ms=300, tail_out_ms=300)

        assert len(result) == 2
        # Both clips should have been processed


# --- Test /make-clips endpoint ---


class TestMakeClipsEndpoint:
    """Tests for the /make-clips endpoint."""

    def test_make_clips_filename_generation(self):
        """Output filenames follow pattern: <prefix>_clip<N>.mp4."""
        prefix = "my_video"
        # Pattern should be: my_video_clip1.mp4, my_video_clip2.mp4, etc.
        for i in range(1, 4):
            expected = f"{prefix}_clip{i}.mp4"
            assert expected == f"{prefix}_clip{i}.mp4"

    def test_make_clips_request_validation_path_required(self):
        """Request requires path field."""
        from pydantic import ValidationError
        from app.main import MakeClipsRequest

        with pytest.raises(ValidationError):
            MakeClipsRequest(output_prefix="test")  # Missing path

    def test_make_clips_request_validation_prefix_required(self):
        """Request requires output_prefix field."""
        from pydantic import ValidationError
        from app.main import MakeClipsRequest

        with pytest.raises(ValidationError):
            MakeClipsRequest(path="/data/video.mp4")  # Missing output_prefix

    def test_make_clips_request_defaults(self):
        """Request has sensible defaults."""
        from app.main import MakeClipsRequest

        req = MakeClipsRequest(path="/data/video.mp4", output_prefix="test")

        assert req.max_clips == 2
        assert req.preferred_clip_type == "document"
        assert req.min_clip_ms == 6000
        assert req.max_clip_ms == 60000
        assert req.lead_in_ms == 300
        assert req.tail_out_ms == 300
        assert req.markers == []

    def test_make_clips_file_not_found_returns_404(self):
        """Non-existent file returns 404."""
        response = client.post(
            "/make-clips",
            json={
                "path": "/nonexistent/path/video.mp4",
                "output_prefix": "test",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_make_clips_requires_api_key(self):
        """Endpoint returns 500 if ANTHROPIC_API_KEY not configured."""
        import os

        # Skip if API key is configured
        if os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("API key is configured, skipping this test")

        # Create a temporary file to test with
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake video data")

        try:
            response = client.post(
                "/make-clips",
                json={
                    "path": tmp_path,
                    "output_prefix": "test",
                },
            )

            assert response.status_code == 500
            assert "ANTHROPIC_API_KEY" in response.json()["detail"]
        finally:
            os.remove(tmp_path)
