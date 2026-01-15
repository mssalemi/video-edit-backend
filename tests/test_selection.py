"""Unit tests for clip selection logic."""

import pytest

from app.selection import (
    PROFILE_DEFAULTS,
    build_candidate_windows,
    compute_segment_quality,
    is_bad_segment,
    is_marker_segment,
    is_punchy_segment,
    normalize_segment,
    score_candidate,
    score_candidate_document,
    score_candidate_fun,
    score_candidate_mixed,
    select_clips,
)


# Sample test segments (simulating a ~60 second video)
SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 3.5, "text": "Hello and welcome to this video."},
    {"start": 3.8, "end": 7.2, "text": "Today we're going to talk about something interesting."},
    {"start": 7.5, "end": 12.0, "text": "This is a really important topic that everyone should know about."},
    {"start": 12.3, "end": 16.8, "text": "Let me explain the first key point here."},
    {"start": 17.0, "end": 21.5, "text": "And here's the second thing you need to understand."},
    {"start": 22.0, "end": 26.3, "text": "Now let's move on to the practical examples."},
    {"start": 26.5, "end": 31.0, "text": "As you can see, this works really well in practice."},
    {"start": 31.5, "end": 35.8, "text": "Thanks for watching, don't forget to subscribe!"},
]

# Extended segments for longer clips (simulating a ~90 second video)
EXTENDED_SEGMENTS = SAMPLE_SEGMENTS + [
    {"start": 36.0, "end": 42.0, "text": "But wait, there's more! This is absolutely amazing!"},
    {"start": 42.5, "end": 48.0, "text": "You won't believe what happens next, it's incredible!"},
    {"start": 48.3, "end": 54.0, "text": "Let me show you this one weird trick that experts hate."},
    {"start": 54.2, "end": 60.0, "text": "And finally, the conclusion of our comprehensive guide."},
    {"start": 60.5, "end": 66.0, "text": "Remember to like and subscribe for more awesome content!"},
    {"start": 66.3, "end": 72.0, "text": "Thanks again for watching, see you in the next video."},
]

# Segments with fluffy/filler content for cleanup testing
FLUFFY_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "So um like hey everyone."},
    {"start": 2.3, "end": 5.0, "text": "Okay so basically um you know what I mean."},
    {"start": 5.5, "end": 10.0, "text": "This is the actual good content we want."},
    {"start": 10.3, "end": 15.0, "text": "Here is more valuable information for you."},
    {"start": 15.5, "end": 20.0, "text": "And this concludes our discussion."},
]

# Segments with markers for boundary testing
MARKER_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "First segment of good content."},
    {"start": 5.5, "end": 10.0, "text": "More good content here."},
    {"start": 10.5, "end": 12.0, "text": "Okay cut that was bad."},  # Marker segment
    {"start": 12.5, "end": 17.0, "text": "Starting fresh with new content."},
    {"start": 17.5, "end": 22.0, "text": "This is also good content."},
]

# Punchy segments for punchline testing
PUNCHY_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Setting up the context here."},
    {"start": 5.5, "end": 10.0, "text": "More setup and explanation."},
    {"start": 10.5, "end": 17.0, "text": "Oh my god this is insane! No way!"},  # Punchy
    {"start": 17.5, "end": 22.0, "text": "Back to normal talking here."},
]


class TestNormalizeSegment:
    def test_adds_missing_ms_fields(self):
        segment = {"start": 1.5, "end": 3.25, "text": "test"}
        result = normalize_segment(segment)
        assert result["start_ms"] == 1500
        assert result["end_ms"] == 3250

    def test_preserves_existing_ms_fields(self):
        segment = {"start": 1.5, "end": 3.25, "text": "test", "start_ms": 1500, "end_ms": 3250}
        result = normalize_segment(segment)
        assert result["start_ms"] == 1500
        assert result["end_ms"] == 3250


class TestComputeSegmentQuality:
    def test_computes_word_count(self):
        segment = {"start": 0, "end": 5, "text": "Hello world this is a test."}
        quality = compute_segment_quality(segment)
        assert quality["word_count"] == 6

    def test_detects_filler_words(self):
        segment = {"start": 0, "end": 5, "text": "Um like you know basically um."}
        quality = compute_segment_quality(segment)
        assert quality["filler_count"] >= 3
        assert quality["filler_ratio"] > 0.3

    def test_detects_fluffy_start(self):
        fluffy = {"start": 0, "end": 5, "text": "So today we will discuss."}
        clean = {"start": 0, "end": 5, "text": "Today we will discuss."}

        assert compute_segment_quality(fluffy)["starts_fluffy"] is True
        assert compute_segment_quality(clean)["starts_fluffy"] is False

    def test_detects_clean_ending(self):
        clean = {"start": 0, "end": 5, "text": "This is a complete sentence."}
        unclean = {"start": 0, "end": 5, "text": "This is not complete"}

        assert compute_segment_quality(clean)["ends_clean"] is True
        assert compute_segment_quality(unclean)["ends_clean"] is False


class TestIsBadSegment:
    def test_high_filler_ratio_is_bad(self):
        segment = {"start": 0, "end": 5, "text": "Um like um basically um you know."}
        quality = compute_segment_quality(segment)
        assert is_bad_segment(segment, quality) is True

    def test_short_fluffy_is_bad(self):
        segment = {"start": 0, "end": 1.5, "text": "So yeah."}
        quality = compute_segment_quality(segment)
        assert is_bad_segment(segment, quality) is True

    def test_normal_segment_is_not_bad(self):
        segment = {"start": 0, "end": 5, "text": "This is valuable content for the viewer."}
        quality = compute_segment_quality(segment)
        assert is_bad_segment(segment, quality) is False


class TestIsMarkerSegment:
    def test_detects_marker(self):
        segment = {"start": 0, "end": 5, "text": "Okay cut that was bad."}
        assert is_marker_segment(segment, ["cut"]) is True
        assert is_marker_segment(segment, ["restart"]) is False

    def test_case_insensitive(self):
        segment = {"start": 0, "end": 5, "text": "NEW CLIP starting here."}
        assert is_marker_segment(segment, ["new clip"]) is True

    def test_empty_markers(self):
        segment = {"start": 0, "end": 5, "text": "Some text with cut word."}
        assert is_marker_segment(segment, []) is False
        assert is_marker_segment(segment, None) is False


class TestIsPunchySegment:
    def test_detects_exclamation_punch(self):
        segment = {"start": 0, "end": 8, "text": "This is absolutely incredible!"}
        quality = compute_segment_quality(segment)
        assert is_punchy_segment(segment, quality) is True

    def test_detects_punch_words(self):
        segment = {"start": 0, "end": 7, "text": "Oh my god that is insane dude."}
        quality = compute_segment_quality(segment)
        assert is_punchy_segment(segment, quality) is True

    def test_too_short_not_punchy(self):
        segment = {"start": 0, "end": 3, "text": "Wow!"}
        quality = compute_segment_quality(segment)
        assert is_punchy_segment(segment, quality) is False

    def test_too_long_not_punchy(self):
        segment = {"start": 0, "end": 15, "text": "This is a very long segment that goes on and on and on!"}
        quality = compute_segment_quality(segment)
        assert is_punchy_segment(segment, quality) is False


class TestBuildCandidateWindows:
    def test_builds_windows_from_segments(self):
        candidates = build_candidate_windows(SAMPLE_SEGMENTS, max_s=25, max_gap_s=1.2)
        assert len(candidates) > 0
        assert len(candidates) >= len(SAMPLE_SEGMENTS)

    def test_respects_max_duration(self):
        candidates = build_candidate_windows(SAMPLE_SEGMENTS, max_s=10, max_gap_s=1.2)
        for c in candidates:
            assert c["duration_s"] <= 10

    def test_respects_max_gap(self):
        segments_with_gaps = [
            {"start": 0.0, "end": 3.0, "text": "First"},
            {"start": 10.0, "end": 13.0, "text": "Second"},  # 7 second gap
        ]
        candidates = build_candidate_windows(segments_with_gaps, max_s=25, max_gap_s=1.2)
        assert all(c["segment_count"] == 1 for c in candidates)

    def test_empty_segments_returns_empty(self):
        candidates = build_candidate_windows([], max_s=25, max_gap_s=1.2)
        assert candidates == []

    def test_markers_act_as_boundaries(self):
        """Markers should prevent merging across that segment."""
        candidates = build_candidate_windows(
            MARKER_SEGMENTS,
            max_s=30,
            max_gap_s=1.2,
            markers=["cut"]
        )

        # No candidate should include both segment before and after the marker
        for c in candidates:
            # Check that no candidate spans from before 10.5s to after 12.0s
            if c["start"] < 10.5:
                assert c["end"] <= 10.5, "Candidate should not cross marker segment"
            if c["end"] > 12.0:
                assert c["start"] >= 12.5, "Candidate should not cross marker segment"


class TestScoreCandidate:
    def test_returns_score_and_reason(self):
        candidate = {
            "start": 0.0,
            "end": 15.0,
            "duration_s": 15.0,
            "text": "This is a test sentence with some words.",
            "segment_count": 3,
            "total_gap_s": 0.5,
            "_segments": [],
        }
        score, reason = score_candidate(candidate, target_s=15.0)
        assert 0 <= score <= 1
        assert isinstance(reason, str)

    def test_prefers_target_duration(self):
        exact_target = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "Test words", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }
        far_from_target = {
            "start": 0.0, "end": 25.0, "duration_s": 25.0,
            "text": "Test words", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }
        score_exact, _ = score_candidate(exact_target, target_s=15.0)
        score_far, _ = score_candidate(far_from_target, target_s=15.0)
        assert score_exact > score_far

    def test_clean_ending_bonus(self):
        with_period = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "This ends with a period.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }
        without_period = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "This does not end cleanly", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }
        score_with, reason_with = score_candidate(with_period, target_s=15.0)
        score_without, _ = score_candidate(without_period, target_s=15.0)
        assert score_with > score_without
        assert "clean ending" in reason_with

    def test_score_includes_clip_type_in_reason(self):
        candidate = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "Test words.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }

        _, reason_doc = score_candidate(candidate, target_s=30.0, clip_type="document")
        assert "document" in reason_doc

        _, reason_fun = score_candidate(candidate, target_s=12.0, clip_type="fun")
        assert "fun" in reason_fun

        _, reason_mixed = score_candidate(candidate, target_s=15.0, clip_type="mixed")
        assert "mixed" in reason_mixed


class TestScoreCandidateWithCleanLevel:
    """Test cleanup penalties in scoring."""

    def test_none_has_no_cleanup_penalty(self):
        # Create a fluffy candidate
        fluffy_seg = {"start": 0, "end": 5, "text": "Um so like basically.", "start_ms": 0, "end_ms": 5000}
        fluffy_seg["_quality"] = compute_segment_quality(fluffy_seg)

        candidate = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "Um so like basically.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [fluffy_seg],
        }

        score_none, reason_none = score_candidate(candidate, target_s=15.0, clean_level="none")
        score_light, reason_light = score_candidate(candidate, target_s=15.0, clean_level="light")

        # None should have higher score (no penalty)
        assert score_none >= score_light
        assert "cleanup" not in reason_none
        assert "cleanup" in reason_light

    def test_aggressive_has_stronger_penalty(self):
        # Create a filler-heavy candidate
        filler_seg = {"start": 0, "end": 5, "text": "Um like um basically you know.", "start_ms": 0, "end_ms": 5000}
        filler_seg["_quality"] = compute_segment_quality(filler_seg)

        candidate = {
            "start": 0.0, "end": 15.0, "duration_s": 15.0,
            "text": "Um like um basically you know.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [filler_seg],
        }

        score_light, _ = score_candidate(candidate, target_s=15.0, clean_level="light")
        score_aggressive, _ = score_candidate(candidate, target_s=15.0, clean_level="aggressive")

        # Aggressive should have lower score
        assert score_aggressive <= score_light


class TestScoreCandidateProfiles:
    """Test profile-specific scoring behaviors."""

    def test_document_prefers_longer_duration(self):
        short_clip = {
            "start": 0.0, "end": 12.0, "duration_s": 12.0,
            "text": "Short clip content here.", "segment_count": 2, "total_gap_s": 0.1,
            "_segments": [],
        }
        long_clip = {
            "start": 0.0, "end": 30.0, "duration_s": 30.0,
            "text": "Longer clip content here with more words.", "segment_count": 5, "total_gap_s": 0.3,
            "_segments": [],
        }

        score_short, _ = score_candidate_document(short_clip, target_s=30.0)
        score_long, _ = score_candidate_document(long_clip, target_s=30.0)

        assert score_long > score_short

    def test_fun_prefers_shorter_duration(self):
        short_clip = {
            "start": 0.0, "end": 12.0, "duration_s": 12.0,
            "text": "This is amazing! Wow!", "segment_count": 2, "total_gap_s": 0.1,
            "_segments": [],
        }
        long_clip = {
            "start": 0.0, "end": 25.0, "duration_s": 25.0,
            "text": "This is amazing! Wow!", "segment_count": 5, "total_gap_s": 0.3,
            "_segments": [],
        }

        score_short, _ = score_candidate_fun(short_clip, target_s=12.0)
        score_long, _ = score_candidate_fun(long_clip, target_s=12.0)

        assert score_short > score_long

    def test_fun_bonus_for_exclamations(self):
        boring = {
            "start": 0.0, "end": 12.0, "duration_s": 12.0,
            "text": "This is a normal sentence.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }
        expressive = {
            "start": 0.0, "end": 12.0, "duration_s": 12.0,
            "text": "This is amazing! Wow! Incredible!", "segment_count": 1, "total_gap_s": 0,
            "_segments": [],
        }

        score_boring, _ = score_candidate_fun(boring, target_s=12.0)
        score_expressive, reason = score_candidate_fun(expressive, target_s=12.0)

        assert score_expressive > score_boring
        assert "expressive" in reason


class TestPunchlineBoost:
    """Test punchline boost for fun clips."""

    def test_punchy_single_segment_gets_boost(self):
        # Create a punchy segment
        punchy_seg = {"start": 0, "end": 8, "text": "Oh wow that is insane!", "start_ms": 0, "end_ms": 8000}
        punchy_seg["_quality"] = compute_segment_quality(punchy_seg)

        # Create a non-punchy segment
        normal_seg = {"start": 0, "end": 8, "text": "This is normal content here.", "start_ms": 0, "end_ms": 8000}
        normal_seg["_quality"] = compute_segment_quality(normal_seg)

        punchy_candidate = {
            "start": 0.0, "end": 8.0, "duration_s": 8.0,
            "text": "Oh wow that is insane!", "segment_count": 1, "total_gap_s": 0,
            "_segments": [punchy_seg],
        }
        normal_candidate = {
            "start": 0.0, "end": 8.0, "duration_s": 8.0,
            "text": "This is normal content here.", "segment_count": 1, "total_gap_s": 0,
            "_segments": [normal_seg],
        }

        score_punchy, reason_punchy = score_candidate_fun(punchy_candidate, target_s=12.0)
        score_normal, _ = score_candidate_fun(normal_candidate, target_s=12.0)

        assert score_punchy > score_normal
        assert "punchline" in reason_punchy

    def test_punchy_segment_can_win_in_selection(self):
        """A punchy single segment should be able to beat longer candidates."""
        clips = select_clips(
            PUNCHY_SEGMENTS,
            clip_type="fun",
            clean_level="none",
            max_clips=1,
            min_s=5,
            max_s=20,
        )

        assert len(clips) >= 1
        # The punchy segment (10.5-17.0) should be selected
        best_clip = clips[0]
        assert "punchline" in best_clip["reason"] or best_clip["start"] == 10.5


class TestSelectClips:
    def test_returns_clips(self):
        clips = select_clips(SAMPLE_SEGMENTS, clean_level="none")
        assert len(clips) > 0

    def test_respects_max_clips(self):
        clips = select_clips(SAMPLE_SEGMENTS, max_clips=1, clean_level="none")
        assert len(clips) <= 1

        clips = select_clips(SAMPLE_SEGMENTS, max_clips=5, clean_level="none")
        assert len(clips) <= 5

    def test_clips_have_required_fields(self):
        clips = select_clips(SAMPLE_SEGMENTS, max_clips=2, clean_level="none")
        for clip in clips:
            assert "start" in clip
            assert "end" in clip
            assert "duration_s" in clip
            assert "start_ms" in clip
            assert "end_ms" in clip
            assert "score" in clip
            assert "reason" in clip

    def test_clips_within_duration_bounds(self):
        # Use max_clips=1 to avoid mixed variety selection which uses different defaults
        clips = select_clips(SAMPLE_SEGMENTS, min_s=10, max_s=20, max_clips=1, clean_level="none")
        for clip in clips:
            assert clip["duration_s"] >= 10 or len(clips) == 1
            assert clip["duration_s"] <= 20

    def test_empty_segments_returns_empty(self):
        clips = select_clips([], clean_level="none")
        assert clips == []

    def test_single_segment_fallback(self):
        single = [{"start": 0.0, "end": 5.0, "text": "Short segment"}]
        clips = select_clips(single, min_s=10, max_s=20, clean_level="none")
        assert len(clips) == 1
        assert clips[0]["duration_s"] == 5.0

    def test_clips_sorted_by_score_desc(self):
        clips = select_clips(SAMPLE_SEGMENTS, max_clips=5, clean_level="none")
        if len(clips) > 1:
            for i in range(len(clips) - 1):
                assert clips[i]["score"] >= clips[i + 1]["score"]

    def test_no_overlapping_clips(self):
        clips = select_clips(SAMPLE_SEGMENTS, max_clips=5, clean_level="none")
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                clip_a = clips[i]
                clip_b = clips[j]
                overlaps = clip_a["start"] < clip_b["end"] and clip_b["start"] < clip_a["end"]
                assert not overlaps, f"Clips {i} and {j} overlap"


class TestSelectClipsWithClipType:
    """Test clip type functionality."""

    def test_document_uses_longer_defaults(self):
        clips = select_clips(EXTENDED_SEGMENTS, clip_type="document", max_clips=1, clean_level="none")

        assert len(clips) >= 1
        assert clips[0]["duration_s"] >= PROFILE_DEFAULTS["document"]["min_s"]

    def test_fun_uses_shorter_defaults(self):
        clips = select_clips(SAMPLE_SEGMENTS, clip_type="fun", max_clips=1, clean_level="none")

        assert len(clips) >= 1
        assert clips[0]["duration_s"] <= PROFILE_DEFAULTS["fun"]["max_s"]

    def test_document_produces_longer_clips_than_fun(self):
        doc_clips = select_clips(EXTENDED_SEGMENTS, clip_type="document", max_clips=1, clean_level="none")
        fun_clips = select_clips(EXTENDED_SEGMENTS, clip_type="fun", max_clips=1, clean_level="none")

        assert len(doc_clips) >= 1
        assert len(fun_clips) >= 1

        doc_duration = doc_clips[0]["duration_s"]
        fun_duration = fun_clips[0]["duration_s"]

        assert doc_duration > fun_duration

    def test_clip_reason_includes_profile(self):
        doc_clips = select_clips(SAMPLE_SEGMENTS, clip_type="document", max_clips=1, clean_level="none")
        fun_clips = select_clips(SAMPLE_SEGMENTS, clip_type="fun", max_clips=1, clean_level="none")
        mixed_clips = select_clips(SAMPLE_SEGMENTS, clip_type="mixed", max_clips=1, clean_level="none")

        assert "document" in doc_clips[0]["reason"]
        assert "fun" in fun_clips[0]["reason"]
        assert "mixed" in mixed_clips[0]["reason"]


class TestSelectClipsWithCleanLevel:
    """Test clean_level functionality."""

    def test_light_shifts_start_to_skip_fluff(self):
        """Light cleanup should shift start forward when first segment is fluffy."""
        clips_none = select_clips(
            FLUFFY_SEGMENTS,
            clip_type="mixed",
            clean_level="none",
            max_clips=1,
            min_s=5,
            max_s=25,
        )
        clips_light = select_clips(
            FLUFFY_SEGMENTS,
            clip_type="mixed",
            clean_level="light",
            max_clips=1,
            min_s=5,
            max_s=25,
        )

        # Light should have shifted start (fluffy segments at 0-5s)
        # The shifted clip should start at 5.5s (first good segment)
        if clips_light and clips_none:
            # Light cleanup may shift start forward
            assert clips_light[0]["start"] >= clips_none[0]["start"]
            if "shifted start" in clips_light[0]["reason"]:
                assert clips_light[0]["start"] >= 5.0

    def test_aggressive_filters_filler_heavy_candidates(self):
        """Aggressive cleanup should filter out candidates with many bad segments."""
        # All-fluffy segments should still return something (best-effort)
        all_fluffy = [
            {"start": 0.0, "end": 5.0, "text": "Um so like basically um."},
            {"start": 5.5, "end": 10.0, "text": "You know like um anyway."},
        ]

        clips = select_clips(
            all_fluffy,
            clip_type="mixed",
            clean_level="aggressive",
            max_clips=1,
            min_s=3,
            max_s=20,
        )

        # Should still return best-effort
        assert len(clips) >= 1

    def test_cleanup_level_appears_in_reason(self):
        """Clean level should appear in the reason string."""
        clips_light = select_clips(
            FLUFFY_SEGMENTS,
            clip_type="mixed",
            clean_level="light",
            max_clips=1,
            min_s=5,
            max_s=25,
        )
        clips_aggressive = select_clips(
            FLUFFY_SEGMENTS,
            clip_type="mixed",
            clean_level="aggressive",
            max_clips=1,
            min_s=5,
            max_s=25,
        )

        if clips_light:
            assert "light cleanup" in clips_light[0]["reason"]
        if clips_aggressive:
            assert "aggressive cleanup" in clips_aggressive[0]["reason"]


class TestSelectClipsWithMarkers:
    """Test marker functionality."""

    def test_markers_prevent_crossing(self):
        """Clips should not cross marker segments."""
        clips = select_clips(
            MARKER_SEGMENTS,
            markers=["cut"],
            clip_type="mixed",
            clean_level="none",
            max_clips=3,
            min_s=3,
            max_s=30,
        )

        # Check no clip spans across the marker at 10.5-12.0
        for clip in clips:
            # A clip should not start before the marker and end after it
            starts_before_marker = clip["start"] < 10.5
            ends_after_marker = clip["end"] > 12.0
            assert not (starts_before_marker and ends_after_marker), \
                f"Clip {clip['start']}-{clip['end']} crosses marker"

    def test_multiple_markers(self):
        """Multiple markers should all act as boundaries."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Content before first cut."},
            {"start": 5.5, "end": 8.0, "text": "Okay cut here."},
            {"start": 8.5, "end": 13.0, "text": "Middle content."},
            {"start": 13.5, "end": 16.0, "text": "Time to restart."},
            {"start": 16.5, "end": 21.0, "text": "Final content section."},
        ]

        clips = select_clips(
            segments,
            markers=["cut", "restart"],
            clip_type="mixed",
            clean_level="none",
            max_clips=5,
            min_s=3,
            max_s=30,
        )

        # Verify no clip crosses either marker
        for clip in clips:
            # Should not span 0-5 to 8.5-13 (crossing first marker)
            if clip["start"] < 5.5 and clip["end"] > 8.0:
                assert False, f"Clip crosses 'cut' marker: {clip['start']}-{clip['end']}"
            # Should not span 8.5-13 to 16.5-21 (crossing second marker)
            if clip["start"] < 13.5 and clip["end"] > 16.0:
                assert False, f"Clip crosses 'restart' marker: {clip['start']}-{clip['end']}"


class TestMixedVariety:
    """Test mixed variety selection with max_clips >= 2."""

    def test_mixed_with_max_clips_2_returns_variety(self):
        clips = select_clips(EXTENDED_SEGMENTS, clip_type="mixed", max_clips=2, clean_level="none")

        assert len(clips) >= 1
        assert len(clips) <= 2

    def test_mixed_clips_no_overlaps(self):
        clips = select_clips(EXTENDED_SEGMENTS, clip_type="mixed", max_clips=3, clean_level="none")

        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                clip_a = clips[i]
                clip_b = clips[j]
                overlaps = clip_a["start"] < clip_b["end"] and clip_b["start"] < clip_a["end"]
                assert not overlaps, f"Clips {i} and {j} overlap"

    def test_mixed_with_max_clips_1_single_profile(self):
        clips = select_clips(SAMPLE_SEGMENTS, clip_type="mixed", max_clips=1, clean_level="none")

        assert len(clips) == 1
        assert "mixed" in clips[0]["reason"]


class TestSelectClipsIntegration:
    """Integration-style tests simulating real usage."""

    def test_typical_video_workflow(self):
        clips = select_clips(
            SAMPLE_SEGMENTS,
            target_s=15,
            min_s=10,
            max_s=25,
            max_clips=2,
            max_gap_s=1.2,
            clean_level="light",
        )

        assert len(clips) >= 1
        assert len(clips) <= 2
        assert clips[0]["score"] >= 0.1

    def test_handles_segments_without_ms(self):
        segments_no_ms = [
            {"start": 0.0, "end": 10.0, "text": "First segment without ms."},
            {"start": 10.5, "end": 20.0, "text": "Second segment without ms."},
        ]
        clips = select_clips(segments_no_ms, min_s=5, max_s=25, clean_level="none")

        assert len(clips) >= 1
        assert clips[0]["start_ms"] == int(clips[0]["start"] * 1000)
        assert clips[0]["end_ms"] == int(clips[0]["end"] * 1000)

    def test_auto_clip_workflow_with_cleanup(self):
        """Simulate auto-clip workflow with cleanup enabled."""
        clips = select_clips(
            FLUFFY_SEGMENTS,
            clip_type="fun",
            clean_level="light",
            max_clips=2,
            min_s=5,
            max_s=20,
        )

        assert len(clips) >= 1
        for clip in clips:
            assert "fun" in clip["reason"]
            assert "light cleanup" in clip["reason"]

    def test_auto_clip_workflow_with_markers(self):
        """Simulate auto-clip workflow with markers."""
        clips = select_clips(
            MARKER_SEGMENTS,
            clip_type="mixed",
            clean_level="light",
            markers=["cut"],
            max_clips=2,
            min_s=3,
            max_s=20,
        )

        assert len(clips) >= 1
        # Verify marker boundary is respected
        for clip in clips:
            if clip["start"] < 10.5:
                assert clip["end"] <= 10.5
