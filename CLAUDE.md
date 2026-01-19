# Project Context for Claude

## What is this?
A local-first FastAPI transcription + video trimming service using ffmpeg and faster-whisper. Includes automatic clip selection via heuristics AND AI-powered segment labeling via Claude API.

## Golden Path: One-Call Pipeline

```bash
curl -s -X POST http://localhost:3000/make-clips \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/my-video.MOV",
    "output_prefix": "my_video",
    "max_clips": 2,
    "preferred_clip_type": "document",
    "lead_in_ms": 800,
    "tail_out_ms": 800
  }' | python3 -m json.tool
```

## Known Limitations
- `/make-clips` currently renders only the **first keep range** per clip (multi-range EDL concatenation per clip is TODO)
- Multi-range clips are planned but rendered as single-range for now

## Tech Stack
- Python 3.11, FastAPI, Uvicorn
- faster-whisper for transcription
- ffmpeg for audio extraction and video trimming
- anthropic SDK for AI planning (ai_labels mode)
- Docker + docker-compose

## Anthropic API Key

Required for `ai_labels` mode and `/make-clips`. Set in environment:

```bash
# Option 1: .env file (recommended)
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Option 2: docker-compose.yml environment section (already wired)
# Just ensure .env exists with the key
```

## Key Files
- `app/main.py` - FastAPI endpoints and Pydantic models
- `app/planner.py` - AI planning logic, labels_to_clips, post-processing functions
- `app/transcribe.py` - Transcription + trimming + chunk_segments
- `app/selection.py` - Clip selection heuristics
- `app/settings.py` - Environment config
- `tests/test_planner.py` - Tests for planner functions (136 tests total)

## Current Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| POST | /transcribe | Transcribe by path (JSON body) |
| POST | /transcribe/upload | Transcribe uploaded file |
| POST | /trim | Trim video by time range |
| POST | /select-clips | Select best clips from segments (heuristic) |
| POST | /auto-clip | Full pipeline: transcribe -> select -> trim |
| POST | /render-edl | Stitch keep ranges, cut out mess-ups |
| POST | /plan-edits | AI planning (stub/heuristic/ai/ai_labels modes) |
| POST | /make-clips | One-call AI pipeline: transcribe -> plan -> render |

## Running
```bash
docker compose up --build
# API at http://localhost:3000
# Mount files in ./data -> /data in container

# Run tests
docker compose run --rm transcriber python -m pytest tests/ -v
```

## AI Labels Mode (Primary Planning Mode)

The `ai_labels` mode in `/plan-edits` is the main AI-powered planning approach:

1. Claude labels each segment with: `{idx, action, tags, story_id}`
2. Actions: "keep", "cut", "unsure"
3. Tags: clean_story, retake_repeat, false_start, filler, outro, intro, topic_shift, etc.
4. Deterministic code converts labels to clips

### Post-Processing Pipeline Order

In `plan_edits_ai_labels()` (planner.py), post-processing runs in this order:

```python
# 1. Outro auto-cut (is_outro_text)
# 2. Label normalization (normalize_labels) - force cut for CUT_FORCING_TAGS
# 3. Labels to clips (labels_to_clips)
# 4. Trailing unsure trimming (trim_trailing_unsure) - document mode only
# 5. Lead-in range drop (drop_short_leadin_ranges) - smart version with exceptions
# 6. Keep range expansion (expand_keep_ranges) - lead_in_ms, tail_out_ms
```

### Key Constants (planner.py)

```python
CUT_FORCING_TAGS = [
    "false_start", "retake_repeat", "filler", "restart_phrase",
    "garbled", "non_story", "meta_commentary", "outro",
]

OUTRO_PHRASES = [
    "let's see", "lets see", "that's it", "thats it", "anyway",
    "okay bye", "ok bye", "cool", "alright so", "alright then",
    "so yeah", "yeah so",
]

MIN_LEADIN_RANGE_MS = 2500  # Threshold for dropping short first ranges

# Defaults used when request params lead_in_ms/tail_out_ms not provided:
DEFAULT_LEAD_IN_MS = 300    # -> lead_in_ms param
DEFAULT_TAIL_OUT_MS = 300   # -> tail_out_ms param
```

### Key Functions (planner.py)

| Function | Purpose |
|----------|---------|
| `plan_edits_ai_labels()` | Main entry point for AI planning |
| `normalize_labels()` | Force cut for CUT_FORCING_TAGS, validate tags |
| `labels_to_clips()` | Convert labels to clip dicts with keep_ms |
| `drop_short_leadin_ranges()` | Remove short first ranges (with exceptions) |
| `expand_keep_ranges()` | Add lead_in_ms/tail_out_ms padding |
| `trim_trailing_unsure()` | Remove trailing "unsure" segments |
| `is_outro_text()` | Detect outro phrases in segment text |

### Unsure Policy

Controls how "unsure" labels are resolved:
- `"keep"` - treat as keep (default for "document" clips)
- `"cut"` - treat as cut (default for "fun" clips)
- `"adjacent"` - follow neighbors (default for "mixed" clips)

## Design Decisions
- Lazy model loading with module-level cache
- Stream copy first for fast trims, re-encode fallback
- Segments include both seconds (float) and milliseconds (int)
- Output clips saved next to input files
- Clip selection is deterministic (no randomness)
- `/auto-clip` calls internal functions directly (no HTTP self-calls)
- `/make-clips` also calls internal functions directly (transcribe_media → plan_edits_ai_labels → trim_video)
- AI labels validated with fallback to heuristic mode
- Post-processing always runs internally (debug=True) to get clip_sources

## Test Coverage

136 tests passing. Key test classes in `tests/test_planner.py`:
- `TestNormalizeLabels` - CUT_FORCING_TAGS normalization
- `TestDropShortLeadinRanges` - Smart lead-in range dropping with exceptions
- `TestExpandKeepRanges` - Keep range expansion with boundary snapping
- `TestMakeClipsEndpoint` - /make-clips integration tests

## Next Steps / Ideas
- Multi-range EDL concatenation in `/make-clips` (see Known Limitations)
- S3/URL input support
- GPU acceleration
- LLM re-ranking of heuristic candidates
