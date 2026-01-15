# Project Context for Claude

## What is this?
A local-first FastAPI transcription + video trimming service using ffmpeg and faster-whisper. Includes automatic clip selection via heuristics.

## Tech Stack
- Python 3.11, FastAPI, Uvicorn
- faster-whisper for transcription
- ffmpeg for audio extraction and video trimming
- Docker + docker-compose

## Key Files
- `app/main.py` - FastAPI endpoints
- `app/transcribe.py` - Core logic (transcription + trimming)
- `app/selection.py` - Clip selection heuristics
- `app/settings.py` - Environment config
- `Dockerfile` / `docker-compose.yml` - Container setup
- `tests/test_selection.py` - Unit tests for selection logic

## Current Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| POST | /transcribe | Transcribe by path (JSON body) |
| POST | /transcribe/upload | Transcribe uploaded file |
| POST | /trim | Trim video by time range |
| POST | /select-clips | Select best clips from segments (heuristic) |
| POST | /auto-clip | Full pipeline: transcribe -> select -> trim |

## Running
```bash
docker compose up --build
# API at http://localhost:3000
# Mount files in ./data -> /data in container
```

## Design Decisions
- Lazy model loading with module-level cache
- Stream copy first for fast trims, re-encode fallback
- Segments include both seconds (float) and milliseconds (int)
- Output clips saved next to input files
- Clip selection is deterministic (no randomness)
- /auto-clip calls internal functions directly (no HTTP self-calls)

## Selection Heuristics (v0)
The `/select-clips` endpoint scores candidates by:
- Duration proximity to target
- Word density (words per second)
- Gap penalty (less silence = better)
- Clean ending bonus (ends with . ! ?)

## Next Steps / Ideas
- LLM-powered clip selection (upgrade from heuristics)
- Batch trimming multiple segments in one call
- S3/URL input support
- GPU acceleration
