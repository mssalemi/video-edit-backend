# fastapi-media-transcriber

A local-first, containerized transcription API using FastAPI, ffmpeg, and faster-whisper.

## Features

- Transcribe video/audio files to text with segment timestamps
- Automatic clip selection using heuristics (no external APIs)
- **Content-aware cleanup**: Skip filler words, fluffy intros, and bad segments
- **Marker-based boundaries**: Use verbal cues like "cut" or "restart" to control clip boundaries
- **Punchline boost**: Fun clips prefer short, punchy segments with exclamations
- **EDL rendering**: Stitch multiple keep ranges into one video, cutting out mess-ups
- Trim video clips by time range
- Full auto-clip pipeline: transcribe -> select -> render
- Two input modes: file path (for mounted volumes) or direct file upload
- Auto language detection or specify language explicitly
- Multiple model sizes (tiny, base, small, medium, large)
- Clean JSON response with full text, segments, and metadata

## Quickstart

```bash
docker compose up --build
```

The API will be available at `http://localhost:3000`.

## API Endpoints

### Health Check

```bash
curl http://localhost:3000/health
```

Response:
```json
{"ok": true}
```

### Transcribe via Path

Place your media files in the `./data` directory (mounted to `/data` in the container):

```bash
curl -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/sample.mp4"}'
```

With optional parameters:

```bash
curl -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/sample.mp4", "language": "en", "model": "base"}'
```

### Transcribe via File Upload

```bash
curl -X POST http://localhost:3000/transcribe/upload \
  -F "file=@./data/sample.mp4"
```

With optional parameters:

```bash
curl -X POST http://localhost:3000/transcribe/upload \
  -F "file=@./data/sample.mp4" \
  -F "language=en" \
  -F "model=small"
```

### Transcription Response Format

```json
{
  "text": "Full transcription text here...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "start_ms": 0,
      "end_ms": 2500,
      "text": "First segment"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "start_ms": 2500,
      "end_ms": 5000,
      "text": "Second segment"
    }
  ],
  "meta": {
    "language": "en",
    "duration_s": 10.5,
    "engine": "faster-whisper",
    "model": "small"
  }
}
```

### Select Clips from Segments

Select the best clip windows from transcript segments using deterministic heuristics:

```bash
curl -X POST http://localhost:3000/select-clips \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 3.5, "text": "Hello and welcome to this video."},
      {"start": 3.8, "end": 7.2, "text": "Today we are going to talk about something interesting."},
      {"start": 7.5, "end": 12.0, "text": "This is a really important topic."},
      {"start": 12.3, "end": 16.8, "text": "Let me explain the first key point."},
      {"start": 17.0, "end": 21.5, "text": "Here is the second thing you need to know."}
    ],
    "clip_type": "mixed",
    "clean_level": "light",
    "max_clips": 2
  }'
```

With markers (creator "cut here" cheat code):

```bash
curl -X POST http://localhost:3000/select-clips \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 5.0, "text": "First good segment here."},
      {"start": 5.5, "end": 10.0, "text": "More good content."},
      {"start": 10.5, "end": 12.0, "text": "Okay cut that was bad."},
      {"start": 12.5, "end": 17.0, "text": "Starting fresh with new content."},
      {"start": 17.5, "end": 22.0, "text": "This is also good content."}
    ],
    "clip_type": "fun",
    "clean_level": "aggressive",
    "markers": ["cut", "restart", "take two"],
    "max_clips": 2
  }'
```

Response:
```json
{
  "clips": [
    {
      "start": 5.5,
      "end": 16.8,
      "duration_s": 11.3,
      "start_ms": 5500,
      "end_ms": 16800,
      "score": 0.72,
      "reason": "mixed + light cleanup: good duration, dense speech, shifted start"
    }
  ]
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| segments | required | Array of transcript segments |
| clip_type | "mixed" | Type of clips: "document", "fun", or "mixed" |
| clean_level | "light" | Cleanup level: "none", "light", or "aggressive" |
| markers | [] | Marker strings that act as hard boundaries (e.g., ["cut", "restart"]) |
| target_s | profile default | Target clip duration in seconds |
| min_s | profile default | Minimum clip duration |
| max_s | profile default | Maximum clip duration |
| max_clips | 3 | Maximum number of clips to return |
| max_gap_s | 1.2 | Max gap between segments when merging |

**Clip Types:**

| Type | Target | Min | Max | Best For |
|------|--------|-----|-----|----------|
| document | 30s | 15s | 60s | Longer, polished clips with clean endings |
| fun | 12s | 6s | 20s | Short, punchy clips with expressive content + punchline boost |
| mixed | 15s | 10s | 25s | Balanced selection; variety when max_clips >= 2 |

**Clean Levels:**

| Level | Behavior |
|-------|----------|
| `"none"` | No cleanup filtering (legacy behavior) |
| `"light"` | Penalize filler-heavy segments; shift start to skip fluffy intros |
| `"aggressive"` | Stronger penalties; filter out candidates with >50% bad segments |

**Markers:**
- Segments containing marker text become hard boundaries
- Clips will not start on, end on, or cross marker segments
- Useful for verbal cues like "cut", "restart", "take two", "that was bad"

**Scoring heuristics:**
- Prefers duration close to target
- Prefers dense speech (more words per second)
- Penalizes large internal gaps
- Bonus for clean endings (sentences ending in `.` `!` `?`)
- **Document:** Stricter gap penalties, stronger clean ending bonus
- **Fun:** Bonus for exclamations (`!` `?`), engaging words, and **punchline boost** for punchy single segments (5-12s with "wow", "insane", etc.)

### Trimming a Video Clip

Trim a video to a specific time range:

```bash
curl -X POST http://localhost:3000/trim \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/test.MOV",
    "start": 4.32,
    "end": 19.84
  }'
```

With custom output filename:

```bash
curl -X POST http://localhost:3000/trim \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/test.MOV",
    "start": 4.32,
    "end": 19.84,
    "output_name": "my_clip.mp4"
  }'
```

Response:
```json
{
  "input": "/data/test.MOV",
  "output": "/data/test_clip.mp4",
  "start": 4.32,
  "end": 19.84,
  "duration_s": 15.52
}
```

**Notes:**
- Output is saved to the same directory as the input file
- If `output_name` is omitted, output is named `{input}_clip.mp4`
- Uses fast stream copy when possible, falls back to re-encoding if needed

### Auto-Clip Pipeline

One-shot endpoint that chains: transcribe -> select-clips -> trim:

```bash
curl -X POST http://localhost:3000/auto-clip \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/test.MOV",
    "clip_type": "fun",
    "clean_level": "light",
    "max_clips": 2
  }'
```

With markers for creator control:

```bash
curl -X POST http://localhost:3000/auto-clip \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/selfie-video.MOV",
    "clip_type": "fun",
    "clean_level": "aggressive",
    "markers": ["cut", "restart", "that was dumb"],
    "max_clips": 3
  }'
```

Response:
```json
{
  "transcript": {
    "text": "Full transcription...",
    "segments": [...],
    "meta": {...}
  },
  "clips": [
    {
      "start": 5.5,
      "end": 14.2,
      "duration_s": 8.7,
      "start_ms": 5500,
      "end_ms": 14200,
      "score": 0.72,
      "reason": "fun + light cleanup: snappy duration, expressive, punchline, shifted start"
    }
  ],
  "renders": [
    {
      "output": "/data/test_clip_1.mp4",
      "start": 5.5,
      "end": 14.2,
      "duration_s": 8.7
    }
  ]
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| path | required | Path to input video |
| language | auto | Language code for transcription |
| model | small | Whisper model to use |
| clip_type | "mixed" | Type of clips: "document", "fun", or "mixed" |
| clean_level | "light" | Cleanup level: "none", "light", or "aggressive" |
| markers | [] | Marker strings that act as hard boundaries |
| target_s | profile default | Target clip duration |
| min_s | profile default | Minimum clip duration |
| max_s | profile default | Maximum clip duration |
| max_clips | 1 | Number of clips to generate |
| max_gap_s | 1.2 | Max gap for segment merging |

**Notes:**
- Output clips are named `{input}_clip_1.mp4`, `{input}_clip_2.mp4`, etc.
- All processing is local (no external API calls)
- Clip type determines duration defaults (see `/select-clips` for profile details)
- Clean level controls filler detection and start shifting
- Markers let you verbally mark "cut points" while recording
- Designed for automation and chaining with other tools

### Render EDL (Edit Decision List)

Stitch together multiple keep ranges into a single output video, cutting out mess-ups:

```bash
curl -X POST http://localhost:3000/render-edl \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/raw-recording.MOV",
    "keep_ms": [
      [0, 5000],
      [8000, 15000],
      [20000, 30000]
    ]
  }'
```

With options:

```bash
curl -X POST http://localhost:3000/render-edl \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/raw-recording.MOV",
    "keep_ms": [
      [0, 5000],
      [5100, 10000],
      [15000, 25000],
      [25500, 35000]
    ],
    "output": "/data/final-edit.mp4",
    "min_segment_ms": 500,
    "merge_gap_ms": 200,
    "max_segments": 10
  }'
```

Response:
```json
{
  "input": "/data/raw-recording.MOV",
  "output": "/data/raw-recording_edl.mp4",
  "kept_ms": [
    [0, 10000],
    [15000, 35000]
  ],
  "duration_s": 30.0,
  "segments_rendered": 2
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| path | required | Path to input video |
| keep_ms | required | List of `[start_ms, end_ms]` pairs to keep |
| output | `{input}_edl.mp4` | Output file path |
| min_segment_ms | 500 | Filter out segments shorter than this |
| merge_gap_ms | 100 | Merge segments within this gap |
| max_segments | unlimited | Maximum segments to render (keeps longest) |
| reencode | false | Force re-encoding (default: stream copy) |

**Processing Pipeline:**
1. **Validate**: Check all ranges have valid start < end
2. **Sort**: Order ranges by start time
3. **Merge**: Combine overlapping or close ranges (within `merge_gap_ms`)
4. **Filter**: Remove segments shorter than `min_segment_ms`
5. **Limit**: Keep only the longest `max_segments` if specified
6. **Render**: Trim each segment and concatenate into final output

**Use Cases:**
- Cut out verbal mess-ups, false starts, or awkward pauses
- Manual editing workflow: transcribe -> review -> mark keeps -> render
- Combine with `/select-clips` output for automated cleanup
- Post-processing after reviewing auto-generated clips

**Notes:**
- Uses fast stream copy by default; falls back to re-encode if needed
- Set `reencode: true` if you encounter audio/video sync issues
- Ranges that overlap or touch are automatically merged
- Output `kept_ms` shows the final merged ranges actually rendered

## Workflow Examples

### Manual: Transcribe, Review, Trim

1. Transcribe to get segments with timestamps
2. Review segments and pick interesting ranges
3. Call `/trim` with your chosen timestamps

### Automatic: One-Shot Clip Generation

```bash
# Generate up to 3 clips of ~15 seconds each
curl -X POST http://localhost:3000/auto-clip \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/interview.mp4", "max_clips": 3, "target_s": 15}'
```

### Programmatic: Custom Selection Logic

1. Call `/transcribe` to get segments
2. Apply your own selection logic
3. Call `/trim` for each clip you want

## File Input

Place media files in the `./data` directory. This folder is mounted to `/data` inside the container.

Supported formats: Any format supported by ffmpeg (mp4, mp3, wav, mkv, webm, mov, etc.)

## Models

Available Whisper models (via faster-whisper):

| Model  | Size   | Speed      | Quality    |
|--------|--------|------------|------------|
| tiny   | ~75MB  | Fastest    | Lower      |
| base   | ~150MB | Fast       | Good       |
| small  | ~500MB | Medium     | Better     |
| medium | ~1.5GB | Slower     | Great      |
| large  | ~3GB   | Slowest    | Best       |

**Default model:** `small`

**Performance notes:**
- First request loads the model (takes a few seconds)
- Subsequent requests reuse the cached model
- Running on CPU; GPU support would significantly speed up transcription
- For long files, expect roughly real-time processing with `small` model on modern CPUs

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable                | Default           | Description                |
|------------------------|-------------------|----------------------------|
| TRANSCRIBE_DEFAULT_MODEL | small            | Default Whisper model      |
| TRANSCRIBE_TMP_DIR     | /tmp/transcriber  | Temp directory for processing |

## Development

Run tests:

```bash
pip install -r requirements.txt
pytest tests/
```

## Future Enhancements

- S3/URL input mode
- GPU support
- Word-level timestamps
- Webhook callbacks for long files
- Batch processing
- LLM-powered clip selection (upgrade from heuristics)

---

## Content-Aware Cleanup (Implemented)

> **Status: Implemented in v0.2**

The clip selection system now includes content-aware cleanup to avoid filler-heavy segments and improve clip quality for selfie/talking-head videos.

### What It Does

When recording selfie videos, creators often produce raw footage with:
- **Filler words** — "uh", "um", "like", "you know", "so", "basically"
- **False starts** — "So today we're going to— actually let me start over"
- **Awkward intros** — Rambling warm-ups before getting to the point
- **Dead air** — Long pauses or silence

The cleanup system detects and handles these automatically.

### Per-Segment Quality Signals

Each segment is analyzed for:

| Signal | Description |
|--------|-------------|
| `word_count` | Total words in segment |
| `filler_count` | Count of filler words (uh, um, like, you know, so, basically, etc.) |
| `filler_ratio` | `filler_count / word_count` |
| `starts_fluffy` | Segment begins with filler or hedge words |
| `ends_clean` | Segment ends with sentence-final punctuation |

### "Bad Segment" Detection

| Rule | Trigger |
|------|---------|
| High filler density | `filler_ratio > 0.25` |
| Short fluff | `duration < 2s` AND `starts_fluffy` |
| Acknowledgement-only | `word_count <= 3` AND `starts_fluffy` |

### Clean Levels

| Level | Behavior |
|-------|----------|
| `"none"` | No cleanup (legacy behavior) |
| `"light"` | Penalize filler-heavy segments; shift start to skip fluffy intros |
| `"aggressive"` | Stronger penalties; filter out candidates with >50% bad segments |

### Marker Boundaries

Pass `markers` array to create hard clip boundaries:

```json
{"markers": ["cut", "restart", "take two", "that was dumb"]}
```

Clips will not cross segments containing marker text (case-insensitive).

### Punchline Boost (Fun Clips)

For `clip_type: "fun"`, single segments (5-12s) with exclamations or punch words get a significant score boost:
- Punch words: "wow", "no way", "insane", "crazy", "bro", "wild", "unreal", "incredible"
- Exclamation endings: `!`

---

## Future: LLM-Powered Judging

Once heuristics are proven, optionally upgrade to an LLM judge:

- Heuristics remain the **candidate generator** (fast, deterministic)
- LLM **re-ranks** top candidates based on `clip_type` and `clean_level`
- Enables nuanced judgments: "this segment sounds hesitant" vs. "this is confident delivery"
- Keeps latency low by only scoring the top N candidates

This keeps the system local-first while allowing opt-in AI enhancement.
