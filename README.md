# fastapi-media-transcriber

A local-first, containerized transcription API using FastAPI, ffmpeg, and faster-whisper.

## Features

- Transcribe video/audio files to text with segment timestamps
- **Chunked transcription**: Split long segments into smaller chunks for finer editing control
- Automatic clip selection using heuristics (no external APIs)
- **Content-aware cleanup**: Skip filler words, fluffy intros, and bad segments
- **Marker-based boundaries**: Use verbal cues like "cut" or "restart" to control clip boundaries
- **Punchline boost**: Fun clips prefer short, punchy segments with exclamations
- **EDL rendering**: Stitch multiple keep ranges into one video, cutting out mess-ups
- **AI planning layer**: Generate edit plans with stub/heuristic/AI modes
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

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/transcribe` | Transcribe media file by path |
| POST | `/transcribe/upload` | Transcribe uploaded file |
| POST | `/select-clips` | Select best clips from segments |
| POST | `/trim` | Trim video by time range |
| POST | `/auto-clip` | Full pipeline: transcribe → select → render |
| POST | `/render-edl` | Stitch keep ranges, cut out mess-ups |
| POST | `/plan-edits` | AI planning layer (stub/heuristic/ai modes) |
| POST | `/make-clips` | **NEW:** One-call AI pipeline: transcribe → plan → render |

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

With chunked granularity (splits long segments >3s into smaller chunks):

```bash
curl -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/sample.mp4", "granularity": "chunked"}'
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

With chunked granularity:

```bash
curl -X POST http://localhost:3000/transcribe/upload \
  -F "file=@./data/sample.mp4" \
  -F "granularity=chunked"
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
    "model": "small",
    "granularity": "default"
  }
}
```

**Transcription Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| path | required | Path to media file (for `/transcribe`) |
| file | required | Uploaded file (for `/transcribe/upload`) |
| language | auto | Language code (e.g., "en", "es") or auto-detect |
| model | "small" | Whisper model: tiny, base, small, medium, large |
| granularity | "default" | `"default"` or `"chunked"` - chunked splits segments >3s |

**Granularity Options:**
| Value | Behavior |
|-------|----------|
| `"default"` | Return segments as-is from Whisper |
| `"chunked"` | Split segments longer than 3000ms into smaller chunks by word boundaries |

Chunked mode is useful when you need finer-grained timestamps for precise editing or AI planning.

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

### Plan Edits (AI Planning Layer)

Generate an edit plan (clips with keep ranges) from transcript segments. Supports four modes:

**Stub mode** (for wiring/testing):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 10.0, "text": "Hello world"},
      {"start": 10.0, "end": 20.0, "text": "More content here"}
    ],
    "mode": "stub"
  }'
```

**Heuristic mode** (deterministic, no external APIs):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 10.0, "text": "Hello and welcome."},
      {"start": 10.0, "end": 20.0, "text": "Let me restart that."},
      {"start": 20.0, "end": 30.0, "text": "Okay, here is the real content."}
    ],
    "mode": "heuristic",
    "max_clips": 2,
    "markers": ["restart"]
  }'
```

**AI mode** (calls Claude API to generate intelligent edit plans):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 10.0, "text": "Hello world"},
      {"start": 10.5, "end": 15.0, "text": "Uh let me restart that"},
      {"start": 15.5, "end": 25.0, "text": "Here is the actual content you want"}
    ],
    "mode": "ai",
    "markers": ["restart"]
  }'
```

> **Note:** AI mode requires `ANTHROPIC_API_KEY` to be set. It uses Claude to detect mess-ups and plan edits intelligently.

**AI Labels mode** (labels segments, then deterministic code converts to clips):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"start": 0.0, "end": 5.0, "text": "Today we will discuss Python."},
      {"start": 5.0, "end": 10.0, "text": "Uh wait, let me restart that."},
      {"start": 10.0, "end": 15.0, "text": "Today we will discuss Python programming."},
      {"start": 15.0, "end": 20.0, "text": "Python is a great language!"},
      {"start": 20.0, "end": 25.0, "text": "Now let me talk about JavaScript."},
      {"start": 25.0, "end": 30.0, "text": "JavaScript is also popular."}
    ],
    "mode": "ai_labels",
    "max_clips": 2
  }'
```

> **Note:** AI Labels mode asks Claude to classify each segment (keep/cut/unsure + tags + story_id), then deterministic code converts labels to clips. This is better for detecting retakes where earlier content should be cut and later versions kept.

**AI Labels with debug output** (see the labels and which segments formed each clip):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [...],
    "mode": "ai_labels",
    "debug": true
  }'
```

**AI Labels with custom unsure_policy** (control how uncertain segments are handled):
```bash
curl -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [...],
    "mode": "ai_labels",
    "unsure_policy": "adjacent"
  }'
```

Response (heuristic mode):
```json
{
  "clips": [
    {
      "clip_id": "550e8400-e29b-41d4-a716-446655440000",
      "clip_type": "mixed",
      "title": "Hello and welcome",
      "keep_ms": [[0, 10000]],
      "total_ms": 10000,
      "reason": "heuristic: mixed selection",
      "confidence": 0.7
    }
  ],
  "meta": {
    "planner": "heuristic",
    "segments_in": 3,
    "max_clips": 2
  }
}
```

Response (AI mode):
```json
{
  "clips": [
    {
      "clip_id": "a1b2c3d4-...",
      "clip_type": "document",
      "title": "Actual Content",
      "keep_ms": [[15500, 25000]],
      "total_ms": 9500,
      "reason": "Skipped restart section, kept coherent content",
      "confidence": 0.85
    }
  ],
  "meta": {
    "planner": "ai",
    "segments_in": 3,
    "max_clips": 2
  }
}
```

Response (AI Labels mode):
```json
{
  "clips": [
    {
      "clip_id": "e5f6g7h8-...",
      "clip_type": "mixed",
      "title": "Today we will discuss Python",
      "keep_ms": [[10000, 20000]],
      "total_ms": 10000,
      "reason": "ai_labels: story 1, tags: ['clean_story']",
      "confidence": 0.85
    },
    {
      "clip_id": "i9j0k1l2-...",
      "clip_type": "mixed",
      "title": "Now let me talk about JavaScript",
      "keep_ms": [[20000, 30000]],
      "total_ms": 10000,
      "reason": "ai_labels: story 2, tags: ['topic_shift', 'clean_story']",
      "confidence": 0.85
    }
  ],
  "meta": {
    "planner": "ai_labels",
    "segments_in": 6,
    "max_clips": 2,
    "labels_count": 6
  }
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| segments | required | Array of transcript segments |
| mode | "heuristic" | Planner mode: "stub", "heuristic", "ai", or "ai_labels" |
| max_clips | 3 | Maximum clips to generate |
| clip_types | ["document", "fun", "mixed"] | Allowed clip types |
| preferred_clip_type | "mixed" | Preferred clip type |
| markers | [] | Marker words for mess-up detection |
| clean_level | "light" | Cleanup level: "none", "light", "aggressive" |
| min_clip_ms | 6000 | Minimum clip duration (ms) |
| max_clip_ms | 60000 | Maximum clip duration (ms) |
| max_keep_ranges | 10 | Max keep ranges per clip |
| enforce_segment_boundaries | true | Snap keep_ms to segment boundaries |
| unsure_policy | (by clip type) | ai_labels only: "keep", "cut", or "adjacent" |
| debug | false | ai_labels only: include labels and clip_sources in meta |
| lead_in_ms | 300 | Expand keep range starts by this amount (clamp to bounds) |
| tail_out_ms | 300 | Expand keep range ends by this amount (clamp to bounds) |

**Planner Modes:**

| Mode | Description |
|------|-------------|
| `stub` | Returns single clip covering entire transcript (for wiring) |
| `heuristic` | Deterministic multi-clip plan using markers, topic shifts, or time buckets |
| `ai` | Calls Claude API to intelligently detect mess-ups and plan edits |
| `ai_labels` | Claude labels each segment (keep/cut + tags + story_id), then deterministic code converts to clips. Best for retake detection. |

**Unsure Policy (ai_labels only):**

When the AI labels a segment as "unsure", this policy determines what to do:

| Policy | Behavior | Default For |
|--------|----------|-------------|
| `"keep"` | Treat unsure as keep (conservative) | "document" clips |
| `"cut"` | Treat unsure as cut (aggressive) | "fun" clips |
| `"adjacent"` | Keep if neighbors keep, cut if neighbors cut | "mixed" clips |

**Debug Output (ai_labels only):**

When `debug: true`, the response includes extra metadata:

```json
{
  "clips": [...],
  "meta": {
    "planner": "ai_labels",
    "labels": [
      {"idx": 0, "action": "keep", "tags": ["intro"], "story_id": 1},
      {"idx": 1, "action": "cut", "tags": ["retake_repeat"], "story_id": 1},
      {"idx": 2, "action": "keep", "tags": ["clean_story"], "story_id": 1}
    ],
    "clip_sources": [
      {
        "clip_index": 0,
        "story_id": 1,
        "kept_segment_indexes": [0, 2],
        "cut_segment_indexes": [1]
      }
    ],
    "unsure_policy": "keep"
  }
}
```

This is useful for debugging why certain segments were included or excluded.

**Validation & Fallback (ai_labels):**

AI-generated clips are automatically validated. If validation fails (empty output, out of bounds, etc.), the system falls back to heuristic mode. Check `meta.planner` to see which mode was actually used:
- `"ai_labels"` - AI labels were used successfully
- `"ai_labels_fallback"` - Fell back to heuristic (check `meta.fallback_reason`)

**Post-Processing (ai_labels):**

After AI labels are converted to clips, several deterministic post-processing steps are applied to produce cleaner, more publishable output:

1. **Label Normalization** — Tags that indicate bad content force `action="cut"` regardless of what the AI returned:
   - Cut-forcing tags: `false_start`, `retake_repeat`, `filler`, `restart_phrase`, `garbled`, `non_story`, `meta_commentary`, `outro`
   - Example: `{"action": "unsure", "tags": ["false_start"]}` → normalized to `{"action": "cut", "tags": ["false_start"]}`
   - Unknown/invalid tags are dropped silently
   - This prevents "unsure + bad tag" segments from being kept due to `unsure_policy=keep`

2. **Outro Auto-Cut** — Segments containing common wrap-up phrases are automatically marked as "cut" with an "outro" tag:
   - "let's see", "that's it", "anyway", "cool", "ok bye", "alright so", "alright then", "so yeah", "yeah so"
   - Short segments (<30 chars) containing these phrases are also cut
   - Prevents awkward "umm, let's see..." endings in clips

3. **Trailing Unsure Trimming** (document mode only) — Removes trailing "unsure" segments from clip ends:
   - Only applies when `preferred_clip_type == "document"` or `unsure_policy == "keep"`
   - If the last N segments in a clip were originally labeled "unsure" by the AI, they are trimmed
   - Prevents clips ending with mumbling, trailing thoughts, or uncertain content
   - Will not trim if it would make the clip shorter than `min_clip_ms`

4. **Smart Lead-in Range Drop** — For clips with multiple keep ranges, short "lead-in" ranges at the start are intelligently removed:
   - If the first range is <2500ms and there are 2+ ranges, it may be dropped
   - **Exceptions** (first range is kept if):
     - The range contains 2+ kept segments
     - Any segment in the range has a `clean_story` tag
   - Prevents clips starting with a tiny fragment before the main content
   - Example: `[[0, 2000], [10000, 25000]]` → `[[10000, 25000]]` (unless exceptions apply)

5. **Keep Range Expansion** — Keep ranges are expanded by `lead_in_ms` and `tail_out_ms`:
   - Start of each range is moved earlier by `lead_in_ms` (default: 300ms)
   - End of each range is moved later by `tail_out_ms` (default: 300ms)
   - Expansion is clamped to transcript bounds
   - Expanded positions are snapped to nearest segment boundaries
   - Creates more natural cuts that don't start/end abruptly mid-word

These rules run automatically in the order listed. They are designed to handle common issues in selfie/talking-head videos where creators trail off, have false starts, or AI labels segments as uncertain.

**Heuristic Strategy:**
1. If markers provided, split at marker segments
2. Otherwise, detect topic shifts via gaps (>3s) or reset phrases ("restart", "take two", etc.)
3. Fall back to equal time buckets
4. For each chunk, use `select_clips` to find best window

**Use Cases:**
- Test integration with stub mode before going live
- Deterministic clip planning with heuristic mode (no API costs)
- Intelligent mess-up detection with AI mode (requires ANTHROPIC_API_KEY)

**Chaining transcribe → plan-edits:**

```bash
# 1. Transcribe and save to file
curl -s -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/my-video.MOV", "granularity": "chunked"}' \
  > /tmp/chunked.json

# 2. Pipe segments into plan-edits
python3 -c '
import json
d = json.load(open("/tmp/chunked.json"))
print(json.dumps({
  "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in d["segments"]],
  "mode": "ai_labels",
  "max_clips": 2,
  "preferred_clip_type": "document",
  "lead_in_ms": 800,
  "tail_out_ms": 800
}))
' | curl -s -X POST http://localhost:3000/plan-edits \
  -H "Content-Type: application/json" \
  -d @- | python3 -m json.tool
```

### Make Clips (One-Call Pipeline)

New one-call endpoint that orchestrates the full clip creation workflow: transcribe → plan-edits → render.

```bash
curl -X POST http://localhost:3000/make-clips \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/my-video.MOV",
    "output_prefix": "my_video"
  }'
```

With options:

```bash
curl -X POST http://localhost:3000/make-clips \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/my-video.MOV",
    "output_prefix": "my_video",
    "max_clips": 3,
    "preferred_clip_type": "document",
    "min_clip_ms": 10000,
    "max_clip_ms": 45000,
    "lead_in_ms": 500,
    "tail_out_ms": 300
  }'
```

Response:
```json
{
  "clips": [
    {
      "clip_id": "a1b2c3d4-...",
      "output_path": "/data/my_video_clip1.mp4",
      "keep_ms": [[5000, 20000]],
      "total_ms": 15000,
      "title": "Today we will discuss Python"
    },
    {
      "clip_id": "e5f6g7h8-...",
      "output_path": "/data/my_video_clip2.mp4",
      "keep_ms": [[25000, 40000]],
      "total_ms": 15000,
      "title": "Now let me talk about JavaScript"
    }
  ],
  "meta": {
    "input_path": "/data/my-video.MOV",
    "output_prefix": "my_video",
    "segments_transcribed": 12,
    "clips_planned": 2,
    "clips_rendered": 2,
    "planner": "ai_labels"
  }
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| path | required | Path to input video |
| output_prefix | required | Prefix for output filenames (e.g., "my_video" → my_video_clip1.mp4) |
| max_clips | 2 | Maximum clips to generate |
| preferred_clip_type | "document" | Preferred clip type |
| markers | [] | Marker words for mess-up detection |
| min_clip_ms | 6000 | Minimum clip duration (ms) |
| max_clip_ms | 60000 | Maximum clip duration (ms) |
| unsure_policy | (by clip type) | How to handle uncertain segments |
| lead_in_ms | 300 | Expand clip starts by this amount |
| tail_out_ms | 300 | Expand clip ends by this amount |
| model | settings default | Whisper model for transcription |
| language | auto | Language code for transcription |

**Key Features:**
- **Deterministic filenames**: Output files are named `<output_prefix>_clip1.mp4`, `<output_prefix>_clip2.mp4`, etc.
- **Uses ai_labels mode**: Leverages Claude API for intelligent segment labeling
- **Full post-processing**: Includes label normalization, lead-in/tail-out expansion, bridge range dropping
- **Automatic cleanup**: Applies outro auto-cut and trailing unsure trimming

> **Note:** Requires `ANTHROPIC_API_KEY` to be set.

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

Environment variables (set in `docker-compose.yml` or `.env`):

| Variable                | Default           | Description                |
|------------------------|-------------------|----------------------------|
| TRANSCRIBE_DEFAULT_MODEL | small            | Default Whisper model      |
| TRANSCRIBE_TMP_DIR     | /tmp/transcriber  | Temp directory for processing |
| MAX_SEGMENT_MS         | 3000              | Max segment duration for chunked mode (ms) |
| ANTHROPIC_API_KEY      | (none)            | API key for AI planner mode |

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
