import logging
import os
import tempfile
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from fastapi.responses import JSONResponse

from app.settings import settings
from app.transcribe import transcribe_media, trim_video
from app.selection import select_clips
from app.edl import render_edl
from app.planner import (
    validate_segments,
    build_ai_plan_prompt,
    plan_edits_stub,
    plan_edits_heuristic,
    plan_edits_ai,
    plan_edits_ai_labels,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Media Transcriber API",
    description="Transcribe video/audio files using faster-whisper",
    version="0.1.0",
)


# --- Type Aliases ---

ClipType = Literal["document", "fun", "mixed"]
CleanLevel = Literal["none", "light", "aggressive"]
PlannerMode = Literal["stub", "heuristic", "ai", "ai_labels"]
UnsurePolicy = Literal["keep", "cut", "adjacent"]
Granularity = Literal["default", "chunked"]


# --- Pydantic Models ---


class HealthResponse(BaseModel):
    ok: bool


class TranscribeRequest(BaseModel):
    path: str
    language: Optional[str] = None
    model: Optional[str] = None
    granularity: Optional[Granularity] = "default"


class SegmentResponse(BaseModel):
    start: float
    end: float
    start_ms: int
    end_ms: int
    text: str


class MetaResponse(BaseModel):
    language: Optional[str]
    duration_s: Optional[float]
    engine: str
    model: str
    granularity: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    segments: list[SegmentResponse]
    meta: MetaResponse


class TrimRequest(BaseModel):
    path: str
    start: float
    end: float
    output_name: Optional[str] = None


class TrimResponse(BaseModel):
    input: str
    output: str
    start: float
    end: float
    duration_s: float


# --- Select Clips Models ---


class SegmentInput(BaseModel):
    start: float
    end: float
    text: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class SelectClipsRequest(BaseModel):
    segments: list[SegmentInput]
    clip_type: Optional[ClipType] = "mixed"
    clean_level: Optional[CleanLevel] = "light"
    markers: Optional[list[str]] = None
    target_s: Optional[float] = None  # Uses profile default if None
    min_s: Optional[float] = None     # Uses profile default if None
    max_s: Optional[float] = None     # Uses profile default if None
    max_clips: int = 3
    max_gap_s: float = 1.2


class ClipResult(BaseModel):
    start: float
    end: float
    duration_s: float
    start_ms: int
    end_ms: int
    score: float
    reason: str


class SelectClipsResponse(BaseModel):
    clips: list[ClipResult]


# --- Auto Clip Models ---


class AutoClipRequest(BaseModel):
    path: str
    language: Optional[str] = None
    model: Optional[str] = None
    clip_type: Optional[ClipType] = "mixed"
    clean_level: Optional[CleanLevel] = "light"
    markers: Optional[list[str]] = None
    target_s: Optional[float] = None  # Uses profile default if None
    min_s: Optional[float] = None     # Uses profile default if None
    max_s: Optional[float] = None     # Uses profile default if None
    max_clips: int = 1
    max_gap_s: float = 1.2


class RenderResult(BaseModel):
    output: str
    start: float
    end: float
    duration_s: float


class AutoClipResponse(BaseModel):
    transcript: TranscribeResponse
    clips: list[ClipResult]
    renders: list[RenderResult]


# --- Render EDL Models ---


class RenderEdlRequest(BaseModel):
    path: str
    keep_ms: list[list[int]]  # List of [start_ms, end_ms] pairs
    output: Optional[str] = None  # Output path (default: {input}_edl.mp4)
    min_segment_ms: int = 500  # Filter out segments shorter than this
    merge_gap_ms: int = 100  # Merge segments within this gap
    max_segments: Optional[int] = None  # Limit number of segments (None = unlimited)
    reencode: bool = False  # Force re-encoding


class RenderEdlResponse(BaseModel):
    input: str
    output: str
    kept_ms: list[list[int]]
    duration_s: float
    segments_rendered: int


# --- Plan Edits Models ---


class PlanEditsRequest(BaseModel):
    segments: list[SegmentInput]
    max_clips: int = 2
    clip_types: list[str] = ["document", "fun", "mixed"]
    preferred_clip_type: str = "document"
    markers: list[str] = []
    clean_level: CleanLevel = "light"
    min_clip_ms: int = 6000
    max_clip_ms: int = 60000
    max_keep_ranges: int = 20
    enforce_segment_boundaries: bool = True
    mode: PlannerMode = "heuristic"
    unsure_policy: Optional[UnsurePolicy] = None  # ai_labels mode only; defaults by clip type
    debug: bool = False  # ai_labels mode only; include labels and clip_sources in meta
    lead_in_ms: int = 300  # Expand keep range start by this amount (clamp to bounds, snap to segment)
    tail_out_ms: int = 300  # Expand keep range end by this amount (clamp to bounds, snap to segment)


class PlannedClip(BaseModel):
    clip_id: str
    clip_type: str
    title: str
    keep_ms: list[list[int]]
    total_ms: int
    reason: str
    confidence: float


class PlanEditsMeta(BaseModel):
    planner: str
    segments_in: int
    max_clips: int
    # Optional fields for ai_labels mode
    labels_count: Optional[int] = None
    unsure_policy: Optional[str] = None
    fallback_reason: Optional[str] = None
    # Debug fields (only present when debug=True)
    labels: Optional[list[dict]] = None
    clip_sources: Optional[list[dict]] = None

    class Config:
        extra = "allow"  # Allow additional fields from planner


class PlanEditsResponse(BaseModel):
    clips: list[PlannedClip]
    meta: PlanEditsMeta


# --- Make Clips Models ---


class MakeClipsRequest(BaseModel):
    """Request model for the make-clips one-call pipeline."""
    path: str  # Video file path (must be accessible in container)
    output_prefix: str  # Prefix for output filenames, e.g., "my_video" -> my_video_clip1.mp4
    # Optional plan-edits parameters
    max_clips: int = 2
    preferred_clip_type: str = "document"
    markers: list[str] = []
    min_clip_ms: int = 6000
    max_clip_ms: int = 60000
    unsure_policy: Optional[UnsurePolicy] = None
    lead_in_ms: int = 300
    tail_out_ms: int = 300
    # Transcription settings
    model: Optional[str] = None  # Whisper model (default from settings)
    language: Optional[str] = None


class MakeClipsClipResult(BaseModel):
    """Result for a single generated clip."""
    clip_id: str
    output_path: str
    keep_ms: list[list[int]]
    total_ms: int
    title: str


class MakeClipsResponse(BaseModel):
    """Response model for make-clips endpoint."""
    clips: list[MakeClipsClipResult]
    meta: dict


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_path(request: TranscribeRequest):
    """
    Transcribe a media file by path.

    The path must be accessible inside the container (e.g., /data/file.mp4).
    """
    try:
        logger.info(f"Received path request: {request.path}")

        # Validate path exists
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.path}",
            )

        # Perform transcription
        logger.info(f"Starting transcription: path={request.path}, model={request.model}, language={request.language}, granularity={request.granularity}")
        result = transcribe_media(request.path, request.model, request.language, request.granularity)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check server logs for details.",
        )


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    granularity: Optional[str] = Form("default"),
):
    """
    Transcribe an uploaded media file.

    Upload a file via multipart form with optional language, model, and granularity fields.
    Set granularity="chunked" to split long segments (>3000ms) into smaller chunks.
    """
    tmp_upload_path: Optional[str] = None

    try:
        logger.info(f"Received file upload: {file.filename}")

        # Ensure temp directory exists
        os.makedirs(settings.TMP_DIR, exist_ok=True)

        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            dir=settings.TMP_DIR,
            delete=False,
        ) as tmp_file:
            tmp_upload_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)

        # Perform transcription
        logger.info(f"Starting transcription: path={tmp_upload_path}, model={model}, language={language}, granularity={granularity}")
        result = transcribe_media(tmp_upload_path, model, language, granularity)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check server logs for details.",
        )
    finally:
        # Cleanup uploaded temp file
        if tmp_upload_path and os.path.exists(tmp_upload_path):
            os.remove(tmp_upload_path)
            logger.debug(f"Cleaned up uploaded file: {tmp_upload_path}")


@app.post("/trim", response_model=TrimResponse)
async def trim_endpoint(request: TrimRequest):
    """
    Trim a video file to a specific time range.

    Uses fast stream copy when possible, falls back to re-encoding if needed.
    Output is saved to the same directory as the input file.
    """
    try:
        logger.info(f"Received trim request: {request.path} [{request.start}s - {request.end}s]")

        # Validate path exists
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.path}",
            )

        # Validate time range
        if request.start < 0:
            raise HTTPException(
                status_code=400,
                detail="start must be >= 0",
            )
        if request.end <= request.start:
            raise HTTPException(
                status_code=400,
                detail="end must be greater than start",
            )

        # Determine output path
        input_dir = os.path.dirname(request.path)
        input_basename = os.path.basename(request.path)
        input_name, _ = os.path.splitext(input_basename)

        if request.output_name:
            output_filename = request.output_name
        else:
            output_filename = f"{input_name}_clip.mp4"

        output_path = os.path.join(input_dir, output_filename)

        # Perform trim
        result = trim_video(
            input_path=request.path,
            start=request.start,
            end=request.end,
            output_path=output_path,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Trim failed")
        raise HTTPException(
            status_code=500,
            detail="Trim failed. Check server logs for details.",
        )


@app.post("/select-clips", response_model=SelectClipsResponse)
async def select_clips_endpoint(request: SelectClipsRequest):
    """
    Select good clip windows from transcript segments.

    Uses deterministic heuristics to identify the best segments for clips.
    No external API calls - pure local computation.

    Clip types:
    - "document": Longer clips (~30s), strict gap penalties, clean endings
    - "fun": Shorter clips (~12s), expressive content, engaging words, punchline boost
    - "mixed": Balanced selection, variety when max_clips >= 2

    Clean levels:
    - "none": No cleanup filtering
    - "light": Penalize filler-heavy segments, shift start to skip fluff
    - "aggressive": Stronger penalties, filter out bad candidates
    """
    try:
        clip_type = request.clip_type or "mixed"
        clean_level = request.clean_level or "light"
        markers = request.markers or []

        logger.info(
            f"Received select-clips request: {len(request.segments)} segments, "
            f"clip_type={clip_type}, clean_level={clean_level}, "
            f"markers={len(markers)}, max_clips={request.max_clips}"
        )

        # Validate inputs
        if not request.segments:
            raise HTTPException(
                status_code=400,
                detail="segments list cannot be empty",
            )
        if request.min_s is not None and request.max_s is not None and request.min_s > request.max_s:
            raise HTTPException(
                status_code=400,
                detail="min_s cannot be greater than max_s",
            )
        if request.max_clips < 1:
            raise HTTPException(
                status_code=400,
                detail="max_clips must be at least 1",
            )

        # Convert pydantic models to dicts
        segments = [s.model_dump() for s in request.segments]

        # Run selection with clip type, clean_level, and markers
        clips = select_clips(
            segments=segments,
            target_s=request.target_s,
            min_s=request.min_s,
            max_s=request.max_s,
            max_clips=request.max_clips,
            max_gap_s=request.max_gap_s,
            clip_type=clip_type,
            clean_level=clean_level,
            markers=markers,
        )

        return {"clips": clips}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Clip selection failed")
        raise HTTPException(
            status_code=500,
            detail="Clip selection failed. Check server logs for details.",
        )


@app.post("/auto-clip", response_model=AutoClipResponse)
async def auto_clip_endpoint(request: AutoClipRequest):
    """
    Automatically transcribe, select clips, and render them.

    Pipeline: transcribe -> select-clips -> trim (for each clip)
    All processing is done locally without external HTTP calls.

    Clip types:
    - "document": Longer clips (~30s), strict gap penalties, clean endings
    - "fun": Shorter clips (~12s), expressive content, engaging words, punchline boost
    - "mixed": Balanced selection, variety when max_clips >= 2

    Clean levels:
    - "none": No cleanup filtering
    - "light": Penalize filler-heavy segments, shift start to skip fluff
    - "aggressive": Stronger penalties, filter out bad candidates

    Markers:
    - Pass marker strings (e.g., ["cut", "restart"]) to create hard boundaries
    - Clips will not cross segments containing marker text
    """
    try:
        clip_type = request.clip_type or "mixed"
        clean_level = request.clean_level or "light"
        markers = request.markers or []

        logger.info(
            f"Received auto-clip request: {request.path}, "
            f"clip_type={clip_type}, clean_level={clean_level}, markers={len(markers)}"
        )

        # Validate path exists
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.path}",
            )

        # Validate inputs
        if request.min_s is not None and request.max_s is not None and request.min_s > request.max_s:
            raise HTTPException(
                status_code=400,
                detail="min_s cannot be greater than max_s",
            )
        if request.max_clips < 1:
            raise HTTPException(
                status_code=400,
                detail="max_clips must be at least 1",
            )

        # Step 1: Transcribe
        logger.info(f"Auto-clip step 1: Transcribing {request.path}")
        transcript_result = transcribe_media(
            request.path,
            request.model,
            request.language,
        )

        # Step 2: Select clips with clip type, clean_level, and markers
        logger.info(
            f"Auto-clip step 2: Selecting clips from {len(transcript_result['segments'])} segments "
            f"with clip_type={clip_type}, clean_level={clean_level}"
        )
        clips = select_clips(
            segments=transcript_result["segments"],
            target_s=request.target_s,
            min_s=request.min_s,
            max_s=request.max_s,
            max_clips=request.max_clips,
            max_gap_s=request.max_gap_s,
            clip_type=clip_type,
            clean_level=clean_level,
            markers=markers,
        )

        # Step 3: Render clips
        logger.info(f"Auto-clip step 3: Rendering {len(clips)} clips")
        input_dir = os.path.dirname(request.path)
        input_basename = os.path.basename(request.path)
        input_name, _ = os.path.splitext(input_basename)

        renders = []
        for i, clip in enumerate(clips, start=1):
            output_filename = f"{input_name}_clip_{i}.mp4"
            output_path = os.path.join(input_dir, output_filename)

            trim_result = trim_video(
                input_path=request.path,
                start=clip["start"],
                end=clip["end"],
                output_path=output_path,
            )

            renders.append({
                "output": trim_result["output"],
                "start": trim_result["start"],
                "end": trim_result["end"],
                "duration_s": trim_result["duration_s"],
            })

        logger.info(f"Auto-clip complete: {len(renders)} clips rendered")

        return {
            "transcript": transcript_result,
            "clips": clips,
            "renders": renders,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Auto-clip failed")
        raise HTTPException(
            status_code=500,
            detail="Auto-clip failed. Check server logs for details.",
        )


@app.post("/render-edl", response_model=RenderEdlResponse)
async def render_edl_endpoint(request: RenderEdlRequest):
    """
    Render a final MP4 from multiple keep ranges.

    Stitches together kept segments to cut out mess-ups.
    Uses ffmpeg concat with stream copy when possible.

    Process:
    1. Validate and normalize ranges
    2. Sort, merge overlapping/close ranges
    3. Filter out segments shorter than min_segment_ms
    4. Limit to max_segments if specified
    5. Trim each segment and concatenate into final output
    """
    try:
        logger.info(
            f"Received render-edl request: {request.path}, "
            f"{len(request.keep_ms)} ranges, min_segment_ms={request.min_segment_ms}"
        )

        # Validate path exists
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.path}",
            )

        # Validate keep_ms is not empty
        if not request.keep_ms:
            raise HTTPException(
                status_code=400,
                detail="keep_ms list cannot be empty",
            )

        # Determine output path
        if request.output:
            output_path = request.output
        else:
            input_dir = os.path.dirname(request.path)
            input_basename = os.path.basename(request.path)
            input_name, _ = os.path.splitext(input_basename)
            output_path = os.path.join(input_dir, f"{input_name}_edl.mp4")

        # Render EDL
        result = render_edl(
            input_path=request.path,
            keep_ms=request.keep_ms,
            output_path=output_path,
            min_segment_ms=request.min_segment_ms,
            merge_gap_ms=request.merge_gap_ms,
            max_segments=request.max_segments,
            reencode=request.reencode,
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("EDL render failed")
        raise HTTPException(
            status_code=500,
            detail="EDL render failed. Check server logs for details.",
        )


@app.post("/plan-edits", response_model=PlanEditsResponse)
async def plan_edits_endpoint(request: PlanEditsRequest):
    """
    Generate an edit plan from transcript segments.

    Returns a list of planned clips, each with keep_ms ranges.
    This enables cutting out mess-ups within longer clips.

    Modes:
    - "stub": Returns trivial one-clip plan (for wiring/testing)
    - "heuristic": Deterministic multi-clip plan without AI
    - "ai": Calls Claude API to generate edit plans with timestamps
    - "ai_labels": Calls Claude to label segments (keep/cut/unsure), then
      deterministic code converts labels to clips. Better for retake detection.

    The response can be used with /render-edl to produce final clips.
    """
    try:
        mode = request.mode or "heuristic"
        logger.info(
            f"Received plan-edits request: {len(request.segments)} segments, "
            f"mode={mode}, max_clips={request.max_clips}"
        )

        # Validate inputs
        if not request.segments:
            raise HTTPException(
                status_code=400,
                detail="segments list cannot be empty",
            )

        if request.max_clips < 1:
            raise HTTPException(
                status_code=400,
                detail="max_clips must be at least 1",
            )

        if request.min_clip_ms >= request.max_clip_ms:
            raise HTTPException(
                status_code=400,
                detail="min_clip_ms must be less than max_clip_ms",
            )

        if request.preferred_clip_type not in request.clip_types:
            raise HTTPException(
                status_code=400,
                detail=f"preferred_clip_type '{request.preferred_clip_type}' must be in clip_types",
            )

        # Convert pydantic models to dicts
        segments = [s.model_dump() for s in request.segments]

        # Validate segments structure
        try:
            validate_segments(segments)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Handle modes
        if mode == "stub":
            result = plan_edits_stub(
                segments=segments,
                preferred_clip_type=request.preferred_clip_type,
            )
            return result

        elif mode == "heuristic":
            result = plan_edits_heuristic(
                segments=segments,
                max_clips=request.max_clips,
                clip_types=request.clip_types,
                preferred_clip_type=request.preferred_clip_type,
                markers=request.markers,
                clean_level=request.clean_level,
                min_clip_ms=request.min_clip_ms,
                max_clip_ms=request.max_clip_ms,
                max_keep_ranges=request.max_keep_ranges,
                enforce_segment_boundaries=request.enforce_segment_boundaries,
            )
            return result

        elif mode == "ai":
            # Call Claude API for AI planning
            try:
                result = plan_edits_ai(
                    segments=segments,
                    max_clips=request.max_clips,
                    clip_types=request.clip_types,
                    preferred_clip_type=request.preferred_clip_type,
                    markers=request.markers,
                    clean_level=request.clean_level,
                    min_clip_ms=request.min_clip_ms,
                    max_clip_ms=request.max_clip_ms,
                    max_keep_ranges=request.max_keep_ranges,
                    enforce_segment_boundaries=request.enforce_segment_boundaries,
                )
                return result
            except ValueError as e:
                raise HTTPException(status_code=500, detail=str(e))

        elif mode == "ai_labels":
            # Call Claude API to label segments, then convert to clips deterministically
            try:
                result = plan_edits_ai_labels(
                    segments=segments,
                    max_clips=request.max_clips,
                    clip_types=request.clip_types,
                    preferred_clip_type=request.preferred_clip_type,
                    markers=request.markers,
                    clean_level=request.clean_level,
                    min_clip_ms=request.min_clip_ms,
                    max_clip_ms=request.max_clip_ms,
                    max_keep_ranges=request.max_keep_ranges,
                    enforce_segment_boundaries=request.enforce_segment_boundaries,
                    unsure_policy=request.unsure_policy,
                    debug=request.debug,
                    lead_in_ms=request.lead_in_ms,
                    tail_out_ms=request.tail_out_ms,
                )
                return result
            except ValueError as e:
                raise HTTPException(status_code=500, detail=str(e))

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Must be 'stub', 'heuristic', 'ai', or 'ai_labels'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Plan edits failed")
        raise HTTPException(
            status_code=500,
            detail="Plan edits failed. Check server logs for details.",
        )


@app.post("/make-clips", response_model=MakeClipsResponse)
async def make_clips_endpoint(request: MakeClipsRequest):
    """
    One-call pipeline: transcribe -> plan-edits -> render clips.

    This endpoint orchestrates the full clip creation workflow:
    1. Transcribes the input video
    2. Plans edits using ai_labels mode
    3. Renders each planned clip to a file

    Output files are named deterministically: <output_prefix>_clip1.mp4, etc.

    Requires ANTHROPIC_API_KEY for the AI planning step.
    """
    try:
        logger.info(f"Received make-clips request: {request.path} -> {request.output_prefix}")

        # Validate input path exists
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.path}",
            )

        # Check for API key
        if not settings.ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY not configured. Required for make-clips.",
            )

        # Step 1: Transcribe
        logger.info(f"Step 1: Transcribing {request.path}")
        whisper_model = request.model or settings.DEFAULT_MODEL
        transcription = transcribe_media(
            request.path,
            model=whisper_model,
            language=request.language,
            granularity="chunked",  # Always use chunked for better segment boundaries
        )

        if not transcription.get("segments"):
            raise HTTPException(
                status_code=400,
                detail="Transcription produced no segments",
            )

        segments = transcription["segments"]
        logger.info(f"Transcription complete: {len(segments)} segments")

        # Step 2: Plan edits using ai_labels
        logger.info(f"Step 2: Planning edits with ai_labels mode")
        plan_result = plan_edits_ai_labels(
            segments=segments,
            max_clips=request.max_clips,
            clip_types=["document", "fun", "mixed"],
            preferred_clip_type=request.preferred_clip_type,
            markers=request.markers,
            clean_level="light",
            min_clip_ms=request.min_clip_ms,
            max_clip_ms=request.max_clip_ms,
            max_keep_ranges=20,
            enforce_segment_boundaries=True,
            unsure_policy=request.unsure_policy,
            debug=False,
            lead_in_ms=request.lead_in_ms,
            tail_out_ms=request.tail_out_ms,
        )

        planned_clips = plan_result.get("clips", [])
        if not planned_clips:
            raise HTTPException(
                status_code=400,
                detail="Planning produced no clips",
            )

        logger.info(f"Planning complete: {len(planned_clips)} clips")

        # Step 3: Render each clip
        logger.info(f"Step 3: Rendering {len(planned_clips)} clips")
        input_dir = os.path.dirname(request.path)
        results = []

        for i, clip in enumerate(planned_clips):
            clip_num = i + 1
            output_filename = f"{request.output_prefix}_clip{clip_num}.mp4"
            output_path = os.path.join(input_dir, output_filename)

            keep_ms = clip.get("keep_ms", [])
            if not keep_ms:
                logger.warning(f"Clip {clip_num} has no keep_ms, skipping")
                continue

            # For multi-range clips, we need to concatenate
            # For now, use the first range only (simplest implementation)
            # TODO: Support multi-range concatenation
            first_range = keep_ms[0]
            start_s = first_range[0] / 1000.0
            end_s = first_range[1] / 1000.0

            logger.info(f"Rendering clip {clip_num}: {start_s:.2f}s - {end_s:.2f}s -> {output_filename}")

            try:
                trim_result = trim_video(
                    input_path=request.path,
                    start=start_s,
                    end=end_s,
                    output_path=output_path,
                )

                results.append(MakeClipsClipResult(
                    clip_id=clip.get("clip_id", f"clip_{clip_num}"),
                    output_path=trim_result["output_path"],
                    keep_ms=keep_ms,
                    total_ms=clip.get("total_ms", int((end_s - start_s) * 1000)),
                    title=clip.get("title", f"Clip {clip_num}"),
                ))
            except Exception as e:
                logger.error(f"Failed to render clip {clip_num}: {e}")
                # Continue with other clips

        if not results:
            raise HTTPException(
                status_code=500,
                detail="Failed to render any clips",
            )

        logger.info(f"Make-clips complete: {len(results)} clips rendered")

        return MakeClipsResponse(
            clips=results,
            meta={
                "input_path": request.path,
                "output_prefix": request.output_prefix,
                "segments_transcribed": len(segments),
                "clips_planned": len(planned_clips),
                "clips_rendered": len(results),
                "planner": plan_result.get("meta", {}).get("planner", "ai_labels"),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Make-clips failed")
        raise HTTPException(
            status_code=500,
            detail=f"Make-clips failed: {str(e)}",
        )
