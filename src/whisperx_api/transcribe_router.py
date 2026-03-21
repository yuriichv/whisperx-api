import sys
from typing import List, Optional

# Global FastAPI app reference removed – state is provided via dependency injection

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse

import logging
import os
import re
import uuid
import shutil
import tempfile
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict


from .config import config
from .state import AppState, get_state

logging.basicConfig(
    level=config.log_level,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ---- Optional imports for optional dependencies ----
try:
    import torch
except Exception:
    torch = None

# PyTorch load workaround (only if torch is available)
if torch is not None:
    _original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _trusted_load

try:
    import whisperx
except Exception:
    whisperx = None  # type: ignore


try:
    from whisperx.diarize import DiarizationPipeline
except ImportError:
    DiarizationPipeline = None


# -----------------------------
# Lifecycle
# -----------------------------
def _select_device(device: Optional[str]) -> str:
    if device:
        return device
    # If torch is unavailable, fall back to CPU
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _select_compute_type(device: str, compute_type: Optional[str]) -> str:
    if compute_type:
        return compute_type
    return "float16" if device == "cuda" else "int8"


# NOTE: startup_load functionality has been moved to ``AppState.startup_load``.
# The original function was removed to avoid duplicate logic.
# Any calls should now use ``await app.state.startup_load()``.


async def shutdown_cleanup() -> None:
    """
    Optional cleanup. Usually not needed.
    """
    return


# -----------------------------
# Helpers
# -----------------------------
def _bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    return f"{uuid.uuid4()}_{name}.tmp"


def _save_upload_once(upload: UploadFile, dst_path: str) -> None:
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _plain_text(result: Dict[str, Any]) -> str:
    return (
        result.get("text")
        or "".join([seg.get("text", "") for seg in result.get("segments", [])])
    ).strip()


def _build_verbose_json(
    result: Dict[str, Any], text: str, language: Optional[str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task": "transcribe",
        "language": result.get("language") or language,
        "text": text,
        "segments": [],
    }
    for i, seg in enumerate(result.get("segments", [])):
        out_seg: Dict[str, Any] = {
            "id": i,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": (seg.get("text") or "").strip(),
        }
        if "words" in seg:
            out_seg["words"] = seg["words"]
        if "speaker" in seg:
            out_seg["speaker"] = seg["speaker"]
        out["segments"].append(out_seg)
    return out


def _build_diarized_json(
    result: Dict[str, Any], speaker_text: str, language: Optional[str]
) -> Dict[str, Any]:
    speakers = set()
    segments_out = []
    for seg in result.get("segments", []):
        spk = seg.get("speaker")
        if spk:
            speakers.add(spk)
        segments_out.append(
            {
                "type": "transcript.text.segment",
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": (seg.get("text") or "").strip(),
                "speaker": spk,
            }
        )
    return {
        "language": result.get("language") or language,
        "text": speaker_text,
        "speakers": sorted(list(speakers)),
        "segments": segments_out,
    }


def _build_diarized_text(result: Dict[str, Any]) -> str:
    segments: List[Dict[str, Any]] = result.get("segments") or []

    lines: List[str] = []
    current_speaker: Optional[str] = None
    current_chunks: List[str] = []

    def flush():
        nonlocal current_speaker, current_chunks
        if current_speaker is None:
            return
        text = " ".join(current_chunks).strip()
        if text:
            lines.append(f"{current_speaker}: {text}")
        current_speaker = None
        current_chunks = []

    for seg in segments:
        speaker = seg.get("speaker") or "UNKNOWN"
        chunk = seg.get("text") or ""
        if not chunk:
            continue

        if current_speaker is None:
            current_speaker = speaker

        if speaker != current_speaker:
            flush()
            current_speaker = speaker

        current_chunks.append(chunk)

    flush()
    return "\n".join(lines)


# -----------------------------
# Lazy-loaders (must be called under GPU_LOCK)
# -----------------------------
def _ensure_align_cached_sync(state: AppState, language_code: str) -> None:

    if whisperx is None:
        raise HTTPException(
            status_code=500,
            detail="whisperx library not available; cannot load alignment model",
        )
    if language_code in state.ALIGN_CACHE:
        return
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=state.DEVICE
    )
    state.ALIGN_CACHE[language_code] = (align_model, metadata)


def _ensure_diarize_pipeline_sync(state: AppState) -> None:

    if getattr(state, "DIARIZE_PIPELINE", None) is not None:
        return

    if DiarizationPipeline is None:
        raise HTTPException(
            status_code=500,
            detail="DiarizationPipeline is not available (install whisperx diarization extras)",
        )
    if not config.hf_token:
        raise HTTPException(
            status_code=500, detail="HF_TOKEN is required for diarization"
        )

    state.DIARIZE_PIPELINE = DiarizationPipeline(
        token=config.hf_token, device=state.DEVICE
    )


# -----------------------------
# Core pipeline (sync, runs in thread; call under GPU_LOCK)
# -----------------------------
def _run_pipeline_sync(
    state: AppState,
    audio,
    language: Optional[str],
    do_align: bool,
    do_diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Dict[str, Any]:
    """
    Full pipeline: ASR, Align, Diarization.
    """

    if state.ASR_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # ASR
    logger.info("make transcribation...")
    kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "print_progress": config.debug,
        "verbose": config.debug,
    }
    if language:
        kwargs["language"] = language
        result = state.ASR_PIPELINE.transcribe(audio, **kwargs)
    logger.info("transcribation done")

    # Align
    if do_align:
        logger.info("make align...")
        detected_lang = result.get("language") or language or "en"
        # либо модель берем из кэша, либо добавляем в кэш и берем из кэша
        _ensure_align_cached_sync(state, detected_lang)
        align_model, metadata = state.ALIGN_CACHE[detected_lang]
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            state.DEVICE,
            return_char_alignments=False,
        )
        logger.info("align done")

    # Diarize
    if do_diarize:
        logger.info("make diarization...")
        _ensure_diarize_pipeline_sync(state)
        diarize_segments = state.DIARIZE_PIPELINE(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("diarization done")

    return result


# -----------------------------
# API
# -----------------------------
router = APIRouter(tags=["audio"])


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    state: AppState = Depends(get_state),
    file: UploadFile = File(...),
    # OpenAI-like fields
    model: Optional[str] = Form(None),  # совместимость
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),  # совместимость. не используется
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),  # совместимость. не используется
    timestamp_granularities: Optional[List[str]] = Form(None),
    # WhisperX extensions
    align: Optional[str] = Form(None),
    diarize: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    """
    OpenAI-compatible: POST /v1/audio/transcriptions
    """
    logger.info(
        f"Request file: {file.filename}, format: {response_format}, model {model}, language: {language}"
    )

    if state.ASR_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # todo: вынести в справочник список поддерживаемых моделей
    if model and model not in (config.default_model, "whisper-1", "whisper-large-v3"):
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    if timestamp_granularities:
        raise HTTPException(
            status_code=400,
            detail="timestamp_granularities is not supported in this server",
        )

    rf = (response_format or "json").strip().lower()
    language = language or config.default_language
    do_align = _bool(align, default=config.default_align)
    do_diarize = _bool(
        diarize, default=(rf == "diarized_json") or config.default_diarize
    )

    tmp_dir = tempfile.mkdtemp(prefix="whisperx_api_")
    in_path = os.path.join(tmp_dir, _safe_filename(file.filename))

    try:
        # Аудио пишем на диск ОДИН раз
        _save_upload_once(file, in_path)
        audio = whisperx.load_audio(in_path)

        # mutex for GPU
        async with state.GPU_LOCK:
            # run sync pipeline in a worker thread so event loop isn't blocked
            result = await asyncio.to_thread(
                _run_pipeline_sync,
                state,
                audio,
                language,
                do_align,
                do_diarize,
                min_speakers,
                max_speakers,
            )

        # Response formats
        if rf == "text":
            text = (
                result.get("text")
                or "".join(
                    [seg.get("text", "") for seg in result.get("segments", [])]
                ).strip()
            )
            return PlainTextResponse(text)

        if rf == "json":
            text = (
                result.get("text")
                or "".join(
                    [seg.get("text", "") for seg in result.get("segments", [])]
                ).strip()
            )
            return JSONResponse({"text": text})

        if rf == "verbose_json":
            text = (
                result.get("text")
                or "".join(
                    [seg.get("text", "") for seg in result.get("segments", [])]
                ).strip()
            )
            return JSONResponse(_build_verbose_json(result, text, language))

        if rf == "diarized_json":
            if not do_diarize:
                raise HTTPException(
                    status_code=400,
                    detail="response_format=diarized_json requires diarize=true (or WHISPERX_DEFAULT_DIARIZE=true)",
                )
            speaker_text = _build_diarized_text(result)
            # speaker_text = "".join([f"{(seg.get('speaker') or '')}: {seg.get('text', '')}\n" for seg in result.get("segments", [])]).strip()
            return JSONResponse(_build_diarized_json(result, speaker_text, language))

        raise HTTPException(
            status_code=400, detail=f"Unsupported response_format: {response_format}"
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
