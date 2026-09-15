"""Transcription pipeline orchestration (sync, runs in worker thread)."""

import logging
from typing import Any

from whisperx_api.asr_conditioning import asr_conditioning_options

from ...config import config
from ...state import AppState
from . import _whisperx
from .schemas import TranscriptionCommand
from .errors import (
    DiarizationUnavailableError,
    HfTokenRequiredError,
    WhisperxUnavailableError,
    ensure_asr_loaded,
)
from .whisperx_types import TranscriptionResult, parse_whisperx_result

logger = logging.getLogger(__name__)


def ensure_align_cached_sync(state: AppState, language_code: str) -> None:
    if _whisperx.whisperx is None:
        raise WhisperxUnavailableError(
            "whisperx library not available; cannot load alignment model"
        )
    if language_code in state.ALIGN_CACHE:
        return
    align_model, metadata = _whisperx.whisperx.load_align_model(
        language_code=language_code, device=state.DEVICE
    )
    state.ALIGN_CACHE[language_code] = (align_model, metadata)


def ensure_diarize_pipeline_sync(state: AppState) -> None:
    if getattr(state, "DIARIZE_PIPELINE", None) is not None:
        return

    if _whisperx.DiarizationPipeline is None:
        raise DiarizationUnavailableError(
            "DiarizationPipeline is not available (install whisperx diarization extras)"
        )
    if not config.hf_token:
        raise HfTokenRequiredError("HF_TOKEN is required for diarization")

    state.DIARIZE_PIPELINE = _whisperx.DiarizationPipeline(
        model_name=config.diarize_model,
        token=config.hf_token,
        device=state.DEVICE,
    )


def run_pipeline_sync(
    state: AppState,
    audio: Any,
    command: TranscriptionCommand,
) -> TranscriptionResult:
    """Full pipeline: ASR, optional align, optional diarization."""
    pipeline = ensure_asr_loaded(state)

    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "print_progress": config.debug,
        "verbose": config.debug,
    }
    if command.language:
        kwargs["language"] = command.language

    logger.info("make transcribation...")
    with asr_conditioning_options(pipeline, command.prompt, command.hotwords):
        raw_result = pipeline.transcribe(audio, **kwargs)
    result = parse_whisperx_result(raw_result)
    logger.info("transcribation done")

    if command.do_align:
        logger.info("make align...")
        detected_lang = result.get("language") or command.language or "en"
        ensure_align_cached_sync(state, detected_lang)
        align_model, metadata = state.ALIGN_CACHE[detected_lang]
        result = parse_whisperx_result(
            _whisperx.whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                state.DEVICE,
                return_char_alignments=False,
            )
        )
        logger.info("align done")

    if command.do_diarize:
        logger.info("make diarization...")
        ensure_diarize_pipeline_sync(state)
        diarize_segments = state.DIARIZE_PIPELINE(
            audio,
            num_speakers=command.num_speakers,
            min_speakers=command.min_speakers,
            max_speakers=command.max_speakers,
        )
        result = parse_whisperx_result(
            _whisperx.whisperx.assign_word_speakers(
                diarize_segments, result, fill_nearest=config.fill_nearest
            )
        )
        logger.info("diarization done")

    return result
