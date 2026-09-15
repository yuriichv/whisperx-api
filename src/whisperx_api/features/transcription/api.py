"""HTTP transport for transcription feature."""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import uuid
from typing import Union

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from ...state import AppState, get_state

from . import _whisperx
from .schemas import (
    DiarizedJsonResponse,
    ErrorResponse,
    ResponseFormat,
    TranscriptionCommand,
    TranscriptionFormInput,
    TranscriptionJsonResponse,
    VerboseJsonResponse,
    get_whisper_tokenizer,
    validate_conditioning_tokens,
)
from .errors import (
    DiarizationUnavailableError,
    HfTokenRequiredError,
    ModelNotLoadedError,
    TranscriptionError,
    WhisperxUnavailableError,
    ensure_asr_loaded,
)
from .responses import (
    build_json_response,
    build_verbose_json,
    diarized_response,
    plain_text,
)
from .service import run_pipeline_sync

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audio"])

get_transcription_command = TranscriptionFormInput.as_form_dependency()


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    return f"{uuid.uuid4()}_{name}.tmp"


def _save_upload_once(upload: UploadFile, dst_path: str) -> None:
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _map_transcription_error(exc: TranscriptionError) -> HTTPException:
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (WhisperxUnavailableError, DiarizationUnavailableError)):
        return HTTPException(status_code=500, detail=str(exc))
    if isinstance(exc, HfTokenRequiredError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/v1/audio/transcriptions",
    response_model=None,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        200: {
            "description": "Transcription result (format depends on response_format)",
            "content": {
                "text/plain": {},
                "application/json": {
                    "schema": {
                        "oneOf": [
                            TranscriptionJsonResponse.model_json_schema(),
                            VerboseJsonResponse.model_json_schema(),
                            DiarizedJsonResponse.model_json_schema(),
                        ]
                    }
                },
            },
        },
    },
)
async def transcriptions(
    state: AppState = Depends(get_state),
    file: UploadFile = File(..., description="Аудиофайл для транскрипции"),
    command: TranscriptionCommand = Depends(get_transcription_command),
) -> Union[
    PlainTextResponse,
    TranscriptionJsonResponse,
    VerboseJsonResponse,
    DiarizedJsonResponse,
]:
    """OpenAI-compatible: POST /v1/audio/transcriptions."""
    logger.info(
        "Request file: %s, format: %s, model %s, language: %s",
        file.filename,
        command.response_format,
        command.model,
        command.language,
    )

    try:
        pipeline = ensure_asr_loaded(state)
    except ModelNotLoadedError as exc:
        raise _map_transcription_error(exc) from exc

    if _whisperx.whisperx is None:
        raise HTTPException(status_code=500, detail="whisperx library not available")

    if command.prompt or command.hotwords:
        validate_conditioning_tokens(
            command.prompt,
            command.hotwords,
            get_whisper_tokenizer(pipeline),
        )

    tmp_dir = tempfile.mkdtemp(prefix="whisperx_api_")
    in_path = os.path.join(tmp_dir, _safe_filename(file.filename or "audio"))

    try:
        _save_upload_once(file, in_path)
        audio = _whisperx.whisperx.load_audio(in_path)

        async with state.GPU_LOCK:
            try:
                result = await asyncio.to_thread(
                    run_pipeline_sync,
                    state,
                    audio,
                    command,
                )
            except TranscriptionError as exc:
                raise _map_transcription_error(exc) from exc

        if command.response_format == ResponseFormat.text:
            return PlainTextResponse(plain_text(result))

        if command.response_format == ResponseFormat.json:
            return build_json_response(result)

        if command.response_format == ResponseFormat.verbose_json:
            text = plain_text(result)
            return build_verbose_json(result, text, command.language)

        if command.response_format == ResponseFormat.diarized_json:
            return diarized_response(result, command.language)

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {command.response_format}",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
