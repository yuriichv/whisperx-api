"""Application errors for the transcription feature."""

from typing import Any

MODEL_NOT_LOADED_MESSAGE = "Model is not loaded yet"


class TranscriptionError(Exception):
    """Base class for transcription domain errors."""


class ModelNotLoadedError(TranscriptionError):
    """ASR model is not loaded yet."""

    def __init__(self, message: str = MODEL_NOT_LOADED_MESSAGE) -> None:
        super().__init__(message)


def ensure_asr_loaded(state: Any) -> Any:
    """Вернуть ASR pipeline или поднять ModelNotLoadedError."""
    if state.ASR_PIPELINE is None:
        raise ModelNotLoadedError()
    return state.ASR_PIPELINE


class WhisperxUnavailableError(TranscriptionError):
    """whisperx library is not installed or failed to import."""


class DiarizationUnavailableError(TranscriptionError):
    """DiarizationPipeline is not available."""


class HfTokenRequiredError(TranscriptionError):
    """HF token is required for diarization but missing."""
