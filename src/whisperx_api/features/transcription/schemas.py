"""Pydantic-контракты внешнего HTTP API транскрипции (OpenAI-compatible)."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from whisperx_api.supported_models import supported_model_ids

from .form_binding import build_form_command_dependency

PROMPT_MAX_CHARS = 500
HOTWORDS_MAX_CHARS = 1000
PROMPT_MAX_TOKENS = 100
HOTWORDS_MAX_TOKENS = 150
COMBINED_MAX_TOKENS = 200

_SPEAKER_COUNT_FIELDS = ("num_speakers", "min_speakers", "max_speakers")


class ResponseFormat(StrEnum):
    json = "json"
    text = "text"
    verbose_json = "verbose_json"
    diarized_json = "diarized_json"


class ErrorResponse(BaseModel):
    """Стандартная ошибка FastAPI (detail)."""

    detail: str


def normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def parse_bool_form(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def count_whisper_tokens(text: str, tokenizer: Any) -> int:
    if not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer must provide encode()")
    encoded = tokenizer.encode(text)
    # faster-whisper Tokenizer.encode → list[int]; hf_tokenizer.encode → Encoding
    if hasattr(encoded, "ids"):
        return len(encoded.ids)
    return len(encoded)


def get_whisper_tokenizer(asr_pipeline: Any) -> Any:
    """Вернуть tokenizer для token-budget validation.

    WhisperX FasterWhisperPipeline хранит faster_whisper.Tokenizer на pipeline;
    при auto-language до первого transcribe — fallback на model.hf_tokenizer.
    """
    pipeline_tokenizer = getattr(asr_pipeline, "tokenizer", None)
    if pipeline_tokenizer is not None:
        return pipeline_tokenizer

    model = getattr(asr_pipeline, "model", None)
    if model is not None:
        hf_tokenizer = getattr(model, "hf_tokenizer", None)
        if hf_tokenizer is not None:
            return hf_tokenizer

    raise HTTPException(
        status_code=503,
        detail="ASR tokenizer is not available for conditioning validation",
    )


def _validate_speaker_params(
    do_diarize: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    if not do_diarize:
        return
    for name, value in zip(
        _SPEAKER_COUNT_FIELDS,
        (num_speakers, min_speakers, max_speakers),
        strict=True,
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ValueError("min_speakers must not exceed max_speakers")


def validate_conditioning_tokens(
    prompt: str | None,
    hotwords: str | None,
    tokenizer: Any,
) -> None:
    prompt_tokens = count_whisper_tokens(prompt, tokenizer) if prompt else 0
    hotwords_tokens = count_whisper_tokens(hotwords, tokenizer) if hotwords else 0
    combined = prompt_tokens + hotwords_tokens

    if prompt and prompt_tokens > PROMPT_MAX_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"prompt exceeds {PROMPT_MAX_TOKENS} Whisper tokens",
        )
    if hotwords and hotwords_tokens > HOTWORDS_MAX_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"hotwords exceeds {HOTWORDS_MAX_TOKENS} Whisper tokens",
        )
    if combined > COMBINED_MAX_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"prompt and hotwords combined exceed {COMBINED_MAX_TOKENS} "
                "Whisper tokens"
            ),
        )


class TranscriptionFormInput(BaseModel):
    """Wire-контракт multipart form-полей (без file)."""

    model: str | None = Field(
        default=None,
        description="OpenAI compatibility; не переключает ASR-модель на сервере",
    )
    language: str | None = Field(
        default=None,
        description="Язык аудио; пустое значение — auto-detect",
    )
    prompt: str | None = Field(
        default=None,
        max_length=PROMPT_MAX_CHARS,
        description="OpenAI-compatible initial context for ASR",
    )
    hotwords: str | None = Field(
        default=None,
        max_length=HOTWORDS_MAX_CHARS,
        description="WhisperX extension: строка терминов для faster-whisper",
    )
    response_format: str = Field(
        default="json",
        description="Формат ответа: json | text | verbose_json | diarized_json",
    )
    temperature: float | None = Field(
        default=None,
        description="OpenAI compatibility; не используется",
    )
    timestamp_granularities: list[str] | None = Field(
        default=None,
        description="OpenAI compatibility; не поддерживается сервером",
    )
    align: str | None = Field(
        default=None,
        description="WhisperX: включить word-level alignment (true/false)",
    )
    diarize: str | None = Field(
        default=None,
        description="WhisperX: включить диаризацию спикеров (true/false)",
    )
    num_speakers: int | None = Field(
        default=None,
        description="Точное число спикеров (валидируется только при diarize)",
    )
    min_speakers: int | None = Field(
        default=None,
        description="Минимальное число спикеров (только при diarize)",
    )
    max_speakers: int | None = Field(
        default=None,
        description="Максимальное число спикеров (только при diarize)",
    )

    @field_validator("language", "prompt", "hotwords", "model", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        return normalize_optional_str(value)

    @field_validator("response_format", mode="before")
    @classmethod
    def _normalize_response_format(cls, value: str | None) -> str:
        return (value or "json").strip().lower()

    def to_command(
        self,
        *,
        default_model: str,
        default_language: str,
        default_align: bool,
        default_diarize: bool,
    ) -> "TranscriptionCommand":
        if self.timestamp_granularities:
            raise ValueError(
                "timestamp_granularities is not supported in this server"
            )

        if self.model and self.model not in supported_model_ids(default_model):
            raise ValueError(f"Unsupported model: {self.model}")

        try:
            rf = ResponseFormat(self.response_format)
        except ValueError:
            raise ValueError(
                f"Unsupported response_format: {self.response_format}"
            ) from None

        language = self.language or normalize_optional_str(default_language)

        do_diarize = parse_bool_form(
            self.diarize,
            default=(rf == ResponseFormat.diarized_json) or default_diarize,
        )
        do_align = parse_bool_form(self.align, default=default_align)
        if do_diarize and self.align is None:
            do_align = True

        _validate_speaker_params(
            do_diarize,
            self.num_speakers,
            self.min_speakers,
            self.max_speakers,
        )

        if rf == ResponseFormat.diarized_json and not do_diarize:
            raise ValueError(
                "response_format=diarized_json requires diarize=true "
                "(or WHISPERX_DEFAULT_DIARIZE=true)"
            )

        return TranscriptionCommand(
            model=self.model,
            language=language,
            prompt=self.prompt,
            hotwords=self.hotwords,
            response_format=rf,
            temperature=self.temperature,
            do_align=do_align,
            do_diarize=do_diarize,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

    @classmethod
    def as_form_dependency(cls) -> Callable[..., "TranscriptionCommand"]:
        from ...config import config

        def resolve(form: TranscriptionFormInput) -> TranscriptionCommand:
            return form.to_command(
                default_model=config.default_model,
                default_language=config.default_language,
                default_align=config.default_align,
                default_diarize=config.default_diarize,
            )

        return build_form_command_dependency(cls, resolve)


class TranscriptionCommand(BaseModel):
    """Application-контракт: нормализованный запрос для пайплайна."""

    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    hotwords: str | None = None
    response_format: ResponseFormat = ResponseFormat.json
    temperature: float | None = None
    do_align: bool = False
    do_diarize: bool = False
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class TranscriptionJsonResponse(BaseModel):
    """response_format=json"""

    text: str


class VerboseJsonWord(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None


class VerboseJsonSegment(BaseModel):
    id: int
    start: float | None = None
    end: float | None = None
    text: str
    words: list[VerboseJsonWord] | None = None
    speaker: str | None = None


class VerboseJsonResponse(BaseModel):
    """response_format=verbose_json"""

    task: Literal["transcribe"] = "transcribe"
    language: str | None = None
    text: str
    segments: list[VerboseJsonSegment]


class DiarizedSegmentOut(BaseModel):
    type: Literal["transcript.text.segment"] = "transcript.text.segment"
    start: float | None = None
    end: float | None = None
    text: str
    speaker: str | None = None


class DiarizedJsonResponse(BaseModel):
    """response_format=diarized_json"""

    language: str | None = None
    text: str
    speakers: list[str]
    segments: list[DiarizedSegmentOut]
