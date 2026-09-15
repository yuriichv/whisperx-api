"""Типы данных whisperx-пайплайна (integration boundary).

Не использовать для HTTP API — см. schemas.py.
"""

from typing import Any

from typing_extensions import TypedDict

from pydantic import TypeAdapter

# --- whisperx segment/word shapes (JSON-like, intentionally partial) ---


class Word(TypedDict, total=False):
    word: str
    start: float
    end: float
    speaker: str
    score: float


class Segment(TypedDict, total=False):
    text: str
    start: float
    end: float
    speaker: str | None
    words: list[Word]


class TranscriptBlock(TypedDict, total=False):
    start: float | None
    end: float | None
    text: str
    speaker: str | None


class TranscriptionResult(TypedDict, total=False):
    language: str
    text: str
    segments: list[Segment]


_result_adapter = TypeAdapter(TranscriptionResult)


def parse_whisperx_result(data: dict[str, Any]) -> TranscriptionResult:
    """Провалидировать ответ whisperx на границе integration adapter."""
    return _result_adapter.validate_python(data)
