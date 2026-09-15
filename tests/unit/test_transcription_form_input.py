"""Unit-тесты Pydantic-контракта transcription API.

Сценарии (spec transcription-api):
- нормализация prompt/hotwords: trim, пустая → None, внутренние пробелы/запятые as-is
- char limits: prompt 500, hotwords 1000
- token budget: prompt ≤100, hotwords ≤150, combined ≤200
- speaker params только при do_diarize
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from fixtures.transcription import (
    TO_COMMAND_DEFAULTS,
    FakeHfTokenizer,
    FakeTokenizer,
)
from whisperx_api.features.transcription.schemas import (
    COMBINED_MAX_TOKENS,
    HOTWORDS_MAX_CHARS,
    HOTWORDS_MAX_TOKENS,
    PROMPT_MAX_CHARS,
    PROMPT_MAX_TOKENS,
    TranscriptionFormInput,
    count_whisper_tokens,
    get_whisper_tokenizer,
    normalize_optional_str,
    validate_conditioning_tokens,
)


class CharTokenizer:
    """1 Whisper token = 1 символ (для предсказуемых лимитов в тестах)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def _form(**overrides) -> TranscriptionFormInput:
    return TranscriptionFormInput(**overrides)


def _command(**overrides):
    return _form(**overrides).to_command(**TO_COMMAND_DEFAULTS)


def test_normalize_optional_str_trim_and_empty():
    assert normalize_optional_str("  hello  ") == "hello"
    assert normalize_optional_str("   ") is None
    assert normalize_optional_str(None) is None


def test_prompt_strip_empty_becomes_none():
    form = _form(prompt="   ")
    assert form.prompt is None


def test_hotwords_preserves_internal_spaces_and_commas():
    form = _form(hotwords="  foo  bar, baz  ")
    assert form.hotwords == "foo  bar, baz"


def test_prompt_max_chars_validation():
    with pytest.raises(ValidationError):
        _form(prompt="x" * (PROMPT_MAX_CHARS + 1))


def test_hotwords_max_chars_validation():
    with pytest.raises(ValidationError):
        _form(hotwords="x" * (HOTWORDS_MAX_CHARS + 1))


def test_prompt_token_limit():
    text = "a" * (PROMPT_MAX_TOKENS + 1)
    with pytest.raises(HTTPException) as exc:
        validate_conditioning_tokens(text, None, CharTokenizer())
    assert exc.value.status_code == 422


def test_hotwords_token_limit():
    text = "a" * (HOTWORDS_MAX_TOKENS + 1)
    with pytest.raises(HTTPException) as exc:
        validate_conditioning_tokens(None, text, CharTokenizer())
    assert exc.value.status_code == 422


def test_combined_token_limit():
    prompt = "a" * PROMPT_MAX_TOKENS
    hotwords = "b" * (COMBINED_MAX_TOKENS - PROMPT_MAX_TOKENS + 1)
    with pytest.raises(HTTPException) as exc:
        validate_conditioning_tokens(prompt, hotwords, CharTokenizer())
    assert exc.value.status_code == 422
    assert "combined" in exc.value.detail


def test_combined_token_limit_within_budget():
    validate_conditioning_tokens("a" * 50, "b" * 100, CharTokenizer())


def test_count_whisper_tokens():
    assert count_whisper_tokens("abcd", CharTokenizer()) == 4


def test_count_whisper_tokens_hf_tokenizer_encoding():
    assert count_whisper_tokens("abcd", FakeHfTokenizer()) == 4


def test_get_whisper_tokenizer_from_pipeline_tokenizer():
    pipeline_tokenizer = FakeTokenizer()
    pipeline = SimpleNamespace(
        tokenizer=pipeline_tokenizer,
        model=SimpleNamespace(hf_tokenizer=FakeHfTokenizer()),
    )
    assert get_whisper_tokenizer(pipeline) is pipeline_tokenizer


def test_get_whisper_tokenizer_fallback_to_hf_tokenizer():
    hf_tokenizer = FakeHfTokenizer()
    pipeline = SimpleNamespace(tokenizer=None, model=SimpleNamespace(hf_tokenizer=hf_tokenizer))
    assert get_whisper_tokenizer(pipeline) is hf_tokenizer


def test_get_whisper_tokenizer_raises_503_when_unavailable():
    pipeline = SimpleNamespace(tokenizer=None, model=SimpleNamespace(hf_tokenizer=None))
    with pytest.raises(HTTPException) as exc:
        get_whisper_tokenizer(pipeline)
    assert exc.value.status_code == 503
    assert "tokenizer" in exc.value.detail


def test_speaker_validation_only_when_diarize():
    with pytest.raises(ValueError, match="num_speakers must be >= 1"):
        _command(diarize="true", num_speakers=0)


def test_speaker_params_ignored_without_diarize():
    _command(num_speakers=0)


def test_unsupported_response_format():
    with pytest.raises(ValueError, match="Unsupported response_format"):
        _command(response_format="unknown")
