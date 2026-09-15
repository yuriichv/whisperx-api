"""Smoke tests for whisperx integration types (Python 3.11 + Pydantic TypeAdapter)."""


def test_type_adapter_initializes_at_import():
    """TypeAdapter(TranscriptionResult) must not fail at module import on py311."""
    from whisperx_api.features.transcription import whisperx_types  # noqa: F401

    assert whisperx_types._result_adapter is not None


def test_parse_whisperx_result_validates_minimal_payload():
    from whisperx_api.features.transcription.whisperx_types import parse_whisperx_result

    result = parse_whisperx_result({"segments": [], "language": "en"})

    assert result["language"] == "en"
    assert result["segments"] == []
