"""Contract tests for transcription HTTP Pydantic schemas."""

import pytest

from fixtures.transcription import TO_COMMAND_DEFAULTS
from whisperx_api.features.transcription.responses import (
    build_json_response,
    build_verbose_json,
)
from whisperx_api.features.transcription.schemas import (
    DiarizedJsonResponse,
    ErrorResponse,
    ResponseFormat,
    TranscriptionCommand,
    TranscriptionFormInput,
    TranscriptionJsonResponse,
    VerboseJsonResponse,
    VerboseJsonWord,
)


def test_transcription_form_input_openapi_schema():
    schema = TranscriptionFormInput.model_json_schema()
    assert "response_format" in schema["properties"]
    assert "hotwords" in schema["properties"]
    assert schema["properties"]["language"]["description"]


def test_transcription_command_openapi_schema():
    schema = TranscriptionCommand.model_json_schema()
    assert "do_align" in schema["properties"]
    assert "do_diarize" in schema["properties"]


def test_error_response_schema():
    schema = ErrorResponse.model_json_schema()
    assert "detail" in schema["properties"]


def test_transcription_form_resolves_diarize_from_format():
    cmd = TranscriptionFormInput(response_format="diarized_json").to_command(
        **TO_COMMAND_DEFAULTS
    )
    assert cmd.response_format == ResponseFormat.diarized_json
    assert cmd.do_diarize is True
    assert cmd.do_align is True


def test_transcription_form_auto_align_when_diarize_explicit():
    cmd = TranscriptionFormInput(diarize="true").to_command(**TO_COMMAND_DEFAULTS)
    assert cmd.do_diarize is True
    assert cmd.do_align is True


def test_transcription_form_align_false_overrides_auto_align():
    cmd = TranscriptionFormInput(diarize="true", align="false").to_command(
        **TO_COMMAND_DEFAULTS
    )
    assert cmd.do_diarize is True
    assert cmd.do_align is False


def test_transcription_form_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        TranscriptionFormInput(model="gpt-4o").to_command(**TO_COMMAND_DEFAULTS)


def test_transcription_json_response():
    body = build_json_response({"segments": [{"text": "hello"}]})
    assert isinstance(body, TranscriptionJsonResponse)
    assert body.text == "hello"


def test_verbose_json_word_typed():
    body = build_verbose_json(
        {
            "language": "en",
            "segments": [
                {
                    "text": "hi",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "hi", "start": 0.0, "end": 0.5}],
                }
            ],
        },
        text="hi",
        language="en",
    )
    assert isinstance(body, VerboseJsonResponse)
    assert isinstance(body.segments[0].words[0], VerboseJsonWord)


def test_diarized_json_openapi_schema_has_required_fields():
    schema = DiarizedJsonResponse.model_json_schema()
    assert "language" in schema["properties"]
    assert "segments" in schema["properties"]
