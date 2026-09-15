"""Contract tests for diarized_json response shape."""

from whisperx_api.features.transcription.schemas import DiarizedJsonResponse
from whisperx_api.features.transcription.responses import diarized_response


def test_diarized_json_contract_keys():
    result = {
        "language": "ru",
        "segments": [
            {
                "text": "Привет мир",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
            }
        ],
    }
    body = diarized_response(result, language="ru")

    assert isinstance(body, DiarizedJsonResponse)
    dumped = body.model_dump()
    assert set(dumped.keys()) == {"language", "text", "speakers", "segments"}
    assert body.speakers == ["SPEAKER_00"]

    segment = body.segments[0]
    assert segment.type == "transcript.text.segment"
    assert segment.text == "Привет мир"
    assert segment.speaker == "SPEAKER_00"
