"""E2E-тесты API-контракта транскрипции через FastAPI-роутер.

Сценарии (spec transcription-api):
- диаризация: num/min/max, diarized_json, word-level split
- prompt/hotwords: проброс в ASR options, char/token limits 500/1000/100/150/200
- prompt + diarize=true → 200 (WhisperX extension)
- auto-language: запрос без language проходит
- response проходит Pydantic model_validate
"""

from whisperx_api.features.transcription.schemas import (
    DiarizedJsonResponse,
    TranscriptionJsonResponse,
    VerboseJsonResponse,
)


def _post(client, **form):
    file = form.pop("file", ("audio.wav", b"\x00\x00\x00\x00", "audio/wav"))
    data = form or {}
    data.setdefault("language", "ru")
    return client.post(
        "/v1/audio/transcriptions",
        data=data,
        files=[("file", file)],
    )


def test_min_max_speakers_determines_3_participants(client):
    """5.1: min_speakers=3/max_speakers=3 → число участников 3 (не 4)."""
    resp = _post(
        client,
        response_format="diarized_json",
        diarize="true",
        min_speakers="3",
        max_speakers="3",
    )
    assert resp.status_code == 200
    body = resp.json()
    # Заглушка вернула 3 участников (SPEAKER_00..02)
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    assert len(body["speakers"]) == 3
    # Пайплайн вызван именно с min/max, num_speakers отсутствует
    call = client._recorder.calls[-1]
    assert call.get("num_speakers") is None
    assert call.get("min_speakers") == 3
    assert call.get("max_speakers") == 3


def test_replicas_in_segment_split_no_text_loss(client):
    """5.2: реплики разных людей в одном сегменте разделены, текст не потерян."""
    resp = _post(
        client,
        response_format="diarized_json",
        diarize="true",
        num_speakers="3",
    )
    assert resp.status_code == 200
    body = resp.json()

    blocks = body["segments"]
    # Первый сегмент со сменой внутри разбит на 2 блока (SPEAKER_00/SPEAKER_01),
    # третий блок — отдельный сегмент SPEAKER_02.
    assert len(blocks) == 3
    assert blocks[0]["speaker"] == "SPEAKER_00"
    assert blocks[1]["speaker"] == "SPEAKER_01"
    assert blocks[2]["speaker"] == "SPEAKER_02"

    # Текст блоков восстанавливает исходную реплику без потери
    assert "Я попробовал это на Биане." in blocks[0]["text"]
    assert blocks[1]["text"] == "Ну давай Андрюх"

    # Текст из первого блока не потерян в diarized text
    assert "Я попробовал это на Биане." in body["text"]
    assert "Ну давай Андрюх" in body["text"]

    # num_speakers имеет приоритет и пробрасывается в пайплайн
    assert client._recorder.calls[-1].get("num_speakers") == 3


def test_num_speakers_priority_over_min_max(client):
    """1.1/#1: num_speakers вместе с min/max — приоритет num_speakers."""
    resp = _post(
        client,
        response_format="diarized_json",
        diarize="true",
        num_speakers="2",
        min_speakers="5",
        max_speakers="5",
    )
    assert resp.status_code == 200
    call = client._recorder.calls[-1]
    # В пайплайн передаются оба, но num_speakers задан (приоритет у pyannote)
    assert call.get("num_speakers") == 2
    assert call.get("min_speakers") == 5
    assert call.get("max_speakers") == 5


def test_speaker_params_ignored_when_no_diarize(client):
    """#2: без диаризации параметры числа спикеров не валидируются и не влияют."""
    resp = _post(client, response_format="json", num_speakers="0")
    # Запрос без диаризации проходит, несмотря на невалидное num_speakers
    assert resp.status_code == 200


def test_no_speaker_params_auto_detection(client):
    """5.3: без параметров числа спикеров запрос проходит (автоопределение)."""
    resp = _post(client, response_format="diarized_json", diarize="true")
    assert resp.status_code == 200
    body = resp.json()
    assert "speakers" in body
    # Пайплайн вызван со всеми параметрами = None (pyannote автоопределение)
    call = client._recorder.calls[-1]
    assert call.get("num_speakers") is None
    assert call.get("min_speakers") is None
    assert call.get("max_speakers") is None


def test_validation_http_400(client):
    """1.3: невалидные значения num/min/max → HTTP 400."""
    cases = [
        {"num_speakers": "0"},
        {"num_speakers": "-1"},
        {"min_speakers": "0"},
        {"max_speakers": "-3"},
        {"min_speakers": "5", "max_speakers": "3"},
    ]
    for form in cases:
        resp = _post(client, diarize="true", **form)
        assert resp.status_code == 400, f"{form} -> {resp.status_code}"


def test_transcription_without_language(client):
    """Фаза 0: auto-detect — запрос без language должен проходить."""
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "json"},
        files=[("file", ("audio.wav", b"\x00\x00\x00\x00", "audio/wav"))],
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


def test_fill_nearest_passed_to_assign_word_speakers(client, monkeypatch):
    """fill_nearest из конфигурации пробрасывается в assign_word_speakers."""
    from whisperx_api.config import config
    from whisperx_api.features.transcription import _whisperx as wx

    captured = {}

    def fake_assign(diarize_df, result, fill_nearest=False):
        captured["fill_nearest"] = fill_nearest
        return result

    monkeypatch.setattr(wx.whisperx, "assign_word_speakers", fake_assign)
    monkeypatch.setattr(config, "fill_nearest", False)

    resp = _post(client, response_format="diarized_json", diarize="true")
    assert resp.status_code == 200
    assert captured.get("fill_nearest") is False


def test_diarize_model_from_config(client):
    """2.2: DiarizationPipeline конструируется с model_name из конфигурации."""
    from whisperx_api.config import config
    from whisperx_api.features.transcription import _whisperx as wx
    from whisperx_api.features.transcription.service import ensure_diarize_pipeline_sync

    captured = {}
    orig_init = wx.DiarizationPipeline.__init__

    def fake_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        self.model = object()

    wx.DiarizationPipeline.__init__ = fake_init
    old_pipeline = client.app.state.DIARIZE_PIPELINE
    old_token = config.hf_token
    try:
        # Сбрасываем пайплайн, чтобы конструктор реально вызвался
        client.app.state.DIARIZE_PIPELINE = None
        config.hf_token = "test-token"
        ensure_diarize_pipeline_sync(client.app.state)
    finally:
        wx.DiarizationPipeline.__init__ = orig_init
        client.app.state.DIARIZE_PIPELINE = old_pipeline
        config.hf_token = old_token

    assert captured["kwargs"].get("model_name") == config.diarize_model
    assert config.diarize_model == "pyannote/speaker-diarization-community-1"
    assert captured["kwargs"].get("token") == "test-token"


def test_prompt_and_hotwords_applied_to_asr_options(client):
    """prompt/hotwords пробрасываются в options.initial_prompt/hotwords на transcribe."""
    resp = _post(
        client,
        prompt="Контекст встречи",
        hotwords="WhisperX, pyannote",
    )
    assert resp.status_code == 200
    prompt, hotwords = client._asr.conditioning_calls[-1]
    assert prompt == "Контекст встречи"
    assert hotwords == "WhisperX, pyannote"


def test_prompt_with_diarize_allowed(client):
    """prompt + diarize=true → 200 (WhisperX extension)."""
    resp = _post(
        client,
        response_format="diarized_json",
        diarize="true",
        prompt="Это совещание",
        num_speakers="3",
    )
    assert resp.status_code == 200
    DiarizedJsonResponse.model_validate(resp.json())


def test_json_response_matches_pydantic_model(client):
    resp = _post(client, response_format="json")
    assert resp.status_code == 200
    TranscriptionJsonResponse.model_validate(resp.json())


def test_verbose_json_response_matches_pydantic_model(client):
    resp = _post(client, response_format="verbose_json", align="true")
    assert resp.status_code == 200
    VerboseJsonResponse.model_validate(resp.json())


def test_hotwords_token_limit_returns_422(client):
    """hotwords > 150 Whisper tokens → HTTP 422."""
    resp = _post(client, hotwords="x" * 151)
    assert resp.status_code == 422


def test_request_without_language_succeeds(client):
    """ASR transcribe вызывается без явного language (auto-detect path)."""
    resp = client.post(
        "/v1/audio/transcriptions",
        data={},
        files=[("file", ("audio.wav", b"\x00\x00\x00\x00", "audio/wav"))],
    )
    assert resp.status_code == 200
    assert client._asr.conditioning_calls
