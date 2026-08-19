"""E2E-тесты API-контракта диаризации через реальный FastAPI-роутер.

Внешние тяжёлые вызовы (whisperx.load_audio/align, DiarizationPipeline,
assign_word_speakers) подменяются на заглушки, чтобы тест проходил в CI без
GPU. Тестируется реальный путь кода: валидация, проброс num/min/max в
whisperx.DiarizationPipeline и гибридный word-level формат diarized_json.
"""


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


def test_diarize_model_from_config(client):
    """2.2: DiarizationPipeline конструируется с model_name из конфигурации."""
    import whisperx_api.transcribe_router as tr
    from whisperx_api.config import config

    captured = {}
    orig_init = tr.DiarizationPipeline.__init__

    def fake_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        self.model = object()

    tr.DiarizationPipeline.__init__ = fake_init
    old_pipeline = client.app.state.DIARIZE_PIPELINE
    old_token = config.hf_token
    try:
        # Сбрасываем пайплайн, чтобы конструктор реально вызвался
        client.app.state.DIARIZE_PIPELINE = None
        config.hf_token = "test-token"
        tr._ensure_diarize_pipeline_sync(client.app.state)
    finally:
        tr.DiarizationPipeline.__init__ = orig_init
        client.app.state.DIARIZE_PIPELINE = old_pipeline
        config.hf_token = old_token

    assert captured["kwargs"].get("model_name") == config.diarize_model
    assert config.diarize_model == "pyannote/speaker-diarization-community-1"
    assert captured["kwargs"].get("token") == "test-token"
