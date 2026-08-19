import os
from types import SimpleNamespace

# Отключаем авторизацию до импорта приложения (deps формируются на импорте)
os.environ["WHISPERX_NO_AUTH"] = "true"

import numpy as np
import pytest
from fastapi.testclient import TestClient

import whisperx_api.transcribe_router as tr
from whisperx_api.config import config

# Если whisperx не установлен в тестовом окружении, подменяем модуль и
# DiarizationPipeline заглушками, чтобы импорт роутера был валиден.
if tr.whisperx is None:
    tr.whisperx = SimpleNamespace()
if tr.DiarizationPipeline is None:

    class _StubDiarizationPipeline:
        def __init__(self, *args, **kwargs):
            pass

    tr.DiarizationPipeline = _StubDiarizationPipeline

from whisperx_api.main import app  # noqa: E402


def make_fake_result():
    """Фейковый результат ASR+align+assign_word_speakers с word.speaker."""
    return {
        "language": "ru",
        "text": "Финальный текст всего аудио",
        "segments": [
            {
                "text": "Я попробовал это на Биане. Ну давай Андрюх",
                "start": 0.0,
                "end": 4.0,
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Я", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"},
                    {"word": "попробовал", "start": 0.3, "end": 0.8, "speaker": "SPEAKER_00"},
                    {"word": "это", "start": 0.9, "end": 1.1, "speaker": "SPEAKER_00"},
                    {"word": "на", "start": 1.2, "end": 1.4, "speaker": "SPEAKER_00"},
                    {"word": "Биане.", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
                    {"word": "Ну", "start": 2.1, "end": 2.3, "speaker": "SPEAKER_01"},
                    {"word": "давай", "start": 2.4, "end": 2.7, "speaker": "SPEAKER_01"},
                    {"word": "Андрюх", "start": 2.8, "end": 3.0, "speaker": "SPEAKER_01"},
                ],
            },
            {
                "text": "Согласен.",
                "start": 4.0,
                "end": 5.0,
                "speaker": "SPEAKER_02",
            },
        ],
    }


class FakeASRPipeline:
    """Заглушка ASR-пайплайна."""

    def __init__(self, result):
        self.result = result

    def transcribe(self, audio, **kwargs):
        return {
            "segments": list(self.result["segments"]),
            "language": self.result["language"],
        }


class RecordingPipeline:
    """Заглушка DiarizationPipeline; записывает kwargs вызова."""

    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 5.0, "speaker": "SPEAKER_02"},
        ]


@pytest.fixture
def client(monkeypatch):
    # Создаём TestClient БЕЗ входа в контекст, чтобы не запускался startup_load
    # (иначе ASR_PIPELINE будет перезаписан None в среде без whisperx).
    test_client = TestClient(app)
    app.state.ASR_PIPELINE = FakeASRPipeline(make_fake_result())
    app.state.DEVICE = "cpu"
    config.no_auth = True
    config.default_align = True
    config.default_diarize = False

    recorder = RecordingPipeline()
    app.state.DIARIZE_PIPELINE = recorder

    # Заглушки внешних вызовов whisperx
    monkeypatch.setattr(
        tr.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32),
        raising=False,
    )
    monkeypatch.setattr(
        tr.whisperx,
        "align",
        lambda segments, am, md, audio, device, return_char_alignments=False: {
            "segments": segments,
            "language": "ru",
        },
        raising=False,
    )
    monkeypatch.setattr(
        tr.whisperx,
        "load_align_model",
        lambda language_code, device: (None, None),
        raising=False,
    )
    monkeypatch.setattr(
        tr.whisperx,
        "assign_word_speakers",
        lambda diarize_df, result, fill_nearest=False: result,
        raising=False,
    )

    test_client._recorder = recorder
    yield test_client
    app.state.ASR_PIPELINE = None
    app.state.DIARIZE_PIPELINE = None
