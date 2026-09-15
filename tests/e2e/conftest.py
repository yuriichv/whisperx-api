import os
from types import SimpleNamespace

# Отключаем авторизацию до импорта приложения (deps формируются на импорте)
os.environ["WHISPERX_NO_AUTH"] = "true"

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fixtures.transcription import (
    FakeASRPipeline,
    RecordingDiarizePipeline,
    make_diarization_e2e_result,
)
from whisperx_api.config import config
from whisperx_api.features.transcription import _whisperx as wx

# Если whisperx не установлен в тестовом окружении, подменяем модуль и
# DiarizationPipeline заглушками, чтобы импорт роутера был валиден.
if wx.whisperx is None:
    wx.whisperx = SimpleNamespace()
if wx.DiarizationPipeline is None:

    class _StubDiarizationPipeline:
        def __init__(self, *args, **kwargs):
            pass

    wx.DiarizationPipeline = _StubDiarizationPipeline

from whisperx_api.main import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Создаём TestClient БЕЗ входа в контекст, чтобы не запускался startup_load
    # (иначе ASR_PIPELINE будет перезаписан None в среде без whisperx).
    test_client = TestClient(app)
    asr = FakeASRPipeline(make_diarization_e2e_result())
    app.state.ASR_PIPELINE = asr
    app.state.DEVICE = "cpu"
    config.no_auth = True
    config.default_align = True
    config.default_diarize = False

    recorder = RecordingDiarizePipeline()
    app.state.DIARIZE_PIPELINE = recorder

    monkeypatch.setattr(
        wx.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32),
        raising=False,
    )
    monkeypatch.setattr(
        wx.whisperx,
        "align",
        lambda segments, am, md, audio, device, return_char_alignments=False: {
            "segments": segments,
            "language": "ru",
        },
        raising=False,
    )
    monkeypatch.setattr(
        wx.whisperx,
        "load_align_model",
        lambda language_code, device: (None, None),
        raising=False,
    )
    monkeypatch.setattr(
        wx.whisperx,
        "assign_word_speakers",
        lambda diarize_df, result, fill_nearest=False: result,
        raising=False,
    )

    test_client._recorder = recorder
    test_client._asr = asr
    yield test_client
    app.state.ASR_PIPELINE = None
    app.state.DIARIZE_PIPELINE = None
