"""Deploy-time VAD method configuration for load_model."""

import asyncio
from types import SimpleNamespace

import pytest

from whisperx_api.config import config
from whisperx_api.features.transcription import _whisperx as wx
from whisperx_api.state import AppState


@pytest.fixture
def whisperx_stub(monkeypatch):
    captured: dict = {}

    def fake_load_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return object()

    ns = SimpleNamespace(load_model=fake_load_model, load_align_model=lambda **k: (None, None))
    monkeypatch.setattr(wx, "whisperx", ns)
    return captured


def test_load_model_omits_vad_method_when_unset(whisperx_stub, monkeypatch):
    monkeypatch.setattr(config, "vad_method", "")

    async def run():
        state = AppState()
        await state.startup_load()

    asyncio.run(run())
    assert "vad_method" not in whisperx_stub["kwargs"]


def test_load_model_passes_vad_method_from_config(whisperx_stub, monkeypatch):
    monkeypatch.setattr(config, "vad_method", "silero")

    async def run():
        state = AppState()
        await state.startup_load()

    asyncio.run(run())
    assert whisperx_stub["kwargs"].get("vad_method") == "silero"


def test_load_model_strips_vad_method(whisperx_stub, monkeypatch):
    monkeypatch.setattr(config, "vad_method", "  pyannote  ")

    async def run():
        state = AppState()
        await state.startup_load()

    asyncio.run(run())
    assert whisperx_stub["kwargs"].get("vad_method") == "pyannote"
