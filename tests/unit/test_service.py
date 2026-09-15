"""Unit tests for transcription service orchestration."""

import pytest

from fixtures.transcription import FakeASRPipeline
from whisperx_api.features.transcription.schemas import TranscriptionCommand
from whisperx_api.features.transcription.errors import ModelNotLoadedError
from whisperx_api.features.transcription.service import run_pipeline_sync
from whisperx_api.state import AppState


def test_run_pipeline_transcribes_without_language():
    """ASR must run even when language is None (auto-detect)."""
    state = AppState()
    state.ASR_PIPELINE = FakeASRPipeline()
    state.DEVICE = "cpu"
    command = TranscriptionCommand(language=None)

    result = run_pipeline_sync(state, audio=[0.0], command=command)

    assert result["segments"][0]["text"] == "hello"
    assert "language" not in state.ASR_PIPELINE.last_kwargs


def test_run_pipeline_applies_conditioning():
    state = AppState()
    pipeline = FakeASRPipeline()
    state.ASR_PIPELINE = pipeline
    state.DEVICE = "cpu"
    command = TranscriptionCommand(prompt="ctx", hotwords="term")

    run_pipeline_sync(state, audio=[0.0], command=command)

    # options восстанавливаются после transcribe (asr_conditioning_options)
    assert pipeline.conditioning_calls == [("ctx", "term")]
    assert pipeline.options.initial_prompt is None
    assert pipeline.options.hotwords is None


def test_run_pipeline_raises_when_model_not_loaded():
    state = AppState()
    state.ASR_PIPELINE = None
    command = TranscriptionCommand(language="ru")

    with pytest.raises(ModelNotLoadedError):
        run_pipeline_sync(state, audio=[0.0], command=command)
