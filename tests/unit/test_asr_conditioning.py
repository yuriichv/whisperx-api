"""Unit-тесты apply/restore ASR conditioning options."""

from types import SimpleNamespace

from whisperx_api.asr_conditioning import asr_conditioning_options


class FakePipeline:
    def __init__(self):
        self.options = SimpleNamespace(initial_prompt="saved-p", hotwords="saved-h")


def test_asr_conditioning_save_set_restore():
    pipeline = FakePipeline()
    with asr_conditioning_options(pipeline, "req-p", "req-h"):
        assert pipeline.options.initial_prompt == "req-p"
        assert pipeline.options.hotwords == "req-h"
    assert pipeline.options.initial_prompt == "saved-p"
    assert pipeline.options.hotwords == "saved-h"


def test_asr_conditioning_none_clears_for_request():
    pipeline = FakePipeline()
    with asr_conditioning_options(pipeline, None, None):
        assert pipeline.options.initial_prompt is None
        assert pipeline.options.hotwords is None
    assert pipeline.options.initial_prompt == "saved-p"
