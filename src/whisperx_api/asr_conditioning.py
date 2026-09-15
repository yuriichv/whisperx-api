"""Per-request apply/restore ASR conditioning options (initial_prompt, hotwords)."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional


@contextmanager
def asr_conditioning_options(
    pipeline: Any,
    prompt: Optional[str],
    hotwords: Optional[str],
) -> Iterator[None]:
    """Set initial_prompt/hotwords for one transcribe call; restore on exit."""
    options = pipeline.options
    saved_prompt = options.initial_prompt
    saved_hotwords = options.hotwords
    try:
        options.initial_prompt = prompt
        options.hotwords = hotwords
        yield
    finally:
        options.initial_prompt = saved_prompt
        options.hotwords = saved_hotwords
