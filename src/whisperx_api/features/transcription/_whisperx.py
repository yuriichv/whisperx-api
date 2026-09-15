"""Optional whisperx/torch imports isolated from router and service orchestration."""

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    import whisperx
except Exception:
    whisperx = None  # type: ignore[assignment]

try:
    from whisperx.diarize import DiarizationPipeline
except ImportError:
    DiarizationPipeline = None  # type: ignore[assignment,misc]
