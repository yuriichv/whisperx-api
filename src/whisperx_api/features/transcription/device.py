"""Device/compute-type selection for whisperx (no AppState dependency)."""

from . import _whisperx


def select_device(device: str | None) -> str:
    if device:
        return device
    torch = _whisperx.torch
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_compute_type(device: str, compute_type: str | None) -> str:
    if compute_type:
        return compute_type
    return "float16" if device == "cuda" else "int8"
