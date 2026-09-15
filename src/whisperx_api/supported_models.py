"""Единый источник ID моделей для HTTP API."""

COMPAT_MODEL_IDS: tuple[str, ...] = ("whisper-1", "whisper-large-v3")

_OWNED_BY: dict[str, str] = {
    "whisper-1": "openai",
    "whisper-large-v3": "local",
}


def supported_model_ids(default_model: str) -> frozenset[str]:
    return frozenset(COMPAT_MODEL_IDS) | {default_model}


def model_catalog_entries(default_model: str) -> list[tuple[str, str]]:
    """(model_id, owned_by) для GET /v1/models."""
    ordered = [default_model]
    for model_id in COMPAT_MODEL_IDS:
        if model_id != default_model:
            ordered.append(model_id)
    return [(model_id, _OWNED_BY.get(model_id, "local")) for model_id in ordered]
