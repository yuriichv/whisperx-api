from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Path

from .schemas.models import ModelObject, ModelsListResponse
from .config import config
from .supported_models import model_catalog_entries

router = APIRouter(tags=["models"])


def _now_epoch() -> int:
    return int(time.time())


def get_models_list() -> list[ModelObject]:
    created = _now_epoch()
    return [
        ModelObject(id=model_id, created=created, owned_by=owned_by)
        for model_id, owned_by in model_catalog_entries(config.default_model)
    ]


def get_model_by_id(model_id: str) -> ModelObject | None:
    for m in get_models_list():
        if m.id == model_id:
            return m
    return None


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """OpenAI-compatible: GET /v1/models"""
    return ModelsListResponse(data=get_models_list())


@router.get("/v1/models/{model_id}", response_model=ModelObject)
async def retrieve_model(
    model_id: str = Path(..., description="Model ID, e.g. whisper-1 or large-v3"),
):
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return model
