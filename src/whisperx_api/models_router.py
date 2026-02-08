from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

import time
from typing import Any, Dict, List

from .config import config

### API
router = APIRouter(tags=["models"])

@router.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible: GET /v1/models
    """
    return JSONResponse({"object": "list", "data": get_models_list()})

@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str = Path(..., description="Model ID, e.g. whisper-1 or large-v3")):
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return JSONResponse(model)

#### Service logic
def _now_epoch() -> int:
    return int(time.time())

def get_models_list() -> List[Dict[str, Any]]:
    """
    Models list to show clients
    """
    created = _now_epoch()
    base_model = config.default_model

    models = [
        {"id": base_model, "object": "model", "created": created, "owned_by": "local"},
        {"id": "whisper-1", "object": "model", "created": created, "owned_by": "openai"},
        {"id": "whisper-large-v3", "object": "model", "created": created, "owned_by": "local"},
    ]
    # Remove dublication by id, example default_model == "whisper-1" or "large-v3"
    uniq: Dict[str, Dict[str, Any]] = {}
    for m in models:
        uniq[m["id"]] = m
    return list(uniq.values())

def get_model_by_id(model_id: str) -> Dict[str, Any] | None:
    for m in get_models_list():
        if m["id"] == model_id:
            return m
    return None