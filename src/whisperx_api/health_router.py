from fastapi import APIRouter, Depends, HTTPException

from .schemas.health import HealthResponse, StatusResponse
from .features.transcription.errors import MODEL_NOT_LOADED_MESSAGE
from .state import AppState, get_state

router = APIRouter(tags=["health"])


@router.get("/live", response_model=StatusResponse)
async def live():
    """Liveness probe – always returns 200 if the process is running."""
    return StatusResponse(status="live")


@router.get("/ready", response_model=StatusResponse)
async def ready(state: AppState = Depends(get_state)):
    """Readiness probe – returns 200 only when the ASR model is loaded."""
    if state.ASR_PIPELINE is None:
        raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED_MESSAGE)
    return StatusResponse(status="ready")


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)):
    """Detailed health endpoint – model loading status and configuration."""
    return HealthResponse(
        model_loaded=state.ASR_PIPELINE is not None,
        device=state.DEVICE,
        compute_type=state.COMPUTE_TYPE,
    )
