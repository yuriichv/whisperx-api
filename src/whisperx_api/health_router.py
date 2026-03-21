from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from .state import AppState, get_state


router = APIRouter(tags=["health"])


@router.get("/live")
async def live():
    """Liveness probe – always returns 200 if the process is running."""
    return JSONResponse(content={"status": "live"})


@router.get("/ready")
async def ready(state: AppState = Depends(get_state)):
    """Readiness probe – returns 200 only when the ASR model is loaded.

    If the model is not yet loaded, returns HTTP 503.
    """
    if state.ASR_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return JSONResponse(content={"status": "ready"})


@router.get("/health")
async def health(state: AppState = Depends(get_state)):
    """Detailed health endpoint – provides model loading status and configuration info."""
    return JSONResponse(
        content={
            "model_loaded": state.ASR_PIPELINE is not None,
            "device": state.DEVICE,
            "compute_type": state.COMPUTE_TYPE,
        }
    )
