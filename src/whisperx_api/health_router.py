from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .transcribe_router import ASR_MODEL, DEVICE, COMPUTE_TYPE

router = APIRouter(tags=["health"])


@router.get("/live")
async def live():
    """Liveness probe – always returns 200 if the process is running."""
    return JSONResponse(content={"status": "live"})


@router.get("/ready")
async def ready():
    """Readiness probe – returns 200 only when the ASR model is loaded.

    If the model is not yet loaded, returns HTTP 503.
    """
    if ASR_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return JSONResponse(content={"status": "ready"})


@router.get("/health")
async def health():
    """Detailed health endpoint – provides model loading status and configuration info."""
    return JSONResponse(
        content={
            "model_loaded": ASR_MODEL is not None,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        }
    )
