from typing import Any, Dict, Tuple, Optional
import asyncio
import logging
from fastapi import Request


class AppState:
    """Container for FastAPI application-wide mutable state.
    All runtime‑mutable globals are stored here via ``app.state``.
    """

    # logger – will be set from main.py after its creation
    logger: Optional[logging.Logger] = None

    # ASR pipeline and related runtime configuration
    ASR_PIPELINE: Any = None
    DEVICE: Optional[str] = None
    COMPUTE_TYPE: Optional[str] = None

    # Cache for alignment models per language code
    ALIGN_CACHE: Dict[str, Tuple[Any, Any]] = {}

    # Diarization pipeline (singleton per process)
    DIARIZE_PIPELINE: Any = None

    # Global GPU lock – instantiated on startup
    GPU_LOCK: asyncio.Lock = asyncio.Lock()

    async def startup_load(self) -> None:
        """Load ASR model and related resources on app start.
        Mirrors the former ``startup_load`` function from ``transcribe_router``.
        """
        # Local imports to avoid circular imports / heavy imports at module load
        from .config import config
        from fastapi import HTTPException
        import logging

        # Optional imports – may be unavailable in the environment
        try:
            import torch
        except Exception:
            torch = None
        try:
            import whisperx
        except Exception:
            whisperx = None

        logger = self.logger or logging.getLogger(__name__)

        # ---- Device selection -------------------------------------------------
        def _select_device(device: Optional[str]) -> str:
            if device:
                return device
            if (
                torch is not None
                and hasattr(torch, "cuda")
                and torch.cuda.is_available()
            ):
                return "cuda"
            return "cpu"

        def _select_compute_type(device: str, compute_type: Optional[str]) -> str:
            if compute_type:
                return compute_type
            return "float16" if device == "cuda" else "int8"

        # Initialise state attributes
        self.DEVICE = _select_device(config.default_device)
        self.COMPUTE_TYPE = _select_compute_type(
            self.DEVICE, config.default_compute_type
        )

        logger.debug("whisperx loading Pipeline...")
        async with self.GPU_LOCK:
            if whisperx is None:
                logger.warning(
                    "whisperx library not available; skipping model loading."
                )
                self.ASR_PIPELINE = None
            else:
                self.ASR_PIPELINE = whisperx.load_model(
                    config.default_model,
                    device=self.DEVICE,
                    compute_type=self.COMPUTE_TYPE,
                    language=config.default_language,
                )
                logger.debug(
                    f"whisperx loaded. Pipeline: {type(self.ASR_PIPELINE).__name__}, model: {type(self.ASR_PIPELINE.model).__name__}"
                )
                # Pre‑load alignment model for default language into cache
                try:
                    language_code = config.default_language
                    if whisperx is None:
                        raise HTTPException(
                            status_code=500,
                            detail="whisperx library not available; cannot load alignment model",
                        )
                    if language_code not in self.ALIGN_CACHE:
                        align_model, metadata = whisperx.load_align_model(
                            language_code=language_code, device=self.DEVICE
                        )
                        self.ALIGN_CACHE[language_code] = (align_model, metadata)
                except Exception as e:
                    logger.error(
                        f"Failed to pre‑load align model for {config.default_language}: {e}"
                    )


# Dependency to expose the shared state via FastAPI's Depends system


def get_state(request: Request) -> "AppState":
    """FastAPI dependency that returns the mutable application state.
    Any router can declare ``state: AppState = Depends(get_state)`` to get
    access to ``request.app.state``.
    """
    return request.app.state
