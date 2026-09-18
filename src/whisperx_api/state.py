from typing import Any, Dict, Tuple, Optional
import asyncio
import logging
from fastapi import Request

from .features.transcription import _whisperx
from .features.transcription.device import select_compute_type, select_device


class AppState:
    """Container for FastAPI application-wide mutable state."""

    logger: Optional[logging.Logger] = None

    ASR_PIPELINE: Any = None
    DEVICE: Optional[str] = None
    COMPUTE_TYPE: Optional[str] = None

    ALIGN_CACHE: Dict[str, Tuple[Any, Any]] = {}
    DIARIZE_PIPELINE: Any = None
    GPU_LOCK: asyncio.Lock = asyncio.Lock()

    async def startup_load(self) -> None:
        """Load ASR model and related resources on app start."""
        from .config import config

        logger = self.logger or logging.getLogger(__name__)
        whisperx = _whisperx.whisperx

        self.DEVICE = select_device(config.default_device or None)
        self.COMPUTE_TYPE = select_compute_type(
            self.DEVICE, config.default_compute_type or None
        )

        logger.debug("whisperx loading Pipeline...")
        async with self.GPU_LOCK:
            if whisperx is None:
                logger.warning(
                    "whisperx library not available; skipping model loading."
                )
                self.ASR_PIPELINE = None
            else:
                load_model_kwargs: dict[str, Any] = {
                    "device": self.DEVICE,
                    "compute_type": self.COMPUTE_TYPE,
                    "language": config.default_language,
                }
                vad_method = (config.vad_method or "").strip()
                if vad_method:
                    load_model_kwargs["vad_method"] = vad_method
                self.ASR_PIPELINE = whisperx.load_model(
                    config.default_model,
                    **load_model_kwargs,
                )
                logger.debug(
                    "whisperx loaded. Pipeline: %s, model: %s",
                    type(self.ASR_PIPELINE).__name__,
                    type(self.ASR_PIPELINE.model).__name__,
                )
                try:
                    language_code = config.default_language
                    if language_code and language_code not in self.ALIGN_CACHE:
                        align_model, metadata = whisperx.load_align_model(
                            language_code=language_code, device=self.DEVICE
                        )
                        self.ALIGN_CACHE[language_code] = (align_model, metadata)
                except Exception as e:
                    logger.error(
                        "Failed to pre-load align model for %s: %s",
                        config.default_language,
                        e,
                    )

        try:
            import nltk

            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("NLTK punkt_tab not found at build time, downloading now...")
            try:
                import nltk

                nltk.download("punkt_tab", quiet=True)
            except Exception as e:
                logger.warning(
                    "Failed to download NLTK punkt_tab: %s — "
                    "sentence splitting in alignment may fail",
                    e,
                )
        except Exception:
            pass


def get_state(request: Request) -> "AppState":
    """FastAPI dependency that returns the mutable application state."""
    return request.app.state
