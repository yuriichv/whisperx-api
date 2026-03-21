import sys

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import secrets

from .config import config
from .models_router import router as models_router
from .transcribe_router import router as transcribe_router
from .transcribe_router import shutdown_cleanup
from .state import AppState

logging.basicConfig(
    level=config.log_level,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != config.api_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# if WHISPERX_NO_AUTH=true disable token verification
deps = [] if config.no_auth else [Depends(verify_token)]
app = FastAPI(title="WhisperX OpenAI-compatible Transcriptions API")

# Attach logger and runtime state to FastAPI app
app.state = AppState()
app.state.logger = logger

# Inject FastAPI app into transcribe router to avoid circular imports
transcribe_router.app = app

# Import health router after app is defined
from .health_router import router as health_router

# routers
app.include_router(models_router, dependencies=deps)
app.include_router(transcribe_router, dependencies=deps)
app.include_router(health_router)


# lifecycle
@app.on_event("startup")
async def _startup():
    if not config.api_token and not config.no_auth:
        # Token gen
        config.api_token = secrets.token_urlsafe(32)
        logger.warning(
            "API_TOKEN not set in environment variables! "
            "Generated temporary token for this session:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Bearer {config.api_token}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Save this token! It will change on next restart."
        )
    else:
        logger.info("API_TOKEN loaded from environment")

    await app.state.startup_load()


@app.on_event("shutdown")
async def _shutdown():
    await shutdown_cleanup()
