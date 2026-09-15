import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .bootstrap import apply_torch_load_workaround, setup_logging
from .config import config
from .features.transcription.api import router as transcribe_router
from .health_router import router as health_router
from .models_router import router as models_router
from .state import AppState

apply_torch_load_workaround()
logger = setup_logging()

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if credentials.credentials != config.api_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


deps = [] if config.no_auth else [Depends(verify_token)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up. Loading model")
    if not config.api_token and not config.no_auth:
        config.api_token = secrets.token_urlsafe(32)
        logger.warning(
            "API_TOKEN not set in environment variables! "
            "Generated temporary token for this session:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Bearer %s\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Save this token! It will change on next restart.",
            config.api_token,
        )
    else:
        logger.info("API_TOKEN loaded from environment")

    await app.state.startup_load()
    yield
    logger.info("Application shutdown complete")


app = FastAPI(
    title="WhisperX OpenAI-compatible Transcriptions API",
    lifespan=lifespan,
)

app.state = AppState()
app.state.logger = logger

app.include_router(models_router, dependencies=deps)
app.include_router(transcribe_router, dependencies=deps)
app.include_router(health_router)
