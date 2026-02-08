from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import secrets

from .config import config
from .models_router import router as models_router
from .transcribe_router import router as transcribe_router
from .transcribe_router import startup_load, shutdown_cleanup

logger = logging.getLogger(__name__)

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
app = FastAPI(title="WhisperX OpenAI-compatible Transcriptions API", dependencies=deps)

# routers
app.include_router(models_router)
app.include_router(transcribe_router)

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

    await startup_load()

@app.on_event("shutdown")
async def _shutdown():
    await shutdown_cleanup()
