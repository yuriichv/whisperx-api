#!/bin/sh
set -eu

APP_HOST="${WHISPERX_APP_HOST:-0.0.0.0}"
APP_PORT="${WHISPERX_APP_PORT:-8000}"

exec uvicorn whisperx_api.main:app \
  --host "$APP_HOST" \
  --port "$APP_PORT" \
  --workers 1
