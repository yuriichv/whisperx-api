#FROM python:3.12-slim
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Кэши (важно для non-root)
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

WORKDIR /app

# ffmpeg нужен для декодирования аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m -u 10001 -s /bin/bash whisper \
 && mkdir -p /app/.cache \
 && chown -R whisper:whisper /app

RUN pip install --no-cache-dir uv

# deps
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# uv venv location (предсказуемо)
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN uv sync --no-dev --no-cache

COPY src/whisperx_api /app/whisperx_api
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \ 
    && chown whisper:whisper /app/entrypoint.sh \
    && chown -R whisper:whisper /app/whisperx_api

EXPOSE 8000

USER whisper

ENTRYPOINT ["/app/entrypoint.sh"]
