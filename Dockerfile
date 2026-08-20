# важно правильную версию torch. Сейчас транзитивно использует 2.8
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Кэши (важно для non-root)
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

# NLTK данные для sentence splitting (punkt_tab)
ENV NLTK_DATA=/app/nltk_data

WORKDIR /app

# ffmpeg нужен для декодирования аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m -u 10001 -s /bin/bash whisper \
 && mkdir -p /app/.cache \
 && chown -R whisper:whisper /app

RUN pip install --no-cache-dir --break-system-packages uv

# deps
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# uv venv location (предсказуемо)
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# ВАЖНО: pyproject/lock должны быть согласованы с torch 2.10 / torchcodec 0.10
RUN uv sync --no-dev --no-cache

# NLTK punkt_tab для sentence splitting в whisperx alignment
# Скачивается один раз при сборке, чтобы рантайм не зависел от сети.
# Важно: nltk.downloader использует каталог из NLTK_DATA только если он уже существует,
# иначе пишет в ~/nltk_data (у root — /root/nltk_data). Поэтому создаём его заранее.
RUN mkdir -p /app/nltk_data \
 && uv run python -m nltk.downloader punkt_tab

COPY src/whisperx_api /app/whisperx_api
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \ 
    && chown whisper:whisper /app/entrypoint.sh \
    && chown -R whisper:whisper /app/whisperx_api \
    && chown -R whisper:whisper /app/nltk_data

EXPOSE 8000

USER whisper

ENTRYPOINT ["/app/entrypoint.sh"]
