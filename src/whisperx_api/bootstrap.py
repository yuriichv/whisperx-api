"""Стартовая инициализация процесса (composition root).

Модуль вызывается из main.py один раз при запуске приложения, до загрузки
моделей и до приёма HTTP-запросов. Сюда вынесены настройки, которые должны
выполниться ровно один раз на весь процесс и не относятся к бизнес-логике
транскрипции или HTTP-роутингу.

Содержит:
- setup_logging() — единая конфигурация логирования (уровень из config);
- apply_torch_load_workaround() — патч torch.load для загрузки чекпоинтов
  Whisper/WhisperX (weights_only=False), иначе PyTorch 2.x может отказать
  в загрузке legacy-весов.

Не импортировать из feature-модулей — только из точки входа (main.py).
"""

import logging
import sys

from .config import config


def setup_logging() -> logging.Logger:
    """Configure process-wide logging once at startup."""
    logging.basicConfig(
        level=config.log_level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    return logging.getLogger("whisperx_api")


def apply_torch_load_workaround() -> None:
    """Allow loading legacy Whisper checkpoints that require weights_only=False."""
    try:
        import torch
    except Exception:
        return

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = trusted_load
