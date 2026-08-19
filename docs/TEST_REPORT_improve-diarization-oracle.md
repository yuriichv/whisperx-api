# Отчёт о тестировании — improve-diarization-oracle

Дата: 2026-08-19
Команда запуска (в контейнере, без локального окружения):

```bash
podman run --rm -v "$PWD":/tmp/app -w /tmp/app python:3.12 \
  sh -c "pip install -q pytest fastapi pydantic pydantic-settings python-multipart httpx numpy && \
  PYTHONPATH=src python -m pytest tests/ -v"
```

Результат: **14 passed, 0 failed** (5 warnings — deprecation FastAPI/Starlette, не связаны с изменением).

## Unit-тесты — formatting.hybrid_word_blocks (`tests/unit/test_formatting.py`)

| Тест | Проверка | Статус |
|------|----------|--------|
| `test_single_speaker_segment_keeps_whole_text` | Один спикер в сегменте → цельный `segment.text`, границы сегмента | PASSED |
| `test_speaker_change_inside_segment_splits` | Смена спикера внутри сегмента → отдельные блоки с корректными `start/end/text/speaker` | PASSED |
| `test_no_text_loss_when_word_without_speaker` | Слово без `word.speaker` → fallback на `segment.text`, без потери реплик | PASSED |
| `test_adjacent_blocks_same_speaker_merged` | Склейка соседних блоков одного спикера (start первого, end последнего) | PASSED |
| `test_segment_without_speaker_preserved` | Сегмент без спикера → `speaker=None`, текст сохранён | PASSED |
| `test_empty_segments` | Пустой вход → пустой результат | PASSED |
| `test_empty_text_segment_skipped` | Пустой текст пропускается | PASSED |

## E2E-тесты — API-контракт через реальный роутер (`tests/e2e/test_transcribe_api.py`)

Внешние тяжёлые вызовы (`whisperx.load_audio/align/load_align_model/assign_word_speakers`, `DiarizationPipeline`) подменяются заглушками, чтобы тесты проходили в CI без GPU и моделей. Тестируется реальный путь кода роутера: валидация, проброс параметров и гибридный word-level формат.

| Тест | Задача | Статус |
|------|--------|--------|
| `test_min_max_speakers_determines_3_participants` | 5.1: `min_speakers=3`/`max_speakers=3` → число участников 3 (не 4); пайплайн вызван с `min=3, max=3`, `num_speakers=None` | PASSED |
| `test_replicas_in_segment_split_no_text_loss` | 5.2: реплики разных людей в одном сегменте разделены на корректные блоки, текст не потерян | PASSED |
| `test_no_speaker_params_auto_detection` | 5.3: без параметров числа спикеров запрос проходит; пайплайн вызван со всеми `None` (автоопределение) | PASSED |
| `test_validation_http_400` | 1.3: `num/min/max < 1`, `min>max` → HTTP 400 | PASSED |
| `test_num_speakers_priority_over_min_max` | 1.1/#1: `num_speakers` + `min/max` — приоритет `num_speakers` | PASSED |
| `test_speaker_params_ignored_when_no_diarize` | #2: без диаризации параметры числа спикеров не валидируются | PASSED |
| `test_diarize_model_from_config` | 2.2: `DiarizationPipeline` конструируется с `model_name` из конфигурации | PASSED |

## Lint

`ruff check src/whisperx_api/formatting.py tests/` — только предсуществующие style-предупреждения (E501 line too long, первазивны в кодовой базе); новых ошибок F401/F821/E999 нет.

## Замечания

1. Полноценный E2E-прогон на реальном проблемном аудио c настоящими моделями (`whisperx` + pyannote, GPU, HF_TOKEN) требует GPU-хоста и файла аудио; в этой среде такие тесты не выполняются и заменяются интеграционными с заглушками внешних моделей.
2. Выявлен предсуществующий баг `_run_pipeline_sync:268-270`: если `language` пуст, `result` не инициализируется → `UnboundLocalError`. Не относится к scope данного change-запроса, но зафиксирован; в тестах обходится передачей `language`.