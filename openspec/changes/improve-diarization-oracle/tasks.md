## 1. Проброс числа спикеров в пайплайн

- [x] 1.1 В `_run_pipeline_sync` оставить проброс `num_speakers`/`min_speakers`/`max_speakers` в `whisperx.DiarizationPipeline` без значений по умолчанию на бэкенде (приоритет `num_speakers` > `min/max`)
- [x] 1.2 Убедиться, что при отсутствии всех параметров `whisperx.DiarizationPipeline` вызывается с `None` (автоопределение pyannote)
- [x] 1.3 Добавить валидацию: `num_speakers >= 1`, `min_speakers >= 1`, `max_speakers >= 1`, `min_speakers <= max_speakers` → HTTP 400 при нарушении
- [x] 1.4 Убедиться, что `whisperx.DiarizationPipeline` остаётся единственным движком диаризации (без кастомных модулей подсчёта/слияния кластеров)

## 2. Конфигурация

- [x] 2.1 Добавить `diarize_model: str = "pyannote/speaker-diarization-community-1"` в `src/whisperx_api/config.py` (env `WHISPERX_DIARIZE_MODEL`)
- [x] 2.2 Передавать `model_name=config.diarize_model` в конструктор `whisperx.DiarizationPipeline`

## 3. Модуль formatting.py — гибридный word-level форматтер

- [x] 3.1 Реализовать `hybrid_word_blocks(segments)`: разрезать сегмент по смене `word.speaker` только при реальной смене спикера
- [x] 3.2 Текст блока: `segment.text` целиком при одном спикере; пересборка из `words[]` при нескольких
- [x] 3.3 Fallback на `segment.text` при отсутствии `word.speaker` — без потери реплик
- [x] 3.4 Склейка соседних блоков одного спикера в один репликовый блок
- [x] 3.5 Перевести `_build_diarized_json`/`_build_diarized_text` на `hybrid_word_blocks` (заменить segment-level версии)
- [x] 3.6 Юнит-тесты: смена спикера внутри сегмента, отсутствие потери текста, склейка, сегмент без speaker

## 4. Документация

- [x] 4.1 Обновить README: контракт `num/min/max` (значения задаёт клиент), автоопределение при отсутствии параметров, гибридный word-level формат, `WHISPERX_DIARIZE_MODEL`
- [x] 4.2 Обновить `docker-compose.yml` примером env `WHISPERX_DIARIZE_MODEL`

## 5. Тесты и отчёт

- [x] 5.1 E2E-тест на проблемном аудио: `min_speakers=3`/`max_speakers=3` → число участников 3 (не 4)
- [x] 5.2 E2E-тест: реплики разных людей в одном сегменте разделены на корректные блоки, текст не потерян
- [x] 5.3 E2E-тест: без параметров числа спикеров запрос проходит (автоопределение)
- [x] 5.4 Запуск всех тестов в контейнере (`podman run ...`) и фиксация отчёта о тестировании