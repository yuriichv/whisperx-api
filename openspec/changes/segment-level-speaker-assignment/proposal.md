## Why

Гибридный word-level форматтер `diarized_json` (change `improve-diarization-oracle`) разбивает реплики по смене `word.speaker`. На границах смены спикеров WhisperX может назначить одному слову «чужого» спикера из-за overlap между forced alignment timestamps и pyannote diarization boundaries — даже при высоком alignment score. Одиночная смена `word.speaker` превращается в заметную ошибку вывода (см. [ADR-001](../../docs/ADR-001-segment-level%20speaker%20assignment.md)). Локальная реализация, использующая `segment.speaker`, этот класс дефектов не проявляет.

## What Changes

- Убрать гибридный word-level форматтер (`hybrid_word_blocks`, `_segment_blocks`) и формировать `diarized_json` по Whisper-сегментам (`segment.speaker` + `segment.text`)
- Склеивать подряд идущие блоки одного спикера в `diarized_json.segments` (как в `text`)
- Сохранить `words[].speaker` в `verbose_json` как диагностическую информацию; не использовать для разбиения реплик
- Сохранить без изменений: авто-align при диаризации, `fill_nearest`, `num_speakers`/`min_speakers`/`max_speakers`, конфигурируемую модель диаризации
- Обновить spec `diarization`: заменить требование word-level атрибуции на segment-level
- Обновить README: описать segment-level поведение `diarized_json`
- Обновить unit/e2e тесты под новый контракт
- Зафиксировать ADR-001 как архитектурное обоснование (документ уже в `docs/`)

## Capabilities

### New Capabilities

_(нет)_

### Modified Capabilities

- `diarization`: заменить требование «Word-level атрибуция текста с корректными границами» на segment-level форматирование `diarized_json`; уточнить, что `words[].speaker` остаётся в `verbose_json`, но не влияет на сборку `diarized_json`

## Impact

- `src/whisperx_api/features/transcription/formatting.py` — удаление гибридного word-level форматтера
- `src/whisperx_api/features/transcription/responses.py` — `diarized_response()` строит блоки из `result.segments` напрямую
- `openspec/specs/diarization/spec.md` — обновление требований форматирования (через delta spec)
- `README.md` — корректировка описания `diarized_json`
- `tests/unit/test_formatting.py` — переписать под segment-level
- `tests/e2e/test_transcribe_api.py` — обновить сценарии, ожидающие word-level split
- **Обратная совместимость**: `diarized_json` снова не разбивает Whisper-сегмент по `word.speaker`; редкая реальная смена спикера внутри одного aligned segment будет отнесена к `segment.speaker` — осознанный trade-off ADR-001
