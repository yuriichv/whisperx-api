## 1. Segment-level форматтер

- [x] 1.1 Удалить `formatting.py` (или `hybrid_word_blocks` и вспомогательные функции) и перенести маппинг `result.segments` → `TranscriptBlock` в `responses.py`; проверить `uv run python -c "from whisperx_api.features.transcription.responses import diarized_response"`
- [x] 1.2 Обновить `diarized_response()`: блоки строятся из `segment.speaker` + `segment.text` + `segment.start`/`segment.end`, без вызова word-level форматтера; проверить unit-тест `diarized_response` на fixture с boundary noise (один сегмент, разные `word.speaker`)

## 2. Сохранить улучшения пайплайна (без изменений, проверить)

- [x] 2.1 Убедиться, что авто-align при `do_diarize=true` и `align is None` остаётся — e2e `test_auto_align_on_diarize` проходит
- [x] 2.2 Убедиться, что `fill_nearest` и `num_speakers`/`min_speakers`/`max_speakers` не затронуты — e2e `test_min_max_speakers_determines_3_participants` проходит

## 3. Тесты

- [x] 3.1 Переписать `tests/unit/test_formatting.py` → segment-level (или переименовать/перенести в `test_responses.py`); добавить сценарий boundary noise из ADR-001; `uv run pytest tests/unit/ -q` проходит
- [x] 3.2 Обновить `tests/e2e/test_transcribe_api.py`: `test_replicas_in_segment_split_no_text_loss` ожидает segment-level (один блок на Whisper-сегмент, текст не теряется); `uv run pytest tests/e2e/ -q` проходит

## 4. Документация

- [x] 4.1 Обновить README: `diarized_json` формируется по Whisper-сегментам (`segment.speaker`), `words[].speaker` — только в `verbose_json`; ссылка на ADR-001
- [x] 4.2 Обновить статус ADR-001: `Proposed` → `Accepted` после merge change

## 5. Верификация

- [x] 5.1 Ручная проверка на проблемном аудио из ADR-001: «Просто интовый массив» — одна реплика SPEAKER_00, без ложного split на «массив»
- [x] 5.2 Ручная проверка: `verbose_json` по-прежнему содержит `words[].speaker` при `diarize=true`
- [x] 5.3 `uv run pytest` — все тесты проходят

## 6. Склейка соседних блоков одного спикера в segments

- [x] 6.1 Добавить `merge_adjacent_same_speaker()` в `responses.py` и вызывать в `diarized_response()` после `segment_blocks()`
- [x] 6.2 Unit-тесты: склейка двух сегментов одного спикера; не склеивать A→B→A; `diarized_response` возвращает merged segments
- [x] 6.3 Обновить delta spec и design: merge в `diarized_json.segments`, не только в `text`
