## Why

Word-level пересборка `diarized_json` (change `improve-diarization-quality`) приводит к потере реплик на границах смены спикера: слова без `word.speaker` отбрасываются, текст собирается из токенов вместо `seg.text`. На реальном аудио после `20.686s` пропадают сегменты «Ну,» и длинная реплика SPEAKER_02, тогда как старая версия с `align=true` и segment-level форматированием отдавала их корректно.

## What Changes

- Убрать `_split_segments_by_word_speaker` и вернуть форматирование `diarized_json` по Whisper-сегментам (`segment.speaker` + `segment.text`)
- Сохранить авто-align при диаризации (`align` не передан → `do_align=true`)
- Сохранить `fill_nearest=True` в `assign_word_speakers`
- Сохранить параметр `num_speakers` в API
- Обновить spec: убрать требование word-level разбивки вывода, заменить на segment-level форматирование
- Обновить README: убрать упоминание word-level split в `diarized_json`

## Capabilities

### New Capabilities

_(нет)_

### Modified Capabilities

- `diarization`: заменить требование word-level форматирования `diarized_json` на segment-level; уточнить, что `words[].speaker` остаётся в `verbose_json`, но не используется для сборки `diarized_json`

## Impact

- `src/whisperx_api/transcribe_router.py` — удаление `_split_segments_by_word_speaker`, откат `_build_diarized_text` и `_build_diarized_json` к работе с `result.segments`
- `openspec/specs/diarization/spec.md` — обновление требований форматирования
- `README.md` — корректировка описания поведения `diarized_json`
- Обратная совместимость: `diarized_json` снова даёт меньше сегментов (по Whisper-сегментам), но без потери текста; два спикера в одном длинном Whisper-сегменте снова могут попасть в одну строку — осознанный trade-off
