## 1. Откат форматтера diarized_json

- [x] 1.1 Удалить функцию `_split_segments_by_word_speaker` из `transcribe_router.py`
- [x] 1.2 Вернуть `_build_diarized_text(result)` — итерация по `result.get("segments")`, текст из `seg.text`
- [x] 1.3 Вернуть `_build_diarized_json(result, ...)` — сегменты из `result.get("segments")`, не из blocks
- [x] 1.4 В endpoint `transcriptions()` убрать вызов `_split_segments_by_word_speaker`, передавать `result` напрямую в builders

## 2. Сохранить улучшения пайплайна (без изменений, проверить)

- [x] 2.1 Убедиться, что авто-align при `do_diarize=true` и `align is None` остаётся
- [x] 2.2 Убедиться, что `fill_nearest=True` в `assign_word_speakers` остаётся
- [x] 2.3 Убедиться, что параметр `num_speakers` в API остаётся

## 3. Документация

- [x] 3.1 Обновить README: `diarized_json` формируется по Whisper-сегментам, не по word-level split
- [x] 3.2 Убрать/скорректировать упоминание word-level speaker split в описании `diarized_json`

## 4. Верификация

- [x] 4.1 Ручная проверка на проблемном аудио: после `20.686s` присутствуют сегменты «Ну,» и «отрасли, давайте...»
- [x] 4.2 Ручная проверка: `verbose_json` по-прежнему содержит `words[].speaker` при `diarize=true`
- [x] 4.3 Ручная проверка: первые три сегмента (SPEAKER_02/04) не деградировали
