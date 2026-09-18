## 1. Упростить форматирование ответа

- [ ] 1.1 Удалить word-level split из `formatting.py` (hybrid `_segment_blocks`, пересборка текста из `words[]`); оставить map Whisper segment → block и склейку соседних сегментов с одним `segment.speaker`
- [ ] 1.2 Обновить `responses.diarized_response` при необходимости и прогнать unit-тесты форматтера — один Whisper segment → один block, speaker/text из segment

## 2. Тесты и контракт

- [ ] 2.1 Переписать/удалить тесты, ожидавшие word-split `diarized_json`; добавить кейс «разные word.speaker, один block по segment»
- [ ] 2.2 Обновить e2e/fixtures под segment-level контракт; проверить, что при `diarize=true` `verbose_json` по-прежнему содержит `words[].speaker`, а `diarized_json` — нет word-split
- [ ] 2.3 Убедиться, что `pytest` по transcription проходит

## 3. VAD-first deploy

- [ ] 3.1 Задать `WHISPERX_VAD_METHOD=silero` в рекомендуемом деплое (`docker-compose.yml` / README)
- [ ] 3.2 Добавить spike/e2e или задокументированный manual regression на фрагмент ADR «Это с нулями или нау?»

## 4. Документация

- [ ] 4.1 Обновить README: `diarized_json` = `segment.speaker`/`segment.text`; `verbose_json` — `words[].speaker` для диагностики; ADR-001; BREAKING; fallback C не реализуется

## 5. OpenSpec sync

- [ ] 5.1 После реализации archive/sync change и `openspec validate`
