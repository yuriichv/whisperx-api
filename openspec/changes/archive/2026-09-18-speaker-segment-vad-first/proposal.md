## Why

WhisperX после `assign_word_speakers()` уже задаёт `segment.speaker` и `segment.text` на границах сегментов ASR. Сервис добавил второй слой: гибридный форматтер режет Whisper-сегмент по `word.speaker` и пересобирает блоки `diarized_json` — это противоречит ADR-001 (вариант A отклонён: локальная ошибка `word.speaker` не должна менять структуру реплики). Принятое решение ADR-001 для текущего этапа: **только** штатный WhisperX (`assign_word_speakers` без пост-обработки `segment.speaker`) и смена VAD на Silero, чтобы границы сегментов давали корректный whole-segment расчёт; кастомную «пересборку» спикера убрать.

## What Changes

- Удалить гибридную word-level модель (`formatting.hybrid_word_blocks`, split/flush по `word.speaker`) — **упрощение** пайплайна ответа
- Формировать `diarized_json` и `diarized_text` напрямую из aligned segments после `assign_word_speakers`: `speaker` = `segment.speaker`, `text` = `segment.text`, границы = `segment.start`/`segment.end`
- Оставить опциональную **склейку только соседних Whisper-сегментов** с одним `segment.speaker` (не меняет атрибуцию, только представление)
- Зафиксировать **VAD-first**: рекомендованный деплой `WHISPERX_VAD_METHOD=silero` + regression на фрагмент «Это с нулями или нау?» из ADR-001
- **Не** вводить post-step переопределения `segment.speaker` по `words[]` (fallback ADR, вариант C — вне scope, отдельный change не планируется)
- Сохранить `words[].speaker` в `verbose_json` **только для диагностики**; не использовать эти метки при сборке `diarized_json`
- Обновить spec/tests/README под segment-level контракт (**BREAKING** для клиентов, зависевших от word-split внутри одного Whisper-сегмента)

## Capabilities

### New Capabilities

_(нет)_

### Modified Capabilities

- `diarization`: segment-level `diarized_json`; `verbose_json` с `words[].speaker` для диагностики

## Impact

- Удаление или существенное упрощение `src/whisperx_api/features/transcription/formatting.py`
- `src/whisperx_api/features/transcription/responses.py` — прямое построение из segments
- `docker-compose.yml`, `README.md` — Silero VAD, описание контракта
- `openspec/specs/diarization/spec.md`
- Unit-тесты форматтера и e2e, завязанные на hybrid split
