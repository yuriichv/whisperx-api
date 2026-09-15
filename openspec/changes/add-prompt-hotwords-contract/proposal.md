## Why

Параметр `prompt` объявлен в API, но не применяется к ASR; `hotwords` отсутствует. Клиенты не могут улучшить распознавание имён, терминов и стиля речи. Контракт запроса/ответа не формализован — параметры разбросаны по `Form(...)`, ответы собираются вручную без typed-моделей, что затрудняет валидацию, тесты и OpenAPI.

## What Changes

- Реализовать проброс **`prompt`** → `initial_prompt` в options ASR-пайплайна whisperx (per-request)
- Добавить WhisperX extension **`hotwords`** (string) → `options.hotwords` (per-request)
- Ввести модуль **`schemas.py`** с Pydantic-моделями **полного** контракта: запрос (все form-поля) и ответы (`json`, `verbose_json`, `diarized_json`, `text`)
- Рефакторинг `transcribe_router.py`: парсинг через Pydantic, сбор ответов через response-модели
- Явно задокументировать: **`prompt` + `diarize=true` разрешены** — WhisperX extension, несовместимо с ограничением OpenAI для `gpt-4o-transcribe-diarize`
- **Без** env-defaults для prompt/hotwords
- Исправить баг: ASR вызывается и при отсутствии явного `language` (auto-detect)
- Валидация conditioning по ADR: `prompt` max 500 chars / 100 tokens, `hotwords` max 1000 chars / 150 tokens, совместно max 200 tokens (см. design.md ADR)

## Capabilities

### New Capabilities

- `transcription-api`: формальный контракт POST `/v1/audio/transcriptions` — Pydantic request/response, prompt, hotwords, валидация, проброс в ASR

### Modified Capabilities

_(нет — диаризация не меняется; prompt/hotwords затрагивают только этап ASR)_

## Impact

- `src/whisperx_api/schemas.py` — новый модуль контрактов
- `src/whisperx_api/transcribe_router.py` — Pydantic, apply/restore ASR options, fix auto-language
- `tests/unit/test_schemas.py`, `tests/e2e/test_transcribe_api.py` — новые/расширенные сценарии
- `README.md` — prompt, hotwords, prompt+diarize extension
- `openspec/specs/transcription-api/spec.md` — новая main spec после archive
