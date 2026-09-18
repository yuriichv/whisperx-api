## Why

Метод VAD для ASR задаётся только дефолтом WhisperX при `load_model`. Оператору нужно переключать `pyannote`, `silero` или другой штатный `vad_method` без правки кода — на этапе деплоя.

## What Changes

- Env-параметр `WHISPERX_VAD_METHOD` (строка, optional): при непустом значении передаётся в `whisperx.load_model(..., vad_method=...)`
- При отсутствии env аргумент `vad_method` в `load_model` не передаётся (поведение WhisperX по умолчанию)
- Документация в README и пример в `docker-compose.yml`

## Capabilities

### New Capabilities

- `transcription`: конфигурация VAD при загрузке ASR pipeline

### Modified Capabilities

_(нет)_

## Impact

- `src/whisperx_api/config.py` — поле `vad_method`
- `src/whisperx_api/state.py` — условный проброс в `load_model`
- Тесты, README, docker-compose
