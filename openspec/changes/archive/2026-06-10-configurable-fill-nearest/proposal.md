## Why

`fill_nearest` в `assign_word_speakers` сейчас захардкожен как `True`. На границах сегментов pyannote это помогает, но на шумных/тишинных участках может назначить неверного спикера. Оператору нужна возможность отключить поведение без правки кода — через env, как для `default_align` и `default_diarize`.

## What Changes

- Добавить env-параметр `WHISPERX_FILL_NEAREST` (bool, default `true`) в `Config`
- Пробросить значение в `whisperx.assign_word_speakers(..., fill_nearest=config.fill_nearest)`
- Документировать в `docker-compose.yml` и README
- Обновить spec: `fill_nearest` управляется конфигурацией сервера, не захардкожен

## Capabilities

### New Capabilities

_(нет)_

### Modified Capabilities

- `diarization`: требование `fill_nearest` — значение берётся из env `WHISPERX_FILL_NEAREST`, default `true`

## Impact

- `src/whisperx_api/config.py` — новое поле `fill_nearest: bool = True`
- `src/whisperx_api/transcribe_router.py` — замена `fill_nearest=True` на `config.fill_nearest`
- `docker-compose.yml` — пример env-переменной
- `README.md` — описание параметра
- Обратная совместимость: без явной установки env поведение не меняется (`true`)
