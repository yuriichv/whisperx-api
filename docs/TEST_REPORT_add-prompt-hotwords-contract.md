# Отчёт о тестировании: add-prompt-hotwords-contract

**Дата:** 2026-09-15  
**Change:** `add-prompt-hotwords-contract`

## Команда

```bash
podman run --rm -v "$PWD":/tmp/app -w /tmp/app python:3.12 sh -c \
  "pip install -q pytest fastapi pydantic pydantic-settings python-multipart httpx numpy && \
   PYTHONPATH=src:tests WHISPERX_NO_AUTH=true python -m pytest tests/ -q"
```

## Результат

```
53 passed, 2 warnings in 2.04s
```

## Покрытие по spec

| Область | Тесты |
|---------|-------|
| Нормализация prompt/hotwords | `tests/unit/test_transcription_form_input.py` |
| Char limits 500/1000 | `tests/unit/test_transcription_form_input.py` |
| Token budget 100/150/200 | `tests/unit/test_transcription_form_input.py`, e2e `test_hotwords_token_limit_returns_422` |
| apply/restore ASR options | `tests/unit/test_asr_conditioning.py`, `tests/unit/test_service.py` |
| prompt/hotwords → options | e2e `test_prompt_and_hotwords_applied_to_asr_options` |
| prompt + diarize | e2e `test_prompt_with_diarize_allowed` |
| auto-language | e2e `test_request_without_language_succeeds` |
| Pydantic response models | e2e `test_*_response_matches_pydantic_model` |
| Дiarization (regression) | существующие e2e в `test_transcribe_api.py` |

## Замечания

- Полный `uv sync` с whisperx в контейнере не использовался; e2e работает на заглушках ASR/diarize (как в предыдущих отчётах).
- Deprecation warnings FastAPI `on_event` — существующие, не связаны с change.
