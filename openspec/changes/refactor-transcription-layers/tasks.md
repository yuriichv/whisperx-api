## 1. Фаза 0 — Hotfix

- [x] 1.1 Исправить вызов `transcribe()` при `language=None`
- [x] 1.2 E2E: запрос без `language` → 200
- [x] 1.3 Unit: `run_pipeline_sync` с `language=None`

## 2. Фаза 1 — Слои

- [x] 2.1 `TranscriptionService` → `service.py`
- [x] 2.2 `responses.py` — все `_build_*`
- [x] 2.3 `errors.py` → HTTP mapping в `api.py`
- [x] 2.4 Убрать дубли `_select_device` из router
- [x] 2.5 Убрать `transcribe_router.app = app`
- [x] 2.6 Один `logging.basicConfig` в `bootstrap.py`

## 3. Фаза 2 — Typed contracts

- [x] 3.1 TypedDict для segment/word/block (внутренний `contracts.py`)
- [x] 3.2 Pydantic BaseModel для внешнего API (`api_schemas.py`)
- [x] 3.3 Contract-тесты схем ответов
- [x] 3.4 `formatting.py` на typed input

## 4. Фаза 3 — Lifecycle и надёжность

- [x] 4.1 `lifespan` вместо `on_event`
- [x] 4.2 `torch.load` workaround в `bootstrap.py`
- [ ] 4.3 `max_upload_bytes` в config (отменено — вне scope diarization spec)
- [x] 4.4 E2E на `fill_nearest=false`
- [x] 4.5 `ruff` + `pyright` в pyproject.toml

## 5. Фаза 4 — Feature-first

- [x] 5.1 Структура `features/transcription/`
- [x] 5.2 Удалены shim `transcribe_router.py` / `formatting.py`; тесты на новых путях
