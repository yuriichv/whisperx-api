# Proposal: refactor-transcription-layers

## Проблема

`transcribe_router.py` совмещал HTTP, оркестрацию пайплайна, lazy-load моделей и форматирование ответов. Обнаружен критический баг: ASR не вызывался при отсутствии `language`.

## Решение

Поэтапный рефакторинг (фазы 0–4):

1. Hotfix вызова `transcribe()` без языка
2. Выделение `features/transcription/{api,service,responses,errors}`
3. Typed contracts (`contracts.py`)
4. Lifespan, лимит upload, quality gate (ruff/pyright)
5. Feature-first структура `features/transcription/`

## Последствия

- Публичный HTTP-контракт не меняется
- Код транскрипции живёт в `features/transcription/`; корневые shim-модули удалены
