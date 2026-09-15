## 1. Спецификация и тестовые сценарии

- [x] 1.1 Описать unit/e2e сценарии для prompt, hotwords, char/token limits (500/1000/100/150/200), prompt+diarize, auto-language в комментариях к тестам (сверка со spec)

## 2. Pydantic-контракт (schemas)

- [x] 2.1 Создать `src/whisperx_api/schemas.py`: `TranscriptionRequest`, nested segment/word models, response models; нормализация prompt/hotwords (strip→None); char limits (prompt 500, hotwords 1000); Whisper token validator (100/150/200) через tokenizer ASR-модели; speaker params при diarize
- [x] 2.2 Добавить dependency `parse_transcription_form` для сборки request из multipart Form + File
- [x] 2.3 Unit-тесты `tests/unit/test_schemas.py`: нормализация hotwords/prompt, char max_length, token budget (per-field и combined), ValidationError → HTTP 422

## 3. ASR: prompt/hotwords и fix language

- [x] 3.1 Вынести apply/restore `initial_prompt`/`hotwords` в helper; вызывать внутри `GPU_LOCK` в `_run_pipeline_sync`
- [x] 3.2 Исправить вызов `transcribe()` без явного `language` (auto-detect)
- [x] 3.3 Unit-тест: mock pipeline options — save/set/restore prompt и hotwords

## 4. Рефакторинг роутера

- [x] 4.1 Перевести `transcriptions` на `TranscriptionRequest` + response Pydantic models вместо ручных dict
- [x] 4.2 Сохранить текущее поведение validation HTTP 400/422 и всех response_format
- [x] 4.3 E2e-тесты: prompt/hotwords пробрасываются в options; prompt+diarize → 200; response проходит model_validate

## 5. Документация и верификация

- [x] 5.1 Обновить `README.md`: ADR limits (500/1000 chars, 100/150/200 tokens, рекомендация 1500 chars суммарно), hotwords нормализация strip-only, prompt+diarize extension, без env-defaults
- [x] 5.2 Запустить полный pytest в контейнере и оформить отчёт о тестировании
