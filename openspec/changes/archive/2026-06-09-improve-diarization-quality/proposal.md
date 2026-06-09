## Why

Сервис некорректно группирует реплики: в одну строку `SPEAKER_X` попадает речь двух разных людей. Основные причины — выключенный align при диаризации (нет word-level speakers), назначение спикера на уровне сегмента Whisper (majority vote), и отсутствие разбивки текста по смене спикера внутри сегмента. Нужно улучшить качество диаризации без смены diarization-модели.

## What Changes

- Автоматически включать `align=true`, когда запрошена диаризация (`diarize=true` или `response_format=diarized_json`), если клиент явно не передал `align=false`
- Пересобирать `diarized_json.text` и сегменты из `words[]` с разбивкой при смене `word.speaker`
- Добавить параметр API `num_speakers` (точное число спикеров) и пробросить его в `DiarizationPipeline`
- Передавать `fill_nearest=true` в `whisperx.assign_word_speakers` для назначения спикера словам на границах сегментов диаризации
- Обновить README: рекомендовать `align=true` при диаризации

## Capabilities

### New Capabilities

- `diarization`: Поведение диаризации в API транскрипции — авто-align, параметры числа спикеров, word-level форматирование вывода, `fill_nearest`

### Modified Capabilities

<!-- Нет существующих spec-файлов в openspec/specs/ -->

## Impact

- `src/whisperx_api/transcribe_router.py` — логика пайплайна, форматирование `diarized_json`, новый form-параметр
- `README.md` — примеры запросов
- Обратная совместимость: клиенты, явно передающие `align=false`, сохраняют текущее поведение; формат `diarized_json` может дать больше сегментов (разбивка по словам) — не breaking, но изменение структуры вывода
