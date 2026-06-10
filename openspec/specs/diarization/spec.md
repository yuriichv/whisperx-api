# Diarization

Speaker diarization behavior for the `/v1/audio/transcriptions` API.

## Requirements

### Requirement: Авто-align при диаризации

Система SHALL автоматически включать этап alignment (`align=true`), когда запрошена диаризация (`diarize=true` или `response_format=diarized_json`), если клиент явно не передал `align=false`.

#### Scenario: Диаризация без явного align

- **WHEN** клиент отправляет запрос с `diarize=true` и не передаёт параметр `align`
- **THEN** пайплайн выполняет alignment перед диаризацией

#### Scenario: Явное отключение align

- **WHEN** клиент отправляет запрос с `diarize=true` и `align=false`
- **THEN** пайплайн пропускает alignment и сохраняет текущее поведение без word-level timestamps

#### Scenario: diarized_json без явного diarize

- **WHEN** клиент отправляет запрос с `response_format=diarized_json` и не передаёт `align`
- **THEN** система включает и диаризацию, и alignment

### Requirement: Параметр num_speakers

API `/v1/audio/transcriptions` SHALL принимать опциональный form-параметр `num_speakers` (целое число ≥ 1) и передавать его в `DiarizationPipeline` как точное число спикеров.

#### Scenario: Задано точное число спикеров

- **WHEN** клиент передаёт `num_speakers=2` и `diarize=true`
- **THEN** диаризация вызывается с `num_speakers=2`

#### Scenario: num_speakers совместим с min/max

- **WHEN** клиент передаёт `num_speakers` вместе с `min_speakers` или `max_speakers`
- **THEN** `num_speakers` имеет приоритет над `min_speakers`/`max_speakers`

#### Scenario: Невалидное значение

- **WHEN** клиент передаёт `num_speakers=0` или отрицательное число
- **THEN** система возвращает HTTP 400 с описанием ошибки

### Requirement: fill_nearest при назначении спикеров

Система SHALL передавать в `whisperx.assign_word_speakers` значение `fill_nearest` из конфигурации сервера (`WHISPERX_FILL_NEAREST`, default `true`). При `fill_nearest=true` словам и сегментам на границах интервалов диаризации без прямого пересечения назначается спикер ближайшего сегмента. При `fill_nearest=false` назначение происходит только при временном overlap.

#### Scenario: fill_nearest включён (default)

- **WHEN** сервис запущен без `WHISPERX_FILL_NEAREST` или с `WHISPERX_FILL_NEAREST=true`
- **THEN** `assign_word_speakers` вызывается с `fill_nearest=true`

#### Scenario: fill_nearest отключён через env

- **WHEN** сервис запущен с `WHISPERX_FILL_NEAREST=false`
- **THEN** `assign_word_speakers` вызывается с `fill_nearest=false`

#### Scenario: Слово на границе сегмента диаризации при fill_nearest=true

- **WHEN** `WHISPERX_FILL_NEAREST=true` и word-level timestamp слова не пересекается ни с одним интервалом диаризации, но находится рядом с ближайшим сегментом
- **THEN** слову назначается спикер ближайшего сегмента диаризации

#### Scenario: Слово на границе без overlap при fill_nearest=false

- **WHEN** `WHISPERX_FILL_NEAREST=false` и word-level timestamp слова не пересекается ни с одним интервалом диаризации
- **THEN** слову не назначается спикер через fill_nearest (остаётся без `speaker`, если нет overlap)

### Requirement: Segment-level форматирование diarized_json

Система SHALL формировать `diarized_json.text` и `diarized_json.segments` на основе Whisper-сегментов (`segment.speaker`, `segment.text`, `segment.start`, `segment.end`), а не на основе пересборки из `words[]`.

#### Scenario: Формирование сегментов из Whisper

- **WHEN** клиент запрашивает `response_format=diarized_json` с включённой диаризацией
- **THEN** каждый Whisper-сегмент с непустым `text` и назначенным `speaker` попадает в `diarized_json.segments`
- **THEN** текст сегмента берётся из `segment.text`, а не из join токенов `words[]`

#### Scenario: Склейка соседних сегментов одного спикера в text

- **WHEN** несколько подряд идущих Whisper-сегментов имеют одинакового `segment.speaker`
- **THEN** `diarized_json.text` объединяет их в одну строку `SPEAKER_X: <text>`
- **THEN** `diarized_json.segments` сохраняет отдельную запись для каждого Whisper-сегмента

#### Scenario: Сегменты без speaker

- **WHEN** Whisper-сегмент не получил `speaker` после диаризации
- **THEN** сегмент включается в вывод с `speaker` = `UNKNOWN` или `null` (как в текущей логике `_build_diarized_text`)

#### Scenario: words[] не используется для diarized_json

- **WHEN** после alignment в сегменте есть `words[]` с разными `word.speaker`
- **THEN** `diarized_json` использует `segment.speaker` (majority overlap), а не разбивает по `word.speaker`
- **THEN** `verbose_json` по-прежнему отдаёт `words[].speaker` для клиентов, которым нужна word-level детализация
