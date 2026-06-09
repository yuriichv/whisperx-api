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

Система SHALL вызывать `whisperx.assign_word_speakers` с `fill_nearest=true` для назначения спикера словам и сегментам на границах интервалов диаризации, где нет прямого временного пересечения.

#### Scenario: Слово на границе сегмента диаризации

- **WHEN** word-level timestamp слова не пересекается ни с одним интервалом диаризации, но находится рядом с ближайшим сегментом
- **THEN** слову назначается спикер ближайшего сегмента диаризации

### Requirement: Word-level форматирование diarized_json

Система SHALL формировать `diarized_json.text` и `diarized_json.segments` на основе `words[]` с разбивкой при смене `word.speaker`, а не только по `segment.speaker` Whisper.

#### Scenario: Смена спикера внутри сегмента Whisper

- **WHEN** после alignment и assign_word_speakers внутри одного Whisper-сегмента слова имеют разных спикеров (`SPEAKER_00`, `SPEAKER_01`)
- **THEN** `diarized_json.text` содержит отдельные строки для каждого непрерывного блока одного спикера
- **THEN** `diarized_json.segments` содержит отдельные записи с корректными `start`, `end`, `speaker` и `text` для каждого блока

#### Scenario: Fallback без words

- **WHEN** alignment отключён (`align=false`) и в результате нет `words[]` со спикерами
- **THEN** система использует текущую логику группировки по `segment.speaker` (без word-level разбивки)

#### Scenario: Склейка соседних слов одного спикера

- **WHEN** несколько подряд идущих слов имеют одинакового спикера
- **THEN** они объединяются в один блок с `start` первого слова и `end` последнего слова
