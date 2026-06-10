## REMOVED Requirements

### Requirement: Word-level форматирование diarized_json

**Reason**: Word-level пересборка вывода приводит к потере реплик (слова без `word.speaker` отбрасываются) и ухудшению текста (join токенов вместо `seg.text`). Segment-level форматирование с `align=true` даёт лучший результат на реальных данных.

**Migration**: `diarized_json` снова формируется по Whisper-сегментам (`segment.speaker` + `segment.text`). Клиентам, которым нужна word-level гранулярность, использовать `verbose_json` с `words[].speaker`.

## ADDED Requirements

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
