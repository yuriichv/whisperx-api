## MODIFIED Requirements

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
