# Diarization

Speaker diarization behavior for the `/v1/audio/transcriptions` API.

## Purpose

Параметры и контракт диаризации: проброс числа спикеров в pyannote, `assign_word_speakers`, форматы `diarized_json` и диагностические метки в `verbose_json`.

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

API `/v1/audio/transcriptions` SHALL принимать опциональный form-параметр `num_speakers` (целое число ≥ 1) и передавать его в `whisperx.DiarizationPipeline` как точное число спикеров. Если задан одновременно с `min_speakers`/`max_speakers`, `num_speakers` имеет приоритет.

#### Scenario: Задано точное число спикеров

- **WHEN** клиент передаёт `num_speakers=3` и `diarize=true`
- **THEN** `whisperx.DiarizationPipeline` вызывается с `num_speakers=3`

#### Scenario: num_speakers вместе с min/max

- **WHEN** клиент передаёт `num_speakers` вместе с `min_speakers` или `max_speakers`
- **THEN** `num_speakers` имеет приоритет над `min_speakers`/`max_speakers`

#### Scenario: Невалидное значение

- **WHEN** клиент передаёт `num_speakers=0` или отрицательное число
- **THEN** система возвращает HTTP 400 с описанием ошибки

### Requirement: Проброс min_speakers/max_speakers без значений по умолчанию

Система SHALL передавать `min_speakers`/`max_speakers` из API в `whisperx.DiarizationPipeline` как есть, не задавая значений по умолчанию на бэкенде. При отсутствии всех параметров числа спикеров (`num_speakers`, `min_speakers`, `max_speakers`) система SHALL вызывать пайплайн с `None`, оставляя определение числа спикеров pyannote (автоопределение).

#### Scenario: Переданы min/max с клиента

- **WHEN** клиент передаёт `min_speakers=3` и `max_speakers=8` и `diarize=true`
- **THEN** `whisperx.DiarizationPipeline` вызывается с `min_speakers=3`, `max_speakers=8`

#### Scenario: Параметры отсутствуют — автоопределение

- **WHEN** клиент передаёт `diarize=true` без `num_speakers`, `min_speakers`, `max_speakers`
- **THEN** `whisperx.DiarizationPipeline` вызывается с `min_speakers=None`, `max_speakers=None` и pyannote сам определяет число спикеров

#### Scenario: Передан только один из min/max

- **WHEN** клиент передаёт только `max_speakers=5`
- **THEN** пайплайн вызывается с `max_speakers=5`, `min_speakers=None`

### Requirement: Валидация min_speakers/max_speakers

Система SHALL валидировать `min_speakers` и `max_speakers` как целые числа ≥ 1, при `min_speakers > max_speakers` возвращать HTTP 400. Валидация `num_speakers`/`min_speakers`/`max_speakers` применяется только при запрошенной диаризации (`do_diarize=true`); без диаризации эти параметры игнорируются и не проверяются.

#### Scenario: min больше max

- **WHEN** клиент передаёт `min_speakers=5` и `max_speakers=3`
- **THEN** система возвращает HTTP 400 с описанием ошибки

#### Scenario: Некорректное значение min/max

- **WHEN** клиент передаёт `min_speakers=0` или отрицательное `max_speakers`
- **THEN** система возвращает HTTP 400 с описанием ошибки

#### Scenario: Валидация только при диаризации

- **WHEN** клиент передаёт `num_speakers=0` без `diarize=true`
- **THEN** система не проверяет параметры числа спикеров и не возвращает HTTP 400

### Requirement: Конфигурируемая модель диаризации

Система SHALL использовать имя diarization-модели из конфигурации сервера (`WHISPERX_DIARIZE_MODEL`, default `pyannote/speaker-diarization-community-1`) при конструировании `whisperx.DiarizationPipeline`.

#### Scenario: Модель из конфигурации

- **WHEN** сервис запущен с `WHISPERX_DIARIZE_MODEL=<model>`
- **THEN** `whisperx.DiarizationPipeline` конструируется с `model_name=<model>`

#### Scenario: Модель по умолчанию

- **WHEN** `WHISPERX_DIARIZE_MODEL` не задан
- **THEN** используется `pyannote/speaker-diarization-community-1`

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

### Requirement: Diarized output по segment.speaker WhisperX

После `whisperx.assign_word_speakers` система SHALL формировать `diarized_json` и текстовый diarized-вывод без переопределения `segment.speaker` и без пересборки `segment.text` из `words[]`. Каждый aligned Whisper-сегмент SHALL давать один репликовый блок с `text` равным `segment.text`, `speaker` равным `segment.speaker`, временными границами сегмента. Система SHALL NOT делить один Whisper-сегмент на несколько блоков только из-за различия `word.speaker` внутри сегмента.

#### Scenario: Разные word.speaker внутри одного Whisper-сегмента

- **WHEN** после `assign_word_speakers` в одном Whisper-сегменте слова имеют разных `word.speaker`, но `segment.speaker` задан WhisperX
- **THEN** `diarized_json` содержит один блок с полным `segment.text` и `segment.speaker`, без word-level нарезки

#### Scenario: Один спикер в сегменте

- **WHEN** все слова сегмента и `segment.speaker` согласованы
- **THEN** блок использует `segment.text` и `segment.speaker` без пересборки из `words[]`

#### Scenario: Склейка соседних Whisper-сегментов

- **WHEN** несколько подряд идущих Whisper-сегментов имеют одинаковый `segment.speaker`
- **THEN** `diarized_json` MAY объединить их в один репликовый блок с `start` первого и `end` последнего сегмента, не изменяя присвоенных WhisperX меток спикера

#### Scenario: Сегмент без speaker

- **WHEN** Whisper-сегмент не получил `segment.speaker` после диаризации
- **THEN** блок отображается с `speaker` равным `UNKNOWN` или `null`, текст `segment.text` сохраняется

### Requirement: Проверка стратегии VAD-first

Для сценариев, где ADR-001 требует исправления whole-segment атрибуции без кастомной пересборки `segment.speaker`, эксплуатация SHALL использовать `WHISPERX_VAD_METHOD=silero` при загрузке ASR. Regression-тесты SHALL включать эталонный фрагмент с текстом «Это с нулями или нау?»: при Silero VAD и штатном `assign_word_speakers` ожидается корректный `segment.speaker` для цельной фразы.

#### Scenario: Эталонный фрагмент на Silero VAD

- **WHEN** ASR загружен с `WHISPERX_VAD_METHOD=silero`, выполнены alignment, диаризация и `assign_word_speakers` без пост-обработки `segment.speaker`
- **THEN** реплика «Это с нулями или нау?» представлена одним блоком с ожидаемым спикером из эталона ADR и без разбиения по последнему слову

### Requirement: Word-level speaker в verbose_json для диагностики

При запросе с диаризацией и alignment система SHALL включать в `verbose_json` поле `words[].speaker` как результат `assign_word_speakers`. Система SHALL NOT использовать `words[].speaker` для нарезки или пересборки `diarized_json` и diarized text. Источник спикера в diarized-форматах — только `segment.speaker`.

#### Scenario: Диагностика расхождения segment и word

- **WHEN** клиент запрашивает `verbose_json` с `diarize=true` и в сегменте `segment.speaker` отличается от части `words[].speaker`
- **THEN** ответ содержит оба уровня меток без изменения структуры Whisper-сегментов
- **THEN** `diarized_json` для того же запроса использует только `segment.speaker` и `segment.text`

#### Scenario: Verbose без word-split по diarized-правилам

- **WHEN** после `assign_word_speakers` в одном Whisper-сегменте несколько `word.speaker`
- **THEN** `verbose_json` сохраняет один Whisper-сегмент с массивом `words[]` и их `speaker`
- **THEN** `diarized_json` не дублирует word-level нарезку этого сегмента