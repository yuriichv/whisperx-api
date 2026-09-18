## MODIFIED Requirements

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

### Requirement: Word-level атрибуция текста с корректными границами

Система SHALL формировать `diarized_json` с разбивкой речи по смене спикера на уровне слов (`word.speaker` из `whisperx.assign_word_speakers`), когда внутри одного Whisper-сегмента присутствуют слова разных спикеров.

#### Scenario: Смена спикера внутри сегмента

- **WHEN** внутри одного Whisper-сегмента реплика «Я попробовал это на Биане.» принадлежит SPEAKER_01, а следующая «Ну, давай, Андрюх,» — SPEAKER_02
- **THEN** `diarized_json` содержит отдельные блоки для SPEAKER_01 и SPEAKER_02 с корректными `start`, `end`, `text`, `speaker`

#### Scenario: Распознанная реплика не теряется

- **WHEN** внутри сегмента слова имеют разных спикеров, включая слова на границах без прямого overlap
- **THEN** ни одна реплика не пропадает из `diarized_json` (fallback на цельный `segment.text` при невозможности разбивки)

#### Scenario: Один спикер в сегменте

- **WHEN** все слова Whisper-сегмента принадлежат одному спикеру
- **THEN** текст блока берётся из `segment.text` целиком, без пересборки из `words[]`

#### Scenario: Склейка соседних блоков одного спикера

- **WHEN** несколько подряд идущих блоков имеют одинакового спикера
- **THEN** `segments` объединяет их в один репликовый блок с `start` первого и `end` последнего

#### Scenario: Сегменты без speaker

- **WHEN** Whisper-сегмент не получил спикера после диаризации
- **THEN** сегмент отображается с `speaker` = `UNKNOWN` или `null`, текст без потери

## REMOVED Requirements

### Requirement: Segment-level форматирование diarized_json

**Reason**: Форматирование строго по Whisper-сегментам (`segment.speaker` majority vote) сливает реплики разных людей внутри одного сегмента и не даёт точных границ смены спикера. Заменяется гибридным word-level форматом с сохранением целостности текста.

**Migration**: `diarized_json` строится через гибридный форматтер: разбивка по `word.speaker` при смене спикера, иначе цельный `segment.text`, склейка соседних блоков одного спикера. Клиенты, которым нужна пер-сегментная гранулярность Whisper без диаризации, используют `verbose_json`.