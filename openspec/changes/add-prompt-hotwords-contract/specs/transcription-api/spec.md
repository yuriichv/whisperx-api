## Purpose

Формальный контракт HTTP API транскрипции `POST /v1/audio/transcriptions`: typed-модели запроса и ответа, проброс prompt/hotwords в ASR whisperx, совместимость с OpenAI multipart и WhisperX extensions.

## ADDED Requirements

### Requirement: Pydantic-модель запроса транскрипции

Система SHALL описывать все form-параметры endpoint `POST /v1/audio/transcriptions` единой Pydantic-моделью `TranscriptionRequest`, включая как минимум: `model`, `language`, `prompt`, `hotwords`, `response_format`, `temperature`, `timestamp_granularities`, `align`, `diarize`, `num_speakers`, `min_speakers`, `max_speakers`. Файл `file` передаётся отдельно (multipart), но валидируется совместно с моделью в dependency роутера.

#### Scenario: Валидный multipart-запрос

- **WHEN** клиент отправляет корректные form-поля и аудиофайл
- **THEN** система строит `TranscriptionRequest` без ручного разбора полей в роутере

#### Scenario: Невалидные form-поля

- **WHEN** клиент передаёт некорректные типы или значения (например, `num_speakers=0` при `diarize=true`)
- **THEN** система возвращает HTTP 422 или HTTP 400 с описанием ошибки до запуска пайплайна

### Requirement: Pydantic-модели ответов по response_format

Система SHALL формировать ответы через Pydantic-модели, соответствующие `response_format`:

- `text` → plain text body
- `json` → `TranscriptionJsonResponse` (`text`)
- `verbose_json` → `TranscriptionVerboseJsonResponse`
- `diarized_json` → `TranscriptionDiarizedJsonResponse`

Сериализация SHALL сохранять текущую JSON-структуру полей, уже используемую клиентами (без breaking rename полей).

#### Scenario: response_format=json

- **WHEN** клиент запрашивает `response_format=json`
- **THEN** ответ валидируется как `TranscriptionJsonResponse` и содержит поле `text`

#### Scenario: response_format=diarized_json

- **WHEN** клиент запрашивает `response_format=diarized_json` с `diarize=true`
- **THEN** ответ валидируется как `TranscriptionDiarizedJsonResponse` с полями `language`, `text`, `speakers`, `segments`

### Requirement: Проброс prompt в ASR

Система SHALL принимать опциональный form-параметр `prompt` (OpenAI-совместимое имя) и передавать его в ASR как `initial_prompt` в options whisperx/faster-whisper **на время одного запроса**. После завершения transcribe система SHALL восстановить предыдущее значение `initial_prompt`.

Пустая строка или строка из одних пробелов SHALL трактоваться как отсутствие prompt.

#### Scenario: Prompt задан

- **WHEN** клиент передаёт `prompt=Это звонок в поддержку. Имена: Иван, Мария`
- **THEN** перед `transcribe()` ASR options получают `initial_prompt` с этим текстом (после strip)

#### Scenario: Prompt не задан

- **WHEN** клиент не передаёт `prompt`
- **THEN** для данного запроса `initial_prompt` устанавливается в `None`, после запроса восстанавливается сохранённое значение

#### Scenario: Prompt с diarize

- **WHEN** клиент передаёт `prompt=...` и `diarize=true`
- **THEN** prompt применяется на этапе ASR, диаризация выполняется после ASR как обычно (WhisperX extension, см. Requirement ниже)

### Requirement: WhisperX extension hotwords

Система SHALL принимать опциональный form-параметр `hotwords` как **одну строку** (WhisperX extension, не OpenAI `keywords`). Значение SHALL передаваться в ASR options как `hotwords` на время одного запроса с последующим restore.

Максимальная длина `hotwords` после нормализации — **1000 символов**; превышение → HTTP 422.

#### Scenario: Hotwords заданы

- **WHEN** клиент передаёт `hotwords=WhisperX, pyannote, OpenAI`
- **THEN** перед `transcribe()` ASR options получают `hotwords` с нормализованной строкой (см. Requirement «Нормализация hotwords»)

#### Scenario: Hotwords не заданы

- **WHEN** клиент не передаёт `hotwords`
- **THEN** для данного запроса `hotwords` устанавливается в `None`, после запроса восстанавливается сохранённое значение

### Requirement: Нормализация hotwords

Система SHALL нормализовать `hotwords` на backend перед валидацией длины и пробросом в faster-whisper:

1. выполнить `strip()` всей строки (удалить только leading/trailing whitespace);
2. если после strip строка пустая — трактовать как `None`;
3. **не изменять** внутренние пробелы;
4. **не парсить** и **не преобразовывать** запятые;
5. передать результат в faster-whisper **как есть** (без дополнительных трансформаций).

#### Scenario: Trim leading/trailing пробелов

- **WHEN** клиент передаёт `hotwords=  WhisperX, pyannote  `
- **THEN** в ASR options попадает строка `WhisperX, pyannote` (без изменения внутреннего пробела после запятой)

#### Scenario: Пустая строка после trim

- **WHEN** клиент передаёт `hotwords=   ` (только пробелы)
- **THEN** `hotwords` трактуется как `None`, ASR options получают `hotwords=None`

#### Scenario: Внутренние пробелы сохраняются

- **WHEN** клиент передаёт `hotwords=foo  bar   baz`
- **THEN** в ASR options попадает строка `foo  bar   baz` без collapse пробелов

#### Scenario: Запятые не парсятся

- **WHEN** клиент передаёт `hotwords=a,b, c,d`
- **THEN** в ASR options попадает строка `a,b, c,d` целиком, без split/join по запятым

#### Scenario: Слишком длинные hotwords

- **WHEN** клиент передаёт `hotwords` длиннее 1000 символов после нормализации
- **THEN** система возвращает HTTP 422 до запуска пайплайна

### Requirement: Нормализация prompt

Система SHALL нормализовать `prompt` перед валидацией и пробросом: `strip()` всей строки; пустая строка после strip → `None`. Внутренние пробелы не изменять.

#### Scenario: Пустой prompt после trim

- **WHEN** клиент передаёт `prompt=   `
- **THEN** `prompt` трактуется как `None`, ASR options получают `initial_prompt=None`

### Requirement: Валидация длины prompt

Система SHALL ограничивать `prompt` длиной **500 символов** после strip. Превышение → HTTP 422.

#### Scenario: Слишком длинный prompt

- **WHEN** клиент передаёт `prompt` длиннее 500 символов
- **THEN** система возвращает HTTP 422 до запуска пайплайна

### Requirement: Валидация Whisper token budget для prompt и hotwords

После нормализации система SHALL проверять conditioning budget через tokenizer ASR-модели (Whisper tokens).

Tokenizer SHALL разрешаться в порядке приоритета:
1. `ASR_PIPELINE.tokenizer` (faster-whisper `Tokenizer` на `FasterWhisperPipeline`);
2. `ASR_PIPELINE.model.hf_tokenizer` (fallback при auto-language до первого `transcribe`).

Если оба недоступны — HTTP 503 до запуска пайплайна.

После нормализации система SHALL проверять conditioning budget (Whisper tokens):

- `prompt` (`initial_prompt`) ≤ **100** tokens;
- `hotwords` ≤ **150** tokens;
- сумма tokens `prompt` + `hotwords` ≤ **200** tokens.

Поля со значением `None` не учитываются в сумме. Превышение любого лимита → HTTP 422.

Документация SHALL рекомендовать клиенту суммарно не более **1500 символов** для `prompt` + `hotwords` (мягкая рекомендация, без блокировки запроса).

#### Scenario: prompt в пределах token limit

- **WHEN** клиент передаёт короткий `prompt` в пределах 100 Whisper tokens
- **THEN** запрос проходит валидацию token budget

#### Scenario: hotwords превышает token limit

- **WHEN** клиент передаёт `hotwords`, которые после tokenize превышают 150 Whisper tokens
- **THEN** система возвращает HTTP 422 до запуска пайплайна

#### Scenario: совместный budget превышен

- **WHEN** клиент передаёт `prompt` и `hotwords`, каждый в отдельности в пределах своего лимита, но сумма tokens > 200
- **THEN** система возвращает HTTP 422 до запуска пайплайна

### Requirement: Отсутствие env-defaults для prompt и hotwords

Система SHALL NOT задавать значения `prompt`/`hotwords` из конфигурации сервера. Единственный источник — параметры запроса клиента.

#### Scenario: Запрос без prompt и hotwords

- **WHEN** клиент не передаёт `prompt` и `hotwords`
- **THEN** ASR options для этих полей устанавливаются в `None` на время запроса (если до запроса не было другого per-request значения)

### Requirement: Документированная несовместимость prompt+diarize с OpenAI

Система SHALL разрешать одновременное использование `prompt` и `diarize=true`. Документация (README и spec) SHALL явно указывать: это **WhisperX extension**; OpenAI запрещает `prompt` для модели `gpt-4o-transcribe-diarize`, но в данном сервисе diarization — отдельный этап pipeline после ASR, технического конфликта нет.

#### Scenario: prompt и diarize вместе

- **WHEN** клиент передаёт `prompt=...`, `diarize=true`, `response_format=diarized_json`
- **THEN** система выполняет ASR с prompt, затем align/diarize; HTTP 200 при успехе

### Requirement: ASR при auto-detect language

Система SHALL вызывать `transcribe()` даже если клиент не передал `language` (auto-detect). Параметр `language` добавляется в kwargs transcribe только при non-empty значении.

#### Scenario: Запрос без language

- **WHEN** клиент отправляет аудио без параметра `language`
- **THEN** ASR transcribe выполняется и возвращает результат с определённым языком

### Requirement: Thread-safe apply/restore options

Изменение `initial_prompt` и `hotwords` в ASR options SHALL выполняться только внутри захвата `GPU_LOCK`, с сохранением и восстановлением предыдущих значений в том же critical section.

#### Scenario: Последовательные запросы с разными prompt

- **WHEN** два запроса с разными `prompt` выполняются последовательно под lock
- **THEN** каждый запрос использует свой prompt, options после каждого запроса восстановлены
