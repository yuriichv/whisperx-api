# WhisperX OpenAI-compatible Transcriptions API

API server for **[WhisperX](https://github.com/m-bain/whisperX)** exposing an OpenAI-compatible **[`/v1/audio/transcriptions`](https://platform.openai.com/docs/api-reference/audio/createTranscription)** endpoint.

**Supported parameters**: `language`, `prompt`, `response_format` = `json` | `text` | `verbose_json` | `diarized_json`.

**WhisperX extensions**: `hotwords`, `align`, `diarize`, `num_speakers`, `min_speakers`, `max_speakers`.

### `prompt` и `hotwords` (conditioning ASR)

- **`prompt`** (OpenAI-compatible) → `initial_prompt` в faster-whisper: краткая подсказка для первого чанка (НЕ system prompt!).
- **`hotwords`** (WhisperX extension, одна строка) → `hotwords` в faster-whisper: термины и имена as-is после `trim`.
- Нормализация backend: `strip()` всей строки; пустая после trim → не применяется; внутренние пробелы и запятые **не изменяются**.
- Env-defaults для prompt/hotwords **нет** — только параметры запроса.

**Лимиты (ADR):**

| Поле | max chars | max Whisper tokens |
|------|-----------|-------------------|
| `prompt` | 500 | 100 |
| `hotwords` | 1000 | 150 |
| совместно | рекомендация ≤ 1500 chars | ≤ 200 |

Превышение hard-лимитов → HTTP 422. Основное ограничение — decoder budget faster-whisper (448 tokens), из которого conditioning не должен занимать слишком много места под транскрипцию.

**`prompt` + `diarize=true` разрешены** — WhisperX extension. OpenAI запрещает `prompt` для `gpt-4o-transcribe-diarize`; здесь diarization — отдельный этап pipeline после ASR, конфликта нет.

Example with prompt and hotwords:

```bash
curl -v http://server/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPERX_API_TOKEN" \
  -F "language=ru" \
  -F "prompt=Совещание команды разработки" \
  -F "hotwords=WhisperX, pyannote, OpenAI" \
  -F "file=@audio.wav"
```

When `diarize=true` (or `response_format=diarized_json`), alignment is enabled automatically unless `align=false` is passed explicitly.

### Контракт числа спикеров (`num_speakers` / `min_speakers` / `max_speakers`)

Число участников разговора **задаёт клиент**. Значения пробрасываются в `whisperx.DiarizationPipeline` без значений по умолчанию на бэкенде:

- `num_speakers` — точное число спикеров; имеет приоритет над `min_speakers`/`max_speakers` (если задан, `min`/`max` игнорируются pyannote).
- `min_speakers` / `max_speakers` — диапазон допустимого числа спикеров.
- Если ни один параметр не передан, пайплайн вызывается с `None`, и **pyannote сам определяет** число спикеров (автоопределение).

Валидация (HTTP 400 при нарушении) применяется **только при запрошенной диаризации**: `num_speakers >= 1`, `min_speakers >= 1`, `max_speakers >= 1`, `min_speakers <= max_speakers`. Без диаризации параметры числа спикеров игнорируются.

### Формат `diarized_json` — гибридный word-level

`diarized_json` строится по результату `whisperx.assign_word_speakers`:

- если внутри одного Whisper-сегмента есть **реальная смена спикера** на уровне слов (`word.speaker`), сегмент разбивается на блоки по смене спикера с точными границами `start`/`end`;
- иначе текст блока берётся целиком из `segment.text` (без потери реплик и без пересборки из токенов);
- соседние блоки одного спикера склеиваются в один репликовый блок;
- сегмент без спикера отображается с `speaker` = `UNKNOWN` или `null`.

Цена компромисса «без потери текста»: если внутри сегмента реально несколько спикеров, но разбивка по `word.speaker` невозможна (например, у части слов нет спикера), фрагмент атрибутируется доминантному спикеру сегмента (`segment.speaker`).

`verbose_json` по-прежнему отдаёт `words[].speaker` для word-level детализации.

**Auth bearer token** support: env | process lifetime generation | disabled.

**Configuration**: see the **`docker-compose.yaml`** example (env + startup parameters). It also includes a Docker run example.

Example request (diarized transcription with 2 speakers):
```bash
curl -v http://server/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPERX_API_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "response_format=diarized_json" \
  -F "diarize=true" \
  -F "num_speakers=2" \
  -F "file=@target_audio_file.m4a" \
  -o whisper.out.json
```

`align` is enabled automatically for diarization. Pass `align=false` only if you need faster processing and accept lower speaker accuracy.

**`WHISPERX_FILL_NEAREST`** (default `true`): when enabled, words and segments without direct time overlap with a diarization interval get the nearest speaker. Disable (`false`) if boundary words are assigned to the wrong speaker on noisy audio.

**`WHISPERX_DIARIZE_MODEL`** (default `pyannote/speaker-diarization-community-1`): имя diarization-модели pyannote, используемой для `whisperx.DiarizationPipeline`. Число участников зависит от модели; при проблемах с разделением спикеров можно указать другую модель.

**`WHISPERX_VAD_METHOD`** (optional): метод VAD для `whisperx.load_model` при старте сервиса (например `silero` или `pyannote`). Если не задан, используется поведение WhisperX по умолчанию.

**Notes**: 
- `model` does not affect behavior and is kept for OpenAI client compatibility: there is only one actual model, configured at application startup (admin-controlled).

- The current version is optimized for **a single GPU device**.
