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

### Формат `diarized_json` — segment-level (ADR-001)

После `whisperx.assign_word_speakers` сервис **не** переопределяет `segment.speaker` и **не** режет Whisper-сегмент по `word.speaker`:

- один aligned Whisper-сегмент → один блок `diarized_json` с `segment.text` и `segment.speaker`;
- соседние Whisper-сегменты с одним `segment.speaker` могут склеиваться в один репликовый блок (только представление);
- сегмент без спикера — `speaker` = `UNKNOWN` или `null`.

**BREAKING:** раньше при смене `word.speaker` внутри сегмента выдавалось несколько блоков; теперь — один блок по upstream `segment.speaker`.

`verbose_json` при диаризации и alignment по-прежнему содержит `words[].speaker` **для диагностики** (расхождение с `segment.speaker`), но эти метки не участвуют в сборке `diarized_json`.

Рекомендуемый деплой для качества атрибуции (VAD-first, см. [docs/ADR-001-speaker-segments.md](docs/ADR-001-speaker-segments.md)): **`WHISPERX_VAD_METHOD=silero`**. Manual regression на эталонной фразе «Это с нулями или нау?» — с `diarized_json`, один блок, ожидаемый спикер реплики из эталона записи.

Fallback ADR (вариант C, override `segment.speaker` по словам) **не реализуется**.

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

**`WHISPERX_VAD_METHOD`** (recommended `silero` for diarization per ADR-001): метод VAD для `whisperx.load_model` при старте сервиса (например `silero` или `pyannote`). Если не задан, используется поведение WhisperX по умолчанию.

**Notes**: 
- `model` does not affect behavior and is kept for OpenAI client compatibility: there is only one actual model, configured at application startup (admin-controlled).

- The current version is optimized for **a single GPU device**.
