# WhisperX OpenAI-compatible Transcriptions API

API server for **[WhisperX](https://github.com/m-bain/whisperX)** exposing an OpenAI-compatible **[`/v1/audio/transcriptions`](https://platform.openai.com/docs/api-reference/audio/createTranscription)** endpoint.

**Supported parameters**: `language`, `response_format` = `json` | `text` | `verbose_json` | `diarized_json`.

**Auth bearer token** support: env | process lifetime generation | disabled.

**Configuration**: see the **`docker-compose.yaml`** example (env + startup parameters). It also includes a Docker run example.

Example request:
```bash
curl -v http://server/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "response_format=diarized_json" \
  -F "align=false" \
  -F "file=@target_audio_file.m4a" \
  -o whisper.out.json
```

**Notes**: 
- `model` does not affect behavior and is kept for OpenAI client compatibility: there is only one actual model, configured at application startup (admin-controlled).

- The current version is optimized for **a single GPU device**.
