# WhisperX OpenAI-compatible Transcriptions API

API server for **[WhisperX](https://github.com/m-bain/whisperX)** exposing an OpenAI-compatible **[`/v1/audio/transcriptions`](https://platform.openai.com/docs/api-reference/audio/createTranscription)** endpoint.

**Supported parameters**: `language`, `response_format` = `json` | `text` | `verbose_json` | `diarized_json`.

**WhisperX extensions**: `align`, `diarize`, `num_speakers`, `min_speakers`, `max_speakers`.

When `diarize=true` (or `response_format=diarized_json`), alignment is enabled automatically unless `align=false` is passed explicitly. Word-level speaker labels are used to split output when speakers change within a Whisper segment.

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

**Notes**: 
- `model` does not affect behavior and is kept for OpenAI client compatibility: there is only one actual model, configured at application startup (admin-controlled).

- The current version is optimized for **a single GPU device**.
