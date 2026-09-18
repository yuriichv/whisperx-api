# Transcription

Deploy-time configuration and ASR pipeline loading for the transcription service.

## Purpose

Конфигурация загрузки ASR pipeline (в том числе VAD) при старте сервиса.

## Requirements

### Requirement: VAD method при загрузке ASR

Система SHALL при старте приложения вызывать `whisperx.load_model` с аргументом `vad_method`, равным значению конфигурации сервера (`WHISPERX_VAD_METHOD`), только если эта переменная задана непустой строкой после нормализации пробелов. Если `WHISPERX_VAD_METHOD` не задан или пуст, аргумент `vad_method` в `load_model` не передаётся, и WhisperX использует поведение по умолчанию библиотеки.

#### Scenario: vad_method задан через env

- **WHEN** сервис запущен с `WHISPERX_VAD_METHOD=silero`
- **THEN** `whisperx.load_model` вызывается с `vad_method="silero"`

#### Scenario: vad_method не задан

- **WHEN** `WHISPERX_VAD_METHOD` отсутствует или пуст
- **THEN** `whisperx.load_model` вызывается без именованного аргумента `vad_method`

#### Scenario: альтернативный штатный метод WhisperX

- **WHEN** сервис запущен с `WHISPERX_VAD_METHOD=pyannote`
- **THEN** `whisperx.load_model` вызывается с `vad_method="pyannote"`
