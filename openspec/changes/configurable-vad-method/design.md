## Context

ASR pipeline создаётся один раз в `AppState.startup_load` через `whisperx.load_model`. WhisperX принимает `vad_method` (например `pyannote`, `silero`); если параметр не передан, библиотека использует свой default.

Прецедент: `WHISPERX_DIARIZE_MODEL` — deploy-time строка в `Config` с префиксом `WHISPERX_`.

## Goals / Non-Goals

**Goals:**

- Deploy-time выбор `vad_method` через env
- Обратная совместимость: без env поведение идентично текущему (не передаём `vad_method`)

**Non-Goals:**

- Per-request override в API
- `vad_options`, `vad_model`, смена default WhisperX в коде
- Валидация списка допустимых методов (WhisperX может добавить новые)

## Decisions

### 1. Имя env: WHISPERX_VAD_METHOD

**Решение:** `vad_method: str = ""` в `Config` → env `WHISPERX_VAD_METHOD`.

Пустая строка после trim → ключ `vad_method` не попадает в вызов `load_model`.

**Альтернатива:** enum только `silero` | `pyannote` — отклонено: пользователь явно просил поддержку «другого» метода, поддерживаемого WhisperX.

### 2. Проброс только при startup

**Решение:** изменение только в `startup_load`, без API form-полей.

## Risks / Trade-offs

- Неверное значение env → ошибка при старте или внутри WhisperX (ожидаемо для misconfiguration)
- Смена VAD влияет на границы сегментов ASR; оператор меняет env осознанно
