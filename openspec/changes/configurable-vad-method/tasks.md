## 1. Configuration

- [x] 1.1 Добавить `vad_method: str = ""` в `src/whisperx_api/config.py` (env: `WHISPERX_VAD_METHOD`)

## 2. Wiring

- [x] 2.1 В `startup_load` передавать `vad_method` в `load_model` только при непустом `config.vad_method`

## 3. Tests

- [x] 3.1 Unit-тест: при заданном `vad_method` аргумент попадает в `load_model`
- [x] 3.2 Unit-тест: при пустом `vad_method` ключ не передаётся

## 4. Docs

- [x] 4.1 README и комментарий в `docker-compose.yml`
