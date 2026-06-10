## 1. Конфигурация

- [x] 1.1 Добавить `fill_nearest: bool = True` в `src/whisperx_api/config.py` (env: `WHISPERX_FILL_NEAREST`)

## 2. Пайплайн

- [x] 2.1 В `_run_pipeline_sync` заменить `fill_nearest=True` на `fill_nearest=config.fill_nearest`

## 3. Документация

- [x] 3.1 Добавить `WHISPERX_FILL_NEAREST` в `docker-compose.yml` с комментарием/default `"true"`
- [x] 3.2 Описать параметр в README (назначение, default, когда отключать)

## 4. Верификация

- [x] 4.1 Без env: диаризация работает как раньше (`fill_nearest=true`)
- [x] 4.2 С `WHISPERX_FILL_NEAREST=false`: в `verbose_json` слова на границах без overlap не получают `speaker` через nearest
