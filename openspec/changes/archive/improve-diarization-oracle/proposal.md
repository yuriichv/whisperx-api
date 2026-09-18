## Why

Качество диаризации на разговорном аудио низкое: (1) pyannote `community-1` без ограничений выделяет лишнего участника (4 вместо 3); (2) текст внутри Whisper-сегмента приписывается одному спикеру по majority vote, из-за чего реплики разных людей сливаются («Я попробовал это на Биане.» ошибочно отнесена к SPEAKER_02); (3) вывод разбит на множество подряд идущих фрагментов одного SPEAKER вместо цельных репликовых блоков. Требование: максимально переиспользовать функционал фреймворков (whisperx + pyannote), минимум кастомных решений.

## What Changes

- **Число участников задаёт клиент** через параметры `num_speakers`/`min_speakers`/`max_speakers`; бэкенд **не задаёт значений по умолчанию** — при их отсутствии передаёт `None` в `whisperx.DiarizationPipeline`, и pyannote сам определяет число спикеров (автоопределение)
- Бэкенд **только пробрасывает** `num/min/max` в `whisperx.DiarizationPipeline` (framework-native), без кастомных модулей подсчёта/слияния кластеров; значения параметров — решение клиента
- Вернуть **гибридное word-level** форматирование `diarized_json` по `word.speaker` (через `whisperx.assign_word_speakers`): резать сегмент только при реальной смене спикера, fallback на цельный `segment.text` — без потери реплик (устранена регрессия `restore-segment-level-diarization`)
- Склеивать соседние блоки одного спикера в один репликовый блок (закрывает фрагментацию `segments`)
- Вынести имя diarization-модели в конфигурацию (env `WHISPERX_DIARIZE_MODEL`), default — текущая

## Capabilities

### New Capabilities

_(нет)_

### Modified Capabilities

- `diarization`: изменить требования — число спикеров передаётся с клиента через `num/min/max` и пробрасывается в `whisperx.DiarizationPipeline` без значений по умолчанию на бэкенде; заменить segment-level форматирование `diarized_json` на гибридное word-level; разрешить конфигурируемую модель диаризации

## Impact

- `src/whisperx_api/transcribe_router.py` — `_run_pipeline_sync`: проброс `num/min/max` в `whisperx.DiarizationPipeline` (без дефолтов на бэкенде); вызов робастного форматтера
- Новый модуль `src/whisperx_api/formatting.py` — гибридный word-level форматтер + склейка блоков
- `src/whisperx_api/config.py` — поле `diarize_model`
- `openspec/specs/diarization/spec.md` — обновление требований числа спикеров и форматирования
- `README.md` — документация контракта `num/min/max`, гибридный word-level формат
- Тесты: unit (форматтер) и e2e spike на проблемном аудио (число участников по `num/min/max`, корректные границы)
