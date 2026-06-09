## 1. Пайплайн: align и diarization

- [x] 1.1 В `transcriptions()` вычислять `do_align=true` автоматически при `do_diarize=true`, если параметр `align` не передан явно
- [x] 1.2 Добавить form-параметр `num_speakers: Optional[int]` в endpoint `/v1/audio/transcriptions`
- [x] 1.3 Добавить валидацию `num_speakers >= 1` с HTTP 400 при невалидном значении
- [x] 1.4 Пробросить `num_speakers` в `_run_pipeline_sync` и вызывать `DiarizationPipeline` с `num_speakers` (приоритет над min/max)
- [x] 1.5 Передавать `fill_nearest=True` в `whisperx.assign_word_speakers`

## 2. Word-level форматирование вывода

- [x] 2.1 Реализовать `_split_segments_by_word_speaker(result) -> List[dict]` с группировкой подряд идущих слов одного спикера
- [x] 2.2 Добавить fallback на segment-level при отсутствии `words[]`
- [x] 2.3 Обновить `_build_diarized_text` для работы с word-split блоками
- [x] 2.4 Обновить `_build_diarized_json` для работы с word-split блоками

## 3. Документация

- [x] 3.1 Обновить README: `align=true` при диаризации, описать `num_speakers`, пример запроса

## 4. Проверка

- [x] 4.1 Ручная проверка: запрос с `diarize=true` без `align` — в ответе есть `words[].speaker` в verbose_json
- [x] 4.2 Ручная проверка: diarized_json с двумя спикерами в одном Whisper-сегменте — разбивка на отдельные строки/сегменты
