## Context

Сейчас после `assign_word_speakers` вызывается `diarized_response` → `hybrid_word_blocks`: при нескольких `word.speaker` в одном Whisper-сегменте текст режется и пересобирается из `words[]`. ADR-001 объясняет, почему это хуже, чем ошибка whole-segment `segment.speaker`, и выбирает минимальное отклонение от upstream: **Silero VAD + штатный `assign_word_speakers`**, без кастомной логики над `segment.speaker`.

См. [docs/ADR-001-speaker-segments.md](../../docs/ADR-001-speaker-segments.md).

## Goals / Non-Goals

**Goals:**

- Убрать второй слой атрибуции/структуры поверх WhisperX
- Единый источник истины для реплики в API: `segment.text` + `segment.speaker` после `assign_word_speakers`
- VAD-first через существующий `WHISPERX_VAD_METHOD`
- Regression на эталонном кейсе ADR

**Non-Goals:**

- Fallback ADR (вариант C: override `segment.speaker` по словам) — не входит в scope и отдельный change не планируется
- Пересборка сегментов по `words[].speaker`
- Изменение `assign_word_speakers`, pyannote, fill_nearest
- Word-level нарезка `verbose_json` (структура segments остаётся Whisper)

## Decisions

### 1. Нет post-processing `segment.speaker`

**Решение:** после `whisperx.assign_word_speakers(...)` segments передаются в построитель ответа как есть. Никакого `refine`, majority по словам, longest-run.

**Обоснование (ADR):** исправление — на стадии VAD/границ сегмента; вариант C из ADR не реализуется в рамках этого change.

### 2. Упростить formatting до segment map + merge

**Решение:** заменить `hybrid_word_blocks` на функцию уровня «Whisper segment → block» с опциональным `_merge_adjacent` по `segment.speaker`. Удалить `_segment_blocks`, split по словам, `_words_text` для нарезки.

```text
audio → VAD (silero) → ASR → align → diarize → assign_word_speakers
                                                      ↓
                                            map segments → diarized_json
                                            (без изменения speaker/text)
```

### 3. VAD через env, без нового default в коде

**Решение:** как в archived configurable-vad-method — `WHISPERX_VAD_METHOD=silero` в compose/README для деплоя, где нужен ADR regression.

### 4. `words[].speaker` только в verbose_json

**Решение:** при `diarize=true` и alignment `build_verbose_json` продолжает пробрасывать `words[].speaker` из результата `assign_word_speakers`. Поле служит диагностике (расхождение с `segment.speaker`, разбор перекрывающейся речи) и **не** участвует в построении `diarized_json` / diarized text.

**Обоснование:** ADR разделяет upstream-атрибуцию реплики (`segment.speaker`) и локальные word-level метки; потребителям word-split в diarized — смотреть verbose, не пересобирать вывод сервисом.

## Risks / Trade-offs

- **[Risk] Silero меняет текст/число сегментов** → regression на репрезентативных записях; откат env
- **[Risk] BREAKING diarized_json** → документация; клиенты теряют word-split внутри Whisper-сегмента
- **[Trade-off]** две реальные реплики в одном Whisper-сегменте останутся одним блоком с одним `segment.speaker` — принято ADR в пользу целостности фразы и upstream-модели

## Migration Plan

1. Упростить formatting + обновить тесты
2. Включить Silero на staging, прогнать эталон ADR
3. Rollback: revert код форматтера; VAD — смена env
