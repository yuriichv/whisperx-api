## Context

Текущий пайплайн (после `improve-diarization-oracle`):

```
ASR → Align (auto) → Diarize → assign_word_speakers(fill_nearest)
                                        │
                                        ▼
                          formatting.hybrid_word_blocks()
                          (split по word.speaker при >1 спикере)
                                        │
                                        ▼
                          build_diarized_text / build_diarized_json
```

`whisperx.assign_word_speakers` назначает `segment.speaker` (majority overlap) и `word.speaker` (per-word max-overlap) независимо. На speaker boundary последнее слово aligned segment может получить `word.speaker` следующего turn, если его длинный interval перекрывает начало следующего diarization segment.

Мотивация и trade-off — [ADR-001](../../docs/ADR-001-segment-level%20speaker%20assignment.md). Ранее segment-level уже применялся (`restore-segment-level-diarization`), но был откачен из-за потери реплик при word-level пересборке текста. ADR-001 предлагает segment-level с другим обоснованием: устойчивость к boundary noise важнее точной word-level гранулярности.

## Goals / Non-Goals

**Goals:**

- Формировать `diarized_json` только по `segment.speaker` + `segment.text` + `segment.start`/`segment.end`
- Устранить ложные смены спикера на границах из-за одиночного `word.speaker`
- Минимальный diff: убрать word-level форматтер, не трогать пайплайн диаризации
- Сохранить `words[].speaker` в `verbose_json`

**Non-Goals:**

- Stabilized word-level diarization (вариант 1 из ADR-001) — отложено до появления измеримых примеров
- Изменения diarization-модели, `fill_nearest`, `num_speakers`
- Word-level split в `verbose_json` — уже есть через `words[]`

## Decisions

### 1. Segment-level форматирование, не гибрид

**Решение:** удалить `hybrid_word_blocks` и связанные функции; `diarized_response()` итерирует `result["segments"]` напрямую, текст из `segment.text`.

**Альтернатива (отклонена):** stabilized word-level — эвристики, пороги, сложнее тестировать; нет подтверждения частоты реальных intra-segment speaker turns (ADR-001).

**Альтернатива (отклонена):** оставить гибридный форматтер — сохраняет boundary noise problem.

### 2. Не повторять регрессию потери текста

**Решение:** не пересобирать текст из `words[]`; всегда брать `segment.text`. Старый `_split_segments_by_word_speaker` отбрасывал слова без `word.speaker` — эта ошибка не воспроизводится при segment-level.

```text
ASR → Align → Diarize → assign_word_speakers
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     result.segments                   verbose_json
     (segment.speaker,                  (words[].speaker
      segment.text)                     для диагностики)
              │
              ▼
     diarized_json (segment-level)
```

### 3. Склейка соседних сегментов одного спикера

**Решение:** после `segment_blocks()` вызывать `merge_adjacent_same_speaker()` — склеивает подряд идущие блоки с одним `speaker` в один репликовый блок (`start` первого, `end` последнего, текст через пробел). Применяется и к `diarized_json.segments`, и к `diarized_json.text` (через общий список blocks).

**Не склеивается:** блоки одного спикера, разделённые репликой другого (A → B → A остаётся тремя блоками, первые два A сливаются только если подряд).

### 4. Удалить formatting.py или упростить

**Решение:** удалить модуль `formatting.py` целиком, если после отката в нём не остаётся логики. Построение блоков — inline в `responses.py` (маппинг segment → TranscriptBlock).

## Risks / Trade-offs

| Риск | Митигация |
|------|-----------|
| Реальная смена спикера внутри одного Whisper-сегмента не разделится | Осознанный trade-off ADR-001; `verbose_json.words[].speaker` доступен для post-processing; условие пересмотра — корпус с измеримыми примерами |
| Клиенты, ожидавшие word-level split в `diarized_json` | Документировать в README; структура возвращается к Whisper-гранулярности |
| E2E тест `test_replicas_in_segment_split_no_text_loss` перестанет проходить | Обновить ожидания: один сегмент → один блок с `segment.speaker` |

## Migration Plan

1. Деплой с segment-level форматтером
2. Клиенты получают более устойчивый diarized output, меньше ложных смен спикера
3. Откат: revert коммита change

## Open Questions

- Нужен ли follow-up spike на stabilized word-level? → Вне скоупа; триггер — условие пересмотра из ADR-001
