## REMOVED Requirements

### Requirement: Word-level атрибуция текста с корректными границами

**Reason:** ADR-001: сервис не должен формировать `diarized_json` разрезанием Whisper-сегмента по `word.speaker`. Итоговый спикер и текст реплики берутся из результата `assign_word_speakers` на уровне сегмента Whisper.

**Migration:** Вместо нескольких блоков внутри одного Whisper-сегмента клиент получает один блок с `segment.text` и `segment.speaker`. Word-level метки для диагностики — в `verbose_json` (`words[].speaker`), не в логике `diarized_json`.

## ADDED Requirements

### Requirement: Diarized output по segment.speaker WhisperX

После `whisperx.assign_word_speakers` система SHALL формировать `diarized_json` и текстовый diarized-вывод без переопределения `segment.speaker` и без пересборки `segment.text` из `words[]`. Каждый aligned Whisper-сегмент SHALL давать один репликовый блок с `text` равным `segment.text`, `speaker` равным `segment.speaker`, временными границами сегмента. Система SHALL NOT делить один Whisper-сегмент на несколько блоков только из-за различия `word.speaker` внутри сегмента.

#### Scenario: Разные word.speaker внутри одного Whisper-сегмента

- **WHEN** после `assign_word_speakers` в одном Whisper-сегменте слова имеют разных `word.speaker`, но `segment.speaker` задан WhisperX
- **THEN** `diarized_json` содержит один блок с полным `segment.text` и `segment.speaker`, без word-level нарезки

#### Scenario: Один спикер в сегменте

- **WHEN** все слова сегмента и `segment.speaker` согласованы
- **THEN** блок использует `segment.text` и `segment.speaker` без пересборки из `words[]`

#### Scenario: Склейка соседних Whisper-сегментов

- **WHEN** несколько подряд идущих Whisper-сегментов имеют одинаковый `segment.speaker`
- **THEN** `diarized_json` MAY объединить их в один репликовый блок с `start` первого и `end` последнего сегмента, не изменяя присвоенных WhisperX меток спикера

#### Scenario: Сегмент без speaker

- **WHEN** Whisper-сегмент не получил `segment.speaker` после диаризации
- **THEN** блок отображается с `speaker` равным `UNKNOWN` или `null`, текст `segment.text` сохраняется

### Requirement: Проверка стратегии VAD-first

Для сценариев, где ADR-001 требует исправления whole-segment атрибуции без кастомной пересборки `segment.speaker`, эксплуатация SHALL использовать `WHISPERX_VAD_METHOD=silero` при загрузке ASR. Regression-тесты SHALL включать эталонный фрагмент с текстом «Это с нулями или нау?»: при Silero VAD и штатном `assign_word_speakers` ожидается корректный `segment.speaker` для цельной фразы.

#### Scenario: Эталонный фрагмент на Silero VAD

- **WHEN** ASR загружен с `WHISPERX_VAD_METHOD=silero`, выполнены alignment, диаризация и `assign_word_speakers` без пост-обработки `segment.speaker`
- **THEN** реплика «Это с нулями или нау?» представлена одним блоком с ожидаемым спикером из эталона ADR и без разбиения по последнему слову

### Requirement: Word-level speaker в verbose_json для диагностики

При запросе с диаризацией и alignment система SHALL включать в `verbose_json` поле `words[].speaker` как результат `assign_word_speakers`. Система SHALL NOT использовать `words[].speaker` для нарезки или пересборки `diarized_json` и diarized text. Источник спикера в diarized-форматах — только `segment.speaker`.

#### Scenario: Диагностика расхождения segment и word

- **WHEN** клиент запрашивает `verbose_json` с `diarize=true` и в сегменте `segment.speaker` отличается от части `words[].speaker`
- **THEN** ответ содержит оба уровня меток без изменения структуры Whisper-сегментов
- **THEN** `diarized_json` для того же запроса использует только `segment.speaker` и `segment.text`

#### Scenario: Verbose без word-split по diarized-правилам

- **WHEN** после `assign_word_speakers` в одном Whisper-сегменте несколько `word.speaker`
- **THEN** `verbose_json` сохраняет один Whisper-сегмент с массивом `words[]` и их `speaker`
- **THEN** `diarized_json` не дублирует word-level нарезку этого сегмента
