import re

from .whisperx_types import Segment, TranscriptBlock, Word

# Гибридный word-level форматтер diarized_json.
# Работает с результатом whisperx.assign_word_speakers: разбивает Whisper-сегмент
# на блоки по смене word.speaker ТОЛЬКО при реальной смене спикера, иначе
# сохраняет цельный segment.text. Соседние блоки одного спикера склеиваются в
# один репликовый блок. Цель — не потерять ни одну реплику при пересборке текста.


def _words_text(words: list[Word]) -> str:
    """Собрать текст блока из слов, аккуратно сжимая пробелы."""
    joined = " ".join(str(w.get("word", "")).strip() for w in words)
    return re.sub(r"\s+", " ", joined).strip()


def _segment_blocks(seg: Segment) -> list[TranscriptBlock]:
    """Разбить один Whisper-сегмент на репликовые блоки."""
    words = seg.get("words") or []
    has_any_speaker = any(w.get("speaker") for w in words)
    has_all_speaker = all(bool(w.get("speaker")) for w in words)
    distinct_speakers = {w.get("speaker") for w in words if w.get("speaker")}

    should_split = bool(
        has_any_speaker and has_all_speaker and len(distinct_speakers) > 1
    )

    if not should_split:
        text = (seg.get("text") or "").strip()
        if not text:
            return []
        return [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
                "speaker": seg.get("speaker"),
            }
        ]

    blocks: list[TranscriptBlock] = []
    current_speaker: str | None = None
    current_words: list[Word] = []

    def flush() -> None:
        nonlocal current_speaker, current_words
        if not current_words:
            return
        text = _words_text(current_words)
        if text:
            blocks.append(
                {
                    "start": current_words[0].get("start"),
                    "end": current_words[-1].get(
                        "end", current_words[-1].get("start")
                    ),
                    "text": text,
                    "speaker": current_speaker,
                }
            )
        current_speaker = None
        current_words = []

    for w in words:
        spk = w.get("speaker")
        if current_speaker is None:
            current_speaker = spk
        if spk != current_speaker:
            flush()
            current_speaker = spk
        current_words.append(w)
    flush()

    return blocks


def _merge_adjacent(blocks: list[TranscriptBlock]) -> list[TranscriptBlock]:
    """Склеить соседние блоки одного спикера в один репликовый блок."""
    merged: list[TranscriptBlock] = []
    for block in blocks:
        if merged and merged[-1].get("speaker") == block.get("speaker"):
            last = merged[-1]
            last["end"] = block.get("end", last.get("end"))
            last["text"] = " ".join(
                [last.get("text", ""), block.get("text", "")]
            ).strip()
            continue
        merged.append(dict(block))
    return merged


def hybrid_word_blocks(segments: list[Segment]) -> list[TranscriptBlock]:
    """Построить репликовые блоки {start, end, text, speaker} из segments."""
    blocks: list[TranscriptBlock] = []
    for seg in segments:
        blocks.extend(_segment_blocks(seg))
    return _merge_adjacent(blocks)
