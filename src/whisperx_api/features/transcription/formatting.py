from .whisperx_types import Segment, TranscriptBlock

# Segment-level форматтер diarized_json после whisperx.assign_word_speakers.
# Один Whisper-сегмент → один блок (segment.text + segment.speaker).
# Соседние блоки одного спикера склеиваются только для представления.


def _segment_to_block(seg: Segment) -> TranscriptBlock | None:
    text = (seg.get("text") or "").strip()
    if not text:
        return None
    return {
        "start": seg.get("start"),
        "end": seg.get("end"),
        "text": text,
        "speaker": seg.get("speaker"),
    }


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


def segment_level_blocks(segments: list[Segment]) -> list[TranscriptBlock]:
    """Построить репликовые блоки {start, end, text, speaker} из Whisper segments."""
    blocks: list[TranscriptBlock] = []
    for seg in segments:
        block = _segment_to_block(seg)
        if block is not None:
            blocks.append(block)
    return _merge_adjacent(blocks)
