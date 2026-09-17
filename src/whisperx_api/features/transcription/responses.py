"""Построители ответов: whisperx_types → Pydantic API models."""

from pydantic import TypeAdapter

from .schemas import (
    DiarizedJsonResponse,
    DiarizedSegmentOut,
    TranscriptionJsonResponse,
    VerboseJsonResponse,
    VerboseJsonSegment,
    VerboseJsonWord,
)
from .whisperx_types import Segment, TranscriptBlock, TranscriptionResult, Word

UNKNOWN_SPEAKER = "UNKNOWN"
_word_adapter = TypeAdapter(VerboseJsonWord)
_segment_adapter = TypeAdapter(DiarizedSegmentOut)


def plain_text(result: TranscriptionResult) -> str:
    return (
        result.get("text")
        or "".join(seg.get("text", "") for seg in result.get("segments", []))
    ).strip()


def _parse_words(words: list[Word]) -> list[VerboseJsonWord]:
    return [_word_adapter.validate_python(word) for word in words]


def _block_to_segment(block: TranscriptBlock) -> DiarizedSegmentOut:
    return _segment_adapter.validate_python(
        {
            **block,
            "text": (block.get("text") or "").strip(),
        }
    )


def build_verbose_json(
    result: TranscriptionResult, text: str, language: str | None
) -> VerboseJsonResponse:
    segments: list[VerboseJsonSegment] = []
    for i, seg in enumerate(result.get("segments", [])):
        words = seg.get("words")
        segments.append(
            VerboseJsonSegment(
                id=i,
                start=seg.get("start"),
                end=seg.get("end"),
                text=(seg.get("text") or "").strip(),
                words=_parse_words(words) if words else None,
                speaker=seg.get("speaker"),
            )
        )
    return VerboseJsonResponse(
        language=result.get("language") or language,
        text=text,
        segments=segments,
    )


def build_json_response(result: TranscriptionResult) -> TranscriptionJsonResponse:
    return TranscriptionJsonResponse(text=plain_text(result))


def build_diarized_json(
    result: TranscriptionResult,
    blocks: list[TranscriptBlock],
    speaker_text: str,
    language: str | None,
) -> DiarizedJsonResponse:
    speakers = {b.get("speaker") for b in blocks if b.get("speaker")}
    segments_out = [_block_to_segment(b) for b in blocks]
    return DiarizedJsonResponse(
        language=result.get("language") or language,
        text=speaker_text,
        speakers=sorted(speakers),
        segments=segments_out,
    )


def build_diarized_text(blocks: list[TranscriptBlock]) -> str:
    lines: list[str] = []
    current_speaker: str | None = None
    current_chunks: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_chunks
        if current_speaker is None:
            return
        text = " ".join(current_chunks).strip()
        if text:
            label = current_speaker or UNKNOWN_SPEAKER
            lines.append(f"{label}: {text}")
        current_speaker = None
        current_chunks = []

    for block in blocks:
        speaker = block.get("speaker")
        chunk = block.get("text") or ""
        if not chunk:
            continue

        if current_speaker is None:
            current_speaker = speaker

        if speaker != current_speaker:
            flush()
            current_speaker = speaker

        current_chunks.append(chunk)

    flush()
    return "\n".join(lines)


def segment_blocks(segments: list[Segment]) -> list[TranscriptBlock]:
    """Построить репликовые блоки из Whisper-сегментов (segment-level speaker)."""
    blocks: list[TranscriptBlock] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
                "speaker": seg.get("speaker"),
            }
        )
    return blocks


def merge_adjacent_same_speaker(blocks: list[TranscriptBlock]) -> list[TranscriptBlock]:
    """Склеить подряд идущие блоки одного спикера в один репликовый блок."""
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


def diarized_response(
    result: TranscriptionResult, language: str | None
) -> DiarizedJsonResponse:
    blocks = merge_adjacent_same_speaker(
        segment_blocks(result.get("segments") or [])
    )
    speaker_text = build_diarized_text(blocks)
    return build_diarized_json(result, blocks, speaker_text, language)
