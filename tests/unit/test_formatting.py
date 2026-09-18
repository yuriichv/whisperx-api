from fixtures.transcription import (
    adr_nulyami_segment,
    biane_speaker_change_segment,
    segment,
    word,
)
from whisperx_api.features.transcription.formatting import segment_level_blocks


def test_single_speaker_segment_keeps_whole_text():
    """Один спикер в сегменте — берётся цельный segment.text."""
    seg = segment(
        "Я попробовал это на Биане.",
        0.0,
        4.0,
        words=[word("Я", 0.0, 0.2, "SPEAKER_01")],
        speaker="SPEAKER_01",
    )
    blocks = segment_level_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Я попробовал это на Биане."
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 4.0


def test_mixed_word_speakers_one_block_by_segment_speaker():
    """Разные word.speaker — один блок по segment.speaker (ADR-001)."""
    seg = biane_speaker_change_segment()
    blocks = segment_level_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["text"] == "Я попробовал это на Биане. Ну давай Андрюх"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 4.0


def test_adr_nulyami_regression_single_block():
    """Regression ADR-001: Silero/VAD даёт segment.speaker=SPEAKER_02 — один блок."""
    seg = adr_nulyami_segment()
    blocks = segment_level_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["speaker"] == "SPEAKER_02"
    assert blocks[0]["text"] == "Это с нулями или нау?"


def test_word_without_speaker_still_one_segment_block():
    """Сегмент не режется по словам даже при частичных word.speaker."""
    seg = segment(
        "Это полное предложение полностью",
        0.0,
        4.0,
        words=[
            word("Это", 0.0, 0.2, "SPEAKER_01"),
            word("полное", 0.3, 0.8, "SPEAKER_01"),
            word("предложение", 0.9, 1.5, "SPEAKER_02"),
            word("полностью", 1.6, 2.0),
        ],
        speaker="SPEAKER_01",
    )
    blocks = segment_level_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Это полное предложение полностью"
    assert blocks[0]["speaker"] == "SPEAKER_01"


def test_adjacent_blocks_same_speaker_merged():
    """Соседние Whisper-сегменты одного спикера склеиваются в один."""
    seg1 = segment("Привет", 0.0, 1.0, speaker="SPEAKER_01")
    seg2 = segment("мир", 1.0, 2.0, speaker="SPEAKER_01")
    blocks = segment_level_blocks([seg1, seg2])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Привет мир"
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0


def test_segment_without_speaker_preserved():
    """Сегмент без спикера сохраняется (speaker None), текст не теряется."""
    seg = segment("Нет спикера", 0.0, 1.0)
    blocks = segment_level_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Нет спикера"
    assert blocks[0]["speaker"] is None


def test_empty_segments():
    assert segment_level_blocks([]) == []


def test_empty_text_segment_skipped():
    seg = segment("   ", 0.0, 1.0, speaker="SPEAKER_01")
    assert segment_level_blocks([seg]) == []
