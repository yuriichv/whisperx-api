from fixtures.transcription import biane_speaker_change_segment, segment, word
from whisperx_api.features.transcription.formatting import hybrid_word_blocks


def test_single_speaker_segment_keeps_whole_text():
    """Один спикер в сегменте — берётся цельный segment.text."""
    seg = segment(
        "Я попробовал это на Биане.",
        0.0,
        4.0,
        words=[word("Я", 0.0, 0.2, "SPEAKER_01")],
        speaker="SPEAKER_01",
    )
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Я попробовал это на Биане."
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 4.0


def test_speaker_change_inside_segment_splits():
    """Смена спикера внутри сегмента — блоки разделяются корректно."""
    seg = biane_speaker_change_segment()
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 2
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["text"] == "Я попробовал это на Биане."
    assert blocks[1]["speaker"] == "SPEAKER_02"
    assert blocks[1]["text"] == "Ну давай Андрюх"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0
    assert blocks[1]["start"] == 2.1
    assert blocks[1]["end"] == 3.0


def test_no_text_loss_when_word_without_speaker():
    """Слово без спикера — fallback на цельный segment.text, без потери."""
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
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Это полное предложение полностью"


def test_adjacent_blocks_same_speaker_merged():
    """Соседние блоки одного спикера склеиваются в один."""
    seg1 = segment("Привет", 0.0, 1.0, speaker="SPEAKER_01")
    seg2 = segment("мир", 1.0, 2.0, speaker="SPEAKER_01")
    blocks = hybrid_word_blocks([seg1, seg2])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Привет мир"
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0


def test_segment_without_speaker_preserved():
    """Сегмент без спикера сохраняется (speaker None), текст не теряется."""
    seg = segment("Нет спикера", 0.0, 1.0)
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Нет спикера"
    assert blocks[0]["speaker"] is None


def test_empty_segments():
    """Пустой список сегментов — пустой список блоков."""
    assert hybrid_word_blocks([]) == []


def test_empty_text_segment_skipped():
    """Сегмент с пустым текстом пропускается."""
    seg = segment("   ", 0.0, 1.0, speaker="SPEAKER_01")
    assert hybrid_word_blocks([seg]) == []
