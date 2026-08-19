from whisperx_api.formatting import hybrid_word_blocks


def _word(text, start, end, speaker=None):
    w = {"word": text, "start": start, "end": end}
    if speaker is not None:
        w["speaker"] = speaker
    return w


def _seg(text, start, end, words=None, speaker=None):
    s = {"text": text, "start": start, "end": end}
    if words is not None:
        s["words"] = words
    if speaker is not None:
        s["speaker"] = speaker
    return s


def test_single_speaker_segment_keeps_whole_text():
    """Один спикер в сегменте — берётся цельный segment.text."""
    seg = _seg(
        "Я попробовал это на Биане.",
        0.0,
        4.0,
        words=[_word("Я", 0.0, 0.2, "SPEAKER_01")],
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
    seg = _seg(
        "Я попробовал это на Биане. Ну давай Андрюх",
        0.0,
        4.0,
        words=[
            _word("Я", 0.0, 0.2, "SPEAKER_01"),
            _word("попробовал", 0.3, 0.8, "SPEAKER_01"),
            _word("это", 0.9, 1.1, "SPEAKER_01"),
            _word("на", 1.2, 1.4, "SPEAKER_01"),
            _word("Биане.", 1.5, 2.0, "SPEAKER_01"),
            _word("Ну", 2.1, 2.3, "SPEAKER_02"),
            _word("давай", 2.4, 2.7, "SPEAKER_02"),
            _word("Андрюх", 2.8, 3.0, "SPEAKER_02"),
        ],
        speaker="SPEAKER_01",
    )
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 2
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["text"] == "Я попробовал это на Биане."
    assert blocks[1]["speaker"] == "SPEAKER_02"
    assert blocks[1]["text"] == "Ну давай Андрюх"
    # Границы блоков — по первому и последнему слову
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0
    assert blocks[1]["start"] == 2.1
    assert blocks[1]["end"] == 3.0


def test_no_text_loss_when_word_without_speaker():
    """Слово без спикера — fallback на цельный segment.text, без потери."""
    seg = _seg(
        "Это полное предложение полностью",
        0.0,
        4.0,
        words=[
            _word("Это", 0.0, 0.2, "SPEAKER_01"),
            _word("полное", 0.3, 0.8, "SPEAKER_01"),
            _word("предложение", 0.9, 1.5, "SPEAKER_02"),
            _word("полностью", 1.6, 2.0),  # нет speaker
        ],
        speaker="SPEAKER_01",
    )
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Это полное предложение полностью"
    # Текст сегмента сохранён целиком


def test_adjacent_blocks_same_speaker_merged():
    """Соседние блоки одного спикера склеиваются в один."""
    seg1 = _seg("Привет", 0.0, 1.0, speaker="SPEAKER_01")
    seg2 = _seg("мир", 1.0, 2.0, speaker="SPEAKER_01")
    blocks = hybrid_word_blocks([seg1, seg2])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Привет мир"
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0


def test_segment_without_speaker_preserved():
    """Сегмент без спикера сохраняется (speaker None), текст не теряется."""
    seg = _seg("Нет спикера", 0.0, 1.0)
    blocks = hybrid_word_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Нет спикера"
    assert blocks[0]["speaker"] is None


def test_empty_segments():
    """Пустой список сегментов — пустой список блоков."""
    assert hybrid_word_blocks([]) == []


def test_empty_text_segment_skipped():
    """Сегмент с пустым текстом пропускается."""
    seg = _seg("   ", 0.0, 1.0, speaker="SPEAKER_01")
    assert hybrid_word_blocks([seg]) == []
