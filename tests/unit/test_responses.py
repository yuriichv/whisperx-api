from fixtures.transcription import segment, word
from whisperx_api.features.transcription.responses import (
    build_verbose_json,
    diarized_response,
    merge_adjacent_same_speaker,
    segment_blocks,
)


def test_segment_block_uses_segment_speaker_and_text():
    seg = segment(
        "Я попробовал это на Биане.",
        0.0,
        4.0,
        words=[word("Я", 0.0, 0.2, "SPEAKER_01")],
        speaker="SPEAKER_01",
    )
    blocks = segment_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Я попробовал это на Биане."
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 4.0


def test_boundary_noise_does_not_split_on_word_speaker():
    """ADR-001: одиночный word.speaker на границе не вызывает split."""
    seg = segment(
        "Просто интовый массив.",
        0.0,
        3.0,
        words=[
            word("Просто", 0.0, 0.5, "SPEAKER_00"),
            word("интовый", 0.6, 1.2, "SPEAKER_00"),
            word("массив.", 1.3, 2.5, "SPEAKER_02"),
        ],
        speaker="SPEAKER_00",
    )
    blocks = segment_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["speaker"] == "SPEAKER_00"
    assert blocks[0]["text"] == "Просто интовый массив."


def test_diarized_response_boundary_noise_single_replica():
    result = {
        "language": "ru",
        "segments": [
            segment(
                "Просто интовый массив.",
                0.0,
                3.0,
                words=[
                    word("Просто", 0.0, 0.5, "SPEAKER_00"),
                    word("интовый", 0.6, 1.2, "SPEAKER_00"),
                    word("массив.", 1.3, 2.5, "SPEAKER_02"),
                ],
                speaker="SPEAKER_00",
            )
        ],
    }
    body = diarized_response(result, language="ru")
    assert len(body.segments) == 1
    assert body.segments[0].speaker == "SPEAKER_00"
    assert body.segments[0].text == "Просто интовый массив."
    assert body.text == "SPEAKER_00: Просто интовый массив."


def test_segment_blocks_maps_each_whisper_segment():
    seg1 = segment("Привет", 0.0, 1.0, speaker="SPEAKER_01")
    seg2 = segment("мир", 1.0, 2.0, speaker="SPEAKER_01")
    blocks = segment_blocks([seg1, seg2])
    assert len(blocks) == 2


def test_merge_adjacent_same_speaker():
    seg1 = segment("Привет", 0.0, 1.0, speaker="SPEAKER_01")
    seg2 = segment("мир", 1.0, 2.0, speaker="SPEAKER_01")
    blocks = merge_adjacent_same_speaker(segment_blocks([seg1, seg2]))
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Привет мир"
    assert blocks[0]["speaker"] == "SPEAKER_01"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 2.0


def test_merge_does_not_join_same_speaker_after_interruption():
    segments = [
        segment("Привет", 0.0, 1.0, speaker="SPEAKER_00"),
        segment("как дела", 1.0, 2.0, speaker="SPEAKER_00"),
        segment("норм", 2.0, 3.0, speaker="SPEAKER_01"),
        segment("отлично", 3.0, 4.0, speaker="SPEAKER_00"),
    ]
    blocks = merge_adjacent_same_speaker(segment_blocks(segments))
    assert len(blocks) == 3
    assert blocks[0]["text"] == "Привет как дела"
    assert blocks[1]["text"] == "норм"
    assert blocks[2]["text"] == "отлично"


def test_diarized_response_merges_adjacent_same_speaker_segments():
    result = {
        "language": "ru",
        "segments": [
            segment("Привет,", 0.0, 1.0, speaker="SPEAKER_00"),
            segment("как дела?", 1.0, 2.0, speaker="SPEAKER_00"),
            segment("Нормально.", 2.0, 3.0, speaker="SPEAKER_01"),
        ],
    }
    body = diarized_response(result, language="ru")
    assert len(body.segments) == 2
    assert body.segments[0].speaker == "SPEAKER_00"
    assert body.segments[0].text == "Привет, как дела?"
    assert body.segments[1].speaker == "SPEAKER_01"


def test_segment_without_speaker_preserved():
    seg = segment("Нет спикера", 0.0, 1.0)
    blocks = segment_blocks([seg])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Нет спикера"
    assert blocks[0]["speaker"] is None


def test_empty_segments():
    assert segment_blocks([]) == []


def test_empty_text_segment_skipped():
    seg = segment("   ", 0.0, 1.0, speaker="SPEAKER_01")
    assert segment_blocks([seg]) == []


def test_verbose_json_preserves_word_speaker():
    result = {
        "language": "ru",
        "segments": [
            segment(
                "Просто интовый массив.",
                0.0,
                3.0,
                words=[
                    word("Просто", 0.0, 0.5, "SPEAKER_00"),
                    word("массив.", 1.3, 2.5, "SPEAKER_02"),
                ],
                speaker="SPEAKER_00",
            )
        ],
    }
    body = build_verbose_json(result, text="Просто интовый массив.", language="ru")
    assert body.segments[0].words is not None
    assert body.segments[0].words[0].speaker == "SPEAKER_00"
    assert body.segments[0].words[1].speaker == "SPEAKER_02"
