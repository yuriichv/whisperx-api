"""Общие фикстуры для unit/contract/e2e тестов transcription."""

from types import SimpleNamespace
from typing import Any

TO_COMMAND_DEFAULTS = {
    "default_model": "large-v3",
    "default_language": "",
    "default_align": False,
    "default_diarize": False,
}


def word(text: str, start: float, end: float, speaker: str | None = None) -> dict:
    w: dict[str, Any] = {"word": text, "start": start, "end": end}
    if speaker is not None:
        w["speaker"] = speaker
    return w


def segment(
    text: str,
    start: float,
    end: float,
    words: list[dict] | None = None,
    speaker: str | None = None,
) -> dict:
    s: dict[str, Any] = {"text": text, "start": start, "end": end}
    if words is not None:
        s["words"] = words
    if speaker is not None:
        s["speaker"] = speaker
    return s


def biane_speaker_change_words(
    speaker_a: str = "SPEAKER_01",
    speaker_b: str = "SPEAKER_02",
) -> list[dict]:
    return [
        word("Я", 0.0, 0.2, speaker_a),
        word("попробовал", 0.3, 0.8, speaker_a),
        word("это", 0.9, 1.1, speaker_a),
        word("на", 1.2, 1.4, speaker_a),
        word("Биане.", 1.5, 2.0, speaker_a),
        word("Ну", 2.1, 2.3, speaker_b),
        word("давай", 2.4, 2.7, speaker_b),
        word("Андрюх", 2.8, 3.0, speaker_b),
    ]


def biane_speaker_change_segment(
    speaker_a: str = "SPEAKER_01",
    speaker_b: str = "SPEAKER_02",
) -> dict:
    return segment(
        "Я попробовал это на Биане. Ну давай Андрюх",
        0.0,
        4.0,
        words=biane_speaker_change_words(speaker_a, speaker_b),
        speaker=speaker_a,
    )


def make_diarization_e2e_result() -> dict:
    return {
        "language": "ru",
        "text": "Финальный текст всего аудио",
        "segments": [
            biane_speaker_change_segment("SPEAKER_00", "SPEAKER_01"),
            {
                "text": "Промежуточная.",
                "start": 4.0,
                "end": 5.0,
                "speaker": "SPEAKER_01",
            },
            {
                "text": "Согласен.",
                "start": 5.0,
                "end": 6.0,
                "speaker": "SPEAKER_02",
            },
        ],
    }


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class FakeHfEncoding:
    def __init__(self, text: str) -> None:
        self.ids = list(range(len(text)))


class FakeHfTokenizer:
    """Имитация tokenizers.Tokenizer (model.hf_tokenizer в faster-whisper)."""

    def encode(self, text: str) -> FakeHfEncoding:
        return FakeHfEncoding(text)


class FakeASRPipeline:
    """Заглушка ASR-пайплайна для unit/e2e тестов."""

    def __init__(
        self,
        result: dict | None = None,
        *,
        pipeline_tokenizer: FakeTokenizer | None = None,
        hf_tokenizer: FakeHfTokenizer | None = None,
    ):
        self.result = result or {
            "language": "en",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
        }
        self.last_kwargs: dict | None = None
        self.tokenizer = pipeline_tokenizer if pipeline_tokenizer is not None else FakeTokenizer()
        self.model = SimpleNamespace(
            hf_tokenizer=hf_tokenizer if hf_tokenizer is not None else FakeHfTokenizer(),
        )
        self.options = SimpleNamespace(initial_prompt=None, hotwords=None)
        self.conditioning_calls: list[tuple] = []

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        self.conditioning_calls.append(
            (self.options.initial_prompt, self.options.hotwords)
        )
        return {
            "segments": list(self.result["segments"]),
            "language": self.result["language"],
        }


class RecordingDiarizePipeline:
    """Заглушка DiarizationPipeline; записывает kwargs вызова."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 5.0, "speaker": "SPEAKER_02"},
        ]
