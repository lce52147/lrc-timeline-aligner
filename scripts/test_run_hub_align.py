"""Executes reviewers/run_hub.py align_song with stubbed model/tensor libraries (no models, no torch)."""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import auto_lrc

SCRIPTS = Path(__file__).resolve().parent
RUN_HUB = SCRIPTS / "reviewers" / "run_hub.py"
FRAMES = 40
SAMPLES = 16_000 * 4


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def cpu(self):
        return self

    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self

    def squeeze(self, _dim):
        return self

    def numpy(self):
        return self


def _fake_libraries(span_count):
    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    torch.int32 = "int32"
    torch.log_softmax = lambda logits, dim: _Tensor((1, FRAMES, 5))
    torch.tensor = lambda values, dtype=None: _Tensor((len(values),))
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    def forced_align(log_probs, targets, blank):
        return [None], [None]

    def merge_tokens(_aligned, _scores, blank=0):
        return [SimpleNamespace(start=index * 2, end=index * 2 + 1, score=-0.1) for index in range(span_count)]

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = SimpleNamespace(forced_align=forced_align, merge_tokens=merge_tokens)
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda *args, **kwargs: None
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCTC = object
    transformers.AutoProcessor = object
    return {"torch": torch, "torchaudio": torchaudio, "huggingface_hub": hub, "transformers": transformers}


def _load_run_hub(span_count):
    with mock.patch.dict(sys.modules, _fake_libraries(span_count)):
        spec = importlib.util.spec_from_file_location("run_hub_under_test", RUN_HUB)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _processor():
    vocab = {"a": 1, "b": 2, "c": 3, "|": 4, "<pad>": 0}
    processor = mock.MagicMock()
    processor.tokenizer.get_vocab.return_value = vocab
    processor.tokenizer.pad_token_id = 0
    processor.return_value = SimpleNamespace(input_values=_Tensor((1, SAMPLES)), attention_mask=None)
    return processor


def _model():
    model = mock.MagicMock()
    model.return_value = SimpleNamespace(logits=_Tensor((1, FRAMES, 5)))
    return model


class RunHubAlignSongTests(unittest.TestCase):
    def _align(self, module):
        waveform = _Tensor((1, SAMPLES))
        with mock.patch.object(module, "decode_audio", return_value=waveform):
            return module.align_song(
                _processor(), _model(), "cpu", Path("song.wav"), ["ab"], [3], "audio-rev"
            )

    def test_opening_evidence_call_receives_entry_index_payload_and_revision(self):
        module = _load_run_hub(span_count=2)
        with mock.patch.object(auto_lrc, "_known_lyric_hubert_opening_evidence", return_value=object()) as evidence:
            rows, frames = self._align(module)
        self.assertEqual(frames, FRAMES)
        evidence.assert_called_once()
        entry_index, payload, revision = evidence.call_args.args
        self.assertEqual(entry_index, 2)
        self.assertEqual([item["char"] for item in payload], ["a", "b"])
        self.assertTrue(all({"start", "end", "score"} <= set(item) for item in payload))
        self.assertIsInstance(revision, str)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry"], 3)
        self.assertIs(rows[0]["opening_supported"], True)
        self.assertIs(rows[0]["trusted"], True)

    def test_none_evidence_marks_the_row_unsupported(self):
        module = _load_run_hub(span_count=2)
        with mock.patch.object(auto_lrc, "_known_lyric_hubert_opening_evidence", return_value=None):
            rows, _frames = self._align(module)
        self.assertIs(rows[0]["opening_supported"], False)
        self.assertIs(rows[0]["trusted"], False)

    def test_real_helper_is_reachable_and_returns_a_bool_flag(self):
        module = _load_run_hub(span_count=2)
        rows, _frames = self._align(module)
        self.assertIsInstance(rows[0]["opening_supported"], bool)

    def test_span_count_mismatch_is_reported(self):
        module = _load_run_hub(span_count=1)
        with self.assertRaisesRegex(RuntimeError, "span mismatch"):
            self._align(module)


if __name__ == "__main__":
    unittest.main()
