import unittest
from types import SimpleNamespace

try:
    import torch
    from torchaudio.models.wav2vec2.components import _get_feature_extractor
    import ctc_align as c
except ImportError:
    torch = None
    c = None

# torchaudio MMS_FA frontend: extractor_mode="layer_norm".
MMS_FA_CONV = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]


def mms_frontend(device: str = "cpu"):
    torch.manual_seed(1234)
    extractor = _get_feature_extractor("layer_norm", MMS_FA_CONV, True).to(device).eval()
    return extractor, SimpleNamespace(model=SimpleNamespace(feature_extractor=extractor))


@unittest.skipUnless(torch is not None and c is not None, "requires torchaudio CTC runtime")
class ChunkedCtcFrontendTests(unittest.TestCase):
    def test_stride_field_and_frame_count_match_the_real_frontend(self):
        extractor, _ = mms_frontend()
        layers = list(extractor.conv_layers)
        self.assertEqual(c.conv_stride_and_field(layers), (320, 400))
        with torch.inference_mode():
            for samples in (399, 400, 719, 720, 16_000, 16_000 * 3 + 77):
                expected = c.conv_output_frames(layers, samples)
                if expected == 0:
                    continue
                features, _ = extractor(torch.zeros((1, samples)), None)
                self.assertEqual(features.shape[1], expected, samples)

    def test_chunked_frontend_reproduces_the_full_length_pass(self):
        extractor, model = mms_frontend()
        torch.manual_seed(7)
        lengths = (
            400,                      # one frame
            320 * 99 + 400,           # exactly one chunk
            320 * 100 + 400,          # one frame past a chunk boundary
            320 * 250 + 399,          # ragged tail
            16_000 * 47 + 123,        # 47 s, many chunks
        )
        waves = [torch.randn((1, n)) for n in lengths]
        with torch.inference_mode():
            full = [extractor(w, None)[0] for w in waves]
        c.configure_chunked_feature_extractor(model, chunk_frames=100)
        with torch.inference_mode():
            for n, wave, reference in zip(lengths, waves, full):
                chunked, chunk_lengths = extractor(wave, None)
                self.assertIsNone(chunk_lengths)
                self.assertEqual(tuple(chunked.shape), tuple(reference.shape), n)
                self.assertEqual(chunked.dtype, torch.float32)
                torch.testing.assert_close(chunked, reference, rtol=1e-5, atol=1e-5, msg=str(n))

    def test_explicit_lengths_keep_the_original_full_pass(self):
        extractor, model = mms_frontend()
        wave = torch.randn((1, 16_000 * 5))
        length = torch.tensor([16_000 * 5])
        with torch.inference_mode():
            reference, reference_lengths = extractor(wave, length)
            c.configure_chunked_feature_extractor(model, chunk_frames=50)
            out, out_lengths = extractor(wave, length)
        self.assertTrue(torch.equal(out, reference))
        self.assertTrue(torch.equal(out_lengths, reference_lengths))

    def test_rewrapping_a_cached_model_wraps_the_original_once(self):
        extractor, model = mms_frontend()
        c.configure_chunked_feature_extractor(model, chunk_frames=50)
        original = extractor._lrc_ctc_original_forward
        c.configure_chunked_feature_extractor(model, chunk_frames=50)
        self.assertIs(extractor._lrc_ctc_original_forward, original)
        wave = torch.randn((1, 320 * 120 + 400))
        with torch.inference_mode():
            out, _ = extractor(wave, None)
        self.assertEqual(out.shape[1], 121)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_cuda_chunks_match_the_cpu_fp32_frontend_and_restore_tf32(self):
        cpu_extractor, _ = mms_frontend("cpu")
        gpu_extractor, gpu_model = mms_frontend("cuda")
        wave = torch.randn((1, 16_000 * 30 + 55))
        with torch.inference_mode():
            reference, _ = cpu_extractor(wave, None)
        before = torch.backends.cudnn.allow_tf32
        c.configure_chunked_feature_extractor(gpu_model, chunk_frames=200)
        with torch.inference_mode():
            out, _ = gpu_extractor(wave, None)
        self.assertEqual(torch.backends.cudnn.allow_tf32, before)
        self.assertEqual(out.device.type, "cuda")
        torch.testing.assert_close(out.cpu(), reference, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
