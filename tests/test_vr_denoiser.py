"""Run with: python -m unittest discover -s tests -v.

Uses NumPy, PyTorch and UVR's audio utilities, without model downloads or the
GUI/ONNX/Demucs dependencies imported by separate.py.
"""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib_v5 import spec_utils


def load_vr_denoiser():
    # Execute the production function, avoiding unrelated separator imports.
    path = Path(__file__).resolve().parents[1] / "separate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "vr_denoiser")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "np": np,
        "torch": torch,
        "cpu": torch.device("cpu"),
        "spec_utils": spec_utils,
        "nets_new": SimpleNamespace(CascadedNet=TestDenoiser),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["vr_denoiser"]


class TestDenoiser(torch.nn.Module):
    """Small float32 network exercising real device transfers and inference."""

    offset = 2

    def __init__(self, n_fft=2048, **kwargs):
        super().__init__()
        self.filter = torch.nn.Conv2d(2, 2, 1, bias=False)
        with torch.no_grad():
            self.filter.weight.fill_(0.25)

    def predict_mask(self, batch):
        return torch.sigmoid(self.filter(batch))[..., self.offset:-self.offset]


class VRDenoiserPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.denoise = staticmethod(load_vr_denoiser())
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.model_path = Path(cls.temp_dir.name) / "denoiser.pth"
        torch.save(TestDenoiser().state_dict(), cls.model_path)
        time = np.arange(65536, dtype=np.float64) / 44100
        cls.audio = np.stack([
            0.3 * np.sin(2 * np.pi * 440 * time),
            0.2 * np.sin(2 * np.pi * 660 * time),
        ])

    def check_denoising(self, device, dtype):
        audio = self.audio.astype(dtype)
        original = audio.copy()
        observed_batches = []
        predict_mask = TestDenoiser.predict_mask

        def record_batch(model, batch):
            observed_batches.append((batch.dtype, batch.device.type, batch.shape[0]))
            return predict_mask(model, batch)

        with patch.object(TestDenoiser, "predict_mask", record_batch):
            output = self.denoise(audio, device, cropsize=16, model_path=self.model_path)

        self.assertEqual(output.shape, audio.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(np.abs(output).max(), 0)
        np.testing.assert_array_equal(audio, original)
        self.assertGreater(len(observed_batches), 1)
        for batch_dtype, batch_device, _ in observed_batches:
            self.assertEqual(batch_dtype, torch.float32)
            self.assertEqual(batch_device, torch.device(device).type)
        self.assertLess(observed_batches[-1][2], 4)  # Also exercise a partial batch.

        reference = self.denoise(self.audio.astype(np.float32), "cpu",
                                 cropsize=16, model_path=self.model_path)
        np.testing.assert_allclose(output, reference, atol=1e-6, rtol=1e-5)

    def test_cpu_float32_audio(self):
        self.check_denoising("cpu", np.float32)

    def test_cpu_float64_audio(self):
        self.check_denoising(torch.device("cpu"), np.float64)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_float32_audio(self):
        self.check_denoising("mps", np.float32)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_float64_audio(self):
        self.check_denoising(torch.device("mps"), np.float64)


if __name__ == "__main__":
    unittest.main()
