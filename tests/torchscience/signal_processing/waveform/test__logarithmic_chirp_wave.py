# tests/torchscience/signal_processing/waveform/test__logarithmic_chirp_wave.py
import scipy.signal
import torch

import torchscience.signal_processing.waveform


class TestLogarithmicChirpWave:
    def test_basic_shape(self):
        result = (
            torchscience.signal_processing.waveform.logarithmic_chirp_wave(
                n=1000, f0=1.0, f1=100.0, sample_rate=1000.0
            )
        )
        assert result.shape == (1000,)

    def test_scipy_comparison(self):
        n = 1000
        f0, f1 = 10.0, 100.0
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = (
            torchscience.signal_processing.waveform.logarithmic_chirp_wave(
                t=t, f0=f0, f1=f1, dtype=torch.float64
            )
        )
        scipy_result = scipy.signal.chirp(
            t.numpy(), f0, 1.0, f1, method="logarithmic"
        )
        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        assert correlation > 0.99

    def test_batched_frequencies(self):
        f0 = torch.tensor([1.0, 5.0, 10.0])
        result = (
            torchscience.signal_processing.waveform.logarithmic_chirp_wave(
                n=1000, f0=f0, f1=100.0, sample_rate=1000.0
            )
        )
        assert result.shape == (3, 1000)
