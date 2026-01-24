# tests/torchscience/signal_processing/waveform/test__hyperbolic_chirp_wave.py
import scipy.signal
import torch
import torchscience.signal_processing.waveform


class TestHyperbolicChirpWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.hyperbolic_chirp_wave(
            n=1000, f0=10.0, f1=1.0, sample_rate=1000.0
        )
        assert result.shape == (1000,)

    def test_scipy_comparison(self):
        n = 1000
        f0, f1 = 10.0, 1.0  # Note: hyperbolic typically sweeps down
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = torchscience.signal_processing.waveform.hyperbolic_chirp_wave(
            t=t, f0=f0, f1=f1, dtype=torch.float64
        )
        scipy_result = scipy.signal.chirp(
            t.numpy(), f0, 1.0, f1, method="hyperbolic"
        )
        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        assert correlation > 0.99

    def test_batched_frequencies(self):
        f0 = torch.tensor([10.0, 20.0, 30.0])
        result = torchscience.signal_processing.waveform.hyperbolic_chirp_wave(
            n=1000, f0=f0, f1=1.0, sample_rate=1000.0
        )
        assert result.shape == (3, 1000)
