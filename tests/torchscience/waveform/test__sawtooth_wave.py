import math

import scipy.signal
import torch
import torchscience.signal_processing.waveform


class TestSawtoothWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.sawtooth_wave(
            n=100, sample_rate=100.0
        )
        assert result.shape == (100,)

    def test_amplitude_range(self):
        result = torchscience.signal_processing.waveform.sawtooth_wave(
            n=1000, frequency=1.0, sample_rate=1000.0, amplitude=2.0
        )
        assert result.max() <= 2.0
        assert result.min() >= -2.0

    def test_scipy_comparison(self):
        n = 1000
        frequency = 5.0
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = torchscience.signal_processing.waveform.sawtooth_wave(
            t=t, frequency=frequency, dtype=torch.float64
        )
        scipy_result = scipy.signal.sawtooth(
            2 * math.pi * frequency * t.numpy()
        )
        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        assert correlation > 0.99
