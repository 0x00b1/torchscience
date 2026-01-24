import math

import pytest
import scipy.signal
import torch
import torchscience.signal_processing.waveform


class TestTriangleWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=100, sample_rate=100.0
        )
        assert result.shape == (100,)

    def test_amplitude_range(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=1000, frequency=1.0, sample_rate=1000.0, amplitude=2.0
        )
        assert result.max() <= 2.0
        assert result.min() >= -2.0

    def test_scipy_comparison(self):
        n = 1000
        frequency = 5.0
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = torchscience.signal_processing.waveform.triangle_wave(
            t=t, frequency=frequency, dtype=torch.float64
        )
        # scipy sawtooth with width=0.5 gives triangle
        scipy_result = scipy.signal.sawtooth(
            2 * math.pi * frequency * t.numpy(), width=0.5
        )
        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        assert correlation > 0.99

    def test_empty_n(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=0, sample_rate=100.0
        )
        assert result.shape == (0,)

    def test_single_sample(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=1, sample_rate=100.0
        )
        assert result.shape == (1,)

    def test_mutually_exclusive_n_t(self):
        with pytest.raises(ValueError):
            torchscience.signal_processing.waveform.triangle_wave(
                n=100, t=torch.linspace(0, 1, 100)
            )

    def test_neither_n_nor_t(self):
        with pytest.raises(ValueError):
            torchscience.signal_processing.waveform.triangle_wave()

    def test_dtype_preservation(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=100, sample_rate=100.0, dtype=torch.float64
        )
        assert result.dtype == torch.float64

    def test_custom_amplitude(self):
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=1000, frequency=1.0, sample_rate=1000.0, amplitude=5.0
        )
        assert result.max() <= 5.0
        assert result.min() >= -5.0

    def test_linearity(self):
        # Triangle wave should have linear segments
        n = 1000
        frequency = 1.0
        sample_rate = 1000.0
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=n, frequency=frequency, sample_rate=sample_rate
        )
        # First half rises linearly - check constant difference
        first_half = result[: n // 2]
        diff = first_half[1:] - first_half[:-1]
        # All differences should be approximately the same (constant slope)
        assert torch.allclose(diff, diff[0].expand_as(diff), atol=1e-5)

    def test_batch_frequency(self):
        frequencies = torch.tensor([1.0, 2.0, 4.0])
        result = torchscience.signal_processing.waveform.triangle_wave(
            n=100, frequency=frequencies, sample_rate=100.0
        )
        assert result.shape == (3, 100)

    def test_with_time_tensor(self):
        t = torch.linspace(0, 1, 100)
        result = torchscience.signal_processing.waveform.triangle_wave(
            t=t, frequency=5.0
        )
        assert result.shape == (100,)
