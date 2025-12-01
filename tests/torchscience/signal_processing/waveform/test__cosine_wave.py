import math

import torch
import torch.testing

import torchscience.signal_processing.waveform


class TestCosineWave:
    """Tests for cosine_wave function."""

    def test_starts_at_one(self):
        """Cosine wave starts at 1.0 with default phase."""
        result = torchscience.signal_processing.waveform.cosine_wave(
            n=100, sample_rate=100.0
        )
        torch.testing.assert_close(
            result[0], torch.tensor(1.0), atol=1e-5, rtol=1e-5
        )

    def test_equals_sine_with_phase_shift(self):
        """cos(x) = sin(x + pi/2)."""
        n = 100
        cosine = torchscience.signal_processing.waveform.cosine_wave(
            n=n, frequency=1.0, sample_rate=100.0, dtype=torch.float64
        )
        sine_shifted = torchscience.signal_processing.waveform.sine_wave(
            n=n,
            frequency=1.0,
            sample_rate=100.0,
            phase=math.pi / 2,
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            cosine, sine_shifted, atol=1e-10, rtol=1e-10
        )

    def test_tensor_parameters(self):
        """Supports tensor parameters like sine_wave."""
        freqs = torch.tensor([1.0, 2.0, 5.0])
        result = torchscience.signal_processing.waveform.cosine_wave(
            n=100, frequency=freqs, sample_rate=100.0
        )
        assert result.shape == (3, 100)
