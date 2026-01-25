# tests/torchscience/signal_processing/waveform/test__gaussian_pulse_wave.py
import math

import torch

import torchscience.signal_processing.waveform


class TestGaussianPulseWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.gaussian_pulse_wave(
            n=100, center=50, std=10.0
        )
        assert result.shape == (100,)

    def test_peak_at_center(self):
        result = torchscience.signal_processing.waveform.gaussian_pulse_wave(
            n=100, center=50, std=10.0, amplitude=1.0
        )
        assert result.argmax() == 50
        assert abs(result[50] - 1.0) < 0.01

    def test_gaussian_shape(self):
        n = 1000
        center = 500
        std = 50.0
        result = torchscience.signal_processing.waveform.gaussian_pulse_wave(
            n=n, center=center, std=std, dtype=torch.float64
        )
        # Check value at 1 std from center
        expected_at_1std = math.exp(-0.5)
        assert abs(result[center + 50] - expected_at_1std) < 0.01

    def test_batched_centers(self):
        centers = torch.tensor([25.0, 50.0, 75.0])
        result = torchscience.signal_processing.waveform.gaussian_pulse_wave(
            n=100, center=centers, std=10.0
        )
        assert result.shape == (3, 100)
