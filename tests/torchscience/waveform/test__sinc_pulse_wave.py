import torch

import torchscience.waveform


class TestSincPulseWave:
    def test_basic_shape(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=100, center=50, bandwidth=0.1
        )
        assert result.shape == (100,)

    def test_peak_at_center(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=100, center=50, bandwidth=0.1, amplitude=1.0
        )
        assert result.argmax() == 50
        assert abs(result[50] - 1.0) < 0.01

    def test_normalized_sinc(self):
        # Normalized sinc: sin(pi*x)/(pi*x) = 1 at x=0
        result = torchscience.waveform.sinc_pulse_wave(
            n=101, center=50, bandwidth=0.5, dtype=torch.float64
        )
        assert abs(result[50] - 1.0) < 1e-10

    def test_batched_centers(self):
        centers = torch.tensor([25.0, 50.0, 75.0])
        result = torchscience.waveform.sinc_pulse_wave(
            n=100, center=centers, bandwidth=0.1
        )
        assert result.shape == (3, 100)

    def test_amplitude_scaling(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=101, center=50, bandwidth=0.5, amplitude=2.5, dtype=torch.float64
        )
        assert abs(result[50] - 2.5) < 1e-10

    def test_sinc_values(self):
        # Test that values away from center follow sinc function
        result = torchscience.waveform.sinc_pulse_wave(
            n=101, center=50, bandwidth=1.0, amplitude=1.0, dtype=torch.float64
        )
        # At k=50, x = 0, sinc(0) = 1
        assert abs(result[50] - 1.0) < 1e-10
        # At k=51, x = 1, sinc(1) = sin(pi)/pi = 0
        assert abs(result[51] - 0.0) < 1e-10
        # At k=52, x = 2, sinc(2) = sin(2*pi)/(2*pi) = 0
        assert abs(result[52] - 0.0) < 1e-10

    def test_empty_output(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=0, center=0, bandwidth=1.0
        )
        assert result.shape == (0,)

    def test_dtype_preservation(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=10, center=5, bandwidth=1.0, dtype=torch.float64
        )
        assert result.dtype == torch.float64

    def test_batched_bandwidth(self):
        bandwidths = torch.tensor([0.1, 0.5, 1.0])
        result = torchscience.waveform.sinc_pulse_wave(
            n=100, center=50, bandwidth=bandwidths
        )
        assert result.shape == (3, 100)

    def test_batched_all_parameters(self):
        centers = torch.tensor([[25.0], [50.0]])  # (2, 1)
        bandwidths = torch.tensor([0.1, 0.5, 1.0])  # (3,)
        amplitudes = torch.tensor([1.0])  # (1,)
        result = torchscience.waveform.sinc_pulse_wave(
            n=100, center=centers, bandwidth=bandwidths, amplitude=amplitudes
        )
        # Broadcasted shape: (2, 3), so output is (2, 3, 100)
        assert result.shape == (2, 3, 100)

    def test_requires_grad(self):
        result = torchscience.waveform.sinc_pulse_wave(
            n=10, center=5, bandwidth=1.0, requires_grad=True
        )
        assert result.requires_grad

    def test_sinc_symmetry(self):
        # Sinc function is symmetric around center
        result = torchscience.waveform.sinc_pulse_wave(
            n=101, center=50, bandwidth=0.5, dtype=torch.float64
        )
        # Check symmetry: result[50-k] should equal result[50+k]
        for k in range(1, 20):
            assert abs(result[50 - k] - result[50 + k]) < 1e-10
