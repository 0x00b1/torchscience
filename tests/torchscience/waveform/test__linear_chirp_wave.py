import math

import scipy.signal
import torch

import torchscience.waveform


class TestLinearChirpWave:
    def test_basic_shape(self):
        result = torchscience.waveform.linear_chirp_wave(
            n=1000, f0=1.0, f1=10.0, sample_rate=1000.0
        )
        assert result.shape == (1000,)

    def test_scipy_comparison(self):
        n = 1000
        sample_rate = 1000.0
        f0, f1 = 1.0, 50.0
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = torchscience.waveform.linear_chirp_wave(
            t=t, f0=f0, f1=f1, dtype=torch.float64
        )
        scipy_result = scipy.signal.chirp(
            t.numpy(), f0, 1.0, f1, method="linear"
        )
        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        assert correlation > 0.99

    def test_batched_frequencies(self):
        f0 = torch.tensor([1.0, 5.0, 10.0])
        result = torchscience.waveform.linear_chirp_wave(
            n=1000, f0=f0, f1=100.0, sample_rate=1000.0
        )
        assert result.shape == (3, 1000)

    def test_n_equals_zero(self):
        """Test that n=0 returns empty tensor."""
        result = torchscience.waveform.linear_chirp_wave(n=0, f0=1.0, f1=10.0)
        assert result.shape == (0,)
        assert result.numel() == 0

    def test_frequency_increases(self):
        """Test that instantaneous frequency increases over time."""
        n = 1000
        sample_rate = 1000.0
        f0, f1 = 1.0, 50.0
        result = torchscience.waveform.linear_chirp_wave(
            n=n, f0=f0, f1=f1, sample_rate=sample_rate, dtype=torch.float64
        )

        # Count zero crossings in first half vs second half
        signs = torch.sign(result)
        signs[signs == 0] = 1
        diff = torch.abs(torch.diff(signs))

        first_half = diff[: n // 2]
        second_half = diff[n // 2 :]

        crossings_first = torch.sum(first_half > 0).item()
        crossings_second = torch.sum(second_half > 0).item()

        # Second half should have more crossings (higher frequency)
        assert crossings_second > crossings_first

    def test_amplitude_scaling(self):
        """Test that amplitude correctly scales the wave."""
        n = 1000
        amplitude = 2.5

        result = torchscience.waveform.linear_chirp_wave(
            n=n, f0=1.0, f1=10.0, amplitude=amplitude, dtype=torch.float64
        )

        # Max and min should be within amplitude bounds
        assert result.max().item() <= amplitude + 1e-10
        assert result.min().item() >= -amplitude - 1e-10

    def test_starts_with_cosine(self):
        """Test that chirp starts at amplitude (cos(0) = 1) with default phase."""
        result = torchscience.waveform.linear_chirp_wave(
            n=100,
            f0=1.0,
            f1=10.0,
            sample_rate=1000.0,
            amplitude=1.0,
            phase=0.0,
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            result[0],
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_phase_offset(self):
        """Test that phase offset works correctly."""
        n = 100
        # With phase=pi/2, cos(pi/2) = 0
        result = torchscience.waveform.linear_chirp_wave(
            n=n,
            f0=1.0,
            f1=10.0,
            sample_rate=1000.0,
            phase=math.pi / 2,
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            result[0],
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_explicit_time_tensor(self):
        """Test with explicit time tensor."""
        t = torch.linspace(0, 1, 100, dtype=torch.float64)
        result = torchscience.waveform.linear_chirp_wave(
            t=t, f0=1.0, f1=10.0, dtype=torch.float64
        )
        assert result.shape == (100,)

    def test_n_and_t_mutually_exclusive(self):
        """Test that providing both n and t raises error."""
        t = torch.linspace(0, 1, 100)
        try:
            torchscience.waveform.linear_chirp_wave(
                n=100, t=t, f0=1.0, f1=10.0
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "mutually exclusive" in str(e)

    def test_neither_n_nor_t_raises(self):
        """Test that providing neither n nor t raises error."""
        try:
            torchscience.waveform.linear_chirp_wave(f0=1.0, f1=10.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Either n or t" in str(e)

    def test_batched_f1(self):
        """Test batched f1 produces correct shape."""
        f1 = torch.tensor([10.0, 50.0, 100.0])
        result = torchscience.waveform.linear_chirp_wave(
            n=500, f0=1.0, f1=f1, sample_rate=1000.0
        )
        assert result.shape == (3, 500)

    def test_batched_amplitude(self):
        """Test batched amplitude produces correct shape."""
        amplitude = torch.tensor([0.5, 1.0, 2.0])
        result = torchscience.waveform.linear_chirp_wave(
            n=500, f0=1.0, f1=10.0, amplitude=amplitude, sample_rate=1000.0
        )
        assert result.shape == (3, 500)

    def test_batched_phase(self):
        """Test batched phase produces correct shape."""
        phase = torch.tensor([0.0, math.pi / 2, math.pi])
        result = torchscience.waveform.linear_chirp_wave(
            n=500, f0=1.0, f1=10.0, phase=phase, sample_rate=1000.0
        )
        assert result.shape == (3, 500)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting of parameters."""
        f0 = torch.tensor([[1.0], [5.0]])  # shape (2, 1)
        f1 = torch.tensor([10.0, 50.0, 100.0])  # shape (3,)
        result = torchscience.waveform.linear_chirp_wave(
            n=100, f0=f0, f1=f1, sample_rate=1000.0
        )
        assert result.shape == (2, 3, 100)

    def test_float64_precision(self):
        """Test float64 maintains precision."""
        result = torchscience.waveform.linear_chirp_wave(
            n=100, f0=1.0, f1=10.0, dtype=torch.float64
        )
        assert result.dtype == torch.float64

    def test_requires_grad(self):
        """Test requires_grad flag."""
        result = torchscience.waveform.linear_chirp_wave(
            n=100, f0=1.0, f1=10.0, requires_grad=True
        )
        assert result.requires_grad

    def test_contiguous_output(self):
        """Test that output tensor is contiguous."""
        result = torchscience.waveform.linear_chirp_wave(
            n=100, f0=1.0, f1=10.0
        )
        assert result.is_contiguous()

    def test_t1_parameter(self):
        """Test t1 affects the chirp rate."""
        n = 1000
        sample_rate = 1000.0
        f0, f1 = 1.0, 50.0

        # With t1=1, frequency reaches f1 at t=1
        result_t1_1 = torchscience.waveform.linear_chirp_wave(
            n=n,
            f0=f0,
            f1=f1,
            t1=1.0,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # With t1=2, frequency reaches f1 at t=2 (slower sweep)
        result_t1_2 = torchscience.waveform.linear_chirp_wave(
            n=n,
            f0=f0,
            f1=f1,
            t1=2.0,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # The slower sweep should have fewer oscillations overall
        signs_1 = torch.sign(result_t1_1)
        signs_1[signs_1 == 0] = 1
        crossings_1 = torch.sum(torch.abs(torch.diff(signs_1)) > 0).item()

        signs_2 = torch.sign(result_t1_2)
        signs_2[signs_2 == 0] = 1
        crossings_2 = torch.sum(torch.abs(torch.diff(signs_2)) > 0).item()

        assert crossings_2 < crossings_1

    def test_negative_sweep(self):
        """Test chirp with decreasing frequency (f1 < f0)."""
        n = 1000
        sample_rate = 1000.0
        f0, f1 = 50.0, 1.0  # Decreasing frequency

        result = torchscience.waveform.linear_chirp_wave(
            n=n, f0=f0, f1=f1, sample_rate=sample_rate, dtype=torch.float64
        )

        # Count zero crossings in first half vs second half
        signs = torch.sign(result)
        signs[signs == 0] = 1
        diff = torch.abs(torch.diff(signs))

        first_half = diff[: n // 2]
        second_half = diff[n // 2 :]

        crossings_first = torch.sum(first_half > 0).item()
        crossings_second = torch.sum(second_half > 0).item()

        # First half should have more crossings (higher frequency)
        assert crossings_first > crossings_second

    def test_sample_rate_effect(self):
        """Test that sample_rate affects the output correctly."""
        n = 100
        f0, f1 = 1.0, 10.0

        result_sr1 = torchscience.waveform.linear_chirp_wave(
            n=n, f0=f0, f1=f1, sample_rate=100.0, dtype=torch.float64
        )
        result_sr2 = torchscience.waveform.linear_chirp_wave(
            n=n, f0=f0, f1=f1, sample_rate=200.0, dtype=torch.float64
        )

        # Higher sample rate means more time points per unit time
        # So the waveform should "stretch out" - fewer oscillations visible
        # Actually with higher sample_rate, we sample a shorter time duration
        # which means we see fewer cycles
        signs_1 = torch.sign(result_sr1)
        signs_1[signs_1 == 0] = 1
        crossings_1 = torch.sum(torch.abs(torch.diff(signs_1)) > 0).item()

        signs_2 = torch.sign(result_sr2)
        signs_2[signs_2 == 0] = 1
        crossings_2 = torch.sum(torch.abs(torch.diff(signs_2)) > 0).item()

        # Higher sample rate = shorter time = fewer oscillations
        assert crossings_2 < crossings_1
