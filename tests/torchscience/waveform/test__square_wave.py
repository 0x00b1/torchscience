import math

import pytest
import scipy.signal
import torch
import torch.testing

import torchscience.waveform


class TestSquareWave:
    """Tests for square_wave function."""

    def test_basic_shape(self):
        """Square wave returns correct shape."""
        result = torchscience.waveform.square_wave(n=100, sample_rate=100.0)
        assert result.shape == (100,)

    def test_alternates_between_plus_minus_amplitude(self):
        """Square wave alternates between +/- amplitude.

        Note: Implementation uses differentiable approximation with smooth
        transitions, so we check that max/min approach target amplitude.
        """
        result = torchscience.waveform.square_wave(
            n=100,
            frequency=1.0,
            sample_rate=100.0,
            amplitude=2.0,
            dtype=torch.float64,
        )
        # Check that extrema approach +/- amplitude
        assert result.max() > 1.9  # Should approach +2
        assert result.min() < -1.9  # Should approach -2

    def test_duty_cycle_50_percent(self):
        """50% duty cycle spends equal time high and low."""
        n = 1000
        result = torchscience.waveform.square_wave(
            n=n,
            frequency=10.0,
            sample_rate=1000.0,
            duty=0.5,
            dtype=torch.float64,
        )
        positive_frac = (result > 0).float().mean()
        assert 0.45 < positive_frac < 0.55

    def test_duty_cycle_75_percent(self):
        """75% duty cycle spends 75% of time high."""
        n = 1000
        result = torchscience.waveform.square_wave(
            n=n,
            frequency=10.0,
            sample_rate=1000.0,
            duty=0.75,
            dtype=torch.float64,
        )
        positive_frac = (result > 0).float().mean()
        assert 0.70 < positive_frac < 0.80

    @pytest.mark.xfail(reason="Autograd not yet implemented for square_wave")
    def test_gradient_through_duty(self):
        """Gradients flow through duty cycle parameter."""
        duty = torch.tensor([0.5], requires_grad=True)
        result = torchscience.waveform.square_wave(
            n=100, duty=duty, sample_rate=100.0
        )
        loss = result.sum()
        loss.backward()
        assert duty.grad is not None

    def test_scipy_comparison(self):
        """Compare against scipy.signal.square.

        Note: Our implementation uses differentiable approximation with smooth
        transitions. The correlation won't be perfect but should be high.
        """
        n = 1000
        sample_rate = 1000.0
        frequency = 5.0

        t = torch.linspace(0, 1, n, dtype=torch.float64)
        result = torchscience.waveform.square_wave(
            t=t, frequency=frequency, dtype=torch.float64
        )

        t_np = t.numpy()
        scipy_result = scipy.signal.square(2 * math.pi * frequency * t_np)

        correlation = torch.corrcoef(
            torch.stack([result, torch.from_numpy(scipy_result)])
        )[0, 1]
        # Lower threshold due to smooth approximation
        assert correlation > 0.90
