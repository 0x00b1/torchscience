# tests/torchscience/signal_processing/filter/test__sosfilt.py
"""Tests for sosfilt (SOS filter application)."""

import math

import torch
from scipy import signal as scipy_signal

from torchscience.filter import sosfilt
from torchscience.filter_design import butterworth_design


class TestSosfiltForward:
    """Test sosfilt forward correctness."""

    def test_matches_scipy(self) -> None:
        """Output should match scipy.signal.sosfilt."""
        # Design a lowpass filter
        sos = butterworth_design(4, 0.2)

        # Create test signal (use float64 to match scipy precision)
        t = torch.linspace(0, 1, 1000, dtype=torch.float64)
        x = torch.sin(2 * math.pi * 5 * t) + 0.5 * torch.sin(
            2 * math.pi * 50 * t
        )

        # Filter with torchscience
        y = sosfilt(sos, x)

        # Filter with scipy
        y_scipy = scipy_signal.sosfilt(sos.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-5, atol=1e-10
        )

    def test_removes_high_frequency(self) -> None:
        """Lowpass filter should attenuate high frequencies."""
        sos = butterworth_design(4, 0.05)  # Very low cutoff (normalized)

        # High frequency signal well above cutoff
        # Sample rate = 1000, Nyquist = 500
        # Signal at 200 Hz = 0.4 normalized, cutoff at 0.05 = 25 Hz
        t = torch.linspace(0, 1, 1000, dtype=torch.float64)
        x = torch.sin(2 * math.pi * 200 * t)

        y = sosfilt(sos, x)

        # Output should have much smaller amplitude (>10x attenuation)
        assert y.abs().max() < 0.1 * x.abs().max()

    def test_passes_low_frequency(self) -> None:
        """Lowpass filter should pass low frequencies."""
        sos = butterworth_design(4, 0.5)  # High cutoff

        # Low frequency signal (after transient)
        t = torch.linspace(0, 1, 1000, dtype=torch.float64)
        x = torch.sin(2 * math.pi * 2 * t)

        y = sosfilt(sos, x)

        # After transient, amplitude should be similar
        assert y[500:].abs().max() > 0.9 * x[500:].abs().max()


class TestSosfiltBatched:
    """Test batched filtering."""

    def test_batch_dimension(self) -> None:
        """Should support batch dimension in input."""
        sos = butterworth_design(4, 0.2)

        # Batched input: (batch, time)
        x = torch.randn(5, 1000)
        y = sosfilt(sos, x)

        assert y.shape == x.shape

    def test_batch_matches_loop(self) -> None:
        """Batched should match filtering each signal separately."""
        sos = butterworth_design(4, 0.2)

        x = torch.randn(3, 500)
        y_batched = sosfilt(sos, x)

        for i in range(3):
            y_single = sosfilt(sos, x[i])
            torch.testing.assert_close(y_batched[i], y_single)


class TestSosfiltDim:
    """Test dim parameter."""

    def test_filter_along_dim0(self) -> None:
        """Should filter along specified dimension."""
        sos = butterworth_design(4, 0.2)

        x = torch.randn(1000, 5)  # Time is dim 0
        y = sosfilt(sos, x, dim=0)

        assert y.shape == x.shape


class TestSosfiltGradients:
    """Test gradient support."""

    def test_gradient_wrt_input(self) -> None:
        """Should have gradient w.r.t. input signal."""
        sos = butterworth_design(4, 0.2)

        x = torch.randn(100, requires_grad=True)
        y = sosfilt(sos, x)

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_wrt_sos(self) -> None:
        """Should have gradient w.r.t. SOS coefficients."""
        sos = butterworth_design(4, 0.2)
        sos = sos.clone().requires_grad_(True)

        x = torch.randn(100)
        y = sosfilt(sos, x)

        loss = y.sum()
        loss.backward()

        assert sos.grad is not None
        assert not torch.isnan(sos.grad).any()
