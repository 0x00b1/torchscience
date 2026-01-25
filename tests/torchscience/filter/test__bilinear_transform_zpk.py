"""Tests for bilinear transform functions."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.filter import (
    bilinear_transform_zpk,
    butterworth_prototype,
)


class TestBilinearTransformZpk:
    """Tests for bilinear_transform_zpk (analog to digital transform)."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("sampling_frequency", [1.0, 2.0, 8000.0])
    def test_matches_scipy(
        self, order: int, sampling_frequency: float
    ) -> None:
        """Should match scipy.signal.bilinear_zpk."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        zeros_d, poles_d, gain_d = bilinear_transform_zpk(
            zeros, poles, gain, sampling_frequency=sampling_frequency
        )

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.bilinear_zpk(
            zeros.numpy(), poles.numpy(), gain.item(), fs=sampling_frequency
        )

        # Digital filter has same number of poles
        assert poles_d.numel() == order

        # Compare poles (sort for consistent ordering)
        p_d_sorted = sorted(poles_d.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p_d_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10, (
                f"Pole mismatch: {p_ts} vs {p_ref}"
            )

        # Compare gain
        assert abs(gain_d.item() - k_sp) < 1e-10

    def test_poles_inside_unit_circle(self) -> None:
        """Digital filter poles should be inside unit circle (stable)."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros_d, poles_d, gain_d = bilinear_transform_zpk(
            zeros, poles, gain, sampling_frequency=2.0
        )

        for pole in poles_d:
            assert abs(pole) < 1.0 + 1e-10, f"Pole {pole} outside unit circle"

    def test_adds_zeros_at_nyquist(self) -> None:
        """Bilinear transform adds zeros at z=-1 (Nyquist)."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros_d, poles_d, gain_d = bilinear_transform_zpk(
            zeros, poles, gain, sampling_frequency=2.0
        )

        # Should have n zeros at z=-1
        assert zeros_d.numel() == poles_d.numel()
        for zero in zeros_d:
            assert abs(zero + 1.0) < 1e-10, f"Zero {zero} not at -1"


class TestBilinearTransformZpkGradients:
    """Test gradients for bilinear_transform_zpk."""

    def test_gradient_wrt_poles(self) -> None:
        """Should have gradient w.r.t. analog poles."""
        # Create simple analog filter with differentiable poles
        poles = torch.tensor(
            [-1.0 + 0.5j, -1.0 - 0.5j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        zeros = torch.empty(0, dtype=poles.dtype)
        gain = torch.tensor(1.0, dtype=torch.float64)

        zeros_d, poles_d, gain_d = bilinear_transform_zpk(
            zeros, poles, gain, sampling_frequency=2.0
        )

        # Compute loss and backward
        loss = poles_d.abs().sum()
        loss.backward()

        assert poles.grad is not None
        assert not torch.isnan(poles.grad).any()
