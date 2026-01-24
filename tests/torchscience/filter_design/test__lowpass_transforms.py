"""Tests for frequency transform functions."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.filter_design import (
    butterworth_prototype,
    lowpass_to_bandpass_zpk,
    lowpass_to_bandstop_zpk,
    lowpass_to_highpass_zpk,
    lowpass_to_lowpass_zpk,
)


class TestLowpassToLowpassZpk:
    """Tests for lowpass_to_lowpass_zpk (lowpass to lowpass frequency scaling)."""

    def test_identity_transform(self) -> None:
        """cutoff_frequency=1.0 should not change the filter."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros2, poles2, gain2 = lowpass_to_lowpass_zpk(
            zeros, poles, gain, cutoff_frequency=1.0
        )

        torch.testing.assert_close(poles2, poles)
        torch.testing.assert_close(gain2, gain)

    def test_frequency_scaling(self) -> None:
        """Poles should scale by cutoff_frequency."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        cutoff_frequency = 2.0
        zeros2, poles2, gain2 = lowpass_to_lowpass_zpk(
            zeros, poles, gain, cutoff_frequency=cutoff_frequency
        )

        # Poles should be scaled by cutoff_frequency
        torch.testing.assert_close(poles2, poles * cutoff_frequency)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("cutoff_frequency", [0.5, 1.0, 2.0, 10.0])
    def test_matches_scipy(self, order: int, cutoff_frequency: float) -> None:
        """Should match scipy.signal.lp2lp_zpk."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        zeros2, poles2, gain2 = lowpass_to_lowpass_zpk(
            zeros, poles, gain, cutoff_frequency=cutoff_frequency
        )

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2lp_zpk(
            zeros.numpy(), poles.numpy(), gain.item(), wo=cutoff_frequency
        )

        # Compare (sort poles for consistent ordering)
        p2_sorted = sorted(poles2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10

        assert abs(gain2.item() - k_sp) < 1e-10


class TestLowpassToLowpassZpkGradients:
    """Test gradients for lowpass_to_lowpass_zpk."""

    def test_gradient_wrt_cutoff_frequency(self) -> None:
        """Should have gradient w.r.t. cutoff_frequency."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        cutoff_frequency = torch.tensor(2.0, requires_grad=True)
        zeros2, poles2, gain2 = lowpass_to_lowpass_zpk(
            zeros, poles, gain, cutoff_frequency=cutoff_frequency
        )

        # Compute loss and backward
        loss = poles2.abs().sum() + gain2
        loss.backward()

        assert cutoff_frequency.grad is not None
        assert not torch.isnan(cutoff_frequency.grad)


class TestLowpassToHighpassZpk:
    """Tests for lowpass_to_highpass_zpk (lowpass to highpass transform)."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("cutoff_frequency", [0.5, 1.0, 2.0, 10.0])
    def test_matches_scipy(self, order: int, cutoff_frequency: float) -> None:
        """Should match scipy.signal.lp2hp_zpk."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        zeros2, poles2, gain2 = lowpass_to_highpass_zpk(
            zeros, poles, gain, cutoff_frequency=cutoff_frequency
        )

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2hp_zpk(
            zeros.numpy(), poles.numpy(), gain.item(), wo=cutoff_frequency
        )

        # Compare zeros (highpass has zeros at origin)
        assert zeros2.numel() == len(z_sp)

        # Compare poles (sort for consistent ordering)
        p2_sorted = sorted(poles2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10

        assert abs(gain2.item() - k_sp) < 1e-10

    def test_adds_zeros_at_origin(self) -> None:
        """Highpass transform adds zeros at s=0."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros2, poles2, gain2 = lowpass_to_highpass_zpk(
            zeros, poles, gain, cutoff_frequency=1.0
        )

        # Should have n zeros at origin (degree difference)
        assert zeros2.numel() == poles.numel()
        for zero in zeros2:
            assert abs(zero) < 1e-10


class TestLowpassToHighpassZpkGradients:
    """Test gradients for lowpass_to_highpass_zpk."""

    def test_gradient_wrt_cutoff_frequency(self) -> None:
        """Should have gradient w.r.t. cutoff_frequency."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        cutoff_frequency = torch.tensor(2.0, requires_grad=True)
        zeros2, poles2, gain2 = lowpass_to_highpass_zpk(
            zeros, poles, gain, cutoff_frequency=cutoff_frequency
        )

        # Compute loss and backward
        loss = poles2.abs().sum() + gain2
        loss.backward()

        assert cutoff_frequency.grad is not None
        assert not torch.isnan(cutoff_frequency.grad)


class TestLowpassToBandpassZpk:
    """Tests for lowpass_to_bandpass_zpk (lowpass to bandpass transform)."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    @pytest.mark.parametrize("center_frequency", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
    def test_matches_scipy(
        self, order: int, center_frequency: float, bandwidth: float
    ) -> None:
        """Should match scipy.signal.lp2bp_zpk."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        zeros2, poles2, gain2 = lowpass_to_bandpass_zpk(
            zeros,
            poles,
            gain,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
        )

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2bp_zpk(
            zeros.numpy(),
            poles.numpy(),
            gain.item(),
            wo=center_frequency,
            bw=bandwidth,
        )

        # Bandpass doubles the order
        assert poles2.numel() == 2 * order
        assert len(p_sp) == 2 * order

        # Compare poles (sort for consistent ordering)
        p2_sorted = sorted(poles2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10, (
                f"Pole mismatch: {p_ts} vs {p_ref}"
            )

        assert abs(gain2.item() - k_sp) < 1e-10

    def test_doubles_filter_order(self) -> None:
        """Bandpass transform doubles the number of poles."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros2, poles2, gain2 = lowpass_to_bandpass_zpk(
            zeros, poles, gain, center_frequency=1.0, bandwidth=0.5
        )

        assert poles2.numel() == 2 * poles.numel()

    def test_adds_zeros_at_origin(self) -> None:
        """Bandpass transform adds zeros at s=0."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros2, poles2, gain2 = lowpass_to_bandpass_zpk(
            zeros, poles, gain, center_frequency=1.0, bandwidth=0.5
        )

        # Should have n zeros at origin
        assert zeros2.numel() == poles.numel()
        for zero in zeros2:
            assert abs(zero) < 1e-10


class TestLowpassToBandpassZpkGradients:
    """Test gradients for lowpass_to_bandpass_zpk."""

    def test_gradient_wrt_center_frequency_and_bandwidth(self) -> None:
        """Should have gradient w.r.t. center_frequency and bandwidth."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        center_frequency = torch.tensor(2.0, requires_grad=True)
        bandwidth = torch.tensor(0.5, requires_grad=True)
        zeros2, poles2, gain2 = lowpass_to_bandpass_zpk(
            zeros,
            poles,
            gain,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
        )

        # Compute loss and backward
        loss = poles2.abs().sum() + gain2
        loss.backward()

        assert center_frequency.grad is not None
        assert bandwidth.grad is not None
        assert not torch.isnan(center_frequency.grad)
        assert not torch.isnan(bandwidth.grad)


class TestLowpassToBandstopZpk:
    """Tests for lowpass_to_bandstop_zpk (lowpass to bandstop transform)."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    @pytest.mark.parametrize("center_frequency", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
    def test_matches_scipy(
        self, order: int, center_frequency: float, bandwidth: float
    ) -> None:
        """Should match scipy.signal.lp2bs_zpk."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        zeros2, poles2, gain2 = lowpass_to_bandstop_zpk(
            zeros,
            poles,
            gain,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
        )

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2bs_zpk(
            zeros.numpy(),
            poles.numpy(),
            gain.item(),
            wo=center_frequency,
            bw=bandwidth,
        )

        # Bandstop doubles the order
        assert poles2.numel() == 2 * order
        assert len(p_sp) == 2 * order

        # Compare poles (sort for consistent ordering)
        p2_sorted = sorted(poles2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10, (
                f"Pole mismatch: {p_ts} vs {p_ref}"
            )

        # Compare zeros (sort for consistent ordering)
        z2_sorted = sorted(zeros2.numpy(), key=lambda x: (x.real, x.imag))
        z_sp_sorted = sorted(z_sp, key=lambda x: (x.real, x.imag))

        for z_ts, z_ref in zip(z2_sorted, z_sp_sorted):
            assert abs(z_ts - z_ref) < 1e-10, (
                f"Zero mismatch: {z_ts} vs {z_ref}"
            )

        assert abs(gain2.item() - k_sp) < 1e-10

    def test_doubles_filter_order(self) -> None:
        """Bandstop transform doubles the number of poles."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        zeros2, poles2, gain2 = lowpass_to_bandstop_zpk(
            zeros, poles, gain, center_frequency=1.0, bandwidth=0.5
        )

        assert poles2.numel() == 2 * poles.numel()

    def test_adds_zeros_on_imaginary_axis(self) -> None:
        """Bandstop transform adds zeros at ±j*center_frequency."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)
        center_frequency = 2.0
        zeros2, poles2, gain2 = lowpass_to_bandstop_zpk(
            zeros,
            poles,
            gain,
            center_frequency=center_frequency,
            bandwidth=0.5,
        )

        # Should have 2n zeros (n pairs at ±j*center_frequency)
        assert zeros2.numel() == 2 * poles.numel()

        # All zeros should be on imaginary axis at ±center_frequency
        for zero in zeros2:
            assert abs(zero.real) < 1e-10
            assert abs(abs(zero.imag) - center_frequency) < 1e-10


class TestLowpassToBandstopZpkGradients:
    """Test gradients for lowpass_to_bandstop_zpk."""

    def test_gradient_wrt_center_frequency_and_bandwidth(self) -> None:
        """Should have gradient w.r.t. center_frequency and bandwidth."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        center_frequency = torch.tensor(2.0, requires_grad=True)
        bandwidth = torch.tensor(0.5, requires_grad=True)
        zeros2, poles2, gain2 = lowpass_to_bandstop_zpk(
            zeros,
            poles,
            gain,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
        )

        # Compute loss and backward
        loss = poles2.abs().sum() + zeros2.abs().sum() + gain2
        loss.backward()

        assert center_frequency.grad is not None
        assert bandwidth.grad is not None
        assert not torch.isnan(center_frequency.grad)
        assert not torch.isnan(bandwidth.grad)
