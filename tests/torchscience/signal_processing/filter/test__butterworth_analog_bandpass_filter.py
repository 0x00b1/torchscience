"""Tests for butterworth_analog_bandpass_filter."""

import math

import pytest
import torch
import torch.testing

import torchscience.signal_processing.filter


class TestButterworthAnalogBandpassFilter:
    """Tests for the Butterworth analog bandpass filter."""

    def test_signature_1_basic(self):
        """Test basic signature 1: n, (omega_p1, omega_p2)."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            4, (0.2, 0.5)
        )
        assert sos.shape == (4, 6)
        assert sos.dtype == torch.float32

    def test_signature_1_different_orders(self):
        """Test different filter orders."""
        for n in [1, 2, 3, 4, 6, 8, 10]:
            sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
                n, (0.2, 0.5)
            )
            assert sos.shape == (n, 6), f"Failed for order {n}"

    def test_signature_2_center_q(self):
        """Test signature 2: n, ((omega, q),)."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            2, ((0.35, 3.0),)
        )
        assert sos.shape == (2, 6)

    def test_signature_3_full_spec(self):
        """Test signature 3: full specification."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            (0.1, 0.2, 0.5, 0.7), (40.0, 1.0)
        )
        assert sos.ndim == 2
        assert sos.shape[1] == 6

    def test_sos_structure_denominator_normalized(self):
        """Test that a0 (coefficient 3) is 1 (normalized)."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.4)
        )
        torch.testing.assert_close(
            sos[:, 3], torch.ones(2), rtol=1e-5, atol=1e-5
        )

    def test_coefficients_finite(self):
        """Test that all coefficients are finite."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            4, (0.2, 0.5)
        )
        assert torch.isfinite(sos).all()

    def test_center_frequency_unity_gain(self):
        """Test unity gain at center frequency omega_0."""
        omega_p1, omega_p2 = 0.2, 0.5
        omega_0 = math.sqrt(omega_p1 * omega_p2)

        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            4, (omega_p1, omega_p2), dtype=torch.float64
        )

        s = 1j * omega_0
        H = 1.0
        for section in sos:
            b0, b1, b2, a0, a1, a2 = section.numpy()
            num = b0 * s**2 + b1 * s + b2
            den = a0 * s**2 + a1 * s + a2
            H *= num / den

        assert abs(abs(H) - 1.0) < 0.1, f"Gain at center frequency: {abs(H)}"

    def test_order_1(self):
        """Test minimum order filter."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            1, (0.2, 0.5)
        )
        assert sos.shape == (1, 6)
        assert torch.isfinite(sos).all()

    def test_invalid_order_zero(self):
        """Test error for order = 0."""
        with pytest.raises(RuntimeError, match="order n must be positive"):
            torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
                0, (0.2, 0.5)
            )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test explicit dtype specification."""
        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.5), dtype=dtype
        )
        assert sos.dtype == dtype

    def test_batched_omega_p1(self):
        """Test batched omega_p1 input."""
        omega_p1 = torch.tensor([0.1, 0.15, 0.2])
        omega_p2 = 0.5

        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            3, (omega_p1, omega_p2)
        )
        assert sos.shape == (3, 3, 6)

    def test_gradient_omega_p1(self):
        """Test gradient flow through omega_p1."""
        omega_p1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, dtype=torch.float64)

        sos = torchscience.signal_processing.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )

        loss = sos.sum()
        loss.backward()

        assert omega_p1.grad is not None
        assert torch.isfinite(omega_p1.grad)
