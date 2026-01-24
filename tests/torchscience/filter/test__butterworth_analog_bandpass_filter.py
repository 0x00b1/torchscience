"""Tests for butterworth_analog_bandpass_filter."""

import math

import pytest
import torch
import torch.testing

import torchscience.filter


class TestButterworthAnalogBandpassFilter:
    """Tests for the Butterworth analog bandpass filter."""

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================

    def test_signature_1_basic(self):
        """Test basic signature 1: n, (omega_p1, omega_p2)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (0.2, 0.5)
        )
        assert sos.shape == (4, 6)
        assert sos.dtype == torch.float32

    def test_signature_1_different_orders(self):
        """Test different filter orders."""
        for n in [1, 2, 3, 4, 6, 8, 10]:
            sos = torchscience.filter.butterworth_analog_bandpass_filter(
                n, (0.2, 0.5)
            )
            assert sos.shape == (n, 6), f"Failed for order {n}"

    def test_signature_2_center_q(self):
        """Test signature 2: n, ((omega, q),)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, ((0.35, 3.0),)
        )
        assert sos.shape == (2, 6)

    def test_signature_3_full_spec(self):
        """Test signature 3: full specification."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            (0.1, 0.2, 0.5, 0.7), (40.0, 1.0)
        )
        # Order is computed automatically, should have at least 1 section
        assert sos.ndim == 2
        assert sos.shape[1] == 6

    # =========================================================================
    # SOS Format Validation Tests
    # =========================================================================

    def test_sos_structure_denominator_normalized(self):
        """Test that a0 (coefficient 3) is 1 (normalized)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.4)
        )
        # a0 should be 1 for normalized SOS
        torch.testing.assert_close(
            sos[:, 3], torch.ones(2), rtol=1e-5, atol=1e-5
        )

    def test_sos_numerator_bandpass_zeros(self):
        """Test that b1 and b2 are 0 (zeros at origin for bandpass)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            3, (0.2, 0.5)
        )
        # For analog bandpass: numerator is s^2, so b1 = b2 = 0
        torch.testing.assert_close(
            sos[:, 1], torch.zeros(3), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            sos[:, 2], torch.zeros(3), rtol=1e-5, atol=1e-5
        )

    def test_coefficients_finite(self):
        """Test that all coefficients are finite."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (0.2, 0.5)
        )
        assert torch.isfinite(sos).all()

    # =========================================================================
    # Mathematical Correctness Tests
    # =========================================================================

    def test_center_frequency_unity_gain(self):
        """Test unity gain at center frequency omega_0."""
        omega_p1, omega_p2 = 0.2, 0.5
        omega_0 = math.sqrt(omega_p1 * omega_p2)

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (omega_p1, omega_p2), dtype=torch.float64
        )

        # Compute transfer function at s = j*omega_0
        s = 1j * omega_0
        H = 1.0
        for section in sos:
            b0, b1, b2, a0, a1, a2 = section.numpy()
            num = b0 * s**2 + b1 * s + b2
            den = a0 * s**2 + a1 * s + a2
            H *= num / den

        # Magnitude should be approximately 1 at center frequency
        assert abs(abs(H) - 1.0) < 0.1, f"Gain at center frequency: {abs(H)}"

    def test_q_factor_bandwidth_relationship(self):
        """Test Q = omega_0 / bandwidth relationship."""
        omega = 0.35
        q = 5.0
        n = 2

        # Compute expected passband from Q
        B = omega / q
        omega_p1 = -B / 2 + math.sqrt(B**2 / 4 + omega**2)
        omega_p2 = omega_p1 + B

        # Get SOS from both signatures
        sos_center_q = torchscience.filter.butterworth_analog_bandpass_filter(
            n, ((omega, q),), dtype=torch.float64
        )
        sos_passband = torchscience.filter.butterworth_analog_bandpass_filter(
            n, (omega_p1, omega_p2), dtype=torch.float64
        )

        torch.testing.assert_close(
            sos_center_q, sos_passband, rtol=1e-5, atol=1e-10
        )

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_order_1(self):
        """Test minimum order filter."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            1, (0.2, 0.5)
        )
        assert sos.shape == (1, 6)
        assert torch.isfinite(sos).all()

    def test_narrow_passband(self):
        """Test with very narrow passband (high Q)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (0.49, 0.51)
        )
        assert torch.isfinite(sos).all()

    def test_wide_passband(self):
        """Test with wide passband (low Q)."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (0.1, 0.9)
        )
        assert torch.isfinite(sos).all()

    # =========================================================================
    # Error Handling
    # =========================================================================

    def test_invalid_order_zero(self):
        """Test error for order = 0."""
        with pytest.raises(RuntimeError, match="order n must be positive"):
            torchscience.filter.butterworth_analog_bandpass_filter(
                0, (0.2, 0.5)
            )

    def test_invalid_order_negative(self):
        """Test error for negative order."""
        with pytest.raises(RuntimeError, match="order n must be positive"):
            torchscience.filter.butterworth_analog_bandpass_filter(
                -1, (0.2, 0.5)
            )

    def test_invalid_first_argument_type(self):
        """Test error for invalid first argument type."""
        with pytest.raises(TypeError):
            torchscience.filter.butterworth_analog_bandpass_filter(
                "invalid",
                (0.2, 0.5),  # type: ignore
            )

    def test_invalid_frequency_spec(self):
        """Test error for invalid frequency specification."""
        with pytest.raises(
            ValueError, match="Invalid frequency specification"
        ):
            torchscience.filter.butterworth_analog_bandpass_filter(
                4,
                (0.2, 0.5, 0.7),  # type: ignore
            )

    # =========================================================================
    # Dtype Tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test explicit dtype specification."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.5), dtype=dtype
        )
        assert sos.dtype == dtype

    # =========================================================================
    # Batching Tests
    # =========================================================================

    def test_batched_omega_p1(self):
        """Test batched omega_p1 input."""
        omega_p1 = torch.tensor([0.1, 0.15, 0.2])
        omega_p2 = 0.5

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            3, (omega_p1, omega_p2)
        )
        assert sos.shape == (3, 3, 6)  # (batch=3, sections=3, coeffs=6)

    def test_batched_omega_p2(self):
        """Test batched omega_p2 input."""
        omega_p1 = 0.2
        omega_p2 = torch.tensor([0.4, 0.5, 0.6])

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )
        assert sos.shape == (3, 2, 6)

    def test_batched_both_frequencies(self):
        """Test both frequencies batched."""
        omega_p1 = torch.tensor([0.1, 0.2])
        omega_p2 = torch.tensor([0.4, 0.5])

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (omega_p1, omega_p2)
        )
        assert sos.shape == (2, 4, 6)

    def test_batched_broadcasting(self):
        """Test broadcasting between omega_p1 and omega_p2."""
        omega_p1 = torch.tensor([[0.1], [0.2]])  # (2, 1)
        omega_p2 = torch.tensor([0.4, 0.5, 0.6])  # (3,)

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )
        assert sos.shape == (2, 3, 2, 6)  # (2, 3, sections=2, coeffs=6)

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_omega_p1(self):
        """Test gradient flow through omega_p1."""
        omega_p1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, dtype=torch.float64)

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )

        # Compute a scalar loss
        loss = sos.sum()
        loss.backward()

        assert omega_p1.grad is not None
        assert torch.isfinite(omega_p1.grad)

    def test_gradient_omega_p2(self):
        """Test gradient flow through omega_p2."""
        omega_p1 = torch.tensor(0.2, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )

        loss = sos.sum()
        loss.backward()

        assert omega_p2.grad is not None
        assert torch.isfinite(omega_p2.grad)

    def test_gradient_both_frequencies(self):
        """Test gradient flow through both frequencies."""
        omega_p1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            3, (omega_p1, omega_p2)
        )

        loss = sos.sum()
        loss.backward()

        assert omega_p1.grad is not None
        assert omega_p2.grad is not None
        assert torch.isfinite(omega_p1.grad)
        assert torch.isfinite(omega_p2.grad)

    def test_gradient_batched(self):
        """Test gradient flow with batched inputs."""
        omega_p1 = torch.tensor(
            [0.1, 0.2, 0.3], requires_grad=True, dtype=torch.float64
        )
        omega_p2 = torch.tensor(
            [0.5, 0.6, 0.7], requires_grad=True, dtype=torch.float64
        )

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )

        loss = sos.sum()
        loss.backward()

        assert omega_p1.grad is not None
        assert omega_p2.grad is not None
        assert omega_p1.grad.shape == omega_p1.shape
        assert omega_p2.grad.shape == omega_p2.shape

    # =========================================================================
    # Gradient Correctness Tests (gradcheck)
    # =========================================================================

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_gradcheck_scalar_inputs(self, n):
        """Test gradient correctness with torch.autograd.gradcheck for scalar inputs."""
        omega_p1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)

        def fn(w1, w2):
            return torchscience.filter.butterworth_analog_bandpass_filter(
                n, (w1, w2)
            )

        # Use gradcheck to verify analytical gradients match numerical gradients
        # eps=1e-5 works better for this function due to complex chain rule through sqrt
        # rtol=0.03 and atol=1e-3 account for complex arithmetic precision and small values
        assert torch.autograd.gradcheck(
            fn, (omega_p1, omega_p2), eps=1e-5, atol=1e-3, rtol=0.03
        )

    def test_gradcheck_different_frequency_ranges(self):
        """Test gradient correctness across different frequency ranges."""
        test_cases = [
            (0.2, 0.5),  # Standard case
            (0.3, 0.7),  # Mid frequencies
            (0.4, 0.6),  # Narrow band
        ]

        for omega_p1_val, omega_p2_val in test_cases:
            omega_p1 = torch.tensor(
                omega_p1_val, requires_grad=True, dtype=torch.float64
            )
            omega_p2 = torch.tensor(
                omega_p2_val, requires_grad=True, dtype=torch.float64
            )

            def fn(w1, w2):
                return torchscience.filter.butterworth_analog_bandpass_filter(
                    2, (w1, w2)
                )

            assert torch.autograd.gradcheck(
                fn, (omega_p1, omega_p2), eps=1e-5, atol=1e-3, rtol=0.03
            ), (
                f"gradcheck failed for omega_p1={omega_p1_val}, omega_p2={omega_p2_val}"
            )

    def test_gradcheck_batched_inputs(self):
        """Test gradient correctness with batched inputs."""
        omega_p1 = torch.tensor(
            [0.2, 0.3], requires_grad=True, dtype=torch.float64
        )
        omega_p2 = torch.tensor(
            [0.5, 0.6], requires_grad=True, dtype=torch.float64
        )

        def fn(w1, w2):
            return torchscience.filter.butterworth_analog_bandpass_filter(
                2, (w1, w2)
            )

        assert torch.autograd.gradcheck(
            fn, (omega_p1, omega_p2), eps=1e-5, atol=1e-3, rtol=0.03
        )

    def test_gradcheck_only_omega_p1(self):
        """Test gradient correctness when only omega_p1 requires grad."""
        omega_p1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, requires_grad=False, dtype=torch.float64)

        def fn(w1):
            return torchscience.filter.butterworth_analog_bandpass_filter(
                3, (w1, omega_p2)
            )

        assert torch.autograd.gradcheck(
            fn, (omega_p1,), eps=1e-5, atol=1e-3, rtol=0.03
        )

    def test_gradcheck_only_omega_p2(self):
        """Test gradient correctness when only omega_p2 requires grad."""
        omega_p1 = torch.tensor(0.2, requires_grad=False, dtype=torch.float64)
        omega_p2 = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)

        def fn(w2):
            return torchscience.filter.butterworth_analog_bandpass_filter(
                3, (omega_p1, w2)
            )

        assert torch.autograd.gradcheck(
            fn, (omega_p2,), eps=1e-5, atol=1e-3, rtol=0.03
        )

    # =========================================================================
    # torch.compile Compatibility Tests
    # =========================================================================

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_basic(self):
        """Test torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.filter.butterworth_analog_bandpass_filter
        )

        result = compiled_fn(4, (0.2, 0.5))
        expected = torchscience.filter.butterworth_analog_bandpass_filter(
            4, (0.2, 0.5)
        )

        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Device Tests
    # =========================================================================

    def test_cpu_device(self):
        """Test CPU device."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.5), device=torch.device("cpu")
        )
        assert sos.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (0.2, 0.5), device=torch.device("cuda")
        )
        assert sos.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_input_tensors(self):
        """Test with CUDA input tensors."""
        omega_p1 = torch.tensor(0.2, device="cuda")
        omega_p2 = torch.tensor(0.5, device="cuda")

        sos = torchscience.filter.butterworth_analog_bandpass_filter(
            2, (omega_p1, omega_p2)
        )
        assert sos.device.type == "cuda"
