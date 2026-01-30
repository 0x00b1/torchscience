import math

import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestIncompleteLegendreEllipticIntegralF:
    """Tests for the incomplete elliptic integral of the first kind F(phi, m)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_value_phi_zero(self):
        """Test F(0, m) = 0 for any m."""
        phi = torch.tensor([0.0], dtype=torch.float64)
        m_values = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.99], dtype=torch.float64
        )
        for m in m_values:
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                phi, m.unsqueeze(0)
            )
            assert result.item() == pytest.approx(0.0, abs=1e-12)

    def test_special_value_m_zero(self):
        """Test F(phi, 0) = phi."""
        phi_values = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        for phi in phi_values:
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                phi.unsqueeze(0), m
            )
            assert result.item() == pytest.approx(phi.item(), rel=1e-10)

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.ellipkinc."""
        phi = torch.tensor(
            [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, math.pi / 2 - 0.01],
            dtype=torch.float64,
        )
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        expected = torch.tensor(
            [scipy.special.ellipkinc(p.item(), m.item()) for p in phi],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_scipy_agreement_various_m(self):
        """Test agreement with scipy for various m values."""
        phi = torch.tensor([1.0], dtype=torch.float64)
        m_values = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        for m in m_values:
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                phi, m.unsqueeze(0)
            )
            expected = scipy.special.ellipkinc(phi.item(), m.item())
            assert result.item() == pytest.approx(expected, rel=1e-8)

    def test_complete_integral_at_pi_over_2(self):
        """Test that F(pi/2, m) equals K(m)."""
        phi = torch.tensor([math.pi / 2], dtype=torch.float64)
        m_values = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        for m in m_values:
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                phi, m.unsqueeze(0)
            )
            # K(m) from scipy is ellipk(m)
            expected = scipy.special.ellipk(m.item())
            assert result.item() == pytest.approx(expected, rel=1e-6)

    def test_odd_function_in_phi(self):
        """Test F(-phi, m) = -F(phi, m)."""
        phi = torch.tensor([0.1, 0.5, 1.0, 1.3], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result_pos = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        result_neg = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            -phi, m
        )
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-10, atol=1e-10
        )

    def test_nan_input(self):
        """Test NaN handling."""
        phi = torch.tensor([float("nan")], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.isnan()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        phi = torch.tensor(
            [0.3, 0.5, 0.8, 1.0], dtype=torch.float64, requires_grad=True
        )
        m = torch.tensor(
            [0.3, 0.4, 0.5, 0.6], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.incomplete_legendre_elliptic_integral_f,
            (phi, m),
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        phi = torch.tensor(
            [0.3, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )
        m = torch.tensor(
            [0.3, 0.4, 0.5], dtype=torch.float64, requires_grad=True
        )
        # Use relaxed tolerance due to finite difference approximation in backward_backward
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.incomplete_legendre_elliptic_integral_f,
            (phi, m),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradient_phi_identity(self):
        """Verify dF/dphi = 1/sqrt(1 - m*sin^2(phi)) numerically."""
        phi = torch.tensor(
            [0.3, 0.5, 0.8, 1.0], dtype=torch.float64, requires_grad=True
        )
        m = torch.tensor([0.5], dtype=torch.float64)
        y = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        grad = torch.autograd.grad(y.sum(), phi)[0]
        # Expected: 1/sqrt(1 - m*sin^2(phi))
        sin_phi = torch.sin(phi.detach())
        expected = 1.0 / torch.sqrt(1.0 - m * sin_phi * sin_phi)
        torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-6)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_dtype(self):
        """Test complex tensor support."""
        phi = torch.randn(5, dtype=torch.complex128)
        m = torch.tensor([0.5 + 0j], dtype=torch.complex128)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.dtype == torch.complex128

    def test_complex_on_real_axis_matches_real(self):
        """Test complex numbers on real axis match real F."""
        phi_real = torch.tensor([0.3, 0.5, 0.8, 1.0], dtype=torch.float64)
        m_real = torch.tensor([0.5], dtype=torch.float64)
        phi_complex = phi_real.to(torch.complex128)
        m_complex = m_real.to(torch.complex128)
        result_real = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi_real, m_real
        )
        result_complex = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi_complex, m_complex
        )
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        phi = torch.randn(10, device="meta")
        m = torch.randn(10, device="meta")
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.shape == phi.shape
        assert result.device == phi.device

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        phi = torch.randn(10, dtype=torch.float32)
        m = torch.tensor([0.5], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                phi, m
            )
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        phi = torch.randn(5, 10, dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torch.vmap(
            lambda p: torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
                p, m
            )
        )(phi)
        expected = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.incomplete_legendre_elliptic_integral_f
        )
        phi = torch.randn(100, dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = compiled_fn(phi, m)
        expected = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        torch.testing.assert_close(result, expected)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        phi = torch.randn(3, 1, dtype=torch.float64)
        m = torch.tensor([[0.1, 0.3, 0.5, 0.7]], dtype=torch.float64)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.shape == (3, 4)

    # =========================================================================
    # dtype tests
    # =========================================================================

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        phi = torch.tensor([0.5, 1.0], dtype=dtype)
        m = torch.tensor([0.5], dtype=dtype)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        phi = torch.tensor([0.5 + 0.1j, 1.0 - 0.1j], dtype=dtype)
        m = torch.tensor([0.5 + 0j], dtype=dtype)
        result = torchscience.special_functions.incomplete_legendre_elliptic_integral_f(
            phi, m
        )
        assert result.dtype == dtype
