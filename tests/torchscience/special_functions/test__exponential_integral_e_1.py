import torch
import torch.testing

import torchscience.special_functions


class TestExponentialIntegralE1:
    """Tests for the exponential integral E_1 function."""

    # =========================================================================
    # Basic forward tests
    # =========================================================================

    def test_forward_positive_values(self):
        """Test E_1 at positive values against known scipy values."""
        # Reference values computed from scipy.special.exp1
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        expected = torch.tensor(
            [
                0.5597735947761608,  # E_1(0.5)
                0.21938393439552062,  # E_1(1.0)
                0.04890051070806112,  # E_1(2.0)
                0.001148295591784439,  # E_1(5.0)
                4.156968929685324e-06,  # E_1(10.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_e_1(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-12)

    def test_forward_small_positive_values(self):
        """Test E_1 at small positive values."""
        x = torch.tensor([0.01, 0.1, 0.2, 0.3], dtype=torch.float64)
        # Reference values from scipy.special.exp1
        expected = torch.tensor(
            [
                4.037929575460108,  # E_1(0.01)
                1.8229239584193906,  # E_1(0.1)
                1.2226505356612035,  # E_1(0.2)
                0.9056768324566622,  # E_1(0.3)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_e_1(x)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_forward_large_positive_values(self):
        """Test E_1 at large positive values (asymptotic region)."""
        x = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        # Should be finite, positive, and very small
        assert torch.isfinite(result).all()
        assert (result > 0).all()
        assert (result < 1e-6).all()
        # Check monotonically decreasing
        assert (result[1:] < result[:-1]).all()

    # =========================================================================
    # Special value tests
    # =========================================================================

    def test_special_value_zero(self):
        """Test E_1(0) = +inf."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert torch.isinf(result).all()
        assert (result > 0).all()

    def test_special_value_positive_inf(self):
        """Test E_1(+inf) = 0."""
        x = torch.tensor([float("inf")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert (result == 0).all()

    def test_special_value_negative_returns_nan(self):
        """Test E_1(x < 0) = nan for real inputs."""
        x = torch.tensor([-1.0, -5.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert torch.isnan(result).all()

    def test_special_value_nan(self):
        """Test E_1(nan) = nan."""
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert torch.isnan(result).all()

    # =========================================================================
    # Relation to Ei tests
    # =========================================================================

    def test_relation_to_ei(self):
        """Test E_1(x) = -Ei(-x) for positive x."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)

        e_1 = torchscience.special_functions.exponential_integral_e_1(x)
        ei_neg_x = torchscience.special_functions.exponential_integral_ei(-x)

        # E_1(x) = -Ei(-x)
        torch.testing.assert_close(e_1, -ei_neg_x, rtol=1e-10, atol=1e-10)

    def test_relation_to_ei_at_boundary(self):
        """Test E_1(x) = -Ei(-x) near algorithm boundaries."""
        # Test near x = 1 (series/CF boundary for E_1)
        x = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)

        e_1 = torchscience.special_functions.exponential_integral_e_1(x)
        ei_neg_x = torchscience.special_functions.exponential_integral_ei(-x)

        torch.testing.assert_close(e_1, -ei_neg_x, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_formula(self):
        """Test gradient d/dx E_1(x) = -e^{-x} / x."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_e_1(x)
        y.sum().backward()

        expected_grad = -torch.exp(-x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_e_1(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradient_at_zero_is_nan(self):
        """Test that gradient at x=0 returns NaN."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.exponential_integral_e_1(x)
        y.backward()
        assert torch.isnan(x.grad).all()

    # =========================================================================
    # Second-order gradient tests
    # =========================================================================

    def test_second_derivative_formula(self):
        """Test d^2/dx^2 E_1(x) = e^{-x} * (x + 1) / x^2."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_e_1(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        expected = (
            torch.exp(-x.detach()) * (x.detach() + 1) / (x.detach() ** 2)
        )
        torch.testing.assert_close(grad2, expected, rtol=1e-6, atol=1e-8)

    def test_gradgradcheck(self):
        """Test second-order gradients with torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_e_1(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real implementation."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.exponential_integral_e_1(
            x_real
        )
        result_complex = (
            torchscience.special_functions.exponential_integral_e_1(x_complex)
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        # Imaginary part should be zero for positive real input
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_general_values(self):
        """Test E_1 at general complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 2.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.exponential_integral_e_1(z)
        # Just verify we get finite results
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_complex_negative_real_axis(self):
        """Test E_1 on negative real axis with small imaginary part."""
        # Complex extension allows evaluation near negative real axis
        z = torch.tensor([-1.0 + 0.01j, -2.0 + 0.01j], dtype=torch.complex128)
        result = torchscience.special_functions.exponential_integral_e_1(z)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_complex_gradient_finite(self):
        """Test that gradients at complex values are finite."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 1.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        result = torchscience.special_functions.exponential_integral_e_1(z)
        result.real.sum().backward()
        assert torch.isfinite(z.grad).all()

    def test_complex_gradcheck(self):
        """Test complex gradient correctness."""
        z = torch.tensor(
            [1.0 + 0.5j, 2.0 + 0.3j],
            dtype=torch.complex128,
            requires_grad=True,
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_e_1(t)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32_support(self):
        """Test float32 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_float64_support(self):
        """Test float64 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_complex64_support(self):
        """Test complex64 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result.real).all()

    def test_complex128_support(self):
        """Test complex128 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors correctly infer output shape."""
        x = torch.empty(3, 4, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    # =========================================================================
    # Properties tests
    # =========================================================================

    def test_monotonicity(self):
        """Test that E_1 is monotonically decreasing for x > 0."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        # Check monotonically decreasing
        assert (result[1:] < result[:-1]).all()

    def test_always_positive_for_positive_x(self):
        """Test that E_1(x) > 0 for all x > 0."""
        x = torch.tensor(
            [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0], dtype=torch.float64
        )
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert (result > 0).all()

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        x = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.shape == (3, 1)

    def test_scalar_input(self):
        """Test scalar (0-dimensional) input."""
        x = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert result.dim() == 0
        assert result.dtype == torch.float64

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_positive(self):
        """Test E_1 at very small positive values."""
        x = torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        # Should be large positive (approaching +inf as x -> 0+)
        assert torch.isfinite(result).all()
        assert (result > 10).all()

    def test_very_large_positive(self):
        """Test E_1 at very large positive values."""
        x = torch.tensor([20.0, 50.0, 100.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        # Should be very small positive values approaching 0
        assert torch.isfinite(result).all()
        assert (result > 0).all()
        assert (result < 1e-6).all()

    def test_boundary_between_algorithms(self):
        """Test values near algorithm transition boundary (x = 1)."""
        x = torch.tensor([0.9, 0.99, 1.0, 1.01, 1.1], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_e_1(x)
        assert torch.isfinite(result).all()
        # Should be monotonically decreasing
        assert (result[1:] < result[:-1]).all()
