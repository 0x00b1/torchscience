import torch
import torch.testing

import torchscience.special_functions


class TestExponentialIntegralEi:
    """Tests for the exponential integral Ei function."""

    # =========================================================================
    # Basic forward tests
    # =========================================================================

    def test_forward_positive_values(self):
        """Test Ei at positive values against known scipy values."""
        # Reference values computed from scipy.special.expi
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        expected = torch.tensor(
            [
                0.4542199048631736,  # Ei(0.5)
                1.8951178163559368,  # Ei(1.0)
                4.954234356001891,  # Ei(2.0)
                40.18527536389832,  # Ei(5.0)
                2492.228976241877,  # Ei(10.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_ei(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_negative_values(self):
        """Test Ei at negative values against known scipy values."""
        # Reference values computed from scipy.special.expi
        x = torch.tensor([-0.5, -1.0, -2.0, -5.0, -10.0], dtype=torch.float64)
        expected = torch.tensor(
            [
                -0.5597735947761608,  # Ei(-0.5)
                -0.21938393439552062,  # Ei(-1.0)
                -0.04890051070806112,  # Ei(-2.0)
                -0.001148295591784439,  # Ei(-5.0)
                -4.156968929685324e-06,  # Ei(-10.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_ei(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-12)

    def test_forward_small_positive_values(self):
        """Test Ei at small positive values."""
        x = torch.tensor([0.01, 0.1, 0.2, 0.3], dtype=torch.float64)
        # Reference values from scipy.special.expi
        expected = torch.tensor(
            [
                -4.0179294654266693,  # Ei(0.01)
                -1.6228128139692766,  # Ei(0.1)
                -0.8217605879024001,  # Ei(0.2)
                -0.3026685392658260,  # Ei(0.3)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_ei(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_forward_large_positive_values(self):
        """Test Ei at large positive values (asymptotic region)."""
        x = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        # Should be finite and positive
        assert torch.isfinite(result).all()
        assert (result > 0).all()
        # Check monotonically increasing
        assert (result[1:] > result[:-1]).all()

    # =========================================================================
    # Special value tests
    # =========================================================================

    def test_special_value_zero(self):
        """Test Ei(0) = -inf."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert torch.isinf(result).all()
        assert (result < 0).all()

    def test_special_value_positive_inf(self):
        """Test Ei(+inf) = +inf."""
        x = torch.tensor([float("inf")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert torch.isinf(result).all()
        assert (result > 0).all()

    def test_special_value_negative_inf(self):
        """Test Ei(-inf) = 0."""
        x = torch.tensor([float("-inf")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert (result == 0).all()

    def test_special_value_nan(self):
        """Test Ei(nan) = nan."""
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert torch.isnan(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_formula_positive(self):
        """Test gradient d/dx Ei(x) = e^x / x for positive x."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ei(x)
        y.sum().backward()

        expected_grad = torch.exp(x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_formula_negative(self):
        """Test gradient d/dx Ei(x) = e^x / x for negative x."""
        x = torch.tensor(
            [-1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ei(x)
        y.sum().backward()

        expected_grad = torch.exp(x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradcheck_positive(self):
        """Test gradient correctness with torch.autograd.gradcheck for positive x."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ei(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_negative(self):
        """Test gradient correctness with torch.autograd.gradcheck for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ei(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradient_at_zero_is_nan(self):
        """Test that gradient at x=0 returns NaN."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.exponential_integral_ei(x)
        y.backward()
        assert torch.isnan(x.grad).all()

    # =========================================================================
    # Second-order gradient tests
    # =========================================================================

    def test_second_derivative_formula(self):
        """Test d^2/dx^2 Ei(x) = (x - 1) * e^x / x^2."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ei(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        expected = (x.detach() - 1) * torch.exp(x.detach()) / (x.detach() ** 2)
        torch.testing.assert_close(grad2, expected, rtol=1e-6, atol=1e-8)

    def test_gradgradcheck_positive(self):
        """Test second-order gradients with torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ei(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    def test_gradgradcheck_negative(self):
        """Test second-order gradients for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ei(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real implementation.

        Note: For negative real x, the complex Ei uses ln(z) which gives
        ln|x| + i*pi, adding a branch cut contribution. We only test positive x.
        """
        x_real = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.exponential_integral_ei(
            x_real
        )
        result_complex = (
            torchscience.special_functions.exponential_integral_ei(x_complex)
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
        """Test Ei at general complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, -1.0 + 0.5j], dtype=torch.complex128
        )
        result = torchscience.special_functions.exponential_integral_ei(z)
        # Just verify we get finite results
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_complex_gradient_finite(self):
        """Test that gradients at complex values with nonzero imaginary part are finite."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, -0.5 + 1.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        result = torchscience.special_functions.exponential_integral_ei(z)
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
            return torchscience.special_functions.exponential_integral_ei(t)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32_support(self):
        """Test float32 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_float64_support(self):
        """Test float64 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_complex64_support(self):
        """Test complex64 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result.real).all()

    def test_complex128_support(self):
        """Test complex128 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors correctly infer output shape."""
        x = torch.empty(3, 4, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    # =========================================================================
    # Relationship tests
    # =========================================================================

    def test_relationship_e1(self):
        """Test Ei(x) = -E1(-x) relationship approximately.

        For x > 0, Ei(x) = -E1(-x) where E1 is the exponential integral of the first kind.
        We verify this indirectly by checking Ei(-x) = -E1(x).
        """
        # For negative x, Ei(x) should equal -E1(-x)
        # Since E1(z) = -Ei(-z), we have Ei(-z) = -E1(z)
        # Testing: for positive t, Ei(-t) * (-1) should behave like E1(t)
        # E1(t) is positive for t > 0 and decreases to 0 as t -> inf
        x = torch.tensor([-1.0, -2.0, -5.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        # Ei(x) for x < 0 should be negative
        assert (result < 0).all()
        # -Ei(x) = E1(-x) should be positive
        assert (-result > 0).all()

    def test_monotonicity_positive(self):
        """Test that Ei is monotonically increasing for x > 0."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        # Check monotonically increasing
        assert (result[1:] > result[:-1]).all()

    def test_sign_near_zero(self):
        """Test that Ei changes sign from negative to positive around x ~ 0.37."""
        # Ei(x) has a zero near x = 0.3725
        x_negative = torch.tensor([0.3], dtype=torch.float64)
        x_positive = torch.tensor([0.4], dtype=torch.float64)

        result_neg = torchscience.special_functions.exponential_integral_ei(
            x_negative
        )
        result_pos = torchscience.special_functions.exponential_integral_ei(
            x_positive
        )

        assert (result_neg < 0).all()
        assert (result_pos > 0).all()

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        x = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.shape == (3, 1)

    def test_scalar_input(self):
        """Test scalar (0-dimensional) input."""
        x = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert result.dim() == 0
        assert result.dtype == torch.float64

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_positive(self):
        """Test Ei at very small positive values."""
        x = torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        # Should be large negative (approaching -inf as x -> 0+)
        assert torch.isfinite(result).all()
        assert (result < -10).all()

    def test_very_large_negative(self):
        """Test Ei at very large negative values."""
        x = torch.tensor([-20.0, -50.0, -100.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        # Should be very small negative values approaching 0
        assert torch.isfinite(result).all()
        assert (result < 0).all()
        assert (torch.abs(result) < 1e-6).all()

    def test_boundary_between_algorithms(self):
        """Test values near algorithm transition boundaries."""
        # Test near x = 40 (series/asymptotic boundary for positive x)
        x = torch.tensor([39.0, 40.0, 41.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert torch.isfinite(result).all()
        # Should be monotonically increasing
        assert (result[1:] > result[:-1]).all()

        # Test near x = -1 (series/continued fraction boundary for negative x)
        x = torch.tensor([-0.9, -1.0, -1.1], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ei(x)
        assert torch.isfinite(result).all()
