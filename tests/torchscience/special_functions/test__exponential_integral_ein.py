import torch
import torch.testing

import torchscience.special_functions


class TestExponentialIntegralEin:
    """Tests for the complementary exponential integral Ein function."""

    # =========================================================================
    # Basic forward tests
    # =========================================================================

    def test_forward_positive_values(self):
        """Test Ein at positive values against known values.

        Ein(x) = sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
        Reference values computed via scipy numerical integration.
        """
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        # Ein(x) values computed from scipy.integrate.quad
        expected = torch.tensor(
            [
                0.4438420791177484,  # Ein(0.5)
                0.7965995992970532,  # Ein(1.0)
                1.3192633561695397,  # Ein(2.0)
                2.1878018729269089,  # Ein(5.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_ein(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_forward_negative_values(self):
        """Test Ein at negative values against known values."""
        x = torch.tensor([-0.5, -1.0, -2.0, -5.0], dtype=torch.float64)
        # Ein(x) values for negative x computed from scipy.integrate.quad
        expected = torch.tensor(
            [
                -0.5701514205215862,  # Ein(-0.5)
                -1.3179021514544038,  # Ein(-1.0)
                -3.6838715105404116,  # Ein(-2.0)
                -37.9986217784675446,  # Ein(-5.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.exponential_integral_ein(x)
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-9)

    def test_forward_small_values(self):
        """Test Ein at small values near zero."""
        x = torch.tensor([0.01, 0.1, 0.2, 0.3], dtype=torch.float64)
        # For small x: Ein(x) ~ x - x^2/4 + x^3/18 - ...
        # Ein(0.01) ~ 0.01 - 0.000025 + ... ~ 0.009975
        result = torchscience.special_functions.exponential_integral_ein(x)
        # Verify finite and reasonable
        assert torch.isfinite(result).all()
        # Ein is approximately x for small x
        torch.testing.assert_close(result[:1], x[:1], rtol=5e-3, atol=1e-4)

    # =========================================================================
    # Special value tests
    # =========================================================================

    def test_special_value_zero(self):
        """Test Ein(0) = 0."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        torch.testing.assert_close(
            result, torch.tensor([0.0], dtype=torch.float64)
        )

    def test_special_value_positive_inf(self):
        """Test Ein(+inf) = +inf."""
        x = torch.tensor([float("inf")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert torch.isinf(result).all()
        assert (result > 0).all()

    def test_special_value_negative_inf(self):
        """Test Ein(-inf) = -inf."""
        x = torch.tensor([float("-inf")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert torch.isinf(result).all()
        assert (result < 0).all()

    def test_special_value_nan(self):
        """Test Ein(nan) = nan."""
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert torch.isnan(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_formula_positive(self):
        """Test gradient d/dx Ein(x) = (1 - e^(-x)) / x for positive x."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ein(x)
        y.sum().backward()

        expected_grad = (1.0 - torch.exp(-x.detach())) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_formula_negative(self):
        """Test gradient d/dx Ein(x) = (1 - e^(-x)) / x for negative x."""
        x = torch.tensor(
            [-1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ein(x)
        y.sum().backward()

        expected_grad = (1.0 - torch.exp(-x.detach())) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_at_zero(self):
        """Test gradient at x=0 is 1 (removable singularity)."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.exponential_integral_ein(x)
        y.backward()
        # lim_{x->0} (1 - e^(-x))/x = 1
        torch.testing.assert_close(
            x.grad,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_gradcheck_positive(self):
        """Test gradient correctness with torch.autograd.gradcheck for positive x."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_negative(self):
        """Test gradient correctness with torch.autograd.gradcheck for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_near_zero(self):
        """Test gradient correctness near x=0."""
        x = torch.tensor(
            [0.01, 0.1, -0.01, -0.1], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Second-order gradient tests
    # =========================================================================

    def test_second_derivative_formula(self):
        """Test d^2/dx^2 Ein(x) = (e^(-x)(x + 1) - 1) / x^2."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.exponential_integral_ein(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        x_val = x.detach()
        expected = (torch.exp(-x_val) * (x_val + 1.0) - 1.0) / (x_val**2)
        torch.testing.assert_close(grad2, expected, rtol=1e-6, atol=1e-8)

    def test_second_derivative_at_zero(self):
        """Test Ein''(0) = -1/2."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.exponential_integral_ein(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        # Ein''(0) = -1/2 from Taylor series Ein(x) = x - x^2/4 + ...
        torch.testing.assert_close(
            grad2,
            torch.tensor([-0.5], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gradgradcheck_positive(self):
        """Test second-order gradients with torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    def test_gradgradcheck_negative(self):
        """Test second-order gradients for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real implementation."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.exponential_integral_ein(
            x_real
        )
        result_complex = (
            torchscience.special_functions.exponential_integral_ein(x_complex)
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

    def test_complex_negative_real_axis_matches_real(self):
        """Test complex numbers on negative real axis match real implementation."""
        x_real = torch.tensor([-0.5, -1.0, -2.0, -5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.exponential_integral_ein(
            x_real
        )
        result_complex = (
            torchscience.special_functions.exponential_integral_ein(x_complex)
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-8, atol=1e-10
        )
        # Imaginary part should be zero for real input
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-8,
        )

    def test_complex_general_values(self):
        """Test Ein at general complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, -1.0 + 0.5j], dtype=torch.complex128
        )
        result = torchscience.special_functions.exponential_integral_ein(z)
        # Just verify we get finite results
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_complex_gradient_finite(self):
        """Test that gradients at complex values are finite."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, -0.5 + 1.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        result = torchscience.special_functions.exponential_integral_ein(z)
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
            return torchscience.special_functions.exponential_integral_ein(t)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32_support(self):
        """Test float32 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_float64_support(self):
        """Test float64 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_complex64_support(self):
        """Test complex64 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result.real).all()

    def test_complex128_support(self):
        """Test complex128 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors correctly infer output shape."""
        x = torch.empty(3, 4, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    # =========================================================================
    # Relationship tests
    # =========================================================================

    def test_relation_to_e1(self):
        """Test Ein(x) = gamma + ln(x) + E1(x) for x > 0.

        gamma is the Euler-Mascheroni constant.
        E1 is the exponential integral of the first kind.
        """
        x = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        gamma = 0.5772156649015328606065120900824024310421593359

        ein = torchscience.special_functions.exponential_integral_ein(x)
        e1 = torchscience.special_functions.exponential_integral_e_1(x)

        # Ein(x) = gamma + ln(x) + E1(x) for x > 0
        expected = gamma + torch.log(x) + e1
        torch.testing.assert_close(ein, expected, rtol=1e-8, atol=1e-8)

    def test_monotonicity_positive(self):
        """Test that Ein is monotonically increasing for x > 0."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        # Check monotonically increasing
        assert (result[1:] > result[:-1]).all()

    def test_monotonicity_negative(self):
        """Test that Ein is monotonically increasing for x < 0."""
        x = torch.tensor(
            [-10.0, -5.0, -2.0, -1.0, -0.5, -0.1], dtype=torch.float64
        )
        result = torchscience.special_functions.exponential_integral_ein(x)
        # Check monotonically increasing
        assert (result[1:] > result[:-1]).all()

    def test_symmetry_series(self):
        """Test Taylor series: Ein(x) = x - x^2/4 + x^3/18 - x^4/96 + ..."""
        x = torch.tensor([0.1], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)

        # Compute partial sum of series
        x_val = 0.1
        # n=1: x, n=2: -x^2/4, n=3: x^3/18, n=4: -x^4/96
        series = (
            x_val
            - x_val**2 / 4
            + x_val**3 / 18
            - x_val**4 / 96
            + x_val**5 / 600
        )
        torch.testing.assert_close(
            result,
            torch.tensor([series], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        x = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.shape == (3, 1)

    def test_scalar_input(self):
        """Test scalar (0-dimensional) input."""
        x = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        assert result.dim() == 0
        assert result.dtype == torch.float64

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_positive(self):
        """Test Ein at very small positive values."""
        x = torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        # For very small x, Ein(x) ~ x
        assert torch.isfinite(result).all()
        # Check approximation Ein(x) ~ x for small x
        torch.testing.assert_close(result, x, rtol=1e-4, atol=1e-10)

    def test_very_small_negative(self):
        """Test Ein at very small negative values."""
        x = torch.tensor([-1e-10, -1e-8, -1e-6], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        # For very small |x|, Ein(x) ~ x
        assert torch.isfinite(result).all()
        # Check approximation Ein(x) ~ x for small x
        torch.testing.assert_close(result, x, rtol=1e-4, atol=1e-10)

    def test_larger_values(self):
        """Test Ein at larger values to verify convergence."""
        x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        # Should be finite
        assert torch.isfinite(result).all()
        # Should be positive and increasing
        assert (result > 0).all()
        assert (result[1:] > result[:-1]).all()

    def test_smooth_at_origin(self):
        """Test that Ein is smooth at x = 0 (entire function)."""
        # Unlike Ei, Ein should be smooth at 0
        x = torch.linspace(-0.1, 0.1, 21, dtype=torch.float64)
        result = torchscience.special_functions.exponential_integral_ein(x)
        # All values should be finite
        assert torch.isfinite(result).all()
        # Result should be continuous (differences should be small)
        diffs = torch.diff(result)
        assert (torch.abs(diffs) < 0.02).all()  # Reasonable continuity
