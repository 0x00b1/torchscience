import torch
import torch.testing

import torchscience.special_functions


class TestSineIntegralSi:
    """Tests for the sine integral Si function."""

    # =========================================================================
    # Basic forward tests
    # =========================================================================

    def test_forward_positive_values(self):
        """Test Si at positive values against known values.

        Si(x) = sum_{n=0}^inf (-1)^n x^(2n+1) / ((2n+1) * (2n+1)!)
        Reference values computed via scipy.special.sici.
        """
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        # Si(x) values from scipy.special.sici
        expected = torch.tensor(
            [
                0.4931074180430667,  # Si(0.5)
                0.9460830703671830,  # Si(1.0)
                1.6054129768026948,  # Si(2.0)
                1.5499312449446073,  # Si(5.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.sine_integral_si(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_forward_negative_values(self):
        """Test Si at negative values against known values (odd function)."""
        x = torch.tensor([-0.5, -1.0, -2.0, -5.0], dtype=torch.float64)
        # Si(x) values: Si(-x) = -Si(x) (odd function)
        expected = torch.tensor(
            [
                -0.4931074180430667,  # Si(-0.5)
                -0.9460830703671830,  # Si(-1.0)
                -1.6054129768026948,  # Si(-2.0)
                -1.5499312449446073,  # Si(-5.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.sine_integral_si(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_forward_small_values(self):
        """Test Si at small values near zero."""
        x = torch.tensor([0.01, 0.1, 0.2, 0.3], dtype=torch.float64)
        # For small x: Si(x) ~ x - x^3/18 + ...
        # Si(0.01) ~ 0.01 - (0.01)^3/18 + ... ~ 0.01
        result = torchscience.special_functions.sine_integral_si(x)
        # Verify finite and reasonable
        assert torch.isfinite(result).all()
        # Si is approximately x for small x
        torch.testing.assert_close(result[:1], x[:1], rtol=5e-3, atol=1e-4)

    def test_forward_large_values(self):
        """Test Si at larger values approaching asymptote."""
        x = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        # Si(x) approaches pi/2 as x -> infinity
        expected = torch.tensor(
            [
                1.6583475942188740,  # Si(10.0)
                1.5481669376186282,  # Si(20.0)
                1.5516170724859358,  # Si(50.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.sine_integral_si(x)
        # Use relaxed tolerance for larger values where series converges more slowly
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Special value tests
    # =========================================================================

    def test_special_value_zero(self):
        """Test Si(0) = 0."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        torch.testing.assert_close(
            result, torch.tensor([0.0], dtype=torch.float64)
        )

    def test_special_value_positive_inf(self):
        """Test Si(+inf) = pi/2."""
        x = torch.tensor([float("inf")], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        pi_2 = torch.tensor([1.5707963267948966], dtype=torch.float64)
        torch.testing.assert_close(result, pi_2, rtol=1e-10, atol=1e-10)

    def test_special_value_negative_inf(self):
        """Test Si(-inf) = -pi/2."""
        x = torch.tensor([float("-inf")], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        neg_pi_2 = torch.tensor([-1.5707963267948966], dtype=torch.float64)
        torch.testing.assert_close(result, neg_pi_2, rtol=1e-10, atol=1e-10)

    def test_special_value_nan(self):
        """Test Si(nan) = nan."""
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        assert torch.isnan(result).all()

    # =========================================================================
    # Odd function tests
    # =========================================================================

    def test_odd_function_property(self):
        """Test that Si(-x) = -Si(x) (odd function)."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        result_pos = torchscience.special_functions.sine_integral_si(x)
        result_neg = torchscience.special_functions.sine_integral_si(-x)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_formula_positive(self):
        """Test gradient d/dx Si(x) = sin(x) / x for positive x."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.sine_integral_si(x)
        y.sum().backward()

        expected_grad = torch.sin(x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_formula_negative(self):
        """Test gradient d/dx Si(x) = sin(x) / x for negative x."""
        x = torch.tensor(
            [-1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.sine_integral_si(x)
        y.sum().backward()

        expected_grad = torch.sin(x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_at_zero(self):
        """Test gradient at x=0 is 1 (removable singularity)."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.sine_integral_si(x)
        y.backward()
        # lim_{x->0} sin(x)/x = 1
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
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_negative(self):
        """Test gradient correctness with torch.autograd.gradcheck for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_near_zero(self):
        """Test gradient correctness near x=0."""
        x = torch.tensor(
            [0.01, 0.1, -0.01, -0.1], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Second-order gradient tests
    # =========================================================================

    def test_second_derivative_formula(self):
        """Test d^2/dx^2 Si(x) = (x*cos(x) - sin(x)) / x^2."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.sine_integral_si(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        x_val = x.detach()
        expected = (x_val * torch.cos(x_val) - torch.sin(x_val)) / (x_val**2)
        torch.testing.assert_close(grad2, expected, rtol=1e-6, atol=1e-8)

    def test_second_derivative_at_zero(self):
        """Test Si''(0) = 0."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.sine_integral_si(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        # Si''(0) = 0 from Taylor series sinc(x) = 1 - x^2/6 + ..., so sinc'(0) = 0
        torch.testing.assert_close(
            grad2,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gradgradcheck_positive(self):
        """Test second-order gradients with torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    def test_gradgradcheck_negative(self):
        """Test second-order gradients for negative x."""
        x = torch.tensor(
            [-0.5, -1.0, -2.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real implementation."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.sine_integral_si(x_real)
        result_complex = torchscience.special_functions.sine_integral_si(
            x_complex
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

        result_real = torchscience.special_functions.sine_integral_si(x_real)
        result_complex = torchscience.special_functions.sine_integral_si(
            x_complex
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
        """Test Si at general complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, -1.0 + 0.5j], dtype=torch.complex128
        )
        result = torchscience.special_functions.sine_integral_si(z)
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
        result = torchscience.special_functions.sine_integral_si(z)
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
            return torchscience.special_functions.sine_integral_si(t)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32_support(self):
        """Test float32 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_float64_support(self):
        """Test float64 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_complex64_support(self):
        """Test complex64 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex64)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result.real).all()

    def test_complex128_support(self):
        """Test complex128 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors correctly infer output shape."""
        x = torch.empty(3, 4, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    # =========================================================================
    # Relationship tests
    # =========================================================================

    def test_monotonicity_positive_small(self):
        """Test that Si is monotonically increasing for small x > 0."""
        x = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        # Check monotonically increasing
        assert (result[1:] > result[:-1]).all()

    def test_symmetry_series(self):
        """Test Taylor series: Si(x) = x - x^3/18 + x^5/600 - ..."""
        x = torch.tensor([0.1], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)

        # Compute partial sum of series
        x_val = 0.1
        # n=0: x, n=1: -x^3/18, n=2: x^5/600, n=3: -x^7/35280
        series = x_val - x_val**3 / 18 + x_val**5 / 600 - x_val**7 / 35280
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
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.shape == (3, 1)

    def test_scalar_input(self):
        """Test scalar (0-dimensional) input."""
        x = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        assert result.dim() == 0
        assert result.dtype == torch.float64

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_positive(self):
        """Test Si at very small positive values."""
        x = torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        # For very small x, Si(x) ~ x
        assert torch.isfinite(result).all()
        # Check approximation Si(x) ~ x for small x
        torch.testing.assert_close(result, x, rtol=1e-4, atol=1e-10)

    def test_very_small_negative(self):
        """Test Si at very small negative values."""
        x = torch.tensor([-1e-10, -1e-8, -1e-6], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        # For very small |x|, Si(x) ~ x
        assert torch.isfinite(result).all()
        # Check approximation Si(x) ~ x for small x
        torch.testing.assert_close(result, x, rtol=1e-4, atol=1e-10)

    def test_larger_values(self):
        """Test Si at larger values to verify convergence."""
        x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        # Should be finite
        assert torch.isfinite(result).all()
        # Should approach pi/2 ~ 1.5708
        pi_2 = 1.5707963267948966
        assert ((result - pi_2).abs() < 0.25).all()

    def test_smooth_at_origin(self):
        """Test that Si is smooth at x = 0 (entire function)."""
        # Si should be smooth at 0
        x = torch.linspace(-0.1, 0.1, 21, dtype=torch.float64)
        result = torchscience.special_functions.sine_integral_si(x)
        # All values should be finite
        assert torch.isfinite(result).all()
        # Result should be continuous (differences should be small)
        diffs = torch.diff(result)
        assert (torch.abs(diffs) < 0.02).all()  # Reasonable continuity
