import torch
import torch.testing

import torchscience.special_functions


class TestCosineIntegralCi:
    """Tests for the cosine integral Ci function."""

    # =========================================================================
    # Basic forward tests
    # =========================================================================

    def test_forward_positive_values(self):
        """Test Ci at positive values against known values.

        Reference values computed via scipy.special.sici (returns si, ci).
        """
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        # Ci(x) values from scipy.special.sici
        expected = torch.tensor(
            [
                -0.17778407877534814,  # Ci(0.5)
                0.33740392290096813,  # Ci(1.0)
                0.42298082631138887,  # Ci(2.0)
                -0.19002974965664387,  # Ci(5.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.cosine_integral_ci(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_forward_small_values(self):
        """Test Ci at small positive values near zero.

        For small x: Ci(x) ~ gamma + ln(x) - x^2/4 + ...
        """
        x = torch.tensor([0.01, 0.1, 0.2, 0.3], dtype=torch.float64)
        # Reference values from scipy.special.sici
        expected = torch.tensor(
            [
                -4.027979520982392,  # Ci(0.01)
                -1.727868386657297,  # Ci(0.1)
                -1.042205595672782,  # Ci(0.2)
                -0.649172932971162,  # Ci(0.3)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.cosine_integral_ci(x)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_forward_large_values(self):
        """Test Ci at larger values."""
        x = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        # Ci(x) oscillates around 0 and approaches 0 for large x
        expected = torch.tensor(
            [
                -0.04545643300445537,  # Ci(10.0)
                0.04441982084535327,  # Ci(20.0)
                -0.005628386324116918,  # Ci(50.0)
            ],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.cosine_integral_ci(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    # =========================================================================
    # Special value tests
    # =========================================================================

    def test_special_value_non_positive_returns_nan(self):
        """Test Ci(x) returns NaN for x <= 0."""
        x = torch.tensor([0.0, -1.0, -0.5], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert torch.isnan(result).all()

    def test_special_value_positive_inf(self):
        """Test Ci(+inf) = 0."""
        x = torch.tensor([float("inf")], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        torch.testing.assert_close(
            result,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_special_value_nan(self):
        """Test Ci(nan) = nan."""
        x = torch.tensor([float("nan")], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert torch.isnan(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_formula_positive(self):
        """Test gradient d/dx Ci(x) = cos(x) / x for positive x."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.cosine_integral_ci(x)
        y.sum().backward()

        expected_grad = torch.cos(x.detach()) / x.detach()
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-8, atol=1e-10
        )

    def test_gradient_small_positive(self):
        """Test gradient for small positive values."""
        x = torch.tensor([0.1, 0.5], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.cosine_integral_ci(x)
        y.sum().backward()

        expected_grad = torch.cos(x.detach()) / x.detach()
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-6, atol=1e-8)

    def test_gradcheck_positive(self):
        """Test gradient correctness with torch.autograd.gradcheck for positive x."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.cosine_integral_ci(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_small_positive(self):
        """Test gradient correctness for small positive x."""
        x = torch.tensor(
            [0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.cosine_integral_ci(t)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Second-order gradient tests
    # =========================================================================

    def test_second_derivative_formula(self):
        """Test d^2/dx^2 Ci(x) = (-x*sin(x) - cos(x)) / x^2."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.cosine_integral_ci(x)

        # Compute first derivative
        (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        # Compute second derivative
        (grad2,) = torch.autograd.grad(grad1.sum(), x)

        x_val = x.detach()
        expected = (-x_val * torch.sin(x_val) - torch.cos(x_val)) / (x_val**2)
        torch.testing.assert_close(grad2, expected, rtol=1e-6, atol=1e-8)

    def test_gradgradcheck_positive(self):
        """Test second-order gradients with torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.cosine_integral_ci(t)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-4, atol=1e-3)

    # =========================================================================
    # Complex tensor tests
    # =========================================================================

    def test_complex_positive_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real implementation."""
        x_real = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = torchscience.special_functions.cosine_integral_ci(x_real)
        result_complex = torchscience.special_functions.cosine_integral_ci(
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

    def test_complex_general_values(self):
        """Test Ci at general complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 1.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.cosine_integral_ci(z)
        # Just verify we get finite results
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_complex_gradient_finite(self):
        """Test that gradients at complex values are finite."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 1.0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        result = torchscience.special_functions.cosine_integral_ci(z)
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
            return torchscience.special_functions.cosine_integral_ci(t)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32_support(self):
        """Test float32 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_float64_support(self):
        """Test float64 dtype support."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_complex64_support(self):
        """Test complex64 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.dtype == torch.complex64
        assert torch.isfinite(result.real).all()

    def test_complex128_support(self):
        """Test complex128 dtype support."""
        x = torch.tensor([0.5 + 0.5j, 1.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.dtype == torch.complex128
        assert torch.isfinite(result.real).all()

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors correctly infer output shape."""
        x = torch.empty(3, 4, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    # =========================================================================
    # Behavior tests
    # =========================================================================

    def test_oscillatory_behavior_large_x(self):
        """Test that Ci oscillates and approaches 0 for large x."""
        x = torch.tensor([10.0, 15.0, 20.0, 25.0, 30.0], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        # All values should be small (approaching 0)
        assert (torch.abs(result) < 0.1).all()

    def test_monotonic_near_zero(self):
        """Test that Ci is increasing near zero (it goes from -inf toward positive)."""
        x = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        # Ci is monotonically increasing for very small x
        assert (result[1:] > result[:-1]).all()

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        x = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.shape == (3, 1)

    def test_scalar_input(self):
        """Test scalar (0-dimensional) input."""
        x = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        assert result.dim() == 0
        assert result.dtype == torch.float64

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_very_small_positive(self):
        """Test Ci at very small positive values."""
        x = torch.tensor([1e-6, 1e-4, 1e-3], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        # For very small x, Ci(x) ~ gamma + ln(x) which is large and negative
        assert torch.isfinite(result).all()
        # All values should be negative for small x
        assert (result < 0).all()

    def test_larger_values(self):
        """Test Ci at larger values to verify convergence."""
        x = torch.tensor([10.0, 20.0, 30.0, 50.0], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)
        # Should be finite
        assert torch.isfinite(result).all()
        # Should be small (approaching 0)
        assert (torch.abs(result) < 0.1).all()

    def test_series_expansion_small_x(self):
        """Test Taylor series: Ci(x) = gamma + ln(x) - x^2/4 + x^4/96 - ..."""
        x = torch.tensor([0.1], dtype=torch.float64)
        result = torchscience.special_functions.cosine_integral_ci(x)

        # Compute partial sum of series
        x_val = 0.1
        gamma = 0.5772156649015328606
        # Ci(x) = gamma + ln(x) - x^2/4 + x^4/96 - x^6/4320 + ...
        series = (
            gamma
            + torch.log(torch.tensor(x_val))
            - x_val**2 / 4
            + x_val**4 / 96
            - x_val**6 / 4320
        )
        torch.testing.assert_close(
            result,
            torch.tensor([series.item()], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )
