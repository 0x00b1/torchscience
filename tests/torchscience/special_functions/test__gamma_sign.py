import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestGammaSign:
    """Tests for the gamma sign function."""

    def test_positive_values(self):
        """Test that gamma_sign(x) = +1 for all x > 0."""
        x = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0], dtype=torch.float64
        )
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.ones(7, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_alternating_signs(self):
        """Test that gamma_sign alternates for negative non-integers."""
        # Between 0 and -1: sign is -1
        # Between -1 and -2: sign is +1
        # Between -2 and -3: sign is -1
        # etc.
        x = torch.tensor(
            [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5], dtype=torch.float64
        )
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.tensor(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_near_integers(self):
        """Test gamma_sign near negative integer poles."""
        # Just to the right of -1: sign is -1
        # Just to the left of -1: sign is +1
        x = torch.tensor([-0.999, -1.001, -1.999, -2.001], dtype=torch.float64)
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_poles_return_nan(self):
        """Test that gamma_sign returns NaN at poles (non-positive integers)."""
        poles = torch.tensor(
            [0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.gamma_sign(poles)
        assert torch.isnan(result).all()

    def test_comparison_with_scipy(self):
        """Test comparison with scipy.special.gammasgn (if available)."""
        pytest.importorskip("scipy")
        from scipy.special import gammasgn

        x = torch.tensor(
            [0.5, 1.0, 2.0, -0.5, -1.5, -2.5, -3.5, 5.0, 10.0],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.tensor(
            [gammasgn(xi.item()) for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_consistent_with_gamma(self):
        """Test that gamma_sign(x) matches sign(gamma(x)) for non-poles."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, -0.5, -1.5, -2.5, -3.5], dtype=torch.float64
        )
        gamma_values = torchscience.special_functions.gamma(x)
        expected_sign = torch.sign(gamma_values)
        result = torchscience.special_functions.gamma_sign(x)
        torch.testing.assert_close(
            result, expected_sign, rtol=1e-10, atol=1e-10
        )

    def test_gradient_is_zero(self):
        """Test that gradient of gamma_sign is zero (piecewise constant)."""
        x = torch.tensor(
            [0.5, 1.5, 2.5, -0.5, -1.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.special_functions.gamma_sign(x)
        y.sum().backward()
        expected_grad = torch.zeros(5, dtype=torch.float64)
        torch.testing.assert_close(
            x.grad, expected_grad, rtol=1e-10, atol=1e-10
        )

    def test_gradcheck(self):
        """Test gradient correctness via torch.autograd.gradcheck."""
        x = torch.tensor(
            [0.5, 1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.gamma_sign(t)

        # Since gradient is 0, gradcheck should pass with appropriate tolerances
        assert torch.autograd.gradcheck(
            func, (x,), eps=1e-6, atol=1e-5, rtol=1e-5
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness via torch.autograd.gradgradcheck."""
        x = torch.tensor(
            [1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True
        )

        def func(t):
            return torchscience.special_functions.gamma_sign(t)

        assert torch.autograd.gradgradcheck(
            func, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_meta_tensor_support(self):
        """Test that meta tensors are supported for shape inference."""
        x = torch.empty(3, 4, dtype=torch.float64, device="meta")
        result = torchscience.special_functions.gamma_sign(x)
        assert result.shape == (3, 4)
        assert result.device == torch.device("meta")
        assert result.dtype == torch.float64

    def test_dtypes(self):
        """Test that various floating-point dtypes are supported."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.tensor([0.5, 1.0, -0.5, -1.5], dtype=dtype)
            result = torchscience.special_functions.gamma_sign(x)
            assert result.dtype == dtype
            # Check values are +1 or -1 (not NaN for these inputs)
            assert torch.all((result == 1.0) | (result == -1.0))

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        x1 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)  # Shape (2, 1)
        x2 = torch.tensor(
            [[0.5, 1.5, 2.5]], dtype=torch.float64
        )  # Shape (1, 3)

        result1 = torchscience.special_functions.gamma_sign(x1)
        result2 = torchscience.special_functions.gamma_sign(x2)
        assert result1.shape == (2, 1)
        assert result2.shape == (1, 3)

    def test_large_negative_values(self):
        """Test gamma_sign for large negative values."""
        # The sign pattern continues alternating
        x = torch.tensor([-100.5, -101.5, -102.5, -103.5], dtype=torch.float64)
        result = torchscience.special_functions.gamma_sign(x)
        # -100.5 is in interval (-101, -100), n=101, sign = (-1)^101 = -1
        # -101.5 is in interval (-102, -101), n=102, sign = (-1)^102 = +1
        expected = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_reflection_formula_sign(self):
        """Test sign consistency with reflection formula for gamma."""
        # The reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        # This implies sign(Gamma(x)) * sign(Gamma(1-x)) = sign(1/sin(pi*x))
        import math

        x_values = [0.3, 0.7, -0.3, -1.3, -2.3]
        for x in x_values:
            x_tensor = torch.tensor([x], dtype=torch.float64)
            one_minus_x_tensor = torch.tensor([1.0 - x], dtype=torch.float64)

            sign_x = torchscience.special_functions.gamma_sign(x_tensor).item()
            sign_1_minus_x = torchscience.special_functions.gamma_sign(
                one_minus_x_tensor
            ).item()

            # sign of 1/sin(pi*x)
            sin_val = math.sin(math.pi * x)
            expected_product = 1.0 if sin_val > 0 else -1.0

            assert sign_x * sign_1_minus_x == expected_product, (
                f"Failed for x={x}"
            )

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_dtype_requires_conversion(self, int_dtype):
        """Test that integer inputs require explicit conversion to float."""
        x_int = torch.tensor([1, 2, 3], dtype=int_dtype)
        with pytest.raises(NotImplementedError):
            torchscience.special_functions.gamma_sign(x_int)

    def test_half_integers_positive(self):
        """Test gamma_sign at positive half-integers."""
        # Gamma is positive for all positive reals
        x = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5], dtype=torch.float64)
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_very_small_positive(self):
        """Test gamma_sign for very small positive values."""
        x = torch.tensor([1e-10, 1e-100, 1e-300], dtype=torch.float64)
        result = torchscience.special_functions.gamma_sign(x)
        expected = torch.ones(3, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_autocast(self):
        """Test that autocast works correctly."""
        x = torch.tensor([0.5, 1.0, 2.0, -0.5, -1.5], dtype=torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.gamma_sign(x)
        # Result should be finite and +1 or -1
        assert torch.all((result == 1.0) | (result == -1.0))
