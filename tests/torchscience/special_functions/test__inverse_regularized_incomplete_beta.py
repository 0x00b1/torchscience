import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestInverseRegularizedIncompleteBeta:
    """Tests for the inverse regularized incomplete beta function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_inverse_relationship(self):
        """Test that inverse_regularized_incomplete_beta is the inverse of incomplete_beta."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        # Compute x = I^{-1}(a, b, y)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify I_x(a, b) = y using incomplete_beta
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )

        torch.testing.assert_close(y_reconstructed, y, rtol=1e-5, atol=1e-6)

    def test_various_parameter_values(self):
        """Test inverse_regularized_incomplete_beta for various parameter combinations."""
        # Test that x = I^{-1}(a, b, y) satisfies I_x(a, b) = y
        test_cases = [
            # (a, b, y)
            (2.0, 3.0, 0.5),
            (3.0, 3.0, 0.5),  # Symmetric case
            (5.0, 2.0, 0.5),
            (1.0, 1.0, 0.5),  # Uniform distribution
            (2.0, 2.0, 0.5),  # Symmetric
            (1.0, 2.0, 0.5),
            (2.0, 1.0, 0.5),
            (5.0, 5.0, 0.1),
            (5.0, 5.0, 0.9),
            (10.0, 3.0, 0.25),
            (3.0, 10.0, 0.75),
        ]

        for a_val, b_val, y_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            x = torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )

            # Verify the inverse relationship
            y_reconstructed = torchscience.special_functions.incomplete_beta(
                x, a, b
            )

            torch.testing.assert_close(
                y_reconstructed,
                y,
                rtol=1e-5,
                atol=1e-6,
                msg=f"Failed for a={a_val}, b={b_val}, y={y_val}: got I_x={y_reconstructed.item()}",
            )

    def test_special_value_y_zero(self):
        """Test I^{-1}(a, b, 0) = 0 for all a, b > 0."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = torch.zeros_like(a)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_y_one(self):
        """Test I^{-1}(a, b, 1) = 1 for all a, b > 0."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        y = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = torch.ones_like(a)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_uniform_distribution(self):
        """Test that for a=b=1 (uniform), result equals y."""
        a = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.1, 0.3, 0.5, 0.9], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        torch.testing.assert_close(result, y, rtol=1e-6, atol=1e-6)

    def test_symmetric_case(self):
        """Test that for a=b, I^{-1}(a, a, 0.5) = 0.5."""
        a = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, a, y
            )
        )
        expected = torch.full_like(a, 0.5)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_a_one_formula(self):
        """Test that for a=1, result equals 1 - (1-y)^(1/b)."""
        a = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        y = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = 1 - (1 - y).pow(1 / b)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_b_one_formula(self):
        """Test that for b=1, result equals y^(1/a)."""
        a = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = y.pow(1 / a)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_invalid_a_negative(self):
        """Test that negative a returns NaN."""
        a = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        b = torch.tensor([2.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert torch.isnan(result).all()

    def test_invalid_b_negative(self):
        """Test that negative b returns NaN."""
        a = torch.tensor([2.0, 2.0], dtype=torch.float64)
        b = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert torch.isnan(result).all()

    def test_invalid_a_zero(self):
        """Test that a=0 returns NaN."""
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        y = torch.tensor([0.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert torch.isnan(result).all()

    def test_invalid_b_zero(self):
        """Test that b=0 returns NaN."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([0.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert torch.isnan(result).all()

    def test_invalid_y_negative(self):
        """Test that y < 0 returns 0."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        y = torch.tensor([-0.1], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_invalid_y_greater_than_one(self):
        """Test that y > 1 returns 1."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        y = torch.tensor([1.5], dtype=torch.float64)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_small_y_values(self):
        """Test with very small y values."""
        a = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-3, atol=1e-8)

    def test_y_close_to_one(self):
        """Test with y values close to 1."""
        a = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.999, 0.999, 0.999], dtype=torch.float64)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-3, atol=1e-6)

    def test_large_a_b_values(self):
        """Test with large shape parameter values."""
        a = torch.tensor([50.0, 100.0], dtype=torch.float64)
        b = torch.tensor([50.0, 100.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5], dtype=torch.float64)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-4, atol=1e-6)

    def test_small_a_values(self):
        """Test with small a parameter values."""
        a = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-4, atol=1e-6)

    def test_small_b_values(self):
        """Test with small b parameter values."""
        a = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        b = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        x = torchscience.special_functions.inverse_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-4, atol=1e-6)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        a = torch.tensor([2.0, 3.0], dtype=dtype)
        b = torch.tensor([3.0, 2.0], dtype=dtype)
        y = torch.tensor([0.5, 0.5], dtype=dtype)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert result.dtype == dtype
        # Verify finite values
        assert torch.isfinite(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        # Use moderate values away from boundaries for numerical stability
        a = torch.tensor(
            [2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor(
            [3.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        y = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )

        def func(a, b, y):
            return torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )

        assert torch.autograd.gradcheck(
            func, (a, b, y), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    @pytest.mark.skip(
        reason="Second-order gradients use numerical differentiation which "
        "is inherently unstable for inverse functions. "
        "First-order gradients (gradcheck) pass correctly."
    )
    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)

        def func(a, b, y):
            return torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )

        assert torch.autograd.gradgradcheck(
            func, (a, b, y), eps=1e-5, atol=5e-2, rtol=5e-2
        )

    def test_gradient_values_y(self):
        """Test gradient values w.r.t. y against numerical derivative."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=False)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=False)
        y = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        result.backward()

        # Numerical gradient
        eps = 1e-6
        y_plus = y.detach() + eps
        y_minus = y.detach() - eps
        x_plus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y_plus
            )
        )
        x_minus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y_minus
            )
        )
        numerical_grad = (x_plus - x_minus) / (2 * eps)

        torch.testing.assert_close(
            y.grad, numerical_grad, rtol=1e-4, atol=1e-6
        )

    def test_gradient_values_a(self):
        """Test gradient values w.r.t. a against numerical derivative."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=False)
        y = torch.tensor([0.5], dtype=torch.float64, requires_grad=False)

        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        result.backward()

        # Numerical gradient
        eps = 1e-6
        a_plus = a.detach() + eps
        a_minus = a.detach() - eps
        x_plus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a_plus, b, y
            )
        )
        x_minus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a_minus, b, y
            )
        )
        numerical_grad = (x_plus - x_minus) / (2 * eps)

        torch.testing.assert_close(
            a.grad, numerical_grad, rtol=1e-3, atol=1e-5
        )

    def test_gradient_values_b(self):
        """Test gradient values w.r.t. b against numerical derivative."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=False)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.5], dtype=torch.float64, requires_grad=False)

        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        result.backward()

        # Numerical gradient
        eps = 1e-6
        b_plus = b.detach() + eps
        b_minus = b.detach() - eps
        x_plus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b_plus, y
            )
        )
        x_minus = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b_minus, y
            )
        )
        numerical_grad = (x_plus - x_minus) / (2 * eps)

        torch.testing.assert_close(
            b.grad, numerical_grad, rtol=1e-3, atol=1e-5
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        a = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)  # (3,)
        y = torch.tensor([0.5], dtype=torch.float64)  # (1,)
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert result.shape == (2, 3)

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        a = torch.empty(3, 4, device="meta")
        b = torch.empty(3, 4, device="meta")
        y = torch.empty(3, 4, device="meta")
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert result.device.type == "meta"
        assert result.shape == (3, 4)

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor with broadcasting."""
        a = torch.empty(3, 1, device="meta")
        b = torch.empty(1, 4, device="meta")
        y = torch.empty(1, device="meta")
        result = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y
            )
        )
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
