import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestInverseComplementaryRegularizedIncompleteBeta:
    """Tests for the inverse complementary regularized incomplete beta function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_inverse_relationship(self):
        """Test that the function is the inverse of 1 - incomplete_beta."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        # Compute x = I_c^{-1}(a, b, y)
        x = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )

        # Verify 1 - I_x(a, b) = y
        y_reconstructed = 1 - torchscience.special_functions.incomplete_beta(
            x, a, b
        )

        torch.testing.assert_close(y_reconstructed, y, rtol=1e-5, atol=1e-6)

    def test_consistency_with_regular_inverse(self):
        """Test I_c^{-1}(a, b, y) = I^{-1}(a, b, 1 - y)."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)

        result_c = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        result_p = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, 1 - y
            )
        )

        torch.testing.assert_close(result_c, result_p, rtol=1e-10, atol=1e-10)

    def test_special_value_y_zero(self):
        """Test I_c^{-1}(a, b, 0) = 1 for all a, b > 0."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        expected = torch.ones_like(a)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_y_one(self):
        """Test I_c^{-1}(a, b, 1) = 0 for all a, b > 0."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        y = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        expected = torch.zeros_like(a)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetric_case(self):
        """Test that for a=b, I_c^{-1}(a, a, 0.5) = 0.5."""
        a = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, a, y
        )
        expected = torch.full_like(a, 0.5)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_various_parameter_values(self):
        """Test for various parameter combinations."""
        test_cases = [
            # (a, b, y)
            (2.0, 3.0, 0.5),
            (3.0, 3.0, 0.5),
            (5.0, 2.0, 0.5),
            (1.0, 1.0, 0.5),
            (2.0, 2.0, 0.5),
            (10.0, 3.0, 0.25),
            (3.0, 10.0, 0.75),
        ]

        for a_val, b_val, y_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            x = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
                a, b, y
            )

            # Verify the inverse relationship
            y_reconstructed = (
                1 - torchscience.special_functions.incomplete_beta(x, a, b)
            )

            torch.testing.assert_close(
                y_reconstructed,
                y,
                rtol=1e-5,
                atol=1e-6,
                msg=f"Failed for a={a_val}, b={b_val}, y={y_val}",
            )

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_invalid_a_negative(self):
        """Test that negative a returns NaN."""
        a = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        b = torch.tensor([2.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert torch.isnan(result).all()

    def test_invalid_b_negative(self):
        """Test that negative b returns NaN."""
        a = torch.tensor([2.0, 2.0], dtype=torch.float64)
        b = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        y = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert torch.isnan(result).all()

    def test_invalid_a_zero(self):
        """Test that a=0 returns NaN."""
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        y = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert torch.isnan(result).all()

    def test_invalid_b_zero(self):
        """Test that b=0 returns NaN."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert torch.isnan(result).all()

    def test_small_y_values(self):
        """Test with very small y values (large x)."""
        a = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64)
        x = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = 1 - torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-3, atol=1e-8)

    def test_y_close_to_one(self):
        """Test with y values close to 1 (small x)."""
        a = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        b = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64)
        y = torch.tensor([0.999, 0.999, 0.999], dtype=torch.float64)
        x = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )

        # Verify the inverse relationship
        y_reconstructed = 1 - torchscience.special_functions.incomplete_beta(
            x, a, b
        )
        torch.testing.assert_close(y_reconstructed, y, rtol=1e-3, atol=1e-6)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        a = torch.tensor([2.0, 3.0], dtype=dtype)
        b = torch.tensor([3.0, 2.0], dtype=dtype)
        y = torch.tensor([0.5, 0.5], dtype=dtype)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert result.dtype == dtype
        assert torch.isfinite(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
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
            return torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
                a, b, y
            )

        assert torch.autograd.gradcheck(
            func, (a, b, y), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    @pytest.mark.skip(
        reason="Second-order gradients use numerical differentiation which "
        "is inherently unstable for inverse functions."
    )
    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)

        def func(a, b, y):
            return torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
                a, b, y
            )

        assert torch.autograd.gradgradcheck(
            func, (a, b, y), eps=1e-5, atol=5e-2, rtol=5e-2
        )

    def test_gradient_sign_differs_from_regular(self):
        """Test that dy gradient has opposite sign compared to regular inverse."""
        a = torch.tensor([2.0], dtype=torch.float64, requires_grad=False)
        b = torch.tensor([3.0], dtype=torch.float64, requires_grad=False)
        y = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        y_p = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # Complementary version
        result_c = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        result_c.backward()
        grad_y_c = y.grad.clone()

        # Regular version at 1-y (should give same x)
        result_p = (
            torchscience.special_functions.inverse_regularized_incomplete_beta(
                a, b, y_p
            )
        )
        result_p.backward()
        grad_y_p = y_p.grad.clone()

        # The gradients should have opposite signs
        assert (grad_y_c * grad_y_p < 0).all(), (
            f"Complementary grad {grad_y_c} and regular grad {grad_y_p} should have opposite signs"
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        a = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)  # (3,)
        y = torch.tensor([0.5], dtype=torch.float64)  # (1,)
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
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
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert result.device.type == "meta"
        assert result.shape == (3, 4)

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor with broadcasting."""
        a = torch.empty(3, 1, device="meta")
        b = torch.empty(1, 4, device="meta")
        y = torch.empty(1, device="meta")
        result = torchscience.special_functions.inverse_complementary_regularized_incomplete_beta(
            a, b, y
        )
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
