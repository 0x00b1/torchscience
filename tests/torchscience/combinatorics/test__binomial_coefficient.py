import pytest
import torch
import torch.testing

import torchscience.combinatorics


class TestBinomialCoefficient:
    """Tests for the binomial coefficient function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_pascals_triangle_row_5(self):
        """Test C(5, k) for k=0..5 matches Pascal's triangle."""
        n = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        k = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor(
            [1.0, 5.0, 10.0, 10.0, 5.0, 1.0], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_against_scipy_formula(self):
        """Test against gamma function formula."""
        n = torch.tensor([10.0, 7.0, 20.0], dtype=torch.float64)
        k = torch.tensor([3.0, 4.0, 10.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.exp(
            torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_0_equals_1(self):
        """Test C(n, 0) = 1 for various n."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0, 100.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_n_equals_1(self):
        """Test C(n, n) = 1 for non-negative integer n."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64)
        k = n.clone()
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.ones(4, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_1_equals_n(self):
        """Test C(n, 1) = n."""
        n = torch.tensor([1.0, 2.0, 5.0, 10.0, 100.0], dtype=torch.float64)
        k = torch.ones(5, dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        torch.testing.assert_close(result, n, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """Test C(n, k) = C(n, n-k)."""
        n = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        result_k = torchscience.combinatorics.binomial_coefficient(n, k)
        result_n_minus_k = torchscience.combinatorics.binomial_coefficient(
            n, n - k
        )
        torch.testing.assert_close(
            result_k, result_n_minus_k, rtol=1e-10, atol=1e-10
        )

    def test_k_negative_returns_zero(self):
        """Test C(n, k) = 0 for k < 0."""
        n = torch.tensor([5.0, 10.0], dtype=torch.float64)
        k = torch.tensor([-1.0, -2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.zeros(2, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_k_greater_than_n_returns_zero(self):
        """Test C(n, k) = 0 for k > n when n >= 0."""
        n = torch.tensor([5.0, 3.0], dtype=torch.float64)
        k = torch.tensor([6.0, 10.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.zeros(2, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_generalized_negative_n(self):
        """Test generalized binomial coefficient with negative n."""
        # C(-0.5, 2) = (-0.5)(-1.5) / 2! = 0.75 / 2 = 0.375
        n = torch.tensor([-0.5], dtype=torch.float64)
        k = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor([0.375], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_generalized_fractional_n(self):
        """Test generalized binomial coefficient with fractional n."""
        # C(0.5, 2) = (0.5)(-0.5) / 2! = -0.25 / 2 = -0.125
        n = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor([-0.125], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        n = torch.tensor([5.0, 10.0], dtype=dtype)
        k = torch.tensor([2.0, 3.0], dtype=dtype)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.dtype == dtype
        expected = torch.tensor([10.0, 120.0], dtype=dtype)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        n = torch.tensor([[5.0], [10.0]], dtype=torch.float64)  # (2, 1)
        k = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)  # (3,)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.shape == (2, 3)
        expected = torch.tensor(
            [[1.0, 5.0, 10.0], [1.0, 10.0, 45.0]], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        n = torch.tensor(
            [5.0, 10.0, 7.0], dtype=torch.float64, requires_grad=True
        )
        k = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def func(n, k):
            return torchscience.combinatorics.binomial_coefficient(n, k)

        assert torch.autograd.gradcheck(
            func, (n, k), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        n = torch.tensor([5.0, 10.0], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)

        def func(n, k):
            return torchscience.combinatorics.binomial_coefficient(n, k)

        # Use slightly looser tolerance for second-order gradients
        # due to numerical precision in digamma/trigamma composition
        assert torch.autograd.gradgradcheck(
            func, (n, k), eps=1e-5, atol=1e-3, rtol=1e-3
        )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        n = torch.empty(3, 4, device="meta")
        k = torch.empty(3, 4, device="meta")
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
