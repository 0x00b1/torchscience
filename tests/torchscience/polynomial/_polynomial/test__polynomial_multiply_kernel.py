# tests/torchscience/polynomial/_polynomial/test__polynomial_multiply_kernel.py
import torch

from torchscience.polynomial import polynomial


class TestPolynomialMultiplyKernel:
    """Tests for polynomial_multiply C++ kernel."""

    def test_multiply_simple(self):
        """Multiply two simple polynomials."""
        # (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        result = p * q
        expected = torch.tensor([3.0, 10.0, 8.0])
        torch.testing.assert_close(result, expected)

    def test_multiply_different_degree(self):
        """Multiply polynomials of different degrees."""
        # (1 + x) * (1 + x + x^2) = 1 + 2x + 2x^2 + x^3
        p = polynomial(torch.tensor([1.0, 1.0]))
        q = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        result = p * q
        expected = torch.tensor([1.0, 2.0, 2.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_multiply_by_constant(self):
        """Multiply by constant polynomial."""
        # 2 * (1 + x + x^2) = 2 + 2x + 2x^2
        p = polynomial(torch.tensor([2.0]))
        q = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        result = p * q
        expected = torch.tensor([2.0, 2.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_multiply_batched(self):
        """Multiply batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [1.0, 1.0]]))
        q = polynomial(torch.tensor([[3.0, 4.0], [1.0, 1.0]]))
        result = p * q
        # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
        # (1 + x)(1 + x) = 1 + 2x + x^2
        expected = torch.tensor([[3.0, 10.0, 8.0], [1.0, 2.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_multiply_broadcast(self):
        """Multiply with broadcasting."""
        p = polynomial(torch.tensor([[1.0, 1.0]]))  # (1, 2)
        q = polynomial(torch.tensor([[2.0], [3.0]]))  # (2, 1)
        result = p * q
        # (1 + x) * 2 = 2 + 2x
        # (1 + x) * 3 = 3 + 3x
        expected = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
        torch.testing.assert_close(result, expected)

    def test_gradcheck(self):
        """Verify first-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_multiply(p, q)

        torch.autograd.gradcheck(fn, (p, q), raise_exception=True)

    def test_gradgradcheck(self):
        """Verify second-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_multiply(p, q)

        torch.autograd.gradgradcheck(fn, (p, q), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 3, device="meta")
        q = torch.empty(2, 5, device="meta")
        result = torch.ops.torchscience.polynomial_multiply(p, q)
        # N + M - 1 = 3 + 5 - 1 = 7
        assert result.shape == (2, 7)
        assert result.device.type == "meta"

    def test_complex(self):
        """Multiply complex polynomials."""
        # (1 + i) * (1 - i) = 1 - i^2 = 1 + 1 = 2
        p = polynomial(torch.tensor([1.0 + 1.0j]))
        q = polynomial(torch.tensor([1.0 - 1.0j]))
        result = p * q
        expected = torch.tensor([2.0 + 0.0j])
        torch.testing.assert_close(result, expected)

    def test_multiply_identity(self):
        """Multiply by identity polynomial (1)."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        one = polynomial(torch.tensor([1.0]))
        result = p * one
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected)
