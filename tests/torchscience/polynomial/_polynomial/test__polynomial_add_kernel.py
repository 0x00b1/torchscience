# tests/torchscience/polynomial/_polynomial/test__polynomial_add_kernel.py
import torch

from torchscience.polynomial import polynomial


class TestPolynomialAddKernel:
    """Tests for polynomial_add C++ kernel."""

    def test_add_same_degree(self):
        """Add polynomials of same degree."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0]))
        result = p + q
        expected = torch.tensor([5.0, 7.0, 9.0])
        torch.testing.assert_close(result, expected)

    def test_add_different_degree(self):
        """Add polynomials of different degrees."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0]))
        result = p + q
        expected = torch.tensor([5.0, 7.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_add_batched(self):
        """Add batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        result = p + q
        expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
        torch.testing.assert_close(result, expected)

    def test_add_broadcast(self):
        """Add with broadcasting."""
        p = polynomial(torch.tensor([[1.0, 2.0]]))
        q = polynomial(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        result = p + q
        expected = torch.tensor([[4.0, 6.0], [6.0, 8.0]])
        torch.testing.assert_close(result, expected)

    def test_gradcheck(self):
        """Verify first-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_add(p, q)

        torch.autograd.gradcheck(fn, (p, q), raise_exception=True)

    def test_gradgradcheck(self):
        """Verify second-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_add(p, q)

        torch.autograd.gradgradcheck(fn, (p, q), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 3, device="meta")
        q = torch.empty(2, 5, device="meta")
        result = torch.ops.torchscience.polynomial_add(p, q)
        assert result.shape == (2, 5)
        assert result.device.type == "meta"

    def test_complex(self):
        """Add complex polynomials."""
        p = polynomial(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        q = polynomial(torch.tensor([5.0 + 6.0j, 7.0 + 8.0j]))
        result = p + q
        expected = torch.tensor([6.0 + 8.0j, 10.0 + 12.0j])
        torch.testing.assert_close(result, expected)
