# tests/torchscience/polynomial/_polynomial/test__polynomial_subtract_kernel.py
import torch

from torchscience.polynomial import polynomial


class TestPolynomialSubtractKernel:
    """Tests for polynomial_subtract C++ kernel."""

    def test_subtract_same_degree(self):
        """Subtract polynomials of same degree."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0]))
        result = p - q
        expected = torch.tensor([-3.0, -3.0, -3.0])
        torch.testing.assert_close(result, expected)

    def test_subtract_different_degree(self):
        """Subtract polynomials of different degrees."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0]))
        result = p - q
        expected = torch.tensor([-3.0, -3.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_subtract_different_degree_reversed(self):
        """Subtract where q has higher degree."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0]))
        result = p - q
        expected = torch.tensor([-3.0, -3.0, -6.0])
        torch.testing.assert_close(result, expected)

    def test_subtract_batched(self):
        """Subtract batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        result = p - q
        expected = torch.tensor([[-4.0, -4.0], [-4.0, -4.0]])
        torch.testing.assert_close(result, expected)

    def test_subtract_broadcast(self):
        """Subtract with broadcasting."""
        p = polynomial(torch.tensor([[1.0, 2.0]]))
        q = polynomial(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        result = p - q
        expected = torch.tensor([[-2.0, -2.0], [-4.0, -4.0]])
        torch.testing.assert_close(result, expected)

    def test_gradcheck(self):
        """Verify first-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_subtract(p, q)

        torch.autograd.gradcheck(fn, (p, q), raise_exception=True)

    def test_gradgradcheck(self):
        """Verify second-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(p, q):
            return torch.ops.torchscience.polynomial_subtract(p, q)

        torch.autograd.gradgradcheck(fn, (p, q), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 3, device="meta")
        q = torch.empty(2, 5, device="meta")
        result = torch.ops.torchscience.polynomial_subtract(p, q)
        assert result.shape == (2, 5)
        assert result.device.type == "meta"

    def test_complex(self):
        """Subtract complex polynomials."""
        p = polynomial(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        q = polynomial(torch.tensor([5.0 + 6.0j, 7.0 + 8.0j]))
        result = p - q
        expected = torch.tensor([-4.0 - 4.0j, -4.0 - 4.0j])
        torch.testing.assert_close(result, expected)

    def test_subtract_self(self):
        """Subtracting a polynomial from itself should give zero."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = p - p
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_backward_negation(self):
        """Verify that grad_q is negated (key difference from add)."""
        p = torch.tensor([[1.0, 2.0]], requires_grad=True)
        q = torch.tensor([[3.0, 4.0]], requires_grad=True)

        result = torch.ops.torchscience.polynomial_subtract(p, q)
        loss = result.sum()
        loss.backward()

        # For subtraction, grad_p should be 1s, grad_q should be -1s
        torch.testing.assert_close(p.grad, torch.tensor([[1.0, 1.0]]))
        torch.testing.assert_close(q.grad, torch.tensor([[-1.0, -1.0]]))
