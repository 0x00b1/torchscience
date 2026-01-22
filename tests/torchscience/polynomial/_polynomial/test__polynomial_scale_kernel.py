# tests/torchscience/polynomial/_polynomial/test__polynomial_scale_kernel.py
import torch

from torchscience.polynomial import polynomial


class TestPolynomialScaleKernel:
    """Tests for polynomial_scale C++ kernel."""

    def test_scale_basic(self):
        """Scale polynomial by a scalar."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        c = torch.tensor(2.0)
        result = p * c
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(result, expected)

    def test_scale_by_zero(self):
        """Scale polynomial by zero."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        c = torch.tensor(0.0)
        result = p * c
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_scale_by_one(self):
        """Scale polynomial by one (identity)."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        c = torch.tensor(1.0)
        result = p * c
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_scale_negative(self):
        """Scale polynomial by negative scalar."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        c = torch.tensor(-2.0)
        result = p * c
        expected = torch.tensor([-2.0, -4.0, -6.0])
        torch.testing.assert_close(result, expected)

    def test_scale_batched(self):
        """Scale batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        c = torch.tensor([2.0, 3.0])
        result = p * c
        expected = torch.tensor([[2.0, 4.0], [9.0, 12.0]])
        torch.testing.assert_close(result, expected)

    def test_scale_broadcast_scalar(self):
        """Scale batched polynomials by single scalar."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        c = torch.tensor(2.0)
        result = p * c
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        torch.testing.assert_close(result, expected)

    def test_gradcheck(self):
        """Verify first-order gradients with cross-terms."""
        # Use float64 for numerical gradient checking
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        c = torch.randn(2, dtype=torch.float64, requires_grad=True)

        def fn(p, c):
            return torch.ops.torchscience.polynomial_scale(p, c)

        torch.autograd.gradcheck(fn, (p, c), raise_exception=True)

    def test_gradcheck_scalar_c(self):
        """Verify gradients when c is a scalar (0-d tensor)."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        # Scalar c must be expanded - kernel expects (B,) shape
        c = torch.randn(2, dtype=torch.float64, requires_grad=True)

        def fn(p, c):
            return torch.ops.torchscience.polynomial_scale(p, c)

        torch.autograd.gradcheck(fn, (p, c), raise_exception=True)

    def test_gradgradcheck(self):
        """Verify second-order gradients (cross-terms are non-zero!)."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        c = torch.randn(2, dtype=torch.float64, requires_grad=True)

        def fn(p, c):
            return torch.ops.torchscience.polynomial_scale(p, c)

        torch.autograd.gradgradcheck(fn, (p, c), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 3, device="meta")
        c = torch.empty(2, device="meta")
        result = torch.ops.torchscience.polynomial_scale(p, c)
        assert result.shape == (2, 3)
        assert result.device.type == "meta"

    def test_meta_tensor_backward(self):
        """Verify meta tensor shape inference for backward."""
        grad_output = torch.empty(2, 3, device="meta")
        p = torch.empty(2, 3, device="meta")
        c = torch.empty(2, device="meta")
        grad_p, grad_c = torch.ops.torchscience.polynomial_scale_backward(
            grad_output, p, c
        )
        assert grad_p.shape == (2, 3)
        assert grad_c.shape == (2,)
        assert grad_p.device.type == "meta"
        assert grad_c.device.type == "meta"

    def test_complex(self):
        """Scale complex polynomials."""
        p = polynomial(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        c = torch.tensor(2.0 + 1.0j)
        result = p * c
        # (1+2j)*(2+1j) = 2+1j+4j+2j^2 = 2+5j-2 = 0+5j
        # (3+4j)*(2+1j) = 6+3j+8j+4j^2 = 6+11j-4 = 2+11j
        expected = torch.tensor([0.0 + 5.0j, 2.0 + 11.0j])
        torch.testing.assert_close(result, expected)

    def test_gradient_cross_terms(self):
        """Explicitly verify that cross-term gradients are non-zero.

        For f(p, c) = c * p:
        - df/dp = c
        - df/dc = p
        - d2f/(dp dc) = 1 (cross-term!)
        """
        p = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True
        )
        c = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        # Forward
        out = torch.ops.torchscience.polynomial_scale(p, c)

        # Check forward output
        expected_out = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float64)
        torch.testing.assert_close(out, expected_out)

        # Backward for grad_p
        grad_out = torch.ones_like(out)
        out.backward(grad_out, retain_graph=True)

        # grad_p[k] = c * grad_out[k] = 2.0 * 1.0 = 2.0
        expected_grad_p = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64)
        torch.testing.assert_close(p.grad, expected_grad_p)

        # grad_c = sum(p[k] * grad_out[k]) = 1+2+3 = 6
        expected_grad_c = torch.tensor([6.0], dtype=torch.float64)
        torch.testing.assert_close(c.grad, expected_grad_c)
