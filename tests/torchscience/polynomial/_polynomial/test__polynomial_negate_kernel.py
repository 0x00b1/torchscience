# tests/torchscience/polynomial/_polynomial/test__polynomial_negate_kernel.py
import torch

from torchscience.polynomial import polynomial


class TestPolynomialNegateKernel:
    """Tests for polynomial_negate C++ kernel."""

    def test_negate_basic(self):
        """Negate a simple polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = -p
        expected = torch.tensor([-1.0, -2.0, -3.0])
        torch.testing.assert_close(result.coeffs, expected)

    def test_negate_batched(self):
        """Negate batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = -p
        expected = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        torch.testing.assert_close(result.coeffs, expected)

    def test_gradcheck(self):
        """Verify first-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

        def fn(p):
            return torch.ops.torchscience.polynomial_negate(p)

        torch.autograd.gradcheck(fn, (p,), raise_exception=True)

    def test_gradgradcheck(self):
        """Verify second-order gradients."""
        p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

        def fn(p):
            return torch.ops.torchscience.polynomial_negate(p)

        torch.autograd.gradgradcheck(fn, (p,), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 3, device="meta")
        result = torch.ops.torchscience.polynomial_negate(p)
        assert result.shape == (2, 3)
        assert result.device.type == "meta"

    def test_complex(self):
        """Negate complex polynomials."""
        p = polynomial(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        result = -p
        expected = torch.tensor([-1.0 - 2.0j, -3.0 - 4.0j])
        torch.testing.assert_close(result.coeffs, expected)

    def test_double_negate(self):
        """Verify double negation returns original polynomial."""
        p = polynomial(torch.tensor([1.0, -2.0, 3.0, -4.0]))
        result = -(-p)
        torch.testing.assert_close(result.coeffs, p.coeffs)

    def test_negate_zeros(self):
        """Negate polynomial with all zero coefficients."""
        p = polynomial(torch.tensor([0.0, 0.0, 0.0]))
        result = -p
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result.coeffs, expected)

    def test_negate_preserves_dtype(self):
        """Verify negate preserves input dtype."""
        for dtype in [torch.float32, torch.float64]:
            p = polynomial(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
            result = -p
            assert result.coeffs.dtype == dtype
