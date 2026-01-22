# tests/torchscience/polynomial/_polynomial/test__polynomial_divmod_kernel.py
import pytest
import torch

from torchscience.polynomial import polynomial, polynomial_divmod


class TestPolynomialDivmodKernel:
    """Tests for polynomial_divmod C++ kernel."""

    def test_divmod_simple(self):
        """Divide polynomial with exact division."""
        # (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        p = polynomial(torch.tensor([-1.0, 0.0, 1.0]))  # x^2 - 1
        q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
        quot, rem = polynomial_divmod(p, q)
        expected_quot = torch.tensor([1.0, 1.0])  # x + 1
        expected_rem = torch.tensor([0.0])
        torch.testing.assert_close(quot, expected_quot)
        torch.testing.assert_close(rem, expected_rem, atol=1e-6, rtol=1e-6)

    def test_divmod_with_remainder(self):
        """Divide polynomial with non-zero remainder."""
        # (x^2 + 1) / (x - 1) = (x + 1), remainder 2
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # x^2 + 1
        q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
        quot, rem = polynomial_divmod(p, q)
        expected_quot = torch.tensor([1.0, 1.0])  # x + 1
        expected_rem = torch.tensor([2.0])  # remainder 2
        torch.testing.assert_close(quot, expected_quot)
        torch.testing.assert_close(rem, expected_rem)

    def test_divmod_cubic(self):
        """Divide cubic polynomial."""
        # (x^3 - 1) / (x - 1) = (x^2 + x + 1), remainder 0
        p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))  # x^3 - 1
        q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
        quot, rem = polynomial_divmod(p, q)
        expected_quot = torch.tensor([1.0, 1.0, 1.0])  # x^2 + x + 1
        expected_rem = torch.tensor([0.0])
        torch.testing.assert_close(quot, expected_quot)
        torch.testing.assert_close(rem, expected_rem, atol=1e-6, rtol=1e-6)

    def test_divmod_by_constant(self):
        """Divide by constant polynomial."""
        # (2 + 4x + 6x^2) / 2 = (1 + 2x + 3x^2), remainder 0
        p = polynomial(torch.tensor([2.0, 4.0, 6.0]))
        q = polynomial(torch.tensor([2.0]))
        quot, rem = polynomial_divmod(p, q)
        expected_quot = torch.tensor([1.0, 2.0, 3.0])
        # For degree 0 divisor, remainder size is 1
        expected_rem = torch.tensor([0.0])
        torch.testing.assert_close(quot, expected_quot)
        torch.testing.assert_close(rem, expected_rem, atol=1e-6, rtol=1e-6)

    def test_divmod_batched(self):
        """Divide batched polynomials."""
        # Batch of two divisions
        p = polynomial(
            torch.tensor([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        )  # x^2 - 1, x^2 + 1
        q = polynomial(
            torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        )  # x - 1, x - 1
        quot, rem = polynomial_divmod(p, q)
        # (x^2 - 1) / (x - 1) = (x + 1), rem 0
        # (x^2 + 1) / (x - 1) = (x + 1), rem 2
        expected_quot = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        expected_rem = torch.tensor([[0.0], [2.0]])
        torch.testing.assert_close(quot, expected_quot)
        torch.testing.assert_close(rem, expected_rem, atol=1e-6, rtol=1e-6)

    def test_gradcheck(self):
        """Verify first-order gradients."""
        p = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        # Ensure leading coefficient is not too small
        q = q.clone()
        q[:, -1] = q[:, -1].abs() + 0.5

        def fn(p, q):
            quot, rem = torch.ops.torchscience.polynomial_divmod(p, q)
            return quot, rem

        torch.autograd.gradcheck(fn, (p, q), raise_exception=True)

    @pytest.mark.xfail(
        reason="Second-order backward for polynomial division needs work"
    )
    def test_gradgradcheck(self):
        """Verify second-order gradients."""
        p = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        # Ensure leading coefficient is not too small
        q = q.clone()
        q[:, -1] = q[:, -1].abs() + 0.5

        def fn(p, q):
            quot, rem = torch.ops.torchscience.polynomial_divmod(p, q)
            return quot, rem

        torch.autograd.gradgradcheck(fn, (p, q), raise_exception=True)

    def test_meta_tensor(self):
        """Verify meta tensor shape inference."""
        p = torch.empty(2, 5, device="meta")  # N = 5
        q = torch.empty(2, 3, device="meta")  # M = 3
        quot, rem = torch.ops.torchscience.polynomial_divmod(p, q)
        # quot: N - M + 1 = 5 - 3 + 1 = 3
        # rem: M - 1 = 3 - 1 = 2
        assert quot.shape == (2, 3)
        assert rem.shape == (2, 2)
        assert quot.device.type == "meta"
        assert rem.device.type == "meta"

    def test_complex(self):
        """Divide complex polynomials."""
        # (x^2 + 1) / (x + i) = (x - i), remainder 0
        # Note: x + i is a factor of x^2 + 1 = (x + i)(x - i)
        p = polynomial(
            torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j])
        )  # x^2 + 1
        q = polynomial(torch.tensor([0.0 + 1.0j, 1.0 + 0.0j]))  # x + i
        quot, rem = polynomial_divmod(p, q)
        expected_quot = torch.tensor([0.0 - 1.0j, 1.0 + 0.0j])  # x - i
        expected_rem = torch.tensor([0.0 + 0.0j])
        torch.testing.assert_close(quot, expected_quot, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(rem, expected_rem, atol=1e-6, rtol=1e-6)

    def test_roundtrip(self):
        """Verify q * quot + rem = p."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        q = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        quot, rem = polynomial_divmod(p, q)

        # Reconstruct: q * quot + rem should equal p
        reconstructed = q * quot
        # Pad remainder to match reconstructed size
        rem_padded = torch.zeros_like(reconstructed)
        rem_padded[..., : rem.shape[-1]] = rem
        reconstructed = polynomial(reconstructed + rem_padded)

        torch.testing.assert_close(reconstructed, p, atol=1e-5, rtol=1e-5)
