"""Tests for ChebyshevPolynomialT arithmetic operations."""

import numpy as np
import pytest
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_add,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_multiply,
    chebyshev_polynomial_t_mulx,
    chebyshev_polynomial_t_negate,
    chebyshev_polynomial_t_pow,
    chebyshev_polynomial_t_scale,
    chebyshev_polynomial_t_subtract,
)


class TestChebyshevPolynomialTAdd:
    """Tests for chebyshev_polynomial_t_add."""

    def test_add_same_degree(self):
        """Add series of same degree."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([4.0, 5.0, 6.0]))
        c = chebyshev_polynomial_t_add(a, b)
        torch.testing.assert_close(c, torch.tensor([5.0, 7.0, 9.0]))

    def test_add_different_degree(self):
        """Add series of different degrees (zero-pad shorter)."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        b = chebyshev_polynomial_t(torch.tensor([3.0, 4.0, 5.0]))
        c = chebyshev_polynomial_t_add(a, b)
        torch.testing.assert_close(c, torch.tensor([4.0, 6.0, 5.0]))

    def test_add_operator(self):
        """Test + operator."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        b = chebyshev_polynomial_t(torch.tensor([3.0, 4.0]))
        c = a + b
        torch.testing.assert_close(c, torch.tensor([4.0, 6.0]))

    def test_add_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebadd."""
        a_coeffs = [1.0, 2.0, 3.0]
        b_coeffs = [4.0, 5.0]

        a = chebyshev_polynomial_t(torch.tensor(a_coeffs))
        b = chebyshev_polynomial_t(torch.tensor(b_coeffs))
        c = chebyshev_polynomial_t_add(a, b)

        c_np = np_cheb.chebadd(a_coeffs, b_coeffs)

        np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-6)


class TestChebyshevPolynomialTSubtract:
    """Tests for chebyshev_polynomial_t_subtract."""

    def test_subtract_same_degree(self):
        """Subtract series of same degree."""
        a = chebyshev_polynomial_t(torch.tensor([5.0, 7.0, 9.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        c = chebyshev_polynomial_t_subtract(a, b)
        torch.testing.assert_close(c, torch.tensor([4.0, 5.0, 6.0]))

    def test_subtract_different_degree(self):
        """Subtract series of different degrees."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        c = chebyshev_polynomial_t_subtract(a, b)
        torch.testing.assert_close(c, torch.tensor([0.0, 0.0, 3.0]))

    def test_subtract_operator(self):
        """Test - operator."""
        a = chebyshev_polynomial_t(torch.tensor([5.0, 6.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        c = a - b
        torch.testing.assert_close(c, torch.tensor([4.0, 4.0]))

    def test_subtract_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebsub."""
        a_coeffs = [5.0, 4.0, 3.0]
        b_coeffs = [1.0, 2.0]

        a = chebyshev_polynomial_t(torch.tensor(a_coeffs))
        b = chebyshev_polynomial_t(torch.tensor(b_coeffs))
        c = chebyshev_polynomial_t_subtract(a, b)

        c_np = np_cheb.chebsub(a_coeffs, b_coeffs)

        np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-6)


class TestChebyshevPolynomialTNegate:
    """Tests for chebyshev_polynomial_t_negate."""

    def test_negate(self):
        """Negate series."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, -2.0, 3.0]))
        b = chebyshev_polynomial_t_negate(a)
        torch.testing.assert_close(b, torch.tensor([-1.0, 2.0, -3.0]))

    def test_negate_operator(self):
        """Test unary - operator."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, -2.0]))
        b = -a
        torch.testing.assert_close(b, torch.tensor([-1.0, 2.0]))


class TestChebyshevPolynomialTScale:
    """Tests for chebyshev_polynomial_t_scale."""

    def test_scale_by_scalar(self):
        """Scale by scalar tensor."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t_scale(a, torch.tensor(2.0))
        torch.testing.assert_close(b, torch.tensor([2.0, 4.0, 6.0]))

    def test_scale_operator(self):
        """Test * operator with scalar."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        b = a * torch.tensor(3.0)
        torch.testing.assert_close(b, torch.tensor([3.0, 6.0]))

    def test_scale_rmul_operator(self):
        """Test scalar * series."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        b = torch.tensor(3.0) * a
        torch.testing.assert_close(b, torch.tensor([3.0, 6.0]))


class TestChebyshevPolynomialTMultiply:
    """Tests for chebyshev_polynomial_t_multiply using linearization."""

    def test_multiply_t0_t0(self):
        """T_0 * T_0 = T_0."""
        a = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
        b = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
        c = chebyshev_polynomial_t_multiply(a, b)
        torch.testing.assert_close(c, torch.tensor([1.0]))

    def test_multiply_t0_t1(self):
        """T_0 * T_1 = T_1."""
        a = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
        b = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        c = chebyshev_polynomial_t_multiply(a, b)
        # Result should be T_1
        torch.testing.assert_close(c, torch.tensor([0.0, 1.0]))

    def test_multiply_t1_t1(self):
        """T_1 * T_1 = 0.5*(T_2 + T_0) = 0.5*T_0 + 0.5*T_2."""
        a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        b = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        c = chebyshev_polynomial_t_multiply(a, b)
        # T_1 * T_1 = 0.5*(T_2 + T_0) -> [0.5, 0, 0.5]
        torch.testing.assert_close(c, torch.tensor([0.5, 0.0, 0.5]))

    def test_multiply_t1_t2(self):
        """T_1 * T_2 = 0.5*(T_3 + T_1)."""
        a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        b = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        c = chebyshev_polynomial_t_multiply(a, b)
        # T_1 * T_2 = 0.5*(T_3 + T_1) -> [0, 0.5, 0, 0.5]
        torch.testing.assert_close(c, torch.tensor([0.0, 0.5, 0.0, 0.5]))

    def test_multiply_linear(self):
        """(1 + T_1) * (2 + 3*T_1)."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))  # 1 + T_1
        b = chebyshev_polynomial_t(torch.tensor([2.0, 3.0]))  # 2 + 3*T_1
        c = chebyshev_polynomial_t_multiply(a, b)
        # = 2*T_0 + 3*T_1 + 2*T_1 + 3*T_1*T_1
        # = 2 + 5*T_1 + 3*0.5*(T_0 + T_2)
        # = 2 + 5*T_1 + 1.5*T_0 + 1.5*T_2
        # = 3.5*T_0 + 5*T_1 + 1.5*T_2
        torch.testing.assert_close(c, torch.tensor([3.5, 5.0, 1.5]))

    def test_multiply_operator(self):
        """Test * operator between series."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        c = a * b
        # (1 + T_1)^2 = 1 + 2*T_1 + T_1^2 = 1 + 2*T_1 + 0.5*(T_0 + T_2)
        # = 1.5*T_0 + 2*T_1 + 0.5*T_2
        torch.testing.assert_close(c, torch.tensor([1.5, 2.0, 0.5]))

    def test_multiply_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebmul."""
        a_coeffs = [1.0, 2.0, 3.0]
        b_coeffs = [4.0, 5.0]

        a = chebyshev_polynomial_t(torch.tensor(a_coeffs))
        b = chebyshev_polynomial_t(torch.tensor(b_coeffs))
        c = chebyshev_polynomial_t_multiply(a, b)

        c_np = np_cheb.chebmul(a_coeffs, b_coeffs)

        np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-6)

    def test_multiply_evaluation_consistency(self):
        """(a*b)(x) == a(x)*b(x) for all x in [-1,1]."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([4.0, -1.0, 2.0]))
        c = chebyshev_polynomial_t_multiply(a, b)

        x = torch.linspace(-1, 1, 20)
        y_product = chebyshev_polynomial_t_evaluate(c, x)
        y_separate = chebyshev_polynomial_t_evaluate(
            a, x
        ) * chebyshev_polynomial_t_evaluate(b, x)

        torch.testing.assert_close(y_product, y_separate, atol=1e-6, rtol=1e-6)


class TestChebyshevPolynomialTMulx:
    """Tests for chebyshev_polynomial_t_mulx (multiply by x)."""

    def test_mulx_t0(self):
        """x * T_0 = T_1."""
        a = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
        b = chebyshev_polynomial_t_mulx(a)
        # x * T_0 = T_1 -> [0, 1]
        torch.testing.assert_close(b, torch.tensor([0.0, 1.0]))

    def test_mulx_t1(self):
        """x * T_1 = 0.5*(T_0 + T_2)."""
        a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        b = chebyshev_polynomial_t_mulx(a)
        # x * T_1 = 0.5*(T_0 + T_2) -> [0.5, 0, 0.5]
        torch.testing.assert_close(b, torch.tensor([0.5, 0.0, 0.5]))

    def test_mulx_t2(self):
        """x * T_2 = 0.5*(T_1 + T_3)."""
        a = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        b = chebyshev_polynomial_t_mulx(a)
        # x * T_2 = 0.5*(T_1 + T_3) -> [0, 0.5, 0, 0.5]
        torch.testing.assert_close(b, torch.tensor([0.0, 0.5, 0.0, 0.5]))

    def test_mulx_linear(self):
        """x * (1 + 2*T_1) = T_1 + 2*0.5*(T_0 + T_2) = T_0 + T_1 + T_2."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))  # 1 + 2*T_1
        b = chebyshev_polynomial_t_mulx(a)
        # x * (1 + 2*T_1) = T_1 + 2*0.5*(T_0 + T_2) = T_0 + T_1 + T_2
        torch.testing.assert_close(b, torch.tensor([1.0, 1.0, 1.0]))

    def test_mulx_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebmulx."""
        coeffs = [1.0, 2.0, 3.0]

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        b = chebyshev_polynomial_t_mulx(a)

        b_np = np_cheb.chebmulx(coeffs)

        np.testing.assert_allclose(b.numpy(), b_np, rtol=1e-6)

    def test_mulx_evaluation_consistency(self):
        """(mulx(a))(x) == x * a(x) for all x in [-1,1]."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t_mulx(a)

        x = torch.linspace(-1, 1, 20)
        y_mulx = chebyshev_polynomial_t_evaluate(b, x)
        y_separate = x * chebyshev_polynomial_t_evaluate(a, x)

        torch.testing.assert_close(y_mulx, y_separate, atol=1e-6, rtol=1e-6)


class TestChebyshevPolynomialTPow:
    """Tests for chebyshev_polynomial_t_pow."""

    def test_pow_zero(self):
        """a^0 = 1 (T_0)."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t_pow(a, 0)
        torch.testing.assert_close(b, torch.tensor([1.0]))

    def test_pow_one(self):
        """a^1 = a."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t_pow(a, 1)
        torch.testing.assert_close(b, a)

    def test_pow_two(self):
        """(1 + T_1)^2."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))  # 1 + T_1
        b = chebyshev_polynomial_t_pow(a, 2)
        # (1 + T_1)^2 = 1 + 2*T_1 + T_1^2 = 1 + 2*T_1 + 0.5*(T_0 + T_2)
        # = 1.5*T_0 + 2*T_1 + 0.5*T_2
        torch.testing.assert_close(b, torch.tensor([1.5, 2.0, 0.5]))

    def test_pow_three(self):
        """(1 + T_1)^3."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        b = chebyshev_polynomial_t_pow(a, 3)

        # Verify by evaluation
        x = torch.linspace(-1, 1, 10)
        y_pow = chebyshev_polynomial_t_evaluate(b, x)
        y_cubed = chebyshev_polynomial_t_evaluate(a, x) ** 3
        torch.testing.assert_close(y_pow, y_cubed, atol=1e-5, rtol=1e-5)

    def test_pow_operator(self):
        """Test ** operator."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        b = a**2
        torch.testing.assert_close(b, torch.tensor([1.5, 2.0, 0.5]))

    def test_pow_negative_raises(self):
        """Negative exponent raises ValueError."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        with pytest.raises(ValueError):
            chebyshev_polynomial_t_pow(a, -1)

    def test_pow_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebpow."""
        coeffs = [1.0, 2.0]

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        b = chebyshev_polynomial_t_pow(a, 4)

        b_np = np_cheb.chebpow(coeffs, 4)

        np.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


class TestChebyshevPolynomialTArithmeticAutograd:
    """Tests for autograd support in arithmetic operations."""

    def test_add_gradcheck(self):
        """Gradcheck for add."""
        a_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            result = chebyshev_polynomial_t_add(
                chebyshev_polynomial_t(a), chebyshev_polynomial_t(b)
            )
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )

    def test_subtract_gradcheck(self):
        """Gradcheck for subtract."""
        a_coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            result = chebyshev_polynomial_t_subtract(
                chebyshev_polynomial_t(a), chebyshev_polynomial_t(b)
            )
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )

    def test_multiply_gradcheck(self):
        """Gradcheck for multiply."""
        a_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            result = chebyshev_polynomial_t_multiply(
                chebyshev_polynomial_t(a), chebyshev_polynomial_t(b)
            )
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )

    def test_mulx_gradcheck(self):
        """Gradcheck for mulx."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            result = chebyshev_polynomial_t_mulx(chebyshev_polynomial_t(c))
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_pow_gradcheck(self):
        """Gradcheck for pow."""
        coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            result = chebyshev_polynomial_t_pow(chebyshev_polynomial_t(c), 3)
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_multiply_gradgradcheck(self):
        """Second-order gradients for multiply."""
        a_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            result = chebyshev_polynomial_t_multiply(
                chebyshev_polynomial_t(a), chebyshev_polynomial_t(b)
            )
            return result.as_subclass(torch.Tensor).sum()

        assert torch.autograd.gradgradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )
