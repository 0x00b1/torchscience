"""Tests for core polynomial operations."""

import numpy as np
import pytest
import torch
from numpy.polynomial import Polynomial as NpPolynomial

from torchscience.polynomial import (
    PolynomialError,
    polynomial,
    polynomial_add,
    polynomial_antiderivative,
    polynomial_compose,
    polynomial_degree,
    polynomial_derivative,
    polynomial_divmod,
    polynomial_equal,
    polynomial_evaluate,
    polynomial_fit,
    polynomial_integral,
    polynomial_multiply,
    polynomial_negate,
    polynomial_pow,
    polynomial_roots,
    polynomial_scale,
    polynomial_subtract,
)


class TestPolynomialConstructor:
    """Tests for polynomial() constructor."""

    def test_single_coefficient(self):
        """Constant polynomial."""
        p = polynomial(torch.tensor([3.0]))
        assert p.coeffs.shape == (1,)
        assert p.coeffs[0] == 3.0

    def test_multiple_coefficients(self):
        """Standard polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert p.coeffs.shape == (3,)
        torch.testing.assert_close(p.coeffs, torch.tensor([1.0, 2.0, 3.0]))

    def test_batched_coefficients(self):
        """Batched polynomials."""
        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = polynomial(coeffs)
        assert p.coeffs.shape == (2, 2)

    def test_empty_raises(self):
        """Empty coefficients raise error."""
        with pytest.raises(PolynomialError):
            polynomial(torch.tensor([]))

    def test_empty_last_dim_raises(self):
        """Empty last dimension raises error."""
        with pytest.raises(PolynomialError):
            polynomial(torch.zeros(3, 0))

    def test_preserves_dtype(self):
        """Dtype is preserved."""
        p = polynomial(torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert p.coeffs.dtype == torch.float64

    def test_preserves_device(self):
        """Device is preserved."""
        coeffs = torch.tensor([1.0, 2.0])
        p = polynomial(coeffs)
        assert p.coeffs.device == coeffs.device


class TestPolynomialArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_degree(self):
        """Add polynomials of same degree."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0]))
        r = polynomial_add(p, q)
        torch.testing.assert_close(r.coeffs, torch.tensor([5.0, 7.0, 9.0]))

    def test_add_different_degree(self):
        """Add polynomials of different degrees."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0, 5.0]))
        r = polynomial_add(p, q)
        torch.testing.assert_close(r.coeffs, torch.tensor([4.0, 6.0, 5.0]))

    def test_add_operator(self):
        """Test + operator."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        r = p + q
        torch.testing.assert_close(r.coeffs, torch.tensor([4.0, 6.0]))

    def test_subtract_same_degree(self):
        """Subtract polynomials of same degree."""
        p = polynomial(torch.tensor([5.0, 7.0, 9.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        r = polynomial_subtract(p, q)
        torch.testing.assert_close(r.coeffs, torch.tensor([4.0, 5.0, 6.0]))

    def test_subtract_different_degree(self):
        """Subtract polynomials of different degrees."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([1.0, 2.0]))
        r = polynomial_subtract(p, q)
        torch.testing.assert_close(r.coeffs, torch.tensor([0.0, 0.0, 3.0]))

    def test_subtract_operator(self):
        """Test - operator."""
        p = polynomial(torch.tensor([5.0, 6.0]))
        q = polynomial(torch.tensor([1.0, 2.0]))
        r = p - q
        torch.testing.assert_close(r.coeffs, torch.tensor([4.0, 4.0]))

    def test_negate(self):
        """Negate polynomial."""
        p = polynomial(torch.tensor([1.0, -2.0, 3.0]))
        r = polynomial_negate(p)
        torch.testing.assert_close(r.coeffs, torch.tensor([-1.0, 2.0, -3.0]))

    def test_negate_operator(self):
        """Test unary - operator."""
        p = polynomial(torch.tensor([1.0, -2.0]))
        r = -p
        torch.testing.assert_close(r.coeffs, torch.tensor([-1.0, 2.0]))

    def test_scale_scalar(self):
        """Scale by scalar."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        r = polynomial_scale(p, torch.tensor(2.0))
        torch.testing.assert_close(r.coeffs, torch.tensor([2.0, 4.0, 6.0]))

    def test_scale_operator(self):
        """Test * operator with scalar."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        r = p * torch.tensor(3.0)
        torch.testing.assert_close(r.coeffs, torch.tensor([3.0, 6.0]))

    def test_scale_rmul_operator(self):
        """Test scalar * polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        r = torch.tensor(3.0) * p
        torch.testing.assert_close(r.coeffs, torch.tensor([3.0, 6.0]))

    def test_multiply_linear(self):
        """Multiply two linear polynomials."""
        # (1 + 2x) * (3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        r = polynomial_multiply(p, q)
        torch.testing.assert_close(r.coeffs, torch.tensor([3.0, 10.0, 8.0]))

    def test_multiply_quadratic(self):
        """Multiply linear by quadratic."""
        # (1 + x) * (1 + 2x + x^2) = 1 + 2x + x^2 + x + 2x^2 + x^3 = 1 + 3x + 3x^2 + x^3
        p = polynomial(torch.tensor([1.0, 1.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 1.0]))
        r = polynomial_multiply(p, q)
        torch.testing.assert_close(
            r.coeffs, torch.tensor([1.0, 3.0, 3.0, 1.0])
        )

    def test_multiply_operator(self):
        """Test * operator between polynomials."""
        p = polynomial(torch.tensor([1.0, 1.0]))
        q = polynomial(torch.tensor([1.0, -1.0]))
        r = p * q
        # (1 + x)(1 - x) = 1 - x^2
        torch.testing.assert_close(r.coeffs, torch.tensor([1.0, 0.0, -1.0]))

    def test_multiply_vs_numpy(self):
        """Compare multiplication against numpy."""
        p_coeffs = [1.0, 2.0, 3.0]
        q_coeffs = [4.0, 5.0]

        p_torch = polynomial(torch.tensor(p_coeffs))
        q_torch = polynomial(torch.tensor(q_coeffs))
        r_torch = polynomial_multiply(p_torch, q_torch)

        p_np = NpPolynomial(p_coeffs)
        q_np = NpPolynomial(q_coeffs)
        r_np = p_np * q_np

        np.testing.assert_allclose(
            r_torch.coeffs.numpy(), r_np.coef, rtol=1e-6
        )

    def test_degree(self):
        """Test polynomial_degree."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert polynomial_degree(p).item() == 2

    def test_degree_constant(self):
        """Degree of constant."""
        p = polynomial(torch.tensor([5.0]))
        assert polynomial_degree(p).item() == 0


class TestPolynomialEvaluation:
    """Tests for polynomial evaluation."""

    def test_evaluate_constant(self):
        """Evaluate constant polynomial."""
        p = polynomial(torch.tensor([3.0]))
        x = torch.tensor([1.0, 2.0, 3.0])
        y = polynomial_evaluate(p, x)
        torch.testing.assert_close(y, torch.tensor([3.0, 3.0, 3.0]))

    def test_evaluate_linear(self):
        """Evaluate linear polynomial."""
        # 1 + 2x at x = 0, 1, 2
        p = polynomial(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.0, 1.0, 2.0])
        y = polynomial_evaluate(p, x)
        torch.testing.assert_close(y, torch.tensor([1.0, 3.0, 5.0]))

    def test_evaluate_quadratic(self):
        """Evaluate quadratic polynomial."""
        # 1 + 2x + 3x^2 at x = 0, 1, 2
        # x=0: 1
        # x=1: 1 + 2 + 3 = 6
        # x=2: 1 + 4 + 12 = 17
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([0.0, 1.0, 2.0])
        y = polynomial_evaluate(p, x)
        torch.testing.assert_close(y, torch.tensor([1.0, 6.0, 17.0]))

    def test_evaluate_call_operator(self):
        """Test __call__ operator."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.0, 1.0])
        y = p(x)
        torch.testing.assert_close(y, torch.tensor([1.0, 3.0]))

    def test_evaluate_scalar(self):
        """Evaluate at scalar point."""
        p = polynomial(torch.tensor([1.0, 2.0, 1.0]))  # (1 + x)^2
        y = polynomial_evaluate(p, torch.tensor(2.0))
        assert y.item() == pytest.approx(9.0)  # (1 + 2)^2 = 9

    def test_evaluate_vs_numpy(self):
        """Compare evaluation against numpy."""
        coeffs = [1.0, -2.0, 3.0, -4.0]
        x = np.linspace(-2, 2, 10)

        p_torch = polynomial(torch.tensor(coeffs))
        y_torch = polynomial_evaluate(p_torch, torch.tensor(x)).numpy()

        p_np = NpPolynomial(coeffs)
        y_np = p_np(x)

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-6)

    def test_evaluate_complex(self):
        """Evaluate with complex coefficients."""
        p = polynomial(torch.tensor([1.0 + 0j, 1j]))  # 1 + ix
        x = torch.tensor([1.0 + 0j, 1j])
        y = polynomial_evaluate(p, x)
        # x=1: 1 + i
        # x=i: 1 + i*i = 1 - 1 = 0
        torch.testing.assert_close(y, torch.tensor([1.0 + 1j, 0.0 + 0j]))


class TestPolynomialCalculus:
    """Tests for derivative, antiderivative, integral."""

    def test_derivative_quadratic(self):
        """Derivative of quadratic."""
        # d/dx(1 + 2x + 3x^2) = 2 + 6x
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        dp = polynomial_derivative(p)
        torch.testing.assert_close(dp.coeffs, torch.tensor([2.0, 6.0]))

    def test_derivative_constant(self):
        """Derivative of constant is zero."""
        p = polynomial(torch.tensor([5.0]))
        dp = polynomial_derivative(p)
        torch.testing.assert_close(dp.coeffs, torch.tensor([0.0]))

    def test_derivative_second_order(self):
        """Second derivative."""
        # d^2/dx^2(1 + 2x + 3x^2 + 4x^3) = 6 + 24x
        p = polynomial(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        d2p = polynomial_derivative(p, order=2)
        torch.testing.assert_close(d2p.coeffs, torch.tensor([6.0, 24.0]))

    def test_derivative_vs_numpy(self):
        """Compare derivative against numpy."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        p_torch = polynomial(torch.tensor(coeffs))
        dp_torch = polynomial_derivative(p_torch)

        p_np = NpPolynomial(coeffs)
        dp_np = p_np.deriv()

        np.testing.assert_allclose(
            dp_torch.coeffs.numpy(), dp_np.coef, rtol=1e-6
        )

    def test_antiderivative_linear(self):
        """Antiderivative of linear."""
        # integral(2 + 6x) = 2x + 3x^2 (with C=0)
        p = polynomial(torch.tensor([2.0, 6.0]))
        ap = polynomial_antiderivative(p)
        torch.testing.assert_close(ap.coeffs, torch.tensor([0.0, 2.0, 3.0]))

    def test_antiderivative_with_constant(self):
        """Antiderivative with integration constant."""
        p = polynomial(torch.tensor([2.0, 6.0]))
        ap = polynomial_antiderivative(p, constant=1.0)
        torch.testing.assert_close(ap.coeffs, torch.tensor([1.0, 2.0, 3.0]))

    def test_antiderivative_vs_numpy(self):
        """Compare antiderivative against numpy."""
        coeffs = [1.0, 2.0, 3.0]

        p_torch = polynomial(torch.tensor(coeffs))
        ap_torch = polynomial_antiderivative(p_torch, constant=0.0)

        p_np = NpPolynomial(coeffs)
        ap_np = p_np.integ()

        np.testing.assert_allclose(
            ap_torch.coeffs.numpy(), ap_np.coef, rtol=1e-6
        )

    def test_definite_integral_quadratic(self):
        """Definite integral of quadratic."""
        # integral_0^1(1 + x^2) = [x + x^3/3]_0^1 = 1 + 1/3 = 4/3
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        result = polynomial_integral(p, torch.tensor(0.0), torch.tensor(1.0))
        assert result.item() == pytest.approx(4.0 / 3.0)

    def test_definite_integral_vs_numpy(self):
        """Compare definite integral against numpy."""
        coeffs = [1.0, 2.0, 3.0]
        a, b = -1.0, 2.0

        p_torch = polynomial(torch.tensor(coeffs))
        result_torch = polynomial_integral(
            p_torch, torch.tensor(a), torch.tensor(b)
        ).item()

        p_np = NpPolynomial(coeffs)
        # NumPy doesn't have definite integral, compute manually
        ap_np = p_np.integ()
        result_np = ap_np(b) - ap_np(a)

        assert result_torch == pytest.approx(result_np, rel=1e-6)


class TestPolynomialBatched:
    """Tests for batched polynomial operations."""

    def test_batched_evaluate(self):
        """Evaluate batched polynomials at multiple points."""
        # Two polynomials: 1+x and 2+3x
        coeffs = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        p = polynomial(coeffs)
        x = torch.tensor([1.0, 2.0])
        y = polynomial_evaluate(p, x)
        # Result shape: (2, 2) - 2 polynomials x 2 points
        # p[0](x) = 1+x: at x=[1,2] -> [2, 3]
        # p[1](x) = 2+3x: at x=[1,2] -> [5, 8]
        assert y.shape == (2, 2)
        expected = torch.tensor([[2.0, 3.0], [5.0, 8.0]])
        torch.testing.assert_close(y, expected)

    def test_batched_add(self):
        """Add batched polynomials."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        r = polynomial_add(p, q)
        expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
        torch.testing.assert_close(r.coeffs, expected)

    def test_batched_multiply(self):
        """Multiply batched polynomials."""
        # (1+x)(1-x) = 1-x^2 for first
        # (2+x)(3+x) = 6+5x+x^2 for second
        p = polynomial(torch.tensor([[1.0, 1.0], [2.0, 1.0]]))
        q = polynomial(torch.tensor([[1.0, -1.0], [3.0, 1.0]]))
        r = polynomial_multiply(p, q)
        expected = torch.tensor([[1.0, 0.0, -1.0], [6.0, 5.0, 1.0]])
        torch.testing.assert_close(r.coeffs, expected)

    def test_batched_derivative(self):
        """Derivative of batched polynomials."""
        # d/dx(1+2x+3x^2) = 2+6x
        # d/dx(4+5x+6x^2) = 5+12x
        p = polynomial(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        dp = polynomial_derivative(p)
        expected = torch.tensor([[2.0, 6.0], [5.0, 12.0]])
        torch.testing.assert_close(dp.coeffs, expected)


class TestPolynomialAutograd:
    """Tests for autograd support."""

    def test_evaluate_grad(self):
        """Gradient through evaluation."""
        coeffs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        p = polynomial(coeffs)
        x = torch.tensor([1.0, 2.0])
        y = polynomial_evaluate(p, x)
        loss = y.sum()
        loss.backward()

        # d(loss)/d(c_i) = sum over x of: x^i
        # c_0: 1 + 1 = 2
        # c_1: 1 + 2 = 3
        # c_2: 1 + 4 = 5
        expected_grad = torch.tensor([2.0, 3.0, 5.0])
        torch.testing.assert_close(coeffs.grad, expected_grad)

    def test_evaluate_gradcheck(self):
        """torch.autograd.gradcheck for evaluation."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(c):
            p = polynomial(c)
            return polynomial_evaluate(
                p, torch.tensor([0.5, 1.5], dtype=torch.float64)
            )

        assert torch.autograd.gradcheck(
            eval_fn, (coeffs,), raise_exception=True
        )

    def test_multiply_gradcheck(self):
        """torch.autograd.gradcheck for multiplication."""
        p_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        q_coeffs = torch.tensor(
            [3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def mul_fn(pc, qc):
            p = polynomial(pc)
            q = polynomial(qc)
            r = polynomial_multiply(p, q)
            return r.coeffs

        assert torch.autograd.gradcheck(
            mul_fn, (p_coeffs, q_coeffs), raise_exception=True
        )

    def test_derivative_gradcheck(self):
        """torch.autograd.gradcheck for derivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def deriv_fn(c):
            p = polynomial(c)
            dp = polynomial_derivative(p)
            return dp.coeffs

        assert torch.autograd.gradcheck(
            deriv_fn, (coeffs,), raise_exception=True
        )

    def test_antiderivative_gradcheck(self):
        """torch.autograd.gradcheck for antiderivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def antideriv_fn(c):
            p = polynomial(c)
            ap = polynomial_antiderivative(p, constant=0.0)
            return ap.coeffs

        assert torch.autograd.gradcheck(
            antideriv_fn, (coeffs,), raise_exception=True
        )

    def test_integral_gradcheck(self):
        """torch.autograd.gradcheck for definite integral."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def integral_fn(c):
            p = polynomial(c)
            return polynomial_integral(
                p,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

        assert torch.autograd.gradcheck(
            integral_fn, (coeffs,), raise_exception=True
        )

    def test_gradgradcheck(self):
        """Second-order gradients (Hessian)."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(c):
            p = polynomial(c)
            return polynomial_evaluate(
                p, torch.tensor([0.5, 1.5], dtype=torch.float64)
            )

        assert torch.autograd.gradgradcheck(
            eval_fn, (coeffs,), raise_exception=True
        )


class TestPolynomialPow:
    """Tests for polynomial_pow."""

    def test_pow_zero(self):
        """p^0 = 1 for any non-zero polynomial."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = polynomial_pow(p, 0)
        expected = polynomial(torch.tensor([1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_pow_one(self):
        """p^1 = p."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = polynomial_pow(p, 1)
        assert polynomial_equal(result, p, tol=1e-6)

    def test_pow_two(self):
        """(1 + x)^2 = 1 + 2x + x^2."""
        p = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
        result = polynomial_pow(p, 2)
        expected = polynomial(torch.tensor([1.0, 2.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_pow_three_binomial(self):
        """(1 + x)^3 = 1 + 3x + 3x^2 + x^3."""
        p = polynomial(torch.tensor([1.0, 1.0]))
        result = polynomial_pow(p, 3)
        expected = polynomial(torch.tensor([1.0, 3.0, 3.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_pow_large_exponent(self):
        """Test binary exponentiation with larger exponent."""
        p = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
        result = polynomial_pow(p, 5)
        # (1+x)^5 = 1 + 5x + 10x^2 + 10x^3 + 5x^4 + x^5
        expected = polynomial(torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_pow_negative_raises(self):
        """Negative exponent raises ValueError."""
        p = polynomial(torch.tensor([1.0, 1.0]))
        with pytest.raises(ValueError):
            polynomial_pow(p, -1)

    def test_pow_operator(self):
        """Test p ** n operator."""
        p = polynomial(torch.tensor([1.0, 1.0]))
        result = p**3
        expected = polynomial(torch.tensor([1.0, 3.0, 3.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)


class TestPolynomialPowAutograd:
    """Tests for polynomial_pow autograd."""

    def test_pow_gradcheck(self):
        """Verify gradients through polynomial_pow."""
        coeffs = torch.tensor(
            [1.0, 2.0], requires_grad=True, dtype=torch.float64
        )

        def pow_sum(c):
            p = polynomial(c)
            result = polynomial_pow(p, 3)
            return result.coeffs.sum()

        assert torch.autograd.gradcheck(pow_sum, (coeffs,), eps=1e-6)

    def test_pow_gradgradcheck(self):
        """Verify second-order gradients through polynomial_pow."""
        coeffs = torch.tensor(
            [1.0, 2.0], requires_grad=True, dtype=torch.float64
        )

        def pow_sum(c):
            p = polynomial(c)
            result = polynomial_pow(p, 3)
            return result.coeffs.sum()

        assert torch.autograd.gradgradcheck(pow_sum, (coeffs,), eps=1e-6)


class TestPolynomialTensorSubclass:
    """Tests for Polynomial as Tensor subclass."""

    def test_polynomial_is_tensor(self):
        """Polynomial should be a Tensor subclass."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(p, torch.Tensor)

    def test_polynomial_indexing_returns_tensor(self):
        """Indexing coefficients returns plain Tensor values."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert p[0] == 1.0
        assert p[1] == 2.0

    def test_polynomial_shape(self):
        """Shape property works like Tensor."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert p.shape == (3,)

    def test_polynomial_clone_preserves_type(self):
        """Clone preserves Polynomial type."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = p.clone()
        assert type(q) == type(p)

    def test_polynomial_to_device_preserves_type(self):
        """Device transfer preserves type."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = p.to(dtype=torch.float64)
        assert type(q) == type(p)

    def test_polynomial_reshape_preserves_type(self):
        """Reshape preserves Polynomial type."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = p.reshape(4)
        assert type(q) == type(p)


class TestPolynomialIntegration:
    """Integration tests combining multiple operations."""

    def test_fit_then_evaluate(self):
        """Fit polynomial and evaluate at original points."""

        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2 * x**2 - 3 * x + 1  # 1 - 3x + 2x^2

        p = polynomial_fit(x, y, degree=2)
        y_fit = polynomial_evaluate(p, x)

        assert torch.allclose(y, y_fit, atol=1e-5)

    def test_divide_then_multiply(self):
        """Division followed by multiplication recovers original."""
        from torchscience.polynomial import polynomial_add

        p = polynomial(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        q = polynomial(torch.tensor([1.0, 1.0]))

        quot, rem = polynomial_divmod(p, q)
        reconstructed = polynomial_add(polynomial_multiply(q, quot), rem)

        assert polynomial_equal(p, reconstructed, tol=1e-6)

    def test_compose_and_roots(self):
        """Composition affects roots predictably."""

        # p(x) = x^2 - 1 has roots at +-1
        p = polynomial(torch.tensor([-1.0, 0.0, 1.0]))
        # q(x) = 2x shifts and scales
        q = polynomial(torch.tensor([0.0, 2.0]))

        composed = polynomial_compose(p, q)  # p(2x) = 4x^2 - 1

        roots = polynomial_roots(composed)
        # Roots of 4x^2 - 1 are +-0.5
        root_reals = roots.real.sort().values
        expected = torch.tensor([-0.5, 0.5])

        assert torch.allclose(root_reals, expected, atol=1e-6)

    def test_pow_and_derivative(self):
        """Power and derivative work together correctly."""
        from torchscience.polynomial import (
            polynomial_derivative,
            polynomial_pow,
        )

        # (1 + x)^3 = 1 + 3x + 3x^2 + x^3
        p = polynomial(torch.tensor([1.0, 1.0]))
        p_cubed = polynomial_pow(p, 3)

        # Derivative: 3 + 6x + 3x^2
        dp = polynomial_derivative(p_cubed)
        expected = polynomial(torch.tensor([3.0, 6.0, 3.0]))

        assert polynomial_equal(dp, expected, tol=1e-6)

    def test_compose_then_evaluate(self):
        """Composition followed by evaluation."""
        from torchscience.polynomial import polynomial_compose

        p = polynomial(torch.tensor([1.0, 2.0, 1.0]))  # 1 + 2x + x^2
        q = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x

        composed = polynomial_compose(p, q)

        # Evaluate at x = 2: composed(2) should equal p(q(2)) = p(3) = 1 + 6 + 9 = 16
        x = torch.tensor([2.0])
        result = polynomial_evaluate(composed, x)
        expected = torch.tensor([16.0])

        assert torch.allclose(result, expected, atol=1e-5)
