"""Tests for torch.compile compatibility of polynomial operations."""

import pytest
import torch

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_multiply,
    hermite_polynomial_h,
    hermite_polynomial_h_evaluate,
    hermite_polynomial_h_multiply,
    legendre_polynomial_p,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_multiply,
    polynomial,
    polynomial_add,
    polynomial_derivative,
    polynomial_evaluate,
    polynomial_multiply,
)


class TestPolynomialCompile:
    """Tests for torch.compile with standard polynomials."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_evaluate_compiles(self):
        """polynomial_evaluate should work with torch.compile."""

        @torch.compile
        def f(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x)

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        result = f(coeffs, x)
        expected = polynomial_evaluate(polynomial(coeffs), x)

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="polynomial_multiply may have graph breaks with dynamic shapes"
    )
    def test_multiply_compiles(self):
        """polynomial_multiply should work with torch.compile."""

        @torch.compile
        def f(a, b):
            pa = polynomial(a)
            pb = polynomial(b)
            return polynomial_multiply(pa, pb)

        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)

        result = f(a, b)
        expected = polynomial_multiply(polynomial(a), polynomial(b))

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="polynomial_add may have graph breaks with dynamic shapes"
    )
    def test_add_compiles(self):
        """polynomial_add should work with torch.compile."""

        @torch.compile
        def f(a, b):
            pa = polynomial(a)
            pb = polynomial(b)
            return polynomial_add(pa, pb)

        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([4.0, 5.0], dtype=torch.float64)

        result = f(a, b)
        expected = polynomial_add(polynomial(a), polynomial(b))

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(reason="polynomial_derivative may have graph breaks")
    def test_derivative_compiles(self):
        """polynomial_derivative should work with torch.compile."""

        @torch.compile
        def f(coeffs):
            p = polynomial(coeffs)
            return polynomial_derivative(p)

        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        result = f(coeffs)
        expected = polynomial_derivative(polynomial(coeffs))

        torch.testing.assert_close(result, expected)


class TestLegendreCompile:
    """Tests for torch.compile with Legendre polynomials."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_evaluate_compiles(self):
        """legendre_polynomial_p_evaluate should work with torch.compile."""

        @torch.compile
        def f(coeffs, x):
            p = legendre_polynomial_p(coeffs)
            return legendre_polynomial_p_evaluate(p, x)

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

        result = f(coeffs, x)
        expected = legendre_polynomial_p_evaluate(
            legendre_polynomial_p(coeffs), x
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_multiply_compiles(self):
        """legendre_polynomial_p_multiply should work with torch.compile."""

        @torch.compile
        def f(a, b):
            pa = legendre_polynomial_p(a)
            pb = legendre_polynomial_p(b)
            return legendre_polynomial_p_multiply(pa, pb)

        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64)

        result = f(a, b)
        expected = legendre_polynomial_p_multiply(
            legendre_polynomial_p(a), legendre_polynomial_p(b)
        )

        torch.testing.assert_close(result, expected)


class TestHermiteCompile:
    """Tests for torch.compile with Hermite polynomials."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_evaluate_compiles(self):
        """hermite_polynomial_h_evaluate should work with torch.compile."""

        @torch.compile
        def f(coeffs, x):
            p = hermite_polynomial_h(coeffs)
            return hermite_polynomial_h_evaluate(p, x)

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

        result = f(coeffs, x)
        expected = hermite_polynomial_h_evaluate(
            hermite_polynomial_h(coeffs), x
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_multiply_compiles(self):
        """hermite_polynomial_h_multiply should work with torch.compile."""

        @torch.compile
        def f(a, b):
            pa = hermite_polynomial_h(a)
            pb = hermite_polynomial_h(b)
            return hermite_polynomial_h_multiply(pa, pb)

        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64)

        result = f(a, b)
        expected = hermite_polynomial_h_multiply(
            hermite_polynomial_h(a), hermite_polynomial_h(b)
        )

        torch.testing.assert_close(result, expected)


class TestChebyshevCompile:
    """Tests for torch.compile with Chebyshev polynomials."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_evaluate_compiles(self):
        """chebyshev_polynomial_t_evaluate should work with torch.compile."""

        @torch.compile
        def f(coeffs, x):
            p = chebyshev_polynomial_t(coeffs)
            return chebyshev_polynomial_t_evaluate(p, x)

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

        result = f(coeffs, x)
        expected = chebyshev_polynomial_t_evaluate(
            chebyshev_polynomial_t(coeffs), x
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_multiply_compiles(self):
        """chebyshev_polynomial_t_multiply should work with torch.compile."""

        @torch.compile
        def f(a, b):
            pa = chebyshev_polynomial_t(a)
            pb = chebyshev_polynomial_t(b)
            return chebyshev_polynomial_t_multiply(pa, pb)

        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64)

        result = f(a, b)
        expected = chebyshev_polynomial_t_multiply(
            chebyshev_polynomial_t(a), chebyshev_polynomial_t(b)
        )

        torch.testing.assert_close(result, expected)


class TestCompileWithGradients:
    """Tests for torch.compile with gradient computation."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="Gradient through compiled polynomial may have issues"
    )
    def test_evaluate_grad_compiles(self):
        """Gradient through polynomial_evaluate should work with torch.compile."""

        @torch.compile
        def f(coeffs, x):
            p = polynomial(coeffs)
            y = polynomial_evaluate(p, x)
            return y.sum()

        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        # Should not raise
        loss = f(coeffs, x)
        loss.backward()

        assert coeffs.grad is not None
        assert coeffs.grad.shape == coeffs.shape

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="Gradient through compiled polynomial multiply may have issues"
    )
    def test_multiply_grad_compiles(self):
        """Gradient through polynomial_multiply should work with torch.compile."""

        @torch.compile
        def f(a, b, x):
            pa = polynomial(a)
            pb = polynomial(b)
            pc = polynomial_multiply(pa, pb)
            y = polynomial_evaluate(pc, x)
            return y.sum()

        a = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.5, 1.0], dtype=torch.float64)

        loss = f(a, b, x)
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None
