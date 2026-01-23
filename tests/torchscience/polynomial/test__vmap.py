"""Tests for torch.vmap compatibility of polynomial operations."""

import pytest
import torch
from torch.func import vmap

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_evaluate,
    hermite_polynomial_h,
    hermite_polynomial_h_evaluate,
    legendre_polynomial_p,
    legendre_polynomial_p_evaluate,
    polynomial,
    polynomial_add,
    polynomial_derivative,
    polynomial_evaluate,
    polynomial_multiply,
)


class TestPolynomialVmap:
    """Tests for torch.vmap with standard polynomials."""

    def test_evaluate_vmap_over_coeffs(self):
        """vmap polynomial_evaluate over batch of coefficient vectors."""

        def single_eval(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(0, None))

        # Batch of 4 polynomials, each degree 2
        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        result = batched_eval(coeffs_batch, x)

        assert result.shape == (4, 3)

        # Verify against manual computation
        for i in range(4):
            expected = polynomial_evaluate(polynomial(coeffs_batch[i]), x)
            torch.testing.assert_close(result[i], expected)

    def test_evaluate_vmap_over_x(self):
        """vmap polynomial_evaluate over batch of x values."""

        def single_eval(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(None, 0))

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        # Batch of 5 different x tensors, each with 3 points
        x_batch = torch.randn(5, 3, dtype=torch.float64)

        result = batched_eval(coeffs, x_batch)

        assert result.shape == (5, 3)

        # Verify against manual computation
        for i in range(5):
            expected = polynomial_evaluate(polynomial(coeffs), x_batch[i])
            torch.testing.assert_close(result[i], expected)

    def test_evaluate_vmap_over_both(self):
        """vmap polynomial_evaluate over both coeffs and x."""

        def single_eval(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(0, 0))

        # Batch of 4 polynomials paired with 4 x tensors
        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x_batch = torch.randn(4, 5, dtype=torch.float64)

        result = batched_eval(coeffs_batch, x_batch)

        assert result.shape == (4, 5)

        # Verify against manual computation
        for i in range(4):
            expected = polynomial_evaluate(
                polynomial(coeffs_batch[i]), x_batch[i]
            )
            torch.testing.assert_close(result[i], expected)

    def test_multiply_vmap(self):
        """vmap polynomial_multiply over batch of coefficient pairs."""

        def single_mul(a, b):
            pa = polynomial(a)
            pb = polynomial(b)
            return polynomial_multiply(pa, pb)

        batched_mul = vmap(single_mul, in_dims=(0, 0))

        # Batch of 3 polynomial pairs
        a_batch = torch.randn(3, 2, dtype=torch.float64)
        b_batch = torch.randn(3, 3, dtype=torch.float64)

        result = batched_mul(a_batch, b_batch)

        # Result should have shape (3, 4) since deg(a*b) = deg(a) + deg(b)
        assert result.shape == (3, 4)

        # Verify against manual computation
        for i in range(3):
            expected = polynomial_multiply(
                polynomial(a_batch[i]), polynomial(b_batch[i])
            )
            torch.testing.assert_close(result[i], expected)

    def test_add_vmap(self):
        """vmap polynomial_add over batch of coefficient pairs."""

        def single_add(a, b):
            pa = polynomial(a)
            pb = polynomial(b)
            return polynomial_add(pa, pb)

        batched_add = vmap(single_add, in_dims=(0, 0))

        # Batch of 4 polynomial pairs of same degree
        a_batch = torch.randn(4, 3, dtype=torch.float64)
        b_batch = torch.randn(4, 3, dtype=torch.float64)

        result = batched_add(a_batch, b_batch)

        assert result.shape == (4, 3)

        # Verify against manual computation
        for i in range(4):
            expected = polynomial_add(
                polynomial(a_batch[i]), polynomial(b_batch[i])
            )
            torch.testing.assert_close(result[i], expected)

    @pytest.mark.xfail(
        reason="polynomial_derivative may not be vmap-compatible due to dynamic output shape"
    )
    def test_derivative_vmap(self):
        """vmap polynomial_derivative over batch of polynomials."""

        def single_deriv(coeffs):
            p = polynomial(coeffs)
            return polynomial_derivative(p)

        batched_deriv = vmap(single_deriv)

        # Batch of 5 polynomials of degree 3
        coeffs_batch = torch.randn(5, 4, dtype=torch.float64)

        result = batched_deriv(coeffs_batch)

        assert result.shape == (5, 3)

        # Verify against manual computation
        for i in range(5):
            expected = polynomial_derivative(polynomial(coeffs_batch[i]))
            torch.testing.assert_close(result[i], expected)


class TestLegendreVmap:
    """Tests for torch.vmap with Legendre polynomials."""

    def test_evaluate_vmap(self):
        """vmap legendre_polynomial_p_evaluate over batch of coefficients."""

        def single_eval(coeffs, x):
            p = legendre_polynomial_p(coeffs)
            return legendre_polynomial_p_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(0, None))

        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

        result = batched_eval(coeffs_batch, x)

        assert result.shape == (4, 3)

        # Verify against manual computation
        for i in range(4):
            expected = legendre_polynomial_p_evaluate(
                legendre_polynomial_p(coeffs_batch[i]), x
            )
            torch.testing.assert_close(result[i], expected)


class TestHermiteVmap:
    """Tests for torch.vmap with Hermite polynomials."""

    def test_evaluate_vmap(self):
        """vmap hermite_polynomial_h_evaluate over batch of coefficients."""

        def single_eval(coeffs, x):
            p = hermite_polynomial_h(coeffs)
            return hermite_polynomial_h_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(0, None))

        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

        result = batched_eval(coeffs_batch, x)

        assert result.shape == (4, 3)

        # Verify against manual computation
        for i in range(4):
            expected = hermite_polynomial_h_evaluate(
                hermite_polynomial_h(coeffs_batch[i]), x
            )
            torch.testing.assert_close(result[i], expected)


class TestChebyshevVmap:
    """Tests for torch.vmap with Chebyshev polynomials."""

    def test_evaluate_vmap(self):
        """vmap chebyshev_polynomial_t_evaluate over batch of coefficients."""

        def single_eval(coeffs, x):
            p = chebyshev_polynomial_t(coeffs)
            return chebyshev_polynomial_t_evaluate(p, x)

        batched_eval = vmap(single_eval, in_dims=(0, None))

        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

        result = batched_eval(coeffs_batch, x)

        assert result.shape == (4, 3)

        # Verify against manual computation
        for i in range(4):
            expected = chebyshev_polynomial_t_evaluate(
                chebyshev_polynomial_t(coeffs_batch[i]), x
            )
            torch.testing.assert_close(result[i], expected)


class TestNestedVmap:
    """Tests for nested vmap operations."""

    def test_double_vmap(self):
        """Double vmap over polynomial evaluation."""

        def single_eval(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x)

        # vmap over x first, then over coeffs
        inner_vmap = vmap(single_eval, in_dims=(None, 0))
        outer_vmap = vmap(inner_vmap, in_dims=(0, None))

        # Batch of 3 polynomials, batch of 4 x tensors
        coeffs_batch = torch.randn(3, 3, dtype=torch.float64)
        x_batch = torch.randn(4, 5, dtype=torch.float64)

        result = outer_vmap(coeffs_batch, x_batch)

        assert result.shape == (3, 4, 5)


class TestVmapWithGradients:
    """Tests for vmap combined with gradient computation."""

    @pytest.mark.xfail(
        reason="vmap+grad combination may have issues with polynomial subclass"
    )
    def test_vmap_grad(self):
        """Gradient computation inside vmap."""

        def eval_and_sum(coeffs, x):
            p = polynomial(coeffs)
            return polynomial_evaluate(p, x).sum()

        # Use jacrev to get jacobian over batch
        from torch.func import grad

        grad_fn = grad(eval_and_sum)
        batched_grad = vmap(grad_fn, in_dims=(0, None))

        coeffs_batch = torch.randn(4, 3, dtype=torch.float64)
        x = torch.tensor([0.5, 1.0], dtype=torch.float64)

        result = batched_grad(coeffs_batch, x)

        assert result.shape == (4, 3)

    @pytest.mark.xfail(
        reason="grad+vmap combination may have issues with polynomial subclass"
    )
    def test_grad_vmap(self):
        """vmap inside gradient computation."""

        def batched_eval_sum(coeffs_batch, x):
            def single_eval(coeffs):
                p = polynomial(coeffs)
                return polynomial_evaluate(p, x).sum()

            return vmap(single_eval)(coeffs_batch).sum()

        coeffs_batch = torch.randn(
            4, 3, dtype=torch.float64, requires_grad=True
        )
        x = torch.tensor([0.5, 1.0], dtype=torch.float64)

        loss = batched_eval_sum(coeffs_batch, x)
        loss.backward()

        assert coeffs_batch.grad is not None
        assert coeffs_batch.grad.shape == coeffs_batch.shape
