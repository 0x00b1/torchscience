"""Tests for polynomial fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchscience.polynomial import (
    polynomial,
    polynomial_equal,
    polynomial_evaluate,
    polynomial_fit,
    polynomial_vandermonde,
)


class TestPolynomialVandermonde:
    """Tests for polynomial_vandermonde."""

    def test_vandermonde_shape(self):
        """Vandermonde matrix has correct shape."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        V = polynomial_vandermonde(x, degree=3)
        assert V.shape == (4, 4)  # (n_points, degree+1)

    def test_vandermonde_values(self):
        """Vandermonde matrix has correct values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        V = polynomial_vandermonde(x, degree=2)

        # V[i, j] = x[i]^j
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # 1^0, 1^1, 1^2
                [1.0, 2.0, 4.0],  # 2^0, 2^1, 2^2
                [1.0, 3.0, 9.0],  # 3^0, 3^1, 3^2
            ]
        )
        assert torch.allclose(V, expected)

    def test_vandermonde_vs_numpy(self):
        """Compare against NumPy's vander."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        V = polynomial_vandermonde(x, degree=3)

        # NumPy vander uses descending order by default
        np_V = np.vander(x.numpy(), N=4, increasing=True)

        assert torch.allclose(V, torch.tensor(np_V, dtype=torch.float32))

    def test_vandermonde_degree_zero(self):
        """Degree 0 gives column of ones."""
        x = torch.tensor([1.0, 2.0, 3.0])
        V = polynomial_vandermonde(x, degree=0)

        expected = torch.ones(3, 1)
        assert torch.allclose(V, expected)


class TestPolynomialFit:
    """Tests for polynomial_fit."""

    def test_fit_exact_linear(self):
        """Fit exact linear data."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 3.0, 5.0, 7.0])  # y = 1 + 2x

        p = polynomial_fit(x, y, degree=1)

        expected = polynomial(torch.tensor([1.0, 2.0]))
        assert polynomial_equal(p, expected, tol=1e-5)

    def test_fit_exact_quadratic(self):
        """Fit exact quadratic data."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2  # y = x^2

        p = polynomial_fit(x, y, degree=2)

        expected = polynomial(torch.tensor([0.0, 0.0, 1.0]))
        assert polynomial_equal(p, expected, tol=1e-5)

    def test_fit_overdetermined(self):
        """Least squares fit with more points than degree."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([0.1, 1.9, 4.1, 8.9, 16.1])  # Noisy x^2

        p = polynomial_fit(x, y, degree=2)

        # Should be close to x^2
        assert p.shape[-1] == 3
        # Evaluate and check residuals are reasonable
        y_fit = polynomial_evaluate(p, x)
        residual = (y - y_fit).abs().mean()
        assert residual < 0.5  # Allow larger residual for noisy data

    def test_fit_vs_numpy(self):
        """Compare against NumPy's polyfit."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = torch.tensor([1.0, 2.5, 6.0, 11.5, 19.0], dtype=torch.float64)

        p = polynomial_fit(x, y, degree=2)

        # NumPy polyfit returns descending order
        np_coeffs = np.polyfit(x.numpy(), y.numpy(), deg=2)
        np_coeffs_ascending = np_coeffs[
            ::-1
        ].copy()  # Copy to avoid negative stride

        assert torch.allclose(
            p,
            torch.tensor(np_coeffs_ascending, dtype=torch.float64),
            atol=1e-10,
        )

    def test_fit_with_weights(self):
        """Weighted least squares fit."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0, 100.0])  # Last point is outlier
        weights = torch.tensor([1.0, 1.0, 1.0, 0.0])  # Zero weight on outlier

        p = polynomial_fit(x, y, degree=1, weights=weights)

        # Should fit 1 + x, ignoring the outlier
        expected = polynomial(torch.tensor([1.0, 1.0]))
        assert polynomial_equal(p, expected, tol=1e-5)

    def test_fit_underdetermined_raises(self):
        """Underdetermined system raises ValueError."""
        x = torch.tensor([0.0, 1.0])
        y = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError):
            polynomial_fit(x, y, degree=3)  # degree >= n_points


class TestFitAutograd:
    """Tests for polynomial_fit autograd."""

    def test_fit_gradcheck(self):
        """Verify gradients through polynomial_fit."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 5.0, 10.0], requires_grad=True, dtype=torch.float64
        )

        def fit_sum(y_val):
            p = polynomial_fit(x, y_val, degree=2)
            return p.sum()

        assert torch.autograd.gradcheck(fit_sum, (y,), eps=1e-6)

    def test_fit_gradgradcheck(self):
        """Verify second-order gradients through polynomial_fit."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 5.0, 10.0], requires_grad=True, dtype=torch.float64
        )

        def fit_sum(y_val):
            p = polynomial_fit(x, y_val, degree=2)
            return p.sum()

        assert torch.autograd.gradgradcheck(fit_sum, (y,), eps=1e-6)

    def test_vandermonde_gradcheck(self):
        """Verify gradients through polynomial_vandermonde."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64
        )

        def vander_sum(x_val):
            V = polynomial_vandermonde(x_val, degree=2)
            return V.sum()

        assert torch.autograd.gradcheck(vander_sum, (x,), eps=1e-6)

    def test_vandermonde_gradgradcheck(self):
        """Verify second-order gradients through polynomial_vandermonde."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64
        )

        def vander_sum(x_val):
            V = polynomial_vandermonde(x_val, degree=2)
            return V.sum()

        assert torch.autograd.gradgradcheck(vander_sum, (x,), eps=1e-6)


class TestFitMultidimensional:
    """Tests for multi-dimensional y values."""

    def test_fit_vector_valued(self):
        """Fit polynomial with vector-valued y."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.stack([x, x**2], dim=-1)  # (4, 2)

        p = polynomial_fit(x, y, degree=2)

        # First column should be ~[0, 1, 0] (for y = x)
        # Second column should be ~[0, 0, 1] (for y = x^2)
        assert p.shape == (3, 2)
