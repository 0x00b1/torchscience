"""Tests for polynomial root finding and related utilities."""

import numpy as np
import pytest
import torch
from numpy.polynomial import Polynomial as NpPolynomial

from torchscience.polynomial import (
    DegreeError,
    polynomial,
    polynomial_equal,
    polynomial_evaluate,
    polynomial_from_roots,
    polynomial_roots,
    polynomial_trim,
)


class TestPolynomialRoots:
    """Tests for polynomial_roots."""

    def test_linear_root(self):
        """Root of linear polynomial."""
        # x - 2 = 0 => x = 2
        p = polynomial(torch.tensor([-2.0, 1.0]))
        roots = polynomial_roots(p)
        assert roots.shape == (1,)
        torch.testing.assert_close(roots.real, torch.tensor([2.0]))
        torch.testing.assert_close(
            roots.imag, torch.tensor([0.0]), atol=1e-6, rtol=0
        )

    def test_quadratic_real_roots(self):
        """Quadratic with real roots."""
        # (x - 1)(x - 2) = x^2 - 3x + 2
        p = polynomial(torch.tensor([2.0, -3.0, 1.0]))
        roots = polynomial_roots(p)
        assert roots.shape == (2,)
        # Roots should be 1 and 2 (in some order)
        roots_real = roots.real.sort().values
        torch.testing.assert_close(
            roots_real, torch.tensor([1.0, 2.0]), atol=1e-6, rtol=0
        )

    def test_quadratic_complex_roots(self):
        """Quadratic with complex roots."""
        # x^2 + 1 = 0 => x = +-i
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        roots = polynomial_roots(p)
        assert roots.shape == (2,)
        # Roots should be i and -i
        assert torch.allclose(roots.real, torch.zeros(2), atol=1e-6)
        roots_imag = roots.imag.abs().sort().values
        torch.testing.assert_close(
            roots_imag, torch.tensor([1.0, 1.0]), atol=1e-6, rtol=0
        )

    def test_cubic_roots(self):
        """Cubic polynomial roots."""
        # (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
        p = polynomial(torch.tensor([-6.0, 11.0, -6.0, 1.0]))
        roots = polynomial_roots(p)
        assert roots.shape == (3,)
        roots_real = roots.real.sort().values
        torch.testing.assert_close(
            roots_real, torch.tensor([1.0, 2.0, 3.0]), atol=1e-5, rtol=0
        )

    def test_double_root(self):
        """Polynomial with double root."""
        # (x - 1)^2 = x^2 - 2x + 1
        p = polynomial(torch.tensor([1.0, -2.0, 1.0]))
        roots = polynomial_roots(p)
        # Both roots should be approximately 1
        torch.testing.assert_close(
            roots.real, torch.tensor([1.0, 1.0]), atol=1e-6, rtol=0
        )

    def test_constant_raises(self):
        """Constant polynomial raises DegreeError."""
        p = polynomial(torch.tensor([5.0]))
        with pytest.raises(DegreeError):
            polynomial_roots(p)

    def test_zero_leading_coeff_raises(self):
        """Zero leading coefficient raises DegreeError."""
        p = polynomial(torch.tensor([1.0, 2.0, 0.0]))
        with pytest.raises(DegreeError):
            polynomial_roots(p)

    def test_roots_vs_numpy(self):
        """Compare roots against numpy."""
        coeffs = [6.0, -5.0, -2.0, 1.0]  # Some cubic

        p_torch = polynomial(torch.tensor(coeffs, dtype=torch.float64))
        roots_torch = polynomial_roots(p_torch).numpy()

        p_np = NpPolynomial(coeffs)
        roots_np = p_np.roots()

        # Sort for comparison (both should be sorted by real part, then imaginary)
        roots_torch_sorted = sorted(
            roots_torch, key=lambda z: (z.real, z.imag)
        )
        roots_np_sorted = sorted(roots_np, key=lambda z: (z.real, z.imag))

        np.testing.assert_allclose(
            np.array(roots_torch_sorted), np.array(roots_np_sorted), rtol=1e-5
        )

    def test_roots_float64_high_degree(self):
        """High-degree polynomial needs float64 for accuracy."""
        # Wilkinson's polynomial: (x-1)(x-2)...(x-n)
        n = 10
        roots_true = torch.arange(1, n + 1, dtype=torch.float64)
        p = polynomial_from_roots(roots_true)

        roots = polynomial_roots(p)
        roots_real = roots.real.sort().values

        torch.testing.assert_close(
            roots_real, roots_true, atol=1e-3, rtol=1e-3
        )


class TestPolynomialFromRoots:
    """Tests for polynomial_from_roots."""

    def test_single_root(self):
        """Single root."""
        # (x - 2) = -2 + x
        roots = torch.tensor([2.0])
        p = polynomial_from_roots(roots)
        torch.testing.assert_close(p, torch.tensor([-2.0, 1.0]))

    def test_two_roots(self):
        """Two roots."""
        # (x - 1)(x - 2) = 2 - 3x + x^2
        roots = torch.tensor([1.0, 2.0])
        p = polynomial_from_roots(roots)
        torch.testing.assert_close(p, torch.tensor([2.0, -3.0, 1.0]))

    def test_complex_roots(self):
        """Complex conjugate roots give real coefficients."""
        # (x - i)(x + i) = x^2 + 1
        roots = torch.tensor([1j, -1j])
        p = polynomial_from_roots(roots)
        # Coefficients should be real (imaginary parts ~0)
        torch.testing.assert_close(
            p.real, torch.tensor([1.0, 0.0, 1.0]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(p.imag, torch.zeros(3), atol=1e-6, rtol=0)

    def test_empty_roots(self):
        """Empty roots gives constant 1."""
        roots = torch.tensor([])
        p = polynomial_from_roots(roots)
        assert p.shape[-1] == 1
        assert p.item() == 1.0

    def test_roundtrip_roots(self):
        """polynomial_roots(polynomial_from_roots(r)) recovers r."""
        roots = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        p = polynomial_from_roots(roots)
        recovered = polynomial_roots(p)
        recovered_sorted = recovered.real.sort().values
        torch.testing.assert_close(recovered_sorted, roots, atol=1e-5, rtol=0)

    def test_roundtrip_polynomial(self):
        """polynomial_from_roots(polynomial_roots(p)) recovers monic p."""
        # Start with monic polynomial
        coeffs = torch.tensor(
            [-6.0, 11.0, -6.0, 1.0], dtype=torch.float64
        )  # (x-1)(x-2)(x-3)
        p = polynomial(coeffs)
        roots = polynomial_roots(p)
        p_recovered = polynomial_from_roots(roots)

        # Should recover original monic polynomial (up to numerical error)
        torch.testing.assert_close(p_recovered.real, coeffs, atol=1e-5, rtol=0)

    def test_batched_from_roots(self):
        """Batched polynomial construction from roots."""
        roots = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = polynomial_from_roots(roots)
        # Shape should be (2, 3) - two polynomials of degree 2
        assert p.shape == (2, 3)

        # First: (x-1)(x-2) = 2 - 3x + x^2
        torch.testing.assert_close(p[0], torch.tensor([2.0, -3.0, 1.0]))
        # Second: (x-3)(x-4) = 12 - 7x + x^2
        torch.testing.assert_close(p[1], torch.tensor([12.0, -7.0, 1.0]))


class TestPolynomialTrim:
    """Tests for polynomial_trim."""

    def test_no_trim_needed(self):
        """No trimming when all coefficients nonzero."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        pt = polynomial_trim(p)
        torch.testing.assert_close(pt, p)

    def test_trim_trailing_zeros(self):
        """Trim trailing zeros."""
        p = polynomial(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        pt = polynomial_trim(p)
        torch.testing.assert_close(pt, torch.tensor([1.0, 2.0]))

    def test_trim_with_tolerance(self):
        """Trim with tolerance."""
        p = polynomial(torch.tensor([1.0, 2.0, 1e-10]))
        pt = polynomial_trim(p, tol=1e-8)
        torch.testing.assert_close(pt, torch.tensor([1.0, 2.0]))

    def test_trim_all_zero(self):
        """Trimming all-zero polynomial leaves single zero."""
        p = polynomial(torch.tensor([0.0, 0.0, 0.0]))
        pt = polynomial_trim(p)
        assert pt.shape[-1] == 1
        assert pt[0] == 0.0

    def test_trim_constant(self):
        """Trimming constant polynomial is no-op."""
        p = polynomial(torch.tensor([5.0]))
        pt = polynomial_trim(p)
        torch.testing.assert_close(pt, torch.tensor([5.0]))


class TestPolynomialEqual:
    """Tests for polynomial_equal."""

    def test_equal_same(self):
        """Same polynomial is equal."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert polynomial_equal(p, q).item()

    def test_equal_different_length_padded(self):
        """Equal when trailing zeros differ."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 0.0]))
        assert polynomial_equal(p, q).item()

    def test_not_equal(self):
        """Different polynomials are not equal."""
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([1.0, 3.0]))
        assert not polynomial_equal(p, q).item()

    def test_equal_with_tolerance(self):
        """Equality with tolerance."""
        # Use float64 for precision with small tolerances
        p = polynomial(torch.tensor([1.0, 2.0], dtype=torch.float64))
        q = polynomial(
            torch.tensor([1.0 + 1e-9, 2.0 - 1e-9], dtype=torch.float64)
        )
        assert polynomial_equal(p, q, tol=1e-8).item()
        assert not polynomial_equal(p, q, tol=1e-10).item()

    def test_batched_equal(self):
        """Batched equality check."""
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))
        result = polynomial_equal(p, q)
        assert result.shape == (2,)
        assert result[0].item()
        assert not result[1].item()


class TestRootsAutograd:
    """Tests for autograd through root finding."""

    def test_roots_gradcheck(self):
        """Gradient check for polynomial_roots."""
        coeffs = torch.tensor(
            [2.0, -3.0, 1.0], dtype=torch.float64, requires_grad=True
        )

        def roots_fn(c):
            p = polynomial(c)
            roots = polynomial_roots(p)
            # Return real and imaginary parts separately
            return roots.real, roots.imag

        # Note: gradcheck may fail for roots close to each other (ill-conditioned)
        assert torch.autograd.gradcheck(
            roots_fn, (coeffs,), raise_exception=True
        )

    def test_from_roots_gradcheck(self):
        """Gradient check for polynomial_from_roots."""
        roots = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def from_roots_fn(r):
            p = polynomial_from_roots(r)
            return p

        assert torch.autograd.gradcheck(
            from_roots_fn, (roots,), raise_exception=True
        )

    def test_roundtrip_gradient(self):
        """Gradient flows through roundtrip."""
        roots = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )

        p = polynomial_from_roots(roots)
        # Evaluate at a point to get scalar for gradient
        x = torch.tensor(0.5, dtype=torch.float64)
        y = polynomial_evaluate(p, x)
        y.backward()

        assert roots.grad is not None
        assert not torch.any(torch.isnan(roots.grad))


class TestEdgeCases:
    """Edge case tests."""

    def test_high_degree_stability(self):
        """Numerical stability for high-degree polynomials."""
        # Create polynomial with known roots
        n = 15
        roots_true = torch.linspace(0.1, 1.5, n, dtype=torch.float64)
        p = polynomial_from_roots(roots_true)

        # Verify evaluation at roots gives ~0
        values = polynomial_evaluate(p, roots_true)
        assert torch.allclose(values, torch.zeros_like(values), atol=1e-8)

    def test_complex_coefficients(self):
        """Polynomial with complex coefficients."""
        coeffs = torch.tensor([1.0 + 1j, 2.0 - 1j, 1.0 + 0j])
        p = polynomial(coeffs)
        roots = polynomial_roots(p)
        assert roots.shape == (2,)

        # Verify roots are actual roots
        for root in roots:
            val = polynomial_evaluate(p, root)
            assert torch.abs(val) < 1e-5
