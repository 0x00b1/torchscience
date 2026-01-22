"""Tests for LegendrePolynomialP root finding."""

import numpy as np
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_companion,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_from_roots,
    legendre_polynomial_p_roots,
)


class TestLegendrePolynomialPCompanion:
    """Tests for legendre_polynomial_p_companion."""

    def test_companion_p2(self):
        """Companion matrix for P_2."""
        c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
        A = legendre_polynomial_p_companion(c)
        assert A.shape == (2, 2)

    def test_companion_p3(self):
        """Companion matrix for P_3."""
        c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # P_3
        A = legendre_polynomial_p_companion(c)
        assert A.shape == (3, 3)

    def test_companion_eigenvalues_are_roots(self):
        """Eigenvalues of companion matrix are roots."""
        # P_2 = (3x^2 - 1)/2, roots at +/- 1/sqrt(3)
        c = legendre_polynomial_p(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        A = legendre_polynomial_p_companion(c)
        eigenvalues = torch.linalg.eigvals(A)

        expected_roots = torch.tensor(
            [1.0 / np.sqrt(3), -1.0 / np.sqrt(3)], dtype=torch.float64
        )
        expected_roots = expected_roots.sort().values
        eigenvalues_sorted = eigenvalues.real.sort().values

        torch.testing.assert_close(
            eigenvalues_sorted, expected_roots, atol=1e-10, rtol=1e-10
        )

    def test_companion_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legcompanion."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 1.0]  # P_4

        c = legendre_polynomial_p(torch.tensor(coeffs, dtype=torch.float64))
        A_torch = legendre_polynomial_p_companion(c)

        A_np = np_leg.legcompanion(coeffs)

        np.testing.assert_allclose(A_torch.numpy(), A_np, rtol=1e-10)


class TestLegendrePolynomialPRoots:
    """Tests for legendre_polynomial_p_roots."""

    def test_roots_p2(self):
        """Roots of P_2."""
        # P_2 = (3x^2 - 1)/2, roots at +/- 1/sqrt(3)
        c = legendre_polynomial_p(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = legendre_polynomial_p_roots(c)

        expected = torch.tensor(
            [1.0 / np.sqrt(3), -1.0 / np.sqrt(3)], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_p3(self):
        """Roots of P_3."""
        # P_3 = (5x^3 - 3x)/2, roots at 0, +/- sqrt(3/5)
        c = legendre_polynomial_p(
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = legendre_polynomial_p_roots(c)

        expected = torch.tensor(
            [0.0, np.sqrt(3 / 5), -np.sqrt(3 / 5)], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_verify_evaluation(self):
        """f(root) is approximately 0 for all roots."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float64)
        c = legendre_polynomial_p(coeffs)
        roots = legendre_polynomial_p_roots(c)

        # Evaluate at roots
        f_at_roots = legendre_polynomial_p_evaluate(c, roots)

        torch.testing.assert_close(
            f_at_roots.abs(),
            torch.zeros_like(f_at_roots.abs()),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_roots_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legroots."""
        coeffs = [1.0, -2.0, 3.0, -1.0, 1.0]

        c = legendre_polynomial_p(torch.tensor(coeffs, dtype=torch.float64))
        roots_torch = legendre_polynomial_p_roots(c)

        roots_np = np_leg.legroots(coeffs)

        # Sort for comparison
        roots_torch_sorted = roots_torch.real.sort().values.numpy()
        roots_np_sorted = np.sort(roots_np.real)

        np.testing.assert_allclose(
            roots_torch_sorted, roots_np_sorted, rtol=1e-8
        )

    def test_roots_gauss_points(self):
        """Roots of P_n are Gauss-Legendre quadrature points."""
        # P_5 has 5 roots that are the 5-point Gauss-Legendre nodes
        c = legendre_polynomial_p(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = legendre_polynomial_p_roots(c)
        roots_real = roots.real.sort().values

        # Known 5-point Gauss-Legendre nodes
        expected = torch.tensor(
            [
                -np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
                -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                0.0,
                np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
            ],
            dtype=torch.float64,
        )
        expected = expected.sort().values

        torch.testing.assert_close(
            roots_real, expected, atol=1e-10, rtol=1e-10
        )


class TestLegendrePolynomialPFromRoots:
    """Tests for legendre_polynomial_p_from_roots."""

    def test_from_roots_single(self):
        """Single root at x=0."""
        roots = torch.tensor([0.0])
        c = legendre_polynomial_p_from_roots(roots)

        # Should be (x - 0) = x = P_1
        x = torch.linspace(-1, 1, 10)
        y = legendre_polynomial_p_evaluate(c, x)

        torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)

    def test_from_roots_two(self):
        """Two roots."""
        roots = torch.tensor([0.5, -0.5])
        c = legendre_polynomial_p_from_roots(roots)

        # Should be (x - 0.5)(x + 0.5) = x^2 - 0.25
        x = torch.linspace(-1, 1, 10)
        y = legendre_polynomial_p_evaluate(c, x)
        y_expected = (x - 0.5) * (x + 0.5)

        torch.testing.assert_close(y, y_expected, atol=1e-5, rtol=1e-5)

    def test_from_roots_roundtrip(self):
        """roots(from_roots(r)) == r."""
        roots_orig = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64)
        c = legendre_polynomial_p_from_roots(roots_orig)
        roots_recovered = legendre_polynomial_p_roots(c)

        roots_orig_sorted = roots_orig.sort().values
        roots_recovered_sorted = roots_recovered.real.sort().values

        torch.testing.assert_close(
            roots_recovered_sorted, roots_orig_sorted, atol=1e-8, rtol=1e-8
        )

    def test_from_roots_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legfromroots."""
        roots = [0.1, -0.2, 0.5, -0.8]

        c_torch = legendre_polynomial_p_from_roots(
            torch.tensor(roots, dtype=torch.float64)
        )
        c_np = np_leg.legfromroots(roots)

        np.testing.assert_allclose(
            c_torch.as_subclass(torch.Tensor).numpy(), c_np, rtol=1e-10
        )

    def test_from_roots_empty(self):
        """Empty roots gives constant 1."""
        roots = torch.tensor([], dtype=torch.float64)
        c = legendre_polynomial_p_from_roots(roots)

        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0], dtype=torch.float64),
        )
