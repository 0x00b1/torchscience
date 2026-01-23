"""Tests for ChebyshevPolynomialT root finding."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_companion,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_from_roots,
    chebyshev_polynomial_t_roots,
)


class TestChebyshevPolynomialTCompanion:
    """Tests for chebyshev_polynomial_t_companion."""

    def test_companion_t2(self):
        """Companion matrix for T_2."""
        # T_2 has roots at x = ±sqrt(2)/2
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        A = chebyshev_polynomial_t_companion(c)
        assert A.shape == (2, 2)

    def test_companion_t3(self):
        """Companion matrix for T_3."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        A = chebyshev_polynomial_t_companion(c)
        assert A.shape == (3, 3)

    def test_companion_eigenvalues_are_roots(self):
        """Eigenvalues of companion matrix are roots."""
        # T_3 = 4x^3 - 3x, roots at 0, ±sqrt(3)/2
        c = chebyshev_polynomial_t(
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        A = chebyshev_polynomial_t_companion(c)
        eigenvalues = torch.linalg.eigvalsh(A)

        expected_roots = torch.tensor(
            [0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2], dtype=torch.float64
        )
        expected_roots = expected_roots.sort().values
        eigenvalues = eigenvalues.sort().values

        torch.testing.assert_close(
            eigenvalues, expected_roots, atol=1e-10, rtol=1e-10
        )

    def test_companion_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebcompanion."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 1.0]  # T_4

        c = chebyshev_polynomial_t(torch.tensor(coeffs, dtype=torch.float64))
        A_torch = chebyshev_polynomial_t_companion(c)

        A_np = np_cheb.chebcompanion(coeffs)

        np.testing.assert_allclose(A_torch.numpy(), A_np, rtol=1e-10)


class TestChebyshevPolynomialTRoots:
    """Tests for chebyshev_polynomial_t_roots."""

    def test_roots_t2(self):
        """Roots of T_2."""
        # T_2 = 2x^2 - 1, roots at ±sqrt(2)/2
        c = chebyshev_polynomial_t(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = chebyshev_polynomial_t_roots(c)

        expected = torch.tensor(
            [np.sqrt(2) / 2, -np.sqrt(2) / 2], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_t3(self):
        """Roots of T_3."""
        # T_3 = 4x^3 - 3x, roots at 0, ±sqrt(3)/2
        c = chebyshev_polynomial_t(
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = chebyshev_polynomial_t_roots(c)

        expected = torch.tensor(
            [0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_verify_evaluation(self):
        """f(root) ≈ 0 for all roots."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float64)
        c = chebyshev_polynomial_t(coeffs)
        roots = chebyshev_polynomial_t_roots(c)

        # Evaluate at roots
        f_at_roots = chebyshev_polynomial_t_evaluate(c, roots)

        torch.testing.assert_close(
            f_at_roots.abs(),
            torch.zeros_like(f_at_roots.abs()),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_roots_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebroots."""
        coeffs = [1.0, -2.0, 3.0, -1.0, 1.0]

        c = chebyshev_polynomial_t(torch.tensor(coeffs, dtype=torch.float64))
        roots_torch = chebyshev_polynomial_t_roots(c)

        roots_np = np_cheb.chebroots(coeffs)

        # Sort for comparison
        roots_torch_sorted = roots_torch.real.sort().values.numpy()
        roots_np_sorted = np.sort(roots_np.real)

        np.testing.assert_allclose(
            roots_torch_sorted, roots_np_sorted, rtol=1e-8
        )


class TestChebyshevPolynomialTFromRoots:
    """Tests for chebyshev_polynomial_t_from_roots."""

    def test_from_roots_single(self):
        """Single root at x=0."""
        roots = torch.tensor([0.0])
        c = chebyshev_polynomial_t_from_roots(roots)

        # Should be (x - 0) = x = T_1
        x = torch.linspace(-1, 1, 10)
        y = chebyshev_polynomial_t_evaluate(c, x)

        torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)

    def test_from_roots_two(self):
        """Two roots."""
        roots = torch.tensor([0.5, -0.5])
        c = chebyshev_polynomial_t_from_roots(roots)

        # Should be (x - 0.5)(x + 0.5) = x^2 - 0.25
        x = torch.linspace(-1, 1, 10)
        y = chebyshev_polynomial_t_evaluate(c, x)
        y_expected = (x - 0.5) * (x + 0.5)

        torch.testing.assert_close(y, y_expected, atol=1e-5, rtol=1e-5)

    def test_from_roots_roundtrip(self):
        """roots(from_roots(r)) == r."""
        roots_orig = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64)
        c = chebyshev_polynomial_t_from_roots(roots_orig)
        roots_recovered = chebyshev_polynomial_t_roots(c)

        roots_orig_sorted = roots_orig.sort().values
        roots_recovered_sorted = roots_recovered.real.sort().values

        torch.testing.assert_close(
            roots_recovered_sorted, roots_orig_sorted, atol=1e-8, rtol=1e-8
        )

    def test_from_roots_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebfromroots."""
        roots = [0.1, -0.2, 0.5, -0.8]

        c_torch = chebyshev_polynomial_t_from_roots(
            torch.tensor(roots, dtype=torch.float64)
        )
        c_np = np_cheb.chebfromroots(roots)

        np.testing.assert_allclose(c_torch.numpy(), c_np, rtol=1e-10)
