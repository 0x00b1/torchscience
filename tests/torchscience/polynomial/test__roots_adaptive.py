import torch

from torchscience.polynomial import polynomial, polynomial_roots


class TestPolynomialRootsAdaptive:
    """Tests for adaptive polynomial root finding."""

    def test_roots_low_degree_uses_companion(self):
        """Low-degree uses companion matrix (default behavior)."""
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
        p = polynomial(coeffs)
        roots = polynomial_roots(p)
        roots_sorted = torch.sort(roots.real)[0]
        expected = torch.tensor([2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_high_degree_correct(self):
        """High-degree polynomial roots are correct using aberth method.

        Note: For very high degrees (>30), building polynomials from roots
        leads to numerical instability. We test aberth at degree 20 where
        it achieves high precision.
        """
        # Use degree 20 which achieves good precision with Chebyshev roots
        n = 20
        k = torch.arange(n, dtype=torch.float64)
        true_roots = torch.cos(torch.pi * (2 * k + 1) / (2 * n))

        # Build monic polynomial from Chebyshev roots
        coeffs = torch.tensor([1.0], dtype=torch.float64)
        for r in true_roots:
            new_coeffs = torch.zeros(len(coeffs) + 1, dtype=torch.float64)
            new_coeffs[1:] = coeffs
            new_coeffs[:-1] -= r * coeffs
            coeffs = new_coeffs

        p = polynomial(coeffs)
        # Force aberth method to test it works
        roots = polynomial_roots(p, method="aberth")

        # Check roots by sorting and comparing
        roots_sorted = torch.sort(roots.real)[0]
        true_roots_sorted = torch.sort(true_roots)[0]
        torch.testing.assert_close(
            roots_sorted, true_roots_sorted, atol=1e-8, rtol=1e-8
        )

    def test_roots_method_parameter(self):
        """Method parameter selects algorithm."""
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
        p = polynomial(coeffs)

        # Explicitly request companion
        roots_companion = polynomial_roots(p, method="companion")

        # Explicitly request aberth
        roots_aberth = polynomial_roots(p, method="aberth")

        # Both should give same roots
        torch.testing.assert_close(
            torch.sort(roots_companion.real)[0],
            torch.sort(roots_aberth.real)[0],
            atol=1e-10,
            rtol=1e-10,
        )

    def test_roots_method_auto_low_degree(self):
        """Auto method uses companion for low degree."""
        # Degree 10 polynomial (below threshold)
        true_roots = torch.linspace(-0.9, 0.9, 10, dtype=torch.float64)
        coeffs = torch.tensor([1.0], dtype=torch.float64)
        for r in true_roots:
            new_coeffs = torch.zeros(len(coeffs) + 1, dtype=torch.float64)
            new_coeffs[1:] = coeffs
            new_coeffs[:-1] -= r * coeffs
            coeffs = new_coeffs

        p = polynomial(coeffs)
        roots = polynomial_roots(p, method="auto")

        # Verify roots
        roots_sorted = torch.sort(roots.real)[0]
        torch.testing.assert_close(
            roots_sorted, true_roots.sort()[0], atol=1e-8, rtol=1e-8
        )

    def test_roots_method_invalid(self):
        """Invalid method raises error."""
        coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
        p = polynomial(coeffs)

        import pytest

        with pytest.raises(ValueError, match="Unknown method"):
            polynomial_roots(p, method="invalid")

    def test_roots_aberth_explicit_high_degree(self):
        """Explicitly request aberth for polynomial built from roots.

        Tests that aberth method can be explicitly requested and produces
        accurate results. Uses degree 25 with relaxed tolerance due to
        numerical conditioning of polynomial built from roots.
        """
        # Use Chebyshev roots which are well-conditioned
        n = 25
        k = torch.arange(n, dtype=torch.float64)
        true_roots = torch.cos(torch.pi * (2 * k + 1) / (2 * n))

        # Build monic polynomial from Chebyshev roots
        coeffs = torch.tensor([1.0], dtype=torch.float64)
        for r in true_roots:
            new_coeffs = torch.zeros(len(coeffs) + 1, dtype=torch.float64)
            new_coeffs[1:] = coeffs
            new_coeffs[:-1] -= r * coeffs
            coeffs = new_coeffs

        p = polynomial(coeffs)
        roots = polynomial_roots(p, method="aberth")

        # Check roots by sorting and comparing
        # Relax tolerance due to numerical conditioning of high-degree polynomials
        roots_sorted = torch.sort(roots.real)[0]
        true_roots_sorted = torch.sort(true_roots)[0]
        torch.testing.assert_close(
            roots_sorted, true_roots_sorted, atol=1e-5, rtol=1e-5
        )

    def test_roots_companion_explicit_high_degree(self):
        """Explicitly request companion for high-degree polynomial."""
        # Build polynomial from known roots
        true_roots = torch.linspace(-0.9, 0.9, 80, dtype=torch.float64)
        coeffs = torch.tensor([1.0], dtype=torch.float64)
        for r in true_roots:
            new_coeffs = torch.zeros(len(coeffs) + 1, dtype=torch.float64)
            new_coeffs[1:] = coeffs
            new_coeffs[:-1] -= r * coeffs
            coeffs = new_coeffs

        p = polynomial(coeffs)
        roots = polynomial_roots(p, method="companion")

        # Verify roots by checking polynomial evaluates to ~0
        p_at_roots = p(roots)
        # Companion may be less accurate for high-degree but should still work
        assert p_at_roots.abs().max() < 1e-3

    def test_roots_complex_coefficients(self):
        """Works with complex coefficients using aberth."""
        # (x - (1+i))(x - (1-i)) = x^2 - 2x + 2
        coeffs = torch.tensor([2.0, -2.0, 1.0], dtype=torch.complex128)
        p = polynomial(coeffs)

        roots_aberth = polynomial_roots(p, method="aberth")
        roots_companion = polynomial_roots(p, method="companion")

        # Sort by imaginary part for comparison
        roots_aberth_sorted = roots_aberth[torch.argsort(roots_aberth.imag)]
        roots_companion_sorted = roots_companion[
            torch.argsort(roots_companion.imag)
        ]

        # Compare with same dtype (both should be complex128 for complex128 input)
        expected = torch.tensor(
            [1.0 - 1.0j, 1.0 + 1.0j], dtype=torch.complex128
        )
        torch.testing.assert_close(
            roots_aberth_sorted, expected, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            roots_companion_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_batched_polynomials(self):
        """Works with batched polynomials."""
        # Batch of 3 quadratics
        coeffs = torch.tensor(
            [
                [2.0, -3.0, 1.0],  # (x-1)(x-2)
                [6.0, -5.0, 1.0],  # (x-2)(x-3)
                [12.0, -7.0, 1.0],  # (x-3)(x-4)
            ],
            dtype=torch.float64,
        )

        p = polynomial(coeffs)

        # Test both methods
        roots_companion = polynomial_roots(p, method="companion")
        roots_aberth = polynomial_roots(p, method="aberth")

        assert roots_companion.shape == (3, 2)
        assert roots_aberth.shape == (3, 2)

        # Both should give same roots (sorted)
        for i in range(3):
            companion_sorted = torch.sort(roots_companion[i].real)[0]
            aberth_sorted = torch.sort(roots_aberth[i].real)[0]
            torch.testing.assert_close(
                companion_sorted, aberth_sorted, atol=1e-9, rtol=1e-9
            )

    def test_roots_linear_polynomial(self):
        """Linear polynomial works with both methods."""
        # 2x + 4 = 0 -> x = -2
        coeffs = torch.tensor([4.0, 2.0], dtype=torch.float64)
        p = polynomial(coeffs)

        roots_companion = polynomial_roots(p, method="companion")
        roots_aberth = polynomial_roots(p, method="aberth")

        torch.testing.assert_close(
            roots_companion.real,
            torch.tensor([-2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            roots_aberth.real,
            torch.tensor([-2.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )
