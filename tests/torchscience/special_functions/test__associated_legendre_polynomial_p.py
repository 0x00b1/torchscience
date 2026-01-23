"""Tests for associated Legendre polynomials."""

import math

import pytest
import torch

from torchscience.special_functions._associated_legendre_polynomial_p import (
    associated_legendre_polynomial_p,
    associated_legendre_polynomial_p_all,
)


class TestAssociatedLegendrePolynomialP:
    """Tests for associated_legendre_polynomial_p function."""

    def test_p00_is_one(self):
        """P_0^0(x) = 1 for all x."""
        x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        result = associated_legendre_polynomial_p(0, 0, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected)

    def test_p10_is_x(self):
        """P_1^0(x) = x."""
        x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        result = associated_legendre_polynomial_p(1, 0, x)
        torch.testing.assert_close(result, x)

    def test_p11(self):
        """P_1^1(x) = -sqrt(1-x^2)."""
        x = torch.tensor([0.0, 0.5, 0.8])
        result = associated_legendre_polynomial_p(1, 1, x)
        expected = -torch.sqrt(1 - x * x)
        torch.testing.assert_close(result, expected)

    def test_p20(self):
        """P_2^0(x) = (3x^2 - 1) / 2."""
        x = torch.tensor([0.0, 0.5, 1.0, -1.0])
        result = associated_legendre_polynomial_p(2, 0, x)
        expected = (3 * x * x - 1) / 2
        torch.testing.assert_close(result, expected)

    def test_p21(self):
        """P_2^1(x) = -3x * sqrt(1-x^2)."""
        x = torch.tensor([0.0, 0.5, 0.8])
        result = associated_legendre_polynomial_p(2, 1, x)
        expected = -3 * x * torch.sqrt(1 - x * x)
        torch.testing.assert_close(result, expected)

    def test_p22(self):
        """P_2^2(x) = 3(1-x^2)."""
        x = torch.tensor([0.0, 0.5, 0.8])
        result = associated_legendre_polynomial_p(2, 2, x)
        expected = 3 * (1 - x * x)
        torch.testing.assert_close(result, expected)

    def test_p30(self):
        """P_3^0(x) = (5x^3 - 3x) / 2."""
        x = torch.tensor([0.0, 0.5, 1.0])
        result = associated_legendre_polynomial_p(3, 0, x)
        expected = (5 * x**3 - 3 * x) / 2
        torch.testing.assert_close(result, expected)

    def test_vs_scipy(self):
        """Compare with scipy.special.lpmv for various (n, m) pairs."""
        try:
            from scipy.special import lpmv
        except ImportError:
            pytest.skip("scipy not available")

        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        test_cases = [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 2),
            (4, 3),
        ]

        for n, m in test_cases:
            result = associated_legendre_polynomial_p(n, m, x)
            expected = torch.tensor(lpmv(m, n, x.numpy()), dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, atol=1e-10, rtol=1e-10
            )

    def test_negative_m(self):
        """Test negative m using symmetry relation."""
        x = torch.tensor([0.0, 0.5], dtype=torch.float64)
        # P_n^{-m} = (-1)^m * (n-m)!/(n+m)! * P_n^m
        # P_2^{-1} = (-1)^1 * (2-1)!/(2+1)! * P_2^1 = -1/6 * P_2^1
        p2_neg1 = associated_legendre_polynomial_p(2, -1, x)
        p2_1 = associated_legendre_polynomial_p(2, 1, x)
        expected = -p2_1 / 6
        torch.testing.assert_close(p2_neg1, expected)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError):
            associated_legendre_polynomial_p(-1, 0, torch.tensor([0.5]))

    def test_m_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            associated_legendre_polynomial_p(2, 3, torch.tensor([0.5]))


class TestAssociatedLegendrePolynomialPAll:
    """Tests for associated_legendre_polynomial_p_all function."""

    def test_shape(self):
        x = torch.tensor([0.5])
        result = associated_legendre_polynomial_p_all(3, x)
        assert result.shape == (1, 4, 4)

    def test_batched_shape(self):
        x = torch.randn(2, 3)
        result = associated_legendre_polynomial_p_all(2, x)
        assert result.shape == (2, 3, 3, 3)

    def test_values_match_single(self):
        """Values should match single-call function."""
        x = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        P_all = associated_legendre_polynomial_p_all(3, x)

        for n in range(4):
            for m in range(n + 1):
                P_single = associated_legendre_polynomial_p(n, m, x)
                torch.testing.assert_close(
                    P_all[..., n, m], P_single, atol=1e-10, rtol=1e-10
                )

    def test_zeros_for_m_greater_than_n(self):
        """P[..., n, m] should be 0 for m > n."""
        x = torch.tensor([0.5])
        P = associated_legendre_polynomial_p_all(3, x)
        assert P[0, 0, 1] == 0
        assert P[0, 0, 2] == 0
        assert P[0, 1, 2] == 0


class TestNormalization:
    """Tests for normalized associated Legendre polynomials."""

    def test_orthonormality(self):
        """Test orthonormality: integral of P_n^m * P_{n'}^m should be delta_{nn'}."""
        # Use Gauss-Legendre quadrature for numerical integration
        try:
            from numpy.polynomial.legendre import leggauss
        except ImportError:
            pytest.skip("numpy not available")

        n_points = 100
        nodes, weights = leggauss(n_points)
        x = torch.tensor(nodes, dtype=torch.float64)
        w = torch.tensor(weights, dtype=torch.float64)

        # Test for m=0
        for n1 in range(4):
            for n2 in range(4):
                P_n1 = associated_legendre_polynomial_p(
                    n1, 0, x, normalized=True
                )
                P_n2 = associated_legendre_polynomial_p(
                    n2, 0, x, normalized=True
                )
                integral = (P_n1 * P_n2 * w).sum()

                if n1 == n2:
                    torch.testing.assert_close(
                        integral,
                        torch.tensor(1.0, dtype=torch.float64),
                        atol=1e-8,
                        rtol=1e-8,
                    )
                else:
                    torch.testing.assert_close(
                        integral,
                        torch.tensor(0.0, dtype=torch.float64),
                        atol=1e-8,
                        rtol=1e-8,
                    )

    def test_normalized_values(self):
        """Test specific normalized values."""
        x = torch.tensor([0.0], dtype=torch.float64)

        # Normalized P_0^0 at x=0
        P_00 = associated_legendre_polynomial_p(0, 0, x, normalized=True)
        # Normalization factor: sqrt(1 / 2)
        expected = math.sqrt(1 / 2)
        torch.testing.assert_close(
            P_00, torch.tensor([expected], dtype=torch.float64)
        )
