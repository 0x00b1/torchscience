"""Tests for spectral differentiation matrices and collocation points."""

import math

import pytest
import torch

from torchscience.polynomial._spectral import (
    chebyshev_differentiation_matrix,
    chebyshev_differentiation_matrix_2,
    chebyshev_differentiation_matrix_scaled,
    chebyshev_points,
    integration_matrix,
    lagrange_differentiation_matrix,
    legendre_differentiation_matrix,
    legendre_gauss_lobatto_points,
    legendre_gauss_points,
    uniform_points,
)


class TestChebyshevPoints:
    """Tests for Chebyshev-Gauss-Lobatto points."""

    def test_n_0(self):
        """n=0 gives single point at 0."""
        x = chebyshev_points(0)
        assert x.shape == (1,)
        torch.testing.assert_close(x, torch.tensor([0.0], dtype=torch.float64))

    def test_n_1(self):
        """n=1 gives endpoints."""
        x = chebyshev_points(1)
        expected = torch.tensor([1.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close(x, expected)

    def test_n_2(self):
        """n=2 gives [1, 0, -1]."""
        x = chebyshev_points(2)
        expected = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close(x, expected)

    def test_endpoints(self):
        """First and last points should be +/- 1."""
        for n in [3, 5, 10, 20]:
            x = chebyshev_points(n)
            torch.testing.assert_close(
                x[0], torch.tensor(1.0, dtype=torch.float64)
            )
            torch.testing.assert_close(
                x[-1], torch.tensor(-1.0, dtype=torch.float64)
            )

    def test_symmetry(self):
        """Points should be symmetric about 0."""
        x = chebyshev_points(8)
        torch.testing.assert_close(x, -x.flip(0), atol=1e-14, rtol=1e-14)

    def test_shape(self):
        """Should return n+1 points."""
        for n in [0, 1, 5, 10, 20]:
            x = chebyshev_points(n)
            assert x.shape == (n + 1,)


class TestChebyshevDifferentiationMatrix:
    """Tests for Chebyshev differentiation matrix."""

    def test_shape(self):
        """Matrix should be (n+1) x (n+1)."""
        for n in [1, 5, 10]:
            D = chebyshev_differentiation_matrix(n)
            assert D.shape == (n + 1, n + 1)

    def test_n_0(self):
        """n=0 gives 1x1 zero matrix."""
        D = chebyshev_differentiation_matrix(0)
        assert D.shape == (1, 1)
        torch.testing.assert_close(D, torch.zeros(1, 1, dtype=torch.float64))

    def test_constant_function(self):
        """Derivative of constant should be zero."""
        D = chebyshev_differentiation_matrix(10)
        f = torch.ones(11, dtype=torch.float64)
        df = D @ f
        torch.testing.assert_close(
            df, torch.zeros(11, dtype=torch.float64), atol=1e-12, rtol=1e-12
        )

    def test_linear_function(self):
        """Derivative of x should be 1."""
        n = 10
        D = chebyshev_differentiation_matrix(n)
        x = chebyshev_points(n)
        df = D @ x
        expected = torch.ones(n + 1, dtype=torch.float64)
        torch.testing.assert_close(df, expected, atol=1e-12, rtol=1e-12)

    def test_quadratic_function(self):
        """Derivative of x^2 should be 2x."""
        n = 10
        D = chebyshev_differentiation_matrix(n)
        x = chebyshev_points(n)
        f = x**2
        df = D @ f
        expected = 2 * x
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_polynomial_exact(self):
        """Differentiation should be exact for polynomials up to degree n."""
        n = 8
        D = chebyshev_differentiation_matrix(n)
        x = chebyshev_points(n)

        # Test x^k for k = 0, 1, ..., n-1
        for k in range(n):
            f = x**k
            df = D @ f
            expected = k * x ** max(0, k - 1) if k > 0 else torch.zeros_like(x)
            torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_sin_function(self):
        """Derivative of sin(x) should approximate cos(x)."""
        n = 20
        D = chebyshev_differentiation_matrix(n)
        x = chebyshev_points(n)
        f = torch.sin(x)
        df = D @ f
        expected = torch.cos(x)
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_row_sum_zero(self):
        """Each row should sum to approximately zero."""
        D = chebyshev_differentiation_matrix(10)
        row_sums = D.sum(dim=1)
        torch.testing.assert_close(
            row_sums,
            torch.zeros(11, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


class TestChebyshevDifferentiationMatrix2:
    """Tests for second-order Chebyshev differentiation matrix."""

    def test_shape(self):
        """Matrix should be (n+1) x (n+1)."""
        D2 = chebyshev_differentiation_matrix_2(10)
        assert D2.shape == (11, 11)

    def test_quadratic_function(self):
        """Second derivative of x^2 should be 2."""
        n = 10
        D2 = chebyshev_differentiation_matrix_2(n)
        x = chebyshev_points(n)
        f = x**2
        d2f = D2 @ f
        expected = 2.0 * torch.ones(n + 1, dtype=torch.float64)
        torch.testing.assert_close(d2f, expected, atol=1e-10, rtol=1e-10)

    def test_cubic_function(self):
        """Second derivative of x^3 should be 6x."""
        n = 10
        D2 = chebyshev_differentiation_matrix_2(n)
        x = chebyshev_points(n)
        f = x**3
        d2f = D2 @ f
        expected = 6.0 * x
        torch.testing.assert_close(d2f, expected, atol=1e-9, rtol=1e-9)

    def test_sin_function(self):
        """Second derivative of sin(x) should approximate -sin(x)."""
        n = 20
        D2 = chebyshev_differentiation_matrix_2(n)
        x = chebyshev_points(n)
        f = torch.sin(x)
        d2f = D2 @ f
        expected = -torch.sin(x)
        torch.testing.assert_close(d2f, expected, atol=1e-8, rtol=1e-8)


class TestChebyshevDifferentiationMatrixScaled:
    """Tests for scaled Chebyshev differentiation matrix."""

    def test_unit_interval(self):
        """[0, 1] interval scaling."""
        n = 10
        D, x = chebyshev_differentiation_matrix_scaled(n, 0.0, 1.0)

        assert D.shape == (n + 1, n + 1)
        assert x.shape == (n + 1,)

        # Points should be in [0, 1]
        assert x.min() >= 0.0
        assert x.max() <= 1.0

        # Test derivative of x^2 on [0, 1]
        f = x**2
        df = D @ f
        expected = 2 * x
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_arbitrary_interval(self):
        """Arbitrary interval [a, b]."""
        n = 10
        a, b = 2.0, 5.0
        D, x = chebyshev_differentiation_matrix_scaled(n, a, b)

        # Points should be in [a, b]
        assert x.min() >= a - 1e-10
        assert x.max() <= b + 1e-10

        # Test derivative of x on [2, 5] should be 1
        f = x
        df = D @ f
        expected = torch.ones(n + 1, dtype=torch.float64)
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)


class TestLegendreGaussLobattoPoints:
    """Tests for Legendre-Gauss-Lobatto points."""

    def test_n_0(self):
        """n=0 gives single point at 0."""
        x = legendre_gauss_lobatto_points(0)
        assert x.shape == (1,)

    def test_n_1(self):
        """n=1 gives endpoints."""
        x = legendre_gauss_lobatto_points(1)
        expected = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(x, expected)

    def test_endpoints(self):
        """First and last points should be -1 and 1."""
        for n in [2, 5, 10]:
            x = legendre_gauss_lobatto_points(n)
            torch.testing.assert_close(
                x[0],
                torch.tensor(-1.0, dtype=torch.float64),
                atol=1e-14,
                rtol=1e-14,
            )
            torch.testing.assert_close(
                x[-1],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-14,
                rtol=1e-14,
            )

    def test_symmetry(self):
        """Points should be symmetric about 0."""
        x = legendre_gauss_lobatto_points(8)
        torch.testing.assert_close(x, -x.flip(0), atol=1e-12, rtol=1e-12)

    def test_ascending_order(self):
        """Points should be in ascending order."""
        x = legendre_gauss_lobatto_points(10)
        assert torch.all(x[1:] > x[:-1])


class TestLegendreGaussPoints:
    """Tests for Legendre-Gauss quadrature points."""

    def test_n_0(self):
        """n=0 gives empty tensor."""
        x = legendre_gauss_points(0)
        assert x.shape == (0,)

    def test_n_1(self):
        """n=1 gives single point at 0."""
        x = legendre_gauss_points(1)
        torch.testing.assert_close(
            x, torch.tensor([0.0], dtype=torch.float64), atol=1e-14, rtol=1e-14
        )

    def test_n_2(self):
        """n=2 gives +/- 1/sqrt(3)."""
        x = legendre_gauss_points(2)
        expected = torch.tensor(
            [-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)], dtype=torch.float64
        )
        torch.testing.assert_close(
            torch.sort(x)[0], expected, atol=1e-14, rtol=1e-14
        )

    def test_interior_points(self):
        """All points should be strictly inside (-1, 1)."""
        for n in [2, 5, 10]:
            x = legendre_gauss_points(n)
            assert x.min() > -1.0
            assert x.max() < 1.0

    def test_symmetry(self):
        """Points should be symmetric about 0."""
        x = legendre_gauss_points(8)
        # For symmetric points, x_i = -x_{n-1-i}
        torch.testing.assert_close(x, -x.flip(0), atol=1e-12, rtol=1e-12)


class TestUniformPoints:
    """Tests for uniformly spaced points."""

    def test_endpoints(self):
        """Should include -1 and 1."""
        x = uniform_points(10)
        torch.testing.assert_close(
            x[0], torch.tensor(-1.0, dtype=torch.float64)
        )
        torch.testing.assert_close(
            x[-1], torch.tensor(1.0, dtype=torch.float64)
        )

    def test_spacing(self):
        """Points should be uniformly spaced."""
        x = uniform_points(10)
        diffs = x[1:] - x[:-1]
        expected_spacing = 2.0 / 10
        torch.testing.assert_close(
            diffs, expected_spacing * torch.ones(10, dtype=torch.float64)
        )


class TestLagrangeDifferentiationMatrix:
    """Tests for general Lagrange differentiation matrix."""

    def test_matches_chebyshev(self):
        """Should match Chebyshev matrix for Chebyshev points."""
        n = 10
        x = chebyshev_points(n)
        D_lagrange = lagrange_differentiation_matrix(x)
        D_chebyshev = chebyshev_differentiation_matrix(n)
        torch.testing.assert_close(
            D_lagrange, D_chebyshev, atol=1e-10, rtol=1e-10
        )

    def test_linear_function(self):
        """Derivative of x should be 1 for any point set."""
        x = uniform_points(10)
        D = lagrange_differentiation_matrix(x)
        df = D @ x
        expected = torch.ones(11, dtype=torch.float64)
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_constant_function(self):
        """Derivative of constant should be zero."""
        x = uniform_points(10)
        D = lagrange_differentiation_matrix(x)
        f = torch.ones(11, dtype=torch.float64)
        df = D @ f
        torch.testing.assert_close(
            df, torch.zeros(11, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )


class TestLegendreDifferentiationMatrix:
    """Tests for Legendre spectral differentiation matrix."""

    def test_returns_tuple(self):
        """Should return (D, x) tuple."""
        result = legendre_differentiation_matrix(10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shape(self):
        """Matrix should be (n+1) x (n+1)."""
        D, x = legendre_differentiation_matrix(10)
        assert D.shape == (11, 11)
        assert x.shape == (11,)

    def test_linear_function(self):
        """Derivative of x should be 1."""
        D, x = legendre_differentiation_matrix(10)
        df = D @ x
        expected = torch.ones(11, dtype=torch.float64)
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_polynomial_exact(self):
        """Should be exact for low-degree polynomials."""
        n = 10
        D, x = legendre_differentiation_matrix(n)

        # Test x^2
        f = x**2
        df = D @ f
        expected = 2 * x
        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)


class TestIntegrationMatrix:
    """Tests for spectral integration matrix."""

    def test_shape(self):
        """Matrix should be (n+1) x (n+1)."""
        x = chebyshev_points(10)
        S = integration_matrix(x)
        assert S.shape == (11, 11)

    def test_constant_function(self):
        """Integral of 1 from x_0 to x_i should be x_i - x_0."""
        x = chebyshev_points(10)
        S = integration_matrix(x)
        f = torch.ones(11, dtype=torch.float64)
        integral = S @ f
        expected = x - x[0]
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_linear_function(self):
        """Integral of x from x_0 to x_i should be (x_i^2 - x_0^2) / 2."""
        x = chebyshev_points(10)
        S = integration_matrix(x)
        integral = S @ x
        expected = (x**2 - x[0] ** 2) / 2
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_zero_at_first_point(self):
        """Integral should be zero at x_0."""
        x = chebyshev_points(10)
        S = integration_matrix(x)
        f = torch.randn(11, dtype=torch.float64)
        integral = S @ f
        torch.testing.assert_close(
            integral[0],
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-14,
            rtol=1e-14,
        )


class TestDeviceAndDtype:
    """Tests for device and dtype handling."""

    def test_chebyshev_float32(self):
        """Should support float32."""
        D = chebyshev_differentiation_matrix(5, dtype=torch.float32)
        assert D.dtype == torch.float32

    def test_chebyshev_points_float32(self):
        """Points should support float32."""
        x = chebyshev_points(5, dtype=torch.float32)
        assert x.dtype == torch.float32

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Should support CUDA device."""
        D = chebyshev_differentiation_matrix(5, device=torch.device("cuda"))
        assert D.device.type == "cuda"
