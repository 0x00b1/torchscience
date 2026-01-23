"""Tests for spherical harmonics."""

import math

import pytest
import torch

from torchscience.special_functions._spherical_harmonic_y import (
    spherical_harmonic_y,
    spherical_harmonic_y_all,
    spherical_harmonic_y_cartesian,
)


class TestSphericalHarmonicY:
    """Tests for spherical_harmonic_y function."""

    def test_y00(self):
        """Y_0^0 = 1/(2*sqrt(pi)) for all angles."""
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2, math.pi], dtype=torch.float64
        )
        phi = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2, math.pi], dtype=torch.float64
        )
        result = spherical_harmonic_y(0, 0, theta, phi)
        expected = torch.full_like(
            theta, 1 / (2 * math.sqrt(math.pi)), dtype=torch.complex128
        )
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_y10(self):
        """Y_1^0 = sqrt(3/(4*pi)) * cos(theta)."""
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2, math.pi], dtype=torch.float64
        )
        phi = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = spherical_harmonic_y(1, 0, theta, phi)
        norm = math.sqrt(3 / (4 * math.pi))
        expected = norm * torch.cos(theta)
        torch.testing.assert_close(
            result.real, expected, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            result.imag,
            torch.zeros_like(expected),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_y11(self):
        """Y_1^1 = -sqrt(3/(8*pi)) * sin(theta) * exp(i*phi)."""
        theta = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        result = spherical_harmonic_y(1, 1, theta, phi)

        # Normalization: sqrt((2*1+1)/(4*pi) * 0!/2!) = sqrt(3/(8*pi))
        norm = math.sqrt(3 / (8 * math.pi))
        # P_1^1(cos(theta)) = -sin(theta)  (Condon-Shortley)
        sin_theta = torch.sin(theta)
        expected = norm * (-sin_theta) * torch.exp(1j * phi)

        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_y1_neg1(self):
        """Y_1^{-1} should satisfy Y_l^{-m} = (-1)^m * conj(Y_l^m)."""
        theta = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([math.pi / 3, math.pi / 4], dtype=torch.float64)

        Y_1_neg1 = spherical_harmonic_y(1, -1, theta, phi)
        Y_1_1 = spherical_harmonic_y(1, 1, theta, phi)

        # Y_1^{-1} = (-1)^1 * conj(Y_1^1) = -conj(Y_1^1)
        expected = -torch.conj(Y_1_1)
        torch.testing.assert_close(Y_1_neg1, expected, atol=1e-10, rtol=1e-10)

    def test_y20(self):
        """Y_2^0 = sqrt(5/(16*pi)) * (3*cos^2(theta) - 1)."""
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )
        phi = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        result = spherical_harmonic_y(2, 0, theta, phi)

        norm = math.sqrt(5 / (16 * math.pi))
        cos_theta = torch.cos(theta)
        # P_2^0(x) = (3x^2 - 1)/2
        expected = norm * (3 * cos_theta**2 - 1)

        torch.testing.assert_close(
            result.real, expected, atol=1e-10, rtol=1e-10
        )

    def test_orthonormality_y00_y10(self):
        """Test orthogonality of Y_0^0 and Y_1^0."""
        # Use numerical integration over sphere
        n_theta = 50
        n_phi = 50
        theta = torch.linspace(0, math.pi, n_theta, dtype=torch.float64)
        phi = torch.linspace(0, 2 * math.pi, n_phi, dtype=torch.float64)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

        Y_00 = spherical_harmonic_y(0, 0, theta_grid, phi_grid)
        Y_10 = spherical_harmonic_y(1, 0, theta_grid, phi_grid)

        # Integration weights: sin(theta) * d_theta * d_phi
        d_theta = math.pi / (n_theta - 1)
        d_phi = 2 * math.pi / (n_phi - 1)
        sin_theta = torch.sin(theta_grid)

        # Integral of conj(Y_00) * Y_10 should be 0
        integrand = torch.conj(Y_00) * Y_10 * sin_theta
        integral = integrand.sum() * d_theta * d_phi

        torch.testing.assert_close(
            integral.real,
            torch.tensor(0.0, dtype=torch.float64),
            atol=0.01,
            rtol=0.01,
        )

    def test_real_y00(self):
        """Real Y_0^0 should equal complex Y_0^0."""
        theta = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        Y_complex = spherical_harmonic_y(0, 0, theta, phi, real=False)
        Y_real = spherical_harmonic_y(0, 0, theta, phi, real=True)

        torch.testing.assert_close(
            Y_complex.real, Y_real, atol=1e-10, rtol=1e-10
        )

    def test_real_y11(self):
        """Real Y_1^1 = sqrt(2) * N * P_1^1 * cos(phi)."""
        theta = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        Y_real = spherical_harmonic_y(1, 1, theta, phi, real=True)

        # Expected: sqrt(2) * sqrt(3/(8*pi)) * (-sin(theta)) * cos(phi)
        norm = math.sqrt(2) * math.sqrt(3 / (8 * math.pi))
        expected = norm * (-torch.sin(theta)) * torch.cos(phi)

        torch.testing.assert_close(Y_real, expected, atol=1e-10, rtol=1e-10)

    def test_real_y1_neg1(self):
        """Real Y_1^{-1} = sqrt(2) * N * P_1^1 * sin(phi)."""
        theta = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)

        Y_real = spherical_harmonic_y(1, -1, theta, phi, real=True)

        # Expected: sqrt(2) * sqrt(3/(8*pi)) * (-sin(theta)) * sin(phi)
        norm = math.sqrt(2) * math.sqrt(3 / (8 * math.pi))
        expected = norm * (-torch.sin(theta)) * torch.sin(phi)

        torch.testing.assert_close(Y_real, expected, atol=1e-10, rtol=1e-10)

    def test_l_negative_raises(self):
        """Negative l should raise ValueError."""
        theta = torch.tensor([0.0])
        phi = torch.tensor([0.0])
        with pytest.raises(ValueError):
            spherical_harmonic_y(-1, 0, theta, phi)

    def test_m_greater_than_l_raises(self):
        """m > l should raise ValueError."""
        theta = torch.tensor([0.0])
        phi = torch.tensor([0.0])
        with pytest.raises(ValueError):
            spherical_harmonic_y(1, 2, theta, phi)


class TestSphericalHarmonicYCartesian:
    """Tests for spherical_harmonic_y_cartesian function."""

    def test_north_pole(self):
        """Test at north pole (0, 0, 1)."""
        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        Y_00 = spherical_harmonic_y_cartesian(0, 0, x, y, z)
        expected = 1 / (2 * math.sqrt(math.pi))
        torch.testing.assert_close(
            Y_00.real,
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_equator(self):
        """Test at equator (1, 0, 0)."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([0.0], dtype=torch.float64)

        # At equator, theta = pi/2, phi = 0
        # Y_1^0 = sqrt(3/(4*pi)) * cos(pi/2) = 0
        Y_10 = spherical_harmonic_y_cartesian(1, 0, x, y, z)
        torch.testing.assert_close(
            Y_10.real,
            torch.tensor([0.0], dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_matches_spherical(self):
        """Cartesian version should match spherical version."""
        theta = torch.tensor([math.pi / 3], dtype=torch.float64)
        phi = torch.tensor([math.pi / 4], dtype=torch.float64)

        # Convert to Cartesian
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        for l in range(3):
            for m in range(-l, l + 1):
                Y_spherical = spherical_harmonic_y(l, m, theta, phi)
                Y_cartesian = spherical_harmonic_y_cartesian(l, m, x, y, z)
                torch.testing.assert_close(
                    Y_spherical, Y_cartesian, atol=1e-10, rtol=1e-10
                )


class TestSphericalHarmonicYAll:
    """Tests for spherical_harmonic_y_all function."""

    def test_shape(self):
        """Test output shape."""
        theta = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        Y_all = spherical_harmonic_y_all(2, theta, phi)
        # (l_max + 1)^2 = 9 harmonics
        assert Y_all.shape == (2, 9)

    def test_values_match_single(self):
        """Values should match single-call function."""
        theta = torch.tensor([math.pi / 3, math.pi / 4], dtype=torch.float64)
        phi = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)

        Y_all = spherical_harmonic_y_all(2, theta, phi)

        for l in range(3):
            for m in range(-l, l + 1):
                idx = l * l + l + m
                Y_single = spherical_harmonic_y(l, m, theta, phi)
                torch.testing.assert_close(
                    Y_all[..., idx], Y_single, atol=1e-10, rtol=1e-10
                )

    def test_real_shape(self):
        """Test real output is real-valued."""
        theta = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        Y_all = spherical_harmonic_y_all(2, theta, phi, real=True)
        assert Y_all.dtype == torch.float64
        assert Y_all.shape == (2, 9)

    def test_l_max_0(self):
        """l_max=0 should give single Y_0^0."""
        theta = torch.tensor([math.pi / 4], dtype=torch.float64)
        phi = torch.tensor([math.pi / 4], dtype=torch.float64)

        Y_all = spherical_harmonic_y_all(0, theta, phi)
        assert Y_all.shape == (1, 1)

        Y_00 = spherical_harmonic_y(0, 0, theta, phi)
        torch.testing.assert_close(Y_all[..., 0], Y_00, atol=1e-10, rtol=1e-10)


class TestSphericalHarmonicVsScipy:
    """Compare with scipy.special.sph_harm if available."""

    def test_vs_scipy(self):
        """Compare with scipy.special.sph_harm for various (l, m) pairs."""
        try:
            from scipy.special import sph_harm
        except ImportError:
            pytest.skip("scipy not available")

        theta = torch.tensor(
            [math.pi / 4, math.pi / 3, math.pi / 2], dtype=torch.float64
        )
        phi = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        test_cases = [
            (0, 0),
            (1, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 1),
            (2, -1),
            (2, 2),
        ]

        for l, m in test_cases:
            result = spherical_harmonic_y(l, m, theta, phi)
            # scipy uses different argument order: sph_harm(m, l, phi, theta)
            expected = torch.tensor(
                sph_harm(m, l, phi.numpy(), theta.numpy()),
                dtype=torch.complex128,
            )
            torch.testing.assert_close(
                result, expected, atol=1e-10, rtol=1e-10
            )
