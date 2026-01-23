"""Tests for Zernike polynomials."""

import math

import pytest
import torch

from torchscience.polynomial._zernike_polynomial_z import (
    ZERNIKE_NAMES,
    nm_to_noll,
    nm_to_osa,
    noll_to_nm,
    osa_to_nm,
    zernike_polynomial_z,
    zernike_polynomial_z_all,
    zernike_polynomial_z_fit,
    zernike_polynomial_z_noll,
    zernike_polynomial_z_osa,
    zernike_polynomial_z_radial,
)


class TestZernikeRadial:
    """Tests for radial Zernike polynomials."""

    def test_r00(self):
        """R_0^0(rho) = 1."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(0, 0, rho)
        expected = torch.ones_like(rho)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_r11(self):
        """R_1^1(rho) = rho."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(1, 1, rho)
        torch.testing.assert_close(result, rho, atol=1e-10, rtol=1e-10)

    def test_r20(self):
        """R_2^0(rho) = 2*rho^2 - 1."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(2, 0, rho)
        expected = 2 * rho**2 - 1
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_r22(self):
        """R_2^2(rho) = rho^2."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(2, 2, rho)
        expected = rho**2
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_r31(self):
        """R_3^1(rho) = 3*rho^3 - 2*rho."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(3, 1, rho)
        expected = 3 * rho**3 - 2 * rho
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_r33(self):
        """R_3^3(rho) = rho^3."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(3, 3, rho)
        expected = rho**3
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_r40(self):
        """R_4^0(rho) = 6*rho^4 - 6*rho^2 + 1."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = zernike_polynomial_z_radial(4, 0, rho)
        expected = 6 * rho**4 - 6 * rho**2 + 1
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_negative_n_raises(self):
        """Negative n should raise ValueError."""
        rho = torch.tensor([0.5])
        with pytest.raises(ValueError):
            zernike_polynomial_z_radial(-1, 0, rho)

    def test_m_greater_than_n_raises(self):
        """m > n should raise ValueError."""
        rho = torch.tensor([0.5])
        with pytest.raises(ValueError):
            zernike_polynomial_z_radial(2, 3, rho)

    def test_odd_n_minus_m_raises(self):
        """n - m odd should raise ValueError."""
        rho = torch.tensor([0.5])
        with pytest.raises(ValueError):
            zernike_polynomial_z_radial(3, 0, rho)


class TestZernikePolynomialZ:
    """Tests for full Zernike polynomials."""

    def test_z00_piston(self):
        """Z_0^0 (piston) should be constant."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )
        result = zernike_polynomial_z(0, 0, rho, theta, normalized=True)
        # Normalized Z_0^0 = sqrt(1) * 1 = 1
        expected = torch.ones_like(rho)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_z11_tilt_x(self):
        """Z_1^1 (tilt) = sqrt(4) * rho * cos(theta)."""
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)
        result = zernike_polynomial_z(1, 1, rho, theta, normalized=True)
        # Normalized: sqrt(2*(1+1)) * rho * cos(theta) = 2 * rho * cos(theta)
        expected = 2 * rho * torch.cos(theta)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_z1_neg1_tilt_y(self):
        """Z_1^{-1} (tilt y) = sqrt(4) * rho * sin(theta)."""
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([math.pi / 4, math.pi / 2], dtype=torch.float64)
        result = zernike_polynomial_z(1, -1, rho, theta, normalized=True)
        # Z_1^{-1} = -N * R * sin(theta) where N = sqrt(4) = 2
        # The negative sign comes from convention
        expected = -2 * rho * torch.sin(theta)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_z20_defocus(self):
        """Z_2^0 (defocus) = sqrt(3) * (2*rho^2 - 1)."""
        rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        theta = torch.zeros_like(rho)
        result = zernike_polynomial_z(2, 0, rho, theta, normalized=True)
        # Normalized: sqrt(2+1) * (2*rho^2 - 1)
        expected = math.sqrt(3) * (2 * rho**2 - 1)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_unnormalized(self):
        """Unnormalized Zernike should not include normalization factor."""
        rho = torch.tensor([0.5, 1.0], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)
        result = zernike_polynomial_z(1, 1, rho, theta, normalized=False)
        # Without normalization: rho * cos(theta)
        expected = rho * torch.cos(theta)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


class TestNollIndexing:
    """Tests for Noll indexing conversion."""

    def test_noll_to_nm_first_few(self):
        """Test first several Noll indices."""
        expected = [
            (1, (0, 0)),  # Piston
            (2, (1, 1)),  # Tilt Y
            (3, (1, -1)),  # Tilt X
            (4, (2, 0)),  # Defocus
            (5, (2, -2)),  # Astigmatism 45
            (6, (2, 2)),  # Astigmatism 0
            (7, (3, -1)),  # Coma Y
            (8, (3, 1)),  # Coma X
            (9, (3, -3)),  # Trefoil Y
            (10, (3, 3)),  # Trefoil X
            (11, (4, 0)),  # Spherical
        ]
        for j, (n, m) in expected:
            result = noll_to_nm(j)
            assert result == (n, m), (
                f"Noll {j}: expected {(n, m)}, got {result}"
            )

    def test_nm_to_noll_roundtrip(self):
        """Converting nm -> noll -> nm should give original."""
        test_cases = [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 2),
            (2, -2),
            (3, 1),
            (3, -1),
        ]
        for n, m in test_cases:
            j = nm_to_noll(n, m)
            n_back, m_back = noll_to_nm(j)
            assert (n, m) == (
                n_back,
                m_back,
            ), f"Roundtrip failed for ({n}, {m}): got ({n_back}, {m_back})"

    def test_noll_index_zero_raises(self):
        """Noll index 0 should raise ValueError."""
        with pytest.raises(ValueError):
            noll_to_nm(0)


class TestOSAIndexing:
    """Tests for OSA/ANSI indexing conversion."""

    def test_osa_to_nm_first_few(self):
        """Test first several OSA indices."""
        expected = [
            (0, (0, 0)),
            (1, (1, -1)),
            (2, (1, 1)),
            (3, (2, -2)),
            (4, (2, 0)),
            (5, (2, 2)),
        ]
        for j, (n, m) in expected:
            result = osa_to_nm(j)
            assert result == (n, m), (
                f"OSA {j}: expected {(n, m)}, got {result}"
            )

    def test_nm_to_osa_roundtrip(self):
        """Converting nm -> osa -> nm should give original."""
        test_cases = [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 2),
            (2, -2),
            (3, 1),
            (3, -1),
        ]
        for n, m in test_cases:
            j = nm_to_osa(n, m)
            n_back, m_back = osa_to_nm(j)
            assert (n, m) == (
                n_back,
                m_back,
            ), f"Roundtrip failed for ({n}, {m}): got ({n_back}, {m_back})"

    def test_osa_index_negative_raises(self):
        """OSA index < 0 should raise ValueError."""
        with pytest.raises(ValueError):
            osa_to_nm(-1)


class TestZernikeNollOSA:
    """Tests for Zernike polynomial evaluation via Noll/OSA index."""

    def test_noll_matches_nm(self):
        """Noll-indexed evaluation should match (n, m) evaluation."""
        rho = torch.tensor([0.5, 0.8], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        for j in range(1, 12):
            n, m = noll_to_nm(j)
            result_noll = zernike_polynomial_z_noll(j, rho, theta)
            result_nm = zernike_polynomial_z(n, m, rho, theta)
            torch.testing.assert_close(
                result_noll, result_nm, atol=1e-10, rtol=1e-10
            )

    def test_osa_matches_nm(self):
        """OSA-indexed evaluation should match (n, m) evaluation."""
        rho = torch.tensor([0.5, 0.8], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

        for j in range(6):
            n, m = osa_to_nm(j)
            result_osa = zernike_polynomial_z_osa(j, rho, theta)
            result_nm = zernike_polynomial_z(n, m, rho, theta)
            torch.testing.assert_close(
                result_osa, result_nm, atol=1e-10, rtol=1e-10
            )


class TestZernikeAll:
    """Tests for batch evaluation of all Zernike polynomials."""

    def test_shape(self):
        """Test output shape for n_max=2."""
        rho = torch.tensor([0.5, 0.8], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)
        result = zernike_polynomial_z_all(2, rho, theta)
        # n_max=2: 6 polynomials (Z_0^0, Z_1^{-1}, Z_1^1, Z_2^{-2}, Z_2^0, Z_2^2)
        assert result.shape == (2, 6)

    def test_values_match_individual(self):
        """Values should match individual evaluations."""
        rho = torch.tensor([0.5, 0.8], dtype=torch.float64)
        theta = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)
        result = zernike_polynomial_z_all(2, rho, theta)

        # Check each polynomial
        idx = 0
        for n in range(3):
            for m in range(-n, n + 1, 2):
                expected = zernike_polynomial_z(n, m, rho, theta)
                torch.testing.assert_close(
                    result[..., idx], expected, atol=1e-10, rtol=1e-10
                )
                idx += 1


class TestZernikeFit:
    """Tests for Zernike polynomial fitting."""

    def test_fit_single_term(self):
        """Fitting data from a single Zernike should recover coefficient."""
        torch.manual_seed(42)
        n_points = 100
        rho = torch.rand(n_points, dtype=torch.float64)
        theta = torch.rand(n_points, dtype=torch.float64) * 2 * math.pi

        # Generate data from Z_2^0 (defocus) with coefficient 2.5
        true_coeff = 2.5
        Z_20 = zernike_polynomial_z(2, 0, rho, theta)
        data = true_coeff * Z_20

        # Fit with n_max=2
        coeffs = zernike_polynomial_z_fit(data, rho, theta, n_max=2)

        # Coefficient for Z_2^0 is at index 4 in OSA ordering
        torch.testing.assert_close(
            coeffs[4],
            torch.tensor(true_coeff, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_fit_roundtrip(self):
        """Fitting and reconstructing should match original data."""
        torch.manual_seed(123)
        n_points = 200
        rho = torch.rand(n_points, dtype=torch.float64)
        theta = torch.rand(n_points, dtype=torch.float64) * 2 * math.pi

        # Generate random Zernike coefficients
        n_max = 3
        true_coeffs = torch.randn(
            (n_max + 1) * (n_max + 2) // 2, dtype=torch.float64
        )

        # Generate wavefront data
        Z_all = zernike_polynomial_z_all(n_max, rho, theta)
        data = Z_all @ true_coeffs

        # Fit
        fitted_coeffs = zernike_polynomial_z_fit(data, rho, theta, n_max=n_max)

        # Reconstruct
        reconstructed = Z_all @ fitted_coeffs

        torch.testing.assert_close(reconstructed, data, atol=1e-8, rtol=1e-8)


class TestZernikeOrthogonality:
    """Tests for orthogonality of Zernike polynomials."""

    def test_orthogonality_normalized(self):
        """Normalized Zernike polynomials should be orthonormal over unit disk."""
        # Use numerical integration
        n_rho = 50
        n_theta = 50

        rho = torch.linspace(0, 1, n_rho, dtype=torch.float64)
        theta = torch.linspace(0, 2 * math.pi, n_theta, dtype=torch.float64)
        rho_grid, theta_grid = torch.meshgrid(rho, theta, indexing="ij")
        rho_flat = rho_grid.flatten()
        theta_flat = theta_grid.flatten()

        # Integration weights (rho * d_rho * d_theta)
        d_rho = 1.0 / (n_rho - 1)
        d_theta = 2 * math.pi / (n_theta - 1)
        weights = rho_flat * d_rho * d_theta / math.pi  # Normalize by pi

        # Test orthogonality between Z_0^0 and Z_2^0
        Z_00 = zernike_polynomial_z(
            0, 0, rho_flat, theta_flat, normalized=True
        )
        Z_20 = zernike_polynomial_z(
            2, 0, rho_flat, theta_flat, normalized=True
        )

        inner_product = (Z_00 * Z_20 * weights).sum()
        torch.testing.assert_close(
            inner_product,
            torch.tensor(0.0, dtype=torch.float64),
            atol=0.05,
            rtol=0.05,
        )

        # Test normalization (Z_0^0 should have norm 1)
        norm_00 = (Z_00**2 * weights).sum()
        torch.testing.assert_close(
            norm_00,
            torch.tensor(1.0, dtype=torch.float64),
            atol=0.05,
            rtol=0.05,
        )


class TestZernikeNames:
    """Tests for Zernike polynomial names."""

    def test_common_names_exist(self):
        """Common aberration names should be in ZERNIKE_NAMES."""
        assert 1 in ZERNIKE_NAMES  # Piston
        assert 4 in ZERNIKE_NAMES  # Defocus
        assert 11 in ZERNIKE_NAMES  # Spherical

    def test_piston_name(self):
        """Noll index 1 should be 'Piston'."""
        assert ZERNIKE_NAMES[1] == "Piston"

    def test_defocus_name(self):
        """Noll index 4 should be 'Defocus'."""
        assert ZERNIKE_NAMES[4] == "Defocus"
