import math

import mpmath
import pytest
import torch
import torch.testing

import torchscience.special_functions


def mpmath_rf(x, y, z):
    """Reference implementation using mpmath."""
    return float(mpmath.elliprf(x, y, z))


class TestCarlsonEllipticIntegralRF:
    """Tests for Carlson's R_F integral."""

    def test_forward_special_value_pi_over_2(self):
        """R_F(0, 1, 1) = pi/2."""
        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, y, z
        )
        expected = torch.tensor([math.pi / 2], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_homogeneity(self):
        """R_F(ax, ay, az) = a^(-1/2) R_F(x, y, z)."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([3.0], dtype=torch.float64)
        a = 4.0

        rf_xyz = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, y, z
        )
        rf_scaled = (
            torchscience.special_functions.carlson_elliptic_integral_r_f(
                a * x, a * y, a * z
            )
        )

        torch.testing.assert_close(
            rf_scaled, rf_xyz / math.sqrt(a), rtol=1e-10, atol=1e-10
        )

    def test_forward_symmetry(self):
        """R_F is symmetric in all arguments."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([3.0], dtype=torch.float64)

        rf_xyz = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, y, z
        )
        rf_xzy = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, z, y
        )
        rf_yxz = torchscience.special_functions.carlson_elliptic_integral_r_f(
            y, x, z
        )
        rf_yzx = torchscience.special_functions.carlson_elliptic_integral_r_f(
            y, z, x
        )
        rf_zxy = torchscience.special_functions.carlson_elliptic_integral_r_f(
            z, x, y
        )
        rf_zyx = torchscience.special_functions.carlson_elliptic_integral_r_f(
            z, y, x
        )

        torch.testing.assert_close(rf_xyz, rf_xzy, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rf_xyz, rf_yxz, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rf_xyz, rf_yzx, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rf_xyz, rf_zxy, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rf_xyz, rf_zyx, rtol=1e-10, atol=1e-10)

    def test_forward_against_mpmath(self):
        """Compare against mpmath reference."""
        test_cases = [
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.5, 1.5, 2.5),
            (0.1, 0.2, 0.3),
        ]
        for x_val, y_val, z_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = (
                torchscience.special_functions.carlson_elliptic_integral_r_f(
                    x, y, z
                )
            )
            expected = mpmath_rf(x_val, y_val, z_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.carlson_elliptic_integral_r_f,
            (x, y, z),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.skip(
        reason="R_D backward_backward uses numerical finite differences; analytical implementation needed"
    )
    def test_gradient_gradient(self):
        """Second-order gradient via gradgradcheck."""
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.carlson_elliptic_integral_r_f,
            (x, y, z),
            eps=1e-4,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_degenerate_to_rc(self):
        """R_F(x, y, y) = R_C(x, y)."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)

        rf = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, y, y
        )
        rc = torchscience.special_functions.carlson_elliptic_integral_r_c(x, y)

        torch.testing.assert_close(rf, rc, rtol=1e-10, atol=1e-10)

    def test_equal_args(self):
        """R_F(x, x, x) = 1/sqrt(x)."""
        x = torch.tensor([4.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_f(
            x, x, x
        )
        expected = torch.tensor([0.5], dtype=torch.float64)  # 1/sqrt(4) = 0.5
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
