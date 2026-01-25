"""Tests for FEM quadrature rules on reference elements."""

import pytest
import torch

from torchscience.partial_differential_equation.finite_element_method import (
    quadrature_points,
)


class TestQuadrature:
    def test_triangle_quadrature_order_1(self):
        """Test triangle quadrature, order 1 (centroid rule)."""
        points, weights = quadrature_points("triangle", order=1)

        assert points.shape == (1, 2)
        assert weights.shape == (1,)
        # Centroid is at (1/3, 1/3)
        assert torch.allclose(
            points[0], torch.tensor([1 / 3, 1 / 3], dtype=torch.float64)
        )
        # Weight is area of reference triangle = 0.5
        assert torch.allclose(
            weights.sum(), torch.tensor(0.5, dtype=torch.float64)
        )

    def test_triangle_quadrature_order_2(self):
        """Test triangle quadrature, order 2."""
        points, weights = quadrature_points("triangle", order=2)

        assert points.shape[0] == 3  # 3 points for order 2
        assert points.shape[1] == 2
        assert torch.allclose(
            weights.sum(), torch.tensor(0.5, dtype=torch.float64)
        )

    def test_line_quadrature(self):
        """Test line element quadrature (Gauss-Legendre on [0,1])."""
        points, weights = quadrature_points("line", order=3)

        assert points.shape[1] == 1
        # Should integrate x^2 exactly (order 3 means exact for poly up to degree 3)
        # integral of x^2 from 0 to 1 = 1/3
        integral = (points[:, 0] ** 2 * weights).sum()
        assert torch.allclose(
            integral, torch.tensor(1 / 3, dtype=torch.float64), atol=1e-10
        )

    def test_tetrahedron_quadrature(self):
        """Test tetrahedron quadrature."""
        points, weights = quadrature_points("tetrahedron", order=1)

        assert points.shape[1] == 3
        # Volume of reference tet is 1/6
        assert torch.allclose(
            weights.sum(), torch.tensor(1 / 6, dtype=torch.float64)
        )


class TestQuadratureQuad:
    """Test quadrature for quadrilateral elements."""

    def test_quad_quadrature_order_1(self):
        """Test quad quadrature, order 1."""
        points, weights = quadrature_points("quad", order=1)

        assert points.shape[1] == 2
        # Area of reference quad [-1,1]^2 is 4
        assert torch.allclose(
            weights.sum(), torch.tensor(4.0, dtype=torch.float64)
        )

    def test_quad_quadrature_order_2(self):
        """Test quad quadrature, order 2 (2x2 = 4 points)."""
        points, weights = quadrature_points("quad", order=2)

        assert points.shape[0] == 4  # 2x2 tensor product
        assert points.shape[1] == 2
        assert torch.allclose(
            weights.sum(), torch.tensor(4.0, dtype=torch.float64)
        )

    def test_quad_integrates_polynomial(self):
        """Test that quad quadrature integrates polynomials exactly."""
        points, weights = quadrature_points("quad", order=3)

        # Integrate x*y over [-1,1]^2, should be 0 (odd function)
        integral = (points[:, 0] * points[:, 1] * weights).sum()
        assert torch.allclose(
            integral, torch.tensor(0.0, dtype=torch.float64), atol=1e-14
        )

        # Integrate x^2 over [-1,1]^2:
        # integral_x from -1 to 1 of x^2 = [x^3/3]_{-1}^{1} = 1/3 - (-1/3) = 2/3
        # integral_y from -1 to 1 of 1 = 2
        # Total = 2/3 * 2 = 4/3
        integral = (points[:, 0] ** 2 * weights).sum()
        assert torch.allclose(
            integral, torch.tensor(4 / 3, dtype=torch.float64), atol=1e-10
        )


class TestQuadratureHexahedron:
    """Test quadrature for hexahedron elements."""

    def test_hexahedron_quadrature_order_1(self):
        """Test hexahedron quadrature, order 1."""
        points, weights = quadrature_points("hexahedron", order=1)

        assert points.shape[1] == 3
        # Volume of reference hexahedron [-1,1]^3 is 8
        assert torch.allclose(
            weights.sum(), torch.tensor(8.0, dtype=torch.float64)
        )

    def test_hexahedron_quadrature_order_2(self):
        """Test hexahedron quadrature, order 2 (2x2x2 = 8 points)."""
        points, weights = quadrature_points("hexahedron", order=2)

        assert points.shape[0] == 8  # 2x2x2 tensor product
        assert points.shape[1] == 3
        assert torch.allclose(
            weights.sum(), torch.tensor(8.0, dtype=torch.float64)
        )


class TestQuadratureTriangle:
    """Additional tests for triangle quadrature."""

    def test_triangle_quadrature_higher_orders(self):
        """Test triangle quadrature for orders 3, 4, 5."""
        for order in [3, 4, 5]:
            points, weights = quadrature_points("triangle", order=order)
            assert points.shape[1] == 2
            assert torch.allclose(
                weights.sum(),
                torch.tensor(0.5, dtype=torch.float64),
                atol=1e-10,
            )

    def test_triangle_points_in_reference_element(self):
        """Test that all quadrature points are inside the reference triangle."""
        for order in [1, 2, 3, 4, 5]:
            points, _ = quadrature_points("triangle", order=order)
            x, y = points[:, 0], points[:, 1]
            # All points should satisfy: x >= 0, y >= 0, x + y <= 1
            assert (x >= -1e-10).all()
            assert (y >= -1e-10).all()
            assert (x + y <= 1 + 1e-10).all()

    def test_triangle_integrates_constant(self):
        """Test that triangle quadrature integrates 1 to get area."""
        for order in [1, 2, 3, 4, 5]:
            _, weights = quadrature_points("triangle", order=order)
            integral = weights.sum()
            assert torch.allclose(
                integral, torch.tensor(0.5, dtype=torch.float64)
            )


class TestQuadratureTetrahedron:
    """Additional tests for tetrahedron quadrature."""

    def test_tetrahedron_quadrature_order_2(self):
        """Test tetrahedron quadrature, order 2."""
        points, weights = quadrature_points("tetrahedron", order=2)

        assert points.shape[0] == 4
        assert points.shape[1] == 3
        assert torch.allclose(
            weights.sum(), torch.tensor(1 / 6, dtype=torch.float64)
        )

    def test_tetrahedron_points_in_reference_element(self):
        """Test that all quadrature points are inside the reference tetrahedron."""
        for order in [1, 2, 3]:
            points, _ = quadrature_points("tetrahedron", order=order)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            # All points should satisfy: x >= 0, y >= 0, z >= 0, x + y + z <= 1
            assert (x >= -1e-10).all()
            assert (y >= -1e-10).all()
            assert (z >= -1e-10).all()
            assert (x + y + z <= 1 + 1e-10).all()


class TestQuadratureLine:
    """Additional tests for line quadrature."""

    def test_line_quadrature_various_orders(self):
        """Test line quadrature for various orders."""
        for order in [1, 2, 3, 4, 5]:
            points, weights = quadrature_points("line", order=order)
            assert points.shape[1] == 1
            # Length of [0, 1] is 1
            assert torch.allclose(
                weights.sum(), torch.tensor(1.0, dtype=torch.float64)
            )

    def test_line_points_in_reference_element(self):
        """Test that all quadrature points are in [0, 1]."""
        for order in [1, 2, 3, 4, 5]:
            points, _ = quadrature_points("line", order=order)
            assert (points >= -1e-10).all()
            assert (points <= 1 + 1e-10).all()

    def test_line_integrates_higher_polynomial(self):
        """Test that line quadrature integrates x^4 correctly with enough points."""
        # Integral of x^4 from 0 to 1 = 1/5
        points, weights = quadrature_points("line", order=5)
        integral = (points[:, 0] ** 4 * weights).sum()
        assert torch.allclose(
            integral, torch.tensor(0.2, dtype=torch.float64), atol=1e-10
        )


class TestQuadratureOptions:
    """Test quadrature function options."""

    def test_dtype_option(self):
        """Test dtype option."""
        points, weights = quadrature_points(
            "triangle", order=1, dtype=torch.float32
        )
        assert points.dtype == torch.float32
        assert weights.dtype == torch.float32

    def test_default_dtype_is_float64(self):
        """Test that default dtype is float64."""
        points, weights = quadrature_points("triangle", order=1)
        assert points.dtype == torch.float64
        assert weights.dtype == torch.float64

    def test_device_option(self):
        """Test device option."""
        points, weights = quadrature_points(
            "triangle", order=1, device=torch.device("cpu")
        )
        assert points.device == torch.device("cpu")
        assert weights.device == torch.device("cpu")

    def test_case_insensitive_element_type(self):
        """Test that element_type is case insensitive."""
        points1, weights1 = quadrature_points("TRIANGLE", order=1)
        points2, weights2 = quadrature_points("triangle", order=1)
        assert torch.allclose(points1, points2)
        assert torch.allclose(weights1, weights2)

    def test_invalid_element_type_raises(self):
        """Test that invalid element type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown element type"):
            quadrature_points("invalid_element", order=1)

    def test_invalid_order_raises(self):
        """Test that unavailable order raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            quadrature_points("triangle", order=100)
