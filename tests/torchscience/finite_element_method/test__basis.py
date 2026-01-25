"""Tests for FEM basis functions."""

import pytest
import torch

from torchscience.finite_element_method import (
    lagrange_basis,
    lagrange_basis_gradient,
)


class TestLagrangeBasis:
    def test_line_p1_partition_of_unity(self):
        """Test that P1 line basis sums to 1."""
        points = torch.rand(10, 1, dtype=torch.float64)
        basis = lagrange_basis("line", order=1, points=points)

        assert basis.shape == (10, 2)  # 2 basis functions
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_line_p1_interpolation(self):
        """Test P1 basis interpolates at nodes."""
        # Nodes for P1 line: 0, 1
        nodes = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        basis = lagrange_basis("line", order=1, points=nodes)

        # Should be identity matrix
        expected = torch.eye(2, dtype=torch.float64)
        assert torch.allclose(basis, expected)

    def test_triangle_p1(self):
        """Test P1 triangle basis."""
        # Reference triangle nodes: (0,0), (1,0), (0,1)
        nodes = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("triangle", order=1, points=nodes)

        assert basis.shape == (3, 3)
        assert torch.allclose(basis, torch.eye(3, dtype=torch.float64))

    def test_triangle_p2(self):
        """Test P2 triangle basis (6 nodes)."""
        points = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        basis = lagrange_basis("triangle", order=2, points=points)

        assert basis.shape == (1, 6)  # 6 P2 basis functions
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(1, dtype=torch.float64)
        )

    def test_gradient_line_p1(self):
        """Test P1 line basis gradient."""
        points = torch.tensor([[0.5]], dtype=torch.float64)
        grad = lagrange_basis_gradient("line", order=1, points=points)

        # Gradients of N1 = 1-x and N2 = x are -1 and 1
        assert grad.shape == (1, 2, 1)
        assert torch.allclose(
            grad[0, 0, 0], torch.tensor(-1.0, dtype=torch.float64)
        )
        assert torch.allclose(
            grad[0, 1, 0], torch.tensor(1.0, dtype=torch.float64)
        )

    def test_gradient_triangle_p1(self):
        """Test P1 triangle basis gradient."""
        points = torch.tensor([[0.3, 0.3]], dtype=torch.float64)
        grad = lagrange_basis_gradient("triangle", order=1, points=points)

        # P1 triangle: N1 = 1-x-y, N2 = x, N3 = y
        # Gradients are constant: [-1,-1], [1,0], [0,1]
        assert grad.shape == (1, 3, 2)
        assert torch.allclose(
            grad[0, 0], torch.tensor([-1.0, -1.0], dtype=torch.float64)
        )
        assert torch.allclose(
            grad[0, 1], torch.tensor([1.0, 0.0], dtype=torch.float64)
        )
        assert torch.allclose(
            grad[0, 2], torch.tensor([0.0, 1.0], dtype=torch.float64)
        )


class TestLagrangeBasisLine:
    """Additional tests for line basis functions."""

    def test_line_p2_partition_of_unity(self):
        """Test that P2 line basis sums to 1."""
        points = torch.rand(10, 1, dtype=torch.float64)
        basis = lagrange_basis("line", order=2, points=points)

        assert basis.shape == (10, 3)  # 3 basis functions
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_line_p2_interpolation(self):
        """Test P2 line basis interpolates at nodes."""
        # Nodes for P2 line: 0, 0.5, 1
        nodes = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        basis = lagrange_basis("line", order=2, points=nodes)

        expected = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)

    def test_line_p1_at_midpoint(self):
        """Test P1 line basis at midpoint gives 0.5, 0.5."""
        points = torch.tensor([[0.5]], dtype=torch.float64)
        basis = lagrange_basis("line", order=1, points=points)

        expected = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        assert torch.allclose(basis, expected)


class TestLagrangeBasisTriangle:
    """Additional tests for triangle basis functions."""

    def test_triangle_p1_partition_of_unity(self):
        """Test that P1 triangle basis sums to 1."""
        points = (
            torch.rand(10, 2, dtype=torch.float64) * 0.5
        )  # Keep in triangle
        basis = lagrange_basis("triangle", order=1, points=points)

        assert basis.shape == (10, 3)
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_triangle_p2_interpolation(self):
        """Test P2 triangle basis interpolates at nodes."""
        # P2 nodes: vertices + edge midpoints
        nodes = torch.tensor(
            [
                [0.0, 0.0],  # vertex 1
                [1.0, 0.0],  # vertex 2
                [0.0, 1.0],  # vertex 3
                [0.5, 0.0],  # midpoint edge 1-2
                [0.5, 0.5],  # midpoint edge 2-3
                [0.0, 0.5],  # midpoint edge 3-1
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("triangle", order=2, points=nodes)

        expected = torch.eye(6, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)

    def test_triangle_p1_at_centroid(self):
        """Test P1 triangle basis at centroid gives 1/3, 1/3, 1/3."""
        points = torch.tensor([[1 / 3, 1 / 3]], dtype=torch.float64)
        basis = lagrange_basis("triangle", order=1, points=points)

        expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64)
        assert torch.allclose(basis, expected)


class TestLagrangeBasisQuad:
    """Tests for quad basis functions."""

    def test_quad_p1_partition_of_unity(self):
        """Test that Q1 quad basis sums to 1."""
        # Random points in [-1, 1]^2
        points = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        basis = lagrange_basis("quad", order=1, points=points)

        assert basis.shape == (10, 4)  # 4 basis functions for Q1
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_quad_p1_interpolation(self):
        """Test Q1 quad basis interpolates at nodes."""
        # Q1 quad nodes: corners of [-1, 1]^2 in tensor product ordering
        # idx = j * n + i, where i varies over x, j over y
        # idx 0: (i=0, j=0) = (-1, -1)
        # idx 1: (i=1, j=0) = (1, -1)
        # idx 2: (i=0, j=1) = (-1, 1)
        # idx 3: (i=1, j=1) = (1, 1)
        nodes = torch.tensor(
            [
                [-1.0, -1.0],  # idx 0
                [1.0, -1.0],  # idx 1
                [-1.0, 1.0],  # idx 2
                [1.0, 1.0],  # idx 3
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("quad", order=1, points=nodes)

        expected = torch.eye(4, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)

    def test_quad_p2_partition_of_unity(self):
        """Test that Q2 quad basis sums to 1."""
        points = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        basis = lagrange_basis("quad", order=2, points=points)

        assert basis.shape == (10, 9)  # 9 basis functions for Q2
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )


class TestLagrangeBasisTetrahedron:
    """Tests for tetrahedron basis functions."""

    def test_tetrahedron_p1_partition_of_unity(self):
        """Test that P1 tetrahedron basis sums to 1."""
        # Random points inside tetrahedron
        points = torch.rand(10, 3, dtype=torch.float64) * 0.3  # Keep in tet
        basis = lagrange_basis("tetrahedron", order=1, points=points)

        assert basis.shape == (10, 4)  # 4 basis functions for P1
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_tetrahedron_p1_interpolation(self):
        """Test P1 tetrahedron basis interpolates at nodes."""
        # Reference tetrahedron nodes
        nodes = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("tetrahedron", order=1, points=nodes)

        expected = torch.eye(4, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)

    def test_tetrahedron_p2_partition_of_unity(self):
        """Test that P2 tetrahedron basis sums to 1."""
        points = torch.rand(10, 3, dtype=torch.float64) * 0.3
        basis = lagrange_basis("tetrahedron", order=2, points=points)

        assert basis.shape == (10, 10)  # 10 basis functions for P2
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_tetrahedron_p2_interpolation(self):
        """Test P2 tetrahedron basis interpolates at nodes."""
        # P2 tetrahedron has 10 nodes: 4 vertices + 6 edge midpoints
        # Reference tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        nodes = torch.tensor(
            [
                # 4 vertices
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                # 6 edge midpoints
                [0.5, 0.0, 0.0],  # edge 0-1
                [0.5, 0.5, 0.0],  # edge 1-2
                [0.0, 0.5, 0.0],  # edge 0-2
                [0.0, 0.0, 0.5],  # edge 0-3
                [0.5, 0.0, 0.5],  # edge 1-3
                [0.0, 0.5, 0.5],  # edge 2-3
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("tetrahedron", order=2, points=nodes)

        assert basis.shape == (10, 10)
        expected = torch.eye(10, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)


class TestLagrangeBasisHexahedron:
    """Tests for hexahedron basis functions."""

    def test_hexahedron_p1_partition_of_unity(self):
        """Test that Q1 hexahedron basis sums to 1."""
        points = torch.rand(10, 3, dtype=torch.float64) * 2 - 1
        basis = lagrange_basis("hexahedron", order=1, points=points)

        assert basis.shape == (10, 8)  # 8 basis functions for Q1
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )

    def test_hexahedron_p1_interpolation(self):
        """Test Q1 hexahedron basis interpolates at nodes."""
        # Q1 hexahedron nodes: corners of [-1, 1]^3 in tensor product ordering
        # idx = k * n^2 + j * n + i, where i varies over x, j over y, k over z
        # idx 0: (i=0, j=0, k=0) = (-1, -1, -1)
        # idx 1: (i=1, j=0, k=0) = (1, -1, -1)
        # idx 2: (i=0, j=1, k=0) = (-1, 1, -1)
        # idx 3: (i=1, j=1, k=0) = (1, 1, -1)
        # idx 4: (i=0, j=0, k=1) = (-1, -1, 1)
        # idx 5: (i=1, j=0, k=1) = (1, -1, 1)
        # idx 6: (i=0, j=1, k=1) = (-1, 1, 1)
        # idx 7: (i=1, j=1, k=1) = (1, 1, 1)
        nodes = torch.tensor(
            [
                [-1.0, -1.0, -1.0],  # idx 0
                [1.0, -1.0, -1.0],  # idx 1
                [-1.0, 1.0, -1.0],  # idx 2
                [1.0, 1.0, -1.0],  # idx 3
                [-1.0, -1.0, 1.0],  # idx 4
                [1.0, -1.0, 1.0],  # idx 5
                [-1.0, 1.0, 1.0],  # idx 6
                [1.0, 1.0, 1.0],  # idx 7
            ],
            dtype=torch.float64,
        )
        basis = lagrange_basis("hexahedron", order=1, points=nodes)

        expected = torch.eye(8, dtype=torch.float64)
        assert torch.allclose(basis, expected, atol=1e-10)

    def test_hexahedron_p2_partition_of_unity(self):
        """Test that Q2 hexahedron basis sums to 1."""
        points = torch.rand(10, 3, dtype=torch.float64) * 2 - 1
        basis = lagrange_basis("hexahedron", order=2, points=points)

        assert basis.shape == (10, 27)  # 27 basis functions for Q2
        assert torch.allclose(
            basis.sum(dim=-1), torch.ones(10, dtype=torch.float64)
        )


class TestLagrangeBasisGradient:
    """Tests for basis function gradients."""

    def test_gradient_line_p2(self):
        """Test P2 line basis gradient."""
        points = torch.tensor([[0.25]], dtype=torch.float64)
        grad = lagrange_basis_gradient("line", order=2, points=points)

        assert grad.shape == (1, 3, 1)
        # P2 line: N1 = (1-x)(1-2x), N2 = 4x(1-x), N3 = x(2x-1)
        # Derivatives: dN1/dx = -3 + 4x, dN2/dx = 4 - 8x, dN3/dx = -1 + 4x
        # At x = 0.25: dN1 = -2, dN2 = 2, dN3 = 0
        expected = torch.tensor([[[-2.0], [2.0], [0.0]]], dtype=torch.float64)
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_triangle_p2(self):
        """Test P2 triangle basis gradient shape."""
        points = torch.rand(5, 2, dtype=torch.float64) * 0.5
        grad = lagrange_basis_gradient("triangle", order=2, points=points)

        assert grad.shape == (5, 6, 2)

    def test_gradient_quad_p1(self):
        """Test Q1 quad basis gradient."""
        points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = lagrange_basis_gradient("quad", order=1, points=points)

        assert grad.shape == (1, 4, 2)
        # Q1 quad basis functions in tensor product ordering (j * n + i)
        # idx 0 (i=0, j=0): N = (1-x)(1-y)/4 at nodes (-1,-1), grad = [-0.25, -0.25]
        # idx 1 (i=1, j=0): N = (1+x)(1-y)/4 at nodes (1,-1), grad = [0.25, -0.25]
        # idx 2 (i=0, j=1): N = (1-x)(1+y)/4 at nodes (-1,1), grad = [-0.25, 0.25]
        # idx 3 (i=1, j=1): N = (1+x)(1+y)/4 at nodes (1,1), grad = [0.25, 0.25]
        expected = torch.tensor(
            [
                [
                    [-0.25, -0.25],  # idx 0: (1-x)(1-y)/4
                    [0.25, -0.25],  # idx 1: (1+x)(1-y)/4
                    [-0.25, 0.25],  # idx 2: (1-x)(1+y)/4
                    [0.25, 0.25],  # idx 3: (1+x)(1+y)/4
                ]
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_tetrahedron_p1(self):
        """Test P1 tetrahedron basis gradient."""
        points = torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float64)
        grad = lagrange_basis_gradient("tetrahedron", order=1, points=points)

        # P1 tet: N1 = 1-x-y-z, N2 = x, N3 = y, N4 = z
        # Gradients are constant:
        # dN1 = [-1, -1, -1], dN2 = [1, 0, 0], dN3 = [0, 1, 0], dN4 = [0, 0, 1]
        assert grad.shape == (1, 4, 3)
        expected = torch.tensor(
            [
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_gradient_hexahedron_p1(self):
        """Test Q1 hexahedron basis gradient shape."""
        points = torch.rand(5, 3, dtype=torch.float64) * 2 - 1
        grad = lagrange_basis_gradient("hexahedron", order=1, points=points)

        assert grad.shape == (5, 8, 3)


class TestLagrangeBasisOptions:
    """Test basis function options and error handling."""

    def test_invalid_element_type_raises(self):
        """Test that invalid element type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown element type"):
            lagrange_basis("invalid", order=1, points=torch.zeros(1, 2))

    def test_invalid_order_raises(self):
        """Test that unsupported order raises ValueError."""
        with pytest.raises(ValueError, match="[Oo]rder"):
            lagrange_basis("line", order=10, points=torch.zeros(1, 1))

    def test_case_insensitive_element_type(self):
        """Test that element_type is case insensitive."""
        points = torch.tensor([[0.5]], dtype=torch.float64)
        basis1 = lagrange_basis("LINE", order=1, points=points)
        basis2 = lagrange_basis("line", order=1, points=points)
        assert torch.allclose(basis1, basis2)

    def test_gradient_invalid_element_type_raises(self):
        """Test that gradient with invalid element type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown element type"):
            lagrange_basis_gradient(
                "invalid", order=1, points=torch.zeros(1, 2)
            )

    def test_points_wrong_dimension_handled(self):
        """Test that points with wrong dimension are handled."""
        # This test documents expected behavior - should raise if dimension mismatch
        with pytest.raises((ValueError, RuntimeError)):
            lagrange_basis("triangle", order=1, points=torch.zeros(1, 3))
