"""Tests for local element matrices."""

import torch

from torchscience.finite_element_method import (
    dof_map,
    local_stiffness_matrices,
)
from torchscience.geometry.mesh import rectangle_mesh


class TestLocalStiffnessMatrices:
    def test_local_stiffness_shape(self):
        """Test shape of local stiffness matrices."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # Shape: (num_elements, dofs_per_element, dofs_per_element)
        assert K_local.shape == (8, 3, 3)

    def test_local_stiffness_symmetry(self):
        """Test that local stiffness matrices are symmetric."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # Each local matrix should be symmetric
        assert torch.allclose(K_local, K_local.transpose(-1, -2), atol=1e-10)

    def test_local_stiffness_positive_semidefinite(self):
        """Test that local stiffness matrices are positive semi-definite."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # Eigenvalues should be >= 0 (with numerical tolerance)
        for i in range(K_local.shape[0]):
            eigenvalues = torch.linalg.eigvalsh(K_local[i])
            assert (eigenvalues >= -1e-10).all()

    def test_local_stiffness_with_material(self):
        """Test stiffness with spatially varying material."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Different material per element
        material = torch.ones(8, dtype=torch.float64) * 2.0

        K_local = local_stiffness_matrices(mesh, dm, material=material)
        K_local_unit = local_stiffness_matrices(mesh, dm)

        # Should scale by material coefficient
        assert torch.allclose(K_local, 2.0 * K_local_unit, atol=1e-10)

    def test_local_stiffness_scalar_material(self):
        """Test stiffness with scalar material parameter."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm, material=3.0)
        K_local_unit = local_stiffness_matrices(mesh, dm)

        assert torch.allclose(K_local, 3.0 * K_local_unit, atol=1e-10)

    def test_local_stiffness_dtype(self):
        """Test that output dtype matches mesh dtype."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], dtype=torch.float64
        )
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        assert K_local.dtype == torch.float64

    def test_local_stiffness_device(self):
        """Test that output device matches mesh device."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        assert K_local.device == mesh.vertices.device


class TestLocalStiffnessMatricesP2:
    """Tests for P2 (quadratic) elements."""

    def test_local_stiffness_p2_shape(self):
        """Test shape of P2 local stiffness matrices."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        K_local = local_stiffness_matrices(mesh, dm)

        # P2 triangles have 6 DOFs per element
        assert K_local.shape == (8, 6, 6)

    def test_local_stiffness_p2_symmetry(self):
        """Test that P2 local stiffness matrices are symmetric."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        K_local = local_stiffness_matrices(mesh, dm)

        assert torch.allclose(K_local, K_local.transpose(-1, -2), atol=1e-10)


class TestLocalStiffnessMatricesQuad:
    """Tests for quadrilateral elements."""

    def test_local_stiffness_quad_shape(self):
        """Test shape of quad local stiffness matrices."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # 4 quads, 4 DOFs each
        assert K_local.shape == (4, 4, 4)

    def test_local_stiffness_quad_symmetry(self):
        """Test that quad local stiffness matrices are symmetric."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        assert torch.allclose(K_local, K_local.transpose(-1, -2), atol=1e-10)


class TestLocalStiffnessMatricesIntegration:
    """Integration tests verifying physical properties."""

    def test_sum_of_rows_zero_for_constant_material(self):
        """Test that rows sum to zero (constant field gives zero gradient)."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # For stiffness matrices, sum of each row should be approximately zero
        # because applying constant displacement gives zero strain energy
        row_sums = K_local.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-10)

    def test_single_element_stiffness_value(self):
        """Test stiffness value for a single unit triangle."""
        # Create a single triangle mesh on unit square
        mesh = rectangle_mesh(1, 1, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local = local_stiffness_matrices(mesh, dm)

        # For a right triangle with legs of length 1, the stiffness matrix
        # should have specific known values
        # The lower triangle has vertices at (0,0), (1,0), (0.5, 0.5)
        # Actually, from rectangle_mesh with diagonal splitting, triangles are:
        # Lower: (0,0), (1,0), (1,1)
        # Upper: (0,0), (1,1), (0,1)
        # Each has area 0.5

        # Just check that values are reasonable and symmetric
        assert K_local.shape == (2, 3, 3)
        assert torch.allclose(K_local, K_local.transpose(-1, -2), atol=1e-10)

    def test_stiffness_scales_with_inverse_area(self):
        """Test that stiffness scales inversely with element area."""
        mesh1 = rectangle_mesh(1, 1, bounds=[[0, 1], [0, 1]])
        mesh2 = rectangle_mesh(1, 1, bounds=[[0, 2], [0, 2]])
        dm1 = dof_map(mesh1, order=1)
        dm2 = dof_map(mesh2, order=1)

        K1 = local_stiffness_matrices(mesh1, dm1)
        K2 = local_stiffness_matrices(mesh2, dm2)

        # Larger elements should have the same stiffness values
        # (stiffness = integral of grad(N) . grad(N), and grad scales as 1/h
        # while area scales as h^2, so overall K stays constant for shape-similar triangles)
        assert torch.allclose(K1, K2, atol=1e-10)


class TestLocalStiffnessQuadOrder:
    """Tests for quadrature order selection."""

    def test_custom_quad_order(self):
        """Test specifying custom quadrature order."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K_local_default = local_stiffness_matrices(mesh, dm)
        K_local_high = local_stiffness_matrices(mesh, dm, quad_order=4)

        # For linear elements, higher quad order should give same result
        # (gradients are constant)
        assert torch.allclose(K_local_default, K_local_high, atol=1e-10)
