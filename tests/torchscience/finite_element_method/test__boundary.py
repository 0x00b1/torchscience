"""Tests for FEM boundary condition utilities."""

import pytest
import torch

from torchscience.finite_element_method import (
    apply_dirichlet_penalty,
    boundary_dofs,
    dof_map,
)
from torchscience.geometry.mesh import rectangle_mesh


class TestBoundaryDofs:
    def test_boundary_dofs_p1(self):
        """Test boundary DOF detection for P1 elements."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        b_dofs = boundary_dofs(mesh, dm)

        # For P1, boundary DOFs = boundary vertices = 8 (perimeter of a 2x2 element grid, 3x3 vertices)
        assert b_dofs.shape == (8,)
        assert b_dofs.dtype == torch.long

    def test_boundary_dofs_p2(self):
        """Test boundary DOF detection for P2 elements."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        b_dofs = boundary_dofs(mesh, dm)

        # P2: boundary vertices + boundary edge DOFs
        # 8 vertices + 8 edges on boundary * 1 DOF per edge = 16
        assert len(b_dofs) == 16

    def test_boundary_dofs_are_valid(self):
        """Test that boundary DOFs are valid indices."""
        mesh = rectangle_mesh(3, 3, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        b_dofs = boundary_dofs(mesh, dm)

        # All DOFs should be in valid range
        assert (b_dofs >= 0).all()
        assert (b_dofs < dm.num_global_dofs).all()

    def test_boundary_dofs_subset(self):
        """Test boundary DOF detection with marker."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Only DOFs where x=0 (left edge)
        def left_edge(coords):
            return torch.isclose(
                coords[:, 0], torch.tensor(0.0, dtype=coords.dtype)
            )

        b_dofs = boundary_dofs(mesh, dm, marker=left_edge)

        # Left edge has 3 vertices for 2x2 mesh
        assert len(b_dofs) == 3

    def test_boundary_dofs_p1_quad_mesh(self):
        """Test boundary DOF detection for P1 on quad mesh."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=1)

        b_dofs = boundary_dofs(mesh, dm)

        # For P1 quads, boundary DOFs = boundary vertices = 8
        assert b_dofs.shape == (8,)
        assert b_dofs.dtype == torch.long

    def test_boundary_dofs_returns_sorted(self):
        """Test that boundary DOFs are returned sorted."""
        mesh = rectangle_mesh(3, 3, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        b_dofs = boundary_dofs(mesh, dm)

        # Should be sorted
        assert torch.all(b_dofs[:-1] <= b_dofs[1:])

    def test_boundary_dofs_marker_right_edge(self):
        """Test boundary DOF detection with marker for right edge."""
        mesh = rectangle_mesh(3, 3, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Only DOFs where x=1 (right edge)
        def right_edge(coords):
            return torch.isclose(
                coords[:, 0], torch.tensor(1.0, dtype=coords.dtype)
            )

        b_dofs = boundary_dofs(mesh, dm, marker=right_edge)

        # Right edge has 4 vertices for 3x3 mesh (4 grid points at x=1)
        assert len(b_dofs) == 4

    def test_boundary_dofs_marker_corner(self):
        """Test boundary DOF detection with marker for single corner."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Only DOF at origin (0, 0)
        def origin(coords):
            return torch.isclose(
                coords[:, 0], torch.tensor(0.0, dtype=coords.dtype)
            ) & torch.isclose(
                coords[:, 1], torch.tensor(0.0, dtype=coords.dtype)
            )

        b_dofs = boundary_dofs(mesh, dm, marker=origin)

        # Should be exactly 1 DOF
        assert len(b_dofs) == 1


class TestApplyDirichletPenalty:
    def test_penalty_modifies_diagonal(self):
        """Test that penalty method adds large value to diagonal."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Create a simple stiffness matrix (identity for testing)
        K = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f = torch.zeros(9, dtype=torch.float64)

        # Apply Dirichlet BC: u = 1.0 at boundary
        bc_dofs = boundary_dofs(mesh, dm)
        bc_values = torch.ones(len(bc_dofs), dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_penalty(K, f, bc_dofs, bc_values)

        # Diagonal at BC DOFs should be much larger
        K_dense = K_mod.to_dense()
        for dof in bc_dofs:
            assert K_dense[dof, dof] > 1e8

    def test_penalty_modifies_rhs(self):
        """Test that penalty method modifies RHS correctly."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        K = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f = torch.zeros(9, dtype=torch.float64)

        bc_dofs = boundary_dofs(mesh, dm)
        bc_values = torch.full((len(bc_dofs),), 2.0, dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_penalty(K, f, bc_dofs, bc_values)

        # RHS at BC DOFs should be penalty * bc_value
        for i, dof in enumerate(bc_dofs):
            assert f_mod[dof] > 1e8  # penalty * 2.0

    def test_penalty_preserves_symmetry(self):
        """Test that penalty method preserves matrix symmetry."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Create symmetric matrix
        K_dense = torch.randn(9, 9, dtype=torch.float64)
        K_dense = (K_dense + K_dense.T) / 2
        K = K_dense.to_sparse_csr()
        f = torch.zeros(9, dtype=torch.float64)

        bc_dofs = torch.tensor([0, 1], dtype=torch.long)
        bc_values = torch.ones(2, dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_penalty(K, f, bc_dofs, bc_values)

        K_mod_dense = K_mod.to_dense()
        assert torch.allclose(K_mod_dense, K_mod_dense.T, atol=1e-10)

    def test_penalty_empty_dofs(self):
        """Test that empty dofs tensor works correctly."""
        K = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f = torch.zeros(9, dtype=torch.float64)

        bc_dofs = torch.tensor([], dtype=torch.long)
        bc_values = torch.tensor([], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_penalty(K, f, bc_dofs, bc_values)

        # Matrix and vector should be unchanged
        assert torch.allclose(K_mod.to_dense(), K.to_dense())
        assert torch.allclose(f_mod, f)

    def test_penalty_device_mismatch_matrix_vector(self):
        """Test that device mismatch between matrix and vector raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f_cuda = torch.zeros(9, dtype=torch.float64, device="cuda")
        bc_dofs = torch.tensor([0, 1], dtype=torch.long)
        bc_values = torch.ones(2, dtype=torch.float64)

        with pytest.raises(
            ValueError, match="matrix and vector must be on the same device"
        ):
            apply_dirichlet_penalty(K_cpu, f_cuda, bc_dofs, bc_values)

    def test_penalty_device_mismatch_matrix_dofs(self):
        """Test that device mismatch between matrix and dofs raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f_cpu = torch.zeros(9, dtype=torch.float64)
        bc_dofs = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        bc_values = torch.ones(2, dtype=torch.float64)

        with pytest.raises(
            ValueError, match="matrix and dofs must be on the same device"
        ):
            apply_dirichlet_penalty(K_cpu, f_cpu, bc_dofs, bc_values)

    def test_penalty_device_mismatch_matrix_values(self):
        """Test that device mismatch between matrix and values raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(9, dtype=torch.float64).to_sparse_csr()
        f_cpu = torch.zeros(9, dtype=torch.float64)
        bc_dofs = torch.tensor([0, 1], dtype=torch.long)
        bc_values = torch.ones(2, dtype=torch.float64, device="cuda")

        with pytest.raises(
            ValueError, match="matrix and values must be on the same device"
        ):
            apply_dirichlet_penalty(K_cpu, f_cpu, bc_dofs, bc_values)
