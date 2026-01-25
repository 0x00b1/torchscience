"""Tests for FEM boundary condition utilities."""

import pytest
import torch

from torchscience.finite_element_method import (
    apply_dirichlet_elimination,
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


class TestApplyDirichletElimination:
    def test_elimination_basic_functionality(self):
        """Test that elimination method yields correct solution."""
        # Simple 3x3 system: K @ u = f
        # K = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        # This is a 1D Laplacian stencil
        K_dense = torch.tensor(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
            dtype=torch.float64,
        )
        K = K_dense.to_sparse_csr()
        f = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

        # Apply BC: u[0] = 0, u[2] = 1
        bc_dofs = torch.tensor([0, 2], dtype=torch.long)
        bc_values = torch.tensor([0.0, 1.0], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

        # Solve the modified system
        u = torch.linalg.solve(K_mod.to_dense(), f_mod)

        # Check boundary values are enforced
        assert torch.isclose(
            u[0], torch.tensor(0.0, dtype=torch.float64), atol=1e-10
        )
        assert torch.isclose(
            u[2], torch.tensor(1.0, dtype=torch.float64), atol=1e-10
        )

        # For this Laplacian with u[0]=0, u[2]=1, the solution should be u[1]=0.5
        assert torch.isclose(
            u[1], torch.tensor(0.5, dtype=torch.float64), atol=1e-10
        )

    def test_elimination_preserves_symmetry(self):
        """Test that elimination method preserves matrix symmetry."""
        # Create symmetric matrix
        K_dense = torch.tensor(
            [[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 5.0]],
            dtype=torch.float64,
        )
        K = K_dense.to_sparse_csr()
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        bc_dofs = torch.tensor([0], dtype=torch.long)
        bc_values = torch.tensor([0.5], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

        K_mod_dense = K_mod.to_dense()
        assert torch.allclose(K_mod_dense, K_mod_dense.T, atol=1e-10)

    def test_elimination_empty_dofs(self):
        """Test that empty dofs tensor works correctly."""
        K = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        bc_dofs = torch.tensor([], dtype=torch.long)
        bc_values = torch.tensor([], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

        # Matrix and vector should be unchanged
        assert torch.allclose(K_mod.to_dense(), K.to_dense())
        assert torch.allclose(f_mod, f)

    def test_elimination_device_mismatch_matrix_vector(self):
        """Test that device mismatch between matrix and vector raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        f_cuda = torch.zeros(4, dtype=torch.float64, device="cuda")
        bc_dofs = torch.tensor([0], dtype=torch.long)
        bc_values = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(
            ValueError, match="matrix and vector must be on the same device"
        ):
            apply_dirichlet_elimination(K_cpu, f_cuda, bc_dofs, bc_values)

    def test_elimination_device_mismatch_matrix_dofs(self):
        """Test that device mismatch between matrix and dofs raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        f_cpu = torch.zeros(4, dtype=torch.float64)
        bc_dofs = torch.tensor([0], dtype=torch.long, device="cuda")
        bc_values = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(
            ValueError, match="matrix and dofs must be on the same device"
        ):
            apply_dirichlet_elimination(K_cpu, f_cpu, bc_dofs, bc_values)

    def test_elimination_device_mismatch_matrix_values(self):
        """Test that device mismatch between matrix and values raises error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        K_cpu = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        f_cpu = torch.zeros(4, dtype=torch.float64)
        bc_dofs = torch.tensor([0], dtype=torch.long)
        bc_values = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        with pytest.raises(
            ValueError, match="matrix and values must be on the same device"
        ):
            apply_dirichlet_elimination(K_cpu, f_cpu, bc_dofs, bc_values)

    def test_elimination_compares_with_penalty(self):
        """Test that elimination and penalty methods give same solution."""
        # Use a simple symmetric positive definite matrix
        K_dense = torch.tensor(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 1.0, 4.0],
            ],
            dtype=torch.float64,
        )
        f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        bc_dofs = torch.tensor([0, 3], dtype=torch.long)
        bc_values = torch.tensor([0.0, 1.0], dtype=torch.float64)

        # Solve with elimination
        K_elim, f_elim = apply_dirichlet_elimination(
            K_dense.to_sparse_csr(), f.clone(), bc_dofs, bc_values
        )
        u_elim = torch.linalg.solve(K_elim.to_dense(), f_elim)

        # Solve with penalty
        K_pen, f_pen = apply_dirichlet_penalty(
            K_dense.to_sparse_csr(), f.clone(), bc_dofs, bc_values
        )
        u_pen = torch.linalg.solve(K_pen.to_dense(), f_pen)

        # Both should satisfy boundary conditions
        assert torch.allclose(u_elim[bc_dofs], bc_values, atol=1e-10)
        assert torch.allclose(
            u_pen[bc_dofs], bc_values, atol=1e-3
        )  # penalty less accurate

        # Solutions should be close (elimination more accurate at BCs)
        assert torch.allclose(u_elim, u_pen, atol=1e-3)

    def test_elimination_row_column_zeroing(self):
        """Test that elimination zeros correct rows and columns."""
        K_dense = torch.tensor(
            [[2.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 4.0]],
            dtype=torch.float64,
        )
        K = K_dense.to_sparse_csr()
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        bc_dofs = torch.tensor([1], dtype=torch.long)
        bc_values = torch.tensor([5.0], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)
        K_mod_dense = K_mod.to_dense()

        # Row 1 should be [0, 1, 0]
        assert torch.allclose(
            K_mod_dense[1, :],
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )

        # Column 1 should be [0, 1, 0]
        assert torch.allclose(
            K_mod_dense[:, 1],
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )

        # f[1] should be the prescribed value
        assert torch.isclose(f_mod[1], torch.tensor(5.0, dtype=torch.float64))

    def test_elimination_rhs_modification(self):
        """Test that RHS is correctly modified for unconstrained DOFs."""
        K_dense = torch.tensor(
            [[2.0, 1.0], [1.0, 2.0]],
            dtype=torch.float64,
        )
        K = K_dense.to_sparse_csr()
        f = torch.tensor([0.0, 0.0], dtype=torch.float64)

        # Constrain DOF 0 to value 3.0
        bc_dofs = torch.tensor([0], dtype=torch.long)
        bc_values = torch.tensor([3.0], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

        # f[0] should be the prescribed value
        assert torch.isclose(f_mod[0], torch.tensor(3.0, dtype=torch.float64))

        # f[1] should be modified: f[1] - K[1,0] * bc_value = 0 - 1 * 3 = -3
        assert torch.isclose(f_mod[1], torch.tensor(-3.0, dtype=torch.float64))

    def test_elimination_multiple_bcs(self):
        """Test elimination with multiple boundary conditions."""
        n = 5
        # Tridiagonal matrix (1D Laplacian)
        K_dense = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            K_dense[i, i] = 2.0
            if i > 0:
                K_dense[i, i - 1] = -1.0
            if i < n - 1:
                K_dense[i, i + 1] = -1.0
        K = K_dense.to_sparse_csr()
        f = torch.zeros(n, dtype=torch.float64)

        # BC: u[0] = 0, u[2] = 0.5, u[4] = 1
        bc_dofs = torch.tensor([0, 2, 4], dtype=torch.long)
        bc_values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

        # Solve and check BCs are satisfied
        u = torch.linalg.solve(K_mod.to_dense(), f_mod)

        assert torch.allclose(u[bc_dofs], bc_values, atol=1e-10)
