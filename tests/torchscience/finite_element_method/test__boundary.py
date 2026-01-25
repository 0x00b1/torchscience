"""Tests for FEM boundary condition utilities."""

import torch

from torchscience.finite_element_method import boundary_dofs, dof_map
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
