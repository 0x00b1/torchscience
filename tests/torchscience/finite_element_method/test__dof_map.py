"""Tests for DOF mapping."""

import pytest
import torch

from torchscience.finite_element_method import DOFMap, dof_map
from torchscience.geometry.mesh import rectangle_mesh


class TestDOFMap:
    def test_p1_triangle_dof_map(self):
        """Test P1 DOF map on triangle mesh."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        assert isinstance(dm, DOFMap)
        assert dm.order == 1
        assert dm.num_global_dofs == 9  # Same as vertices for P1
        assert dm.dofs_per_element == 3
        assert dm.local_to_global.shape == (8, 3)  # 8 elements, 3 DOFs each

    def test_p2_triangle_dof_map(self):
        """Test P2 DOF map on triangle mesh."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        assert dm.order == 2
        assert dm.dofs_per_element == 6  # 3 vertices + 3 edge midpoints
        # P2 has vertices + edge midpoints
        # 9 vertices + 16 unique edges (6 horizontal + 6 vertical + 4 diagonal)
        assert dm.num_global_dofs == 25  # 9 vertices + 16 edges

    def test_dof_map_continuity(self):
        """Test that shared DOFs are properly identified."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Two elements sharing an edge should share 2 DOFs
        # Check that local_to_global has repeated indices
        all_dofs = dm.local_to_global.flatten()
        unique_dofs = torch.unique(all_dofs)
        assert len(unique_dofs) == dm.num_global_dofs


class TestDOFMapP1:
    """Additional tests for P1 elements."""

    def test_p1_local_to_global_matches_elements(self):
        """For P1, local_to_global should equal mesh.elements."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # For P1, DOFs are at vertices, so local_to_global = elements
        assert torch.equal(dm.local_to_global, mesh.elements)

    def test_p1_quad_dof_map(self):
        """Test P1 DOF map on quad mesh."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=1)

        assert dm.order == 1
        assert dm.num_global_dofs == 9  # Same as vertices for Q1
        assert dm.dofs_per_element == 4  # 4 vertices per quad
        assert dm.local_to_global.shape == (4, 4)  # 4 elements, 4 DOFs each

    def test_p1_single_element(self):
        """Test P1 DOF map on single element mesh."""
        mesh = rectangle_mesh(1, 1, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        assert dm.num_global_dofs == 4  # 4 vertices
        assert dm.local_to_global.shape == (2, 3)  # 2 triangles, 3 DOFs each


class TestDOFMapP2:
    """Tests for P2 elements."""

    def test_p2_single_element(self):
        """Test P2 DOF map on single element mesh."""
        mesh = rectangle_mesh(1, 1, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        # 4 vertices + 5 unique edges = 9 DOFs
        # Note: diagonal is shared by both triangles
        assert dm.num_global_dofs == 9
        assert dm.dofs_per_element == 6
        assert dm.local_to_global.shape == (2, 6)

    def test_p2_continuity(self):
        """Test that P2 edge DOFs are shared between adjacent elements."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)

        # The number of unique DOFs should match num_global_dofs
        all_dofs = dm.local_to_global.flatten()
        unique_dofs = torch.unique(all_dofs)
        assert len(unique_dofs) == dm.num_global_dofs

    def test_p2_quad_dof_map(self):
        """Test P2 DOF map on quad mesh."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=2)

        # Q2 has (order+1)^2 = 9 DOFs per element
        assert dm.dofs_per_element == 9
        # For 2x2 quad mesh: 9 vertices + 12 edges + 4 interior = 25
        assert dm.num_global_dofs == 25
        assert dm.local_to_global.shape == (4, 9)


class TestDOFMapDiscontinuous:
    """Tests for discontinuous Galerkin (DG) DOF maps."""

    def test_dg_p1_no_sharing(self):
        """Test that DG DOF map has no shared DOFs."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1, continuity="discontinuous")

        # Each element has its own DOFs
        expected_dofs = mesh.num_elements * dm.dofs_per_element
        assert dm.num_global_dofs == expected_dofs
        assert dm.local_to_global.shape == (8, 3)

        # All DOFs should be unique
        all_dofs = dm.local_to_global.flatten()
        unique_dofs = torch.unique(all_dofs)
        assert len(unique_dofs) == dm.num_global_dofs

    def test_dg_p2_no_sharing(self):
        """Test that P2 DG DOF map has no shared DOFs."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2, continuity="discontinuous")

        expected_dofs = mesh.num_elements * 6  # 8 elements * 6 DOFs
        assert dm.num_global_dofs == expected_dofs


class TestDOFMapElementTypes:
    """Tests for different element types."""

    def test_element_type_stored(self):
        """Test that element_type is stored in DOFMap."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)
        assert dm.element_type == "triangle"

        mesh_quad = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm_quad = dof_map(mesh_quad, order=1)
        assert dm_quad.element_type == "quad"


class TestDOFMapErrors:
    """Tests for error handling."""

    def test_invalid_continuity(self):
        """Test that invalid continuity raises error."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        with pytest.raises(ValueError, match="continuity"):
            dof_map(mesh, order=1, continuity="invalid")

    def test_unsupported_order(self):
        """Test that very high order raises error."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        with pytest.raises(ValueError, match="[Oo]rder"):
            dof_map(mesh, order=10)
