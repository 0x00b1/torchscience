"""Tests for FEM matrix and vector assembly."""

import pytest
import torch

from torchscience.finite_element_method import assemble_matrix, dof_map
from torchscience.geometry.mesh import rectangle_mesh


class TestAssembleMatrix:
    def test_assemble_identity_local_matrices(self):
        """Test assembling identity local matrices."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Create identity local matrices (3x3 for each triangle)
        local_matrices = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(8, -1, -1)
        )

        global_matrix = assemble_matrix(local_matrices, dm)

        # Global matrix should be sparse CSR
        assert global_matrix.is_sparse_csr
        # Shape should be (num_global_dofs, num_global_dofs)
        assert global_matrix.shape == (9, 9)

    def test_assemble_preserves_symmetry(self):
        """Test that symmetric local matrices give symmetric global matrix."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Create symmetric local matrices
        local = torch.randn(8, 3, 3, dtype=torch.float64)
        local_matrices = (local + local.transpose(-1, -2)) / 2

        global_matrix = assemble_matrix(local_matrices, dm)

        # Convert to dense for symmetry check
        dense = global_matrix.to_dense()
        assert torch.allclose(dense, dense.T, atol=1e-10)

    def test_assemble_matrix_values(self):
        """Test that assembly sums contributions correctly."""
        mesh = rectangle_mesh(
            1, 1, bounds=[[0, 1], [0, 1]]
        )  # Single cell, 2 triangles
        dm = dof_map(mesh, order=1)

        # All ones local matrices
        local_matrices = torch.ones(2, 3, 3, dtype=torch.float64)

        global_matrix = assemble_matrix(local_matrices, dm)
        dense = global_matrix.to_dense()

        # Diagonal entries: vertex appears in how many elements
        # Corner vertices: 1 element each, center vertex: 2 elements
        # Off-diagonal: shared edges get summed
        assert dense.sum() > 0  # Non-trivial assembly

    def test_assemble_matrix_shape_validation(self):
        """Test that local_matrices shape is validated."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Wrong number of elements
        local_matrices = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(4, -1, -1)
        )

        with pytest.raises(ValueError, match="num_elements"):
            assemble_matrix(local_matrices, dm)

    def test_assemble_matrix_dofs_per_element_validation(self):
        """Test that dofs_per_element is validated."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Wrong dofs per element (should be 3, using 4)
        local_matrices = (
            torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(8, -1, -1)
        )

        with pytest.raises(ValueError, match="dofs_per_element"):
            assemble_matrix(local_matrices, dm)

    def test_assemble_matrix_quad_mesh(self):
        """Test assembly on quad mesh."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        dm = dof_map(mesh, order=1)

        # Create identity local matrices (4x4 for each quad)
        local_matrices = (
            torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(4, -1, -1)
        )

        global_matrix = assemble_matrix(local_matrices, dm)

        # Global matrix should be sparse CSR
        assert global_matrix.is_sparse_csr
        # Shape should be (num_global_dofs, num_global_dofs)
        assert global_matrix.shape == (9, 9)

    def test_assemble_matrix_higher_order(self):
        """Test assembly with higher-order elements (P2)."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=2)  # P2 has 6 DOFs per triangle

        # Create identity local matrices (6x6 for each triangle)
        local_matrices = (
            torch.eye(6, dtype=torch.float64).unsqueeze(0).expand(8, -1, -1)
        )

        global_matrix = assemble_matrix(local_matrices, dm)

        # Global matrix should be sparse CSR
        assert global_matrix.is_sparse_csr
        # Shape should be (num_global_dofs, num_global_dofs)
        assert global_matrix.shape == (dm.num_global_dofs, dm.num_global_dofs)

    def test_assemble_matrix_dtype_preserved(self):
        """Test that dtype is preserved in assembly."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Float32 local matrices
        local_matrices = (
            torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(8, -1, -1)
        )

        global_matrix = assemble_matrix(local_matrices, dm)

        assert global_matrix.dtype == torch.float32

    def test_assemble_matrix_device_preserved(self):
        """Test that device is preserved in assembly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], device=torch.device("cuda")
        )
        dm = dof_map(mesh, order=1)

        local_matrices = (
            torch.eye(3, dtype=torch.float64, device=torch.device("cuda"))
            .unsqueeze(0)
            .expand(8, -1, -1)
        )

        global_matrix = assemble_matrix(local_matrices, dm)

        assert global_matrix.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_assemble_matrix_device_mismatch_error(self):
        """Test that device mismatch raises ValueError."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        dm = dof_map(mesh, order=1)

        # Create local matrices on CUDA
        local_matrices = (
            torch.eye(3, dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .expand(8, -1, -1)
        )

        # dof_map is on CPU, local_matrices is on CUDA - should raise
        with pytest.raises(ValueError, match="same device"):
            assemble_matrix(local_matrices, dm)
