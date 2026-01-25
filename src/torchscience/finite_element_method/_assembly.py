"""FEM matrix and vector assembly utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from torchscience.finite_element_method._dof_map import DOFMap


def assemble_matrix(
    local_matrices: Tensor,
    dof_map: DOFMap,
) -> Tensor:
    """Assemble global sparse matrix from local element matrices.

    Parameters
    ----------
    local_matrices : Tensor
        Local element matrices, shape (num_elements, dofs_per_element, dofs_per_element).
    dof_map : DOFMap
        DOF mapping from local to global indices.

    Returns
    -------
    Tensor
        Global sparse matrix in CSR format, shape (num_global_dofs, num_global_dofs).

    Notes
    -----
    The returned tensor uses PyTorch's sparse CSR format, which is currently
    in beta. CSR format is efficient for FEM systems due to its row-based
    storage enabling fast matrix-vector products and iterative solvers.

    Raises
    ------
    ValueError
        If local_matrices shape doesn't match the DOF map dimensions.
    ValueError
        If local_matrices and dof_map are on different devices.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import dof_map, assemble_matrix
    >>> mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> dm = dof_map(mesh, order=1)
    >>> local_K = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(dm.local_to_global.shape[0], -1, -1)
    >>> K = assemble_matrix(local_K, dm)
    >>> K.shape
    torch.Size([16, 16])

    """
    # Validate inputs
    num_elements = dof_map.local_to_global.shape[0]
    dofs_per_element = dof_map.dofs_per_element
    num_global_dofs = dof_map.num_global_dofs

    if local_matrices.shape[0] != num_elements:
        raise ValueError(
            f"local_matrices has {local_matrices.shape[0]} elements, "
            f"but dof_map has num_elements={num_elements}"
        )
    if local_matrices.shape[1] != dofs_per_element:
        raise ValueError(
            f"local_matrices has shape[1]={local_matrices.shape[1]}, "
            f"but dof_map has dofs_per_element={dofs_per_element}"
        )
    if local_matrices.shape[2] != dofs_per_element:
        raise ValueError(
            f"local_matrices has shape[2]={local_matrices.shape[2]}, "
            f"but dof_map has dofs_per_element={dofs_per_element}"
        )

    # Validate device compatibility
    if local_matrices.device != dof_map.local_to_global.device:
        raise ValueError(
            f"local_matrices and dof_map must be on the same device, "
            f"got {local_matrices.device} and {dof_map.local_to_global.device}"
        )

    device = local_matrices.device
    dtype = local_matrices.dtype

    # Build COO indices and values
    # For each element e and local DOFs i, j:
    #   global_matrix[local_to_global[e, i], local_to_global[e, j]] += local_matrices[e, i, j]

    # Expand local_to_global to get row and column indices for all entries
    # row_idx: local_to_global[e, i] repeated for each j
    # col_idx: local_to_global[e, j] repeated for each i
    local_to_global = (
        dof_map.local_to_global
    )  # (num_elements, dofs_per_element)

    # Row indices: repeat each row index dofs_per_element times
    row_idx = local_to_global.unsqueeze(-1).expand(
        -1, -1, dofs_per_element
    )  # (num_elements, dofs_per_element, dofs_per_element)

    # Column indices: repeat each column index dofs_per_element times
    col_idx = local_to_global.unsqueeze(-2).expand(
        -1, dofs_per_element, -1
    )  # (num_elements, dofs_per_element, dofs_per_element)

    # Flatten all indices and values
    rows = row_idx.reshape(-1)  # (num_elements * dofs_per_element^2,)
    cols = col_idx.reshape(-1)  # (num_elements * dofs_per_element^2,)
    values = local_matrices.reshape(-1)  # (num_elements * dofs_per_element^2,)

    # Create sparse COO tensor
    indices = torch.stack([rows, cols], dim=0)

    sparse_coo = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_global_dofs, num_global_dofs),
        dtype=dtype,
        device=device,
    )

    # Coalesce to sum duplicate entries, then convert to CSR
    return sparse_coo.coalesce().to_sparse_csr()
