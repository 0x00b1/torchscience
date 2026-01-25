"""Local element matrices for finite element methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from torchscience.finite_element_method._basis import (
    lagrange_basis,
    lagrange_basis_gradient,
)
from torchscience.finite_element_method._dof_map import DOFMap
from torchscience.finite_element_method._quadrature import quadrature_points
from torchscience.geometry.mesh import Mesh

if TYPE_CHECKING:
    pass


def local_stiffness_matrices(
    mesh: Mesh,
    dof_map: DOFMap,
    material: Tensor | float = 1.0,
    quad_order: int | None = None,
) -> Tensor:
    """Compute local element stiffness matrices.

    Computes K_e[i,j] = integral(grad(N_i) . grad(N_j) * material) dV
    for each element.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    dof_map : DOFMap
        DOF mapping.
    material : Tensor or float, optional
        Material coefficient (diffusivity). Can be per-element tensor
        of shape (num_elements,) or scalar. Default 1.0.
    quad_order : int, optional
        Quadrature order. If None, uses 2*order.

    Returns
    -------
    Tensor
        Local stiffness matrices, shape (num_elements, dofs_per_element, dofs_per_element).

    Notes
    -----
    For Poisson equation -div(kappa * grad(u)) = f, the stiffness matrix integrates
    kappa * grad(N_i) . grad(N_j) over each element.

    The computation uses isoparametric mapping from reference to physical elements:
    - Reference to physical: x = sum(N_i * x_i) where x_i are element vertices
    - Jacobian: J = dx/dxi = sum(x_i @ grad_N_i.T)
    - Physical gradient: grad_phys = J^{-T} @ grad_ref
    - Integration: K_ij = sum_q(w_q * det(J) * kappa * grad_N_i . grad_N_j)

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import dof_map, local_stiffness_matrices
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
    >>> dm = dof_map(mesh, order=1)
    >>> K_local = local_stiffness_matrices(mesh, dm)
    >>> K_local.shape
    torch.Size([8, 3, 3])
    """
    element_type = mesh.element_type.lower()
    # Handle case where order might be stored as tensor in tensorclass
    order = int(dof_map.order)
    num_elements = mesh.num_elements
    dofs_per_element = int(dof_map.dofs_per_element)
    dim = mesh.dim

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Determine quadrature order
    if quad_order is None:
        quad_order = 2 * order

    # Handle material parameter
    if isinstance(material, (int, float)):
        material_tensor = torch.full(
            (num_elements,), float(material), dtype=dtype, device=device
        )
    else:
        material_tensor = material.to(dtype=dtype, device=device)

    # Get quadrature points and weights for reference element
    quad_points, quad_weights = quadrature_points(
        element_type, quad_order, dtype=dtype, device=device
    )
    num_quad_points = quad_points.shape[0]

    # Evaluate basis gradients at quadrature points (reference coordinates)
    # Shape: (num_quad_points, dofs_per_element, dim)
    grad_ref = lagrange_basis_gradient(element_type, order, quad_points)

    # Get element vertex coordinates
    # Shape: (num_elements, nodes_per_element, dim)
    element_vertices = mesh.vertices[mesh.elements]

    # For triangles and tetrahedra with linear mapping, Jacobian is constant
    # For quads and hexahedra, Jacobian varies within element
    if element_type in ("triangle", "tetrahedron"):
        K_local = _compute_stiffness_simplex(
            element_vertices,
            grad_ref,
            quad_weights,
            material_tensor,
            dofs_per_element,
            dim,
        )
    elif element_type in ("quad", "hexahedron"):
        K_local = _compute_stiffness_tensor_product(
            element_vertices,
            grad_ref,
            quad_points,
            quad_weights,
            material_tensor,
            dofs_per_element,
            dim,
            order,
            element_type,
        )
    else:
        raise ValueError(f"Unsupported element type: {element_type}")

    return K_local


def _compute_stiffness_simplex(
    element_vertices: Tensor,
    grad_ref: Tensor,
    quad_weights: Tensor,
    material: Tensor,
    dofs_per_element: int,
    dim: int,
) -> Tensor:
    """Compute stiffness for simplex elements (triangle, tetrahedron).

    For simplex elements with linear geometric mapping, the Jacobian is constant
    within each element, so we can compute it once and apply to all quadrature points.
    """
    num_elements = element_vertices.shape[0]
    device = element_vertices.device
    dtype = element_vertices.dtype

    # Compute Jacobian for each element
    # For a simplex, the Jacobian maps from reference to physical space
    # J = [x1-x0, x2-x0, ...] for triangle/tet

    if dim == 2:
        # Triangle: J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
        # where vertices are v0=(x0,y0), v1=(x1,y1), v2=(x2,y2)
        v0 = element_vertices[:, 0, :]  # (num_elements, 2)
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]

        # Jacobian columns are edge vectors
        J = torch.stack([v1 - v0, v2 - v0], dim=-1)  # (num_elements, 2, 2)
    else:  # dim == 3
        # Tetrahedron
        v0 = element_vertices[:, 0, :]
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]
        v3 = element_vertices[:, 3, :]

        J = torch.stack(
            [v1 - v0, v2 - v0, v3 - v0], dim=-1
        )  # (num_elements, 3, 3)

    # Compute determinant and inverse transpose of Jacobian
    det_J = torch.linalg.det(J)  # (num_elements,)
    abs_det_J = torch.abs(det_J)  # Use absolute value for integration measure

    # J^{-T} for transforming gradients
    # grad_phys = J^{-T} @ grad_ref
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)  # (num_elements, dim, dim)

    # Transform gradients from reference to physical space
    # grad_ref: (num_quad_points, dofs_per_element, dim)
    # J_inv_T: (num_elements, dim, dim)
    # We want grad_phys: (num_elements, num_quad_points, dofs_per_element, dim)
    # grad_phys[e, q, i, :] = J_inv_T[e] @ grad_ref[q, i, :]

    # Expand for batch matmul
    # grad_ref: (Q, D, dim) -> (1, Q, D, dim, 1)
    # J_inv_T: (E, dim, dim) -> (E, 1, 1, dim, dim)
    grad_ref_expanded = grad_ref.unsqueeze(0).unsqueeze(
        -1
    )  # (1, Q, D, dim, 1)
    J_inv_T_expanded = J_inv_T.unsqueeze(1).unsqueeze(1)  # (E, 1, 1, dim, dim)

    # Batch matrix-vector multiply
    grad_phys = (J_inv_T_expanded @ grad_ref_expanded).squeeze(
        -1
    )  # (E, Q, D, dim)

    # Compute stiffness matrix: K[i,j] = sum_q w_q * |J| * kappa * grad_i . grad_j
    # grad_phys: (E, Q, D, dim)

    # Compute dot products: grad_i . grad_j for all i, j pairs
    # Using einsum: K = sum over q of w_q * |J| * kappa * grad[q,i,:] . grad[q,j,:]

    # quad_weights: (Q,)
    # abs_det_J: (E,)
    # material: (E,)

    # Weight factor: w_q * |J| * kappa for each element and quadrature point
    # Shape: (E, Q)
    weight_factors = (
        quad_weights.unsqueeze(0)
        * abs_det_J.unsqueeze(-1)
        * material.unsqueeze(-1)
    )

    # Compute K using einsum
    # K[e,i,j] = sum_q weight_factors[e,q] * grad_phys[e,q,i,d] * grad_phys[e,q,j,d]
    K_local = torch.einsum(
        "eq,eqid,eqjd->eij", weight_factors, grad_phys, grad_phys
    )

    return K_local


def _compute_stiffness_tensor_product(
    element_vertices: Tensor,
    grad_ref: Tensor,
    quad_points: Tensor,
    quad_weights: Tensor,
    material: Tensor,
    dofs_per_element: int,
    dim: int,
    order: int,
    element_type: str,
) -> Tensor:
    """Compute stiffness for tensor product elements (quad, hexahedron).

    For tensor product elements, the Jacobian varies within the element,
    so we compute it at each quadrature point.
    """

    num_elements = element_vertices.shape[0]
    num_quad_points = quad_points.shape[0]
    nodes_per_element = element_vertices.shape[1]
    device = element_vertices.device
    dtype = element_vertices.dtype

    # Evaluate geometric basis functions and their gradients at quadrature points
    # For isoparametric elements, use linear (order 1) basis for geometry
    # unless using higher-order geometric mapping
    geom_order = 1  # Linear geometry mapping

    # Get geometric basis gradients
    # Shape: (num_quad_points, nodes_per_geom_element, dim)
    grad_geom = lagrange_basis_gradient(element_type, geom_order, quad_points)
    nodes_per_geom = grad_geom.shape[1]

    # Compute Jacobian at each quadrature point for each element
    # x = sum_a N_a * x_a => dx/dxi = sum_a x_a * (dN_a/dxi)
    # J[d1, d2] = sum_a vertex[a, d1] * grad_N[a, d2]
    # J: (num_elements, num_quad_points, dim, dim)

    # element_vertices: (E, nodes_per_geom, dim)
    # grad_geom: (Q, nodes_per_geom, dim)

    # Use only the corner vertices for geometry (first nodes_per_geom nodes)
    geom_vertices = element_vertices[:, :nodes_per_geom, :]

    # Compute Jacobian: J[e,q,i,j] = sum_a vertex[e,a,i] * grad_geom[q,a,j]
    J = torch.einsum("eai,qaj->eqij", geom_vertices, grad_geom)

    # Compute determinant and inverse transpose
    det_J = torch.linalg.det(J)  # (E, Q)
    abs_det_J = torch.abs(det_J)

    # J^{-T}
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)  # (E, Q, dim, dim)

    # Transform gradients from reference to physical space
    # grad_ref: (Q, D, dim) where D = dofs_per_element
    # J_inv_T: (E, Q, dim, dim)
    # grad_phys[e, q, i, :] = J_inv_T[e, q] @ grad_ref[q, i, :]

    # Reshape for batch matmul
    grad_ref_expanded = grad_ref.unsqueeze(0).unsqueeze(
        -1
    )  # (1, Q, D, dim, 1)
    J_inv_T_expanded = J_inv_T.unsqueeze(2)  # (E, Q, 1, dim, dim)

    grad_phys = (J_inv_T_expanded @ grad_ref_expanded).squeeze(
        -1
    )  # (E, Q, D, dim)

    # Weight factors: w_q * |J_eq| * kappa_e
    # quad_weights: (Q,)
    # abs_det_J: (E, Q)
    # material: (E,)
    weight_factors = (
        quad_weights.unsqueeze(0) * abs_det_J * material.unsqueeze(-1)
    )  # (E, Q)

    # Compute stiffness matrix
    K_local = torch.einsum(
        "eq,eqid,eqjd->eij", weight_factors, grad_phys, grad_phys
    )

    return K_local


def local_mass_matrices(
    mesh: Mesh,
    dof_map: DOFMap,
    density: Tensor | float = 1.0,
    quad_order: int | None = None,
) -> Tensor:
    """Compute local element mass matrices.

    Computes M_e[i,j] = integral(N_i * N_j * density) dV
    for each element.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    dof_map : DOFMap
        DOF mapping.
    density : Tensor or float, optional
        Density coefficient. Can be per-element tensor
        of shape (num_elements,) or scalar. Default 1.0.
    quad_order : int, optional
        Quadrature order. If None, uses 2*order.

    Returns
    -------
    Tensor
        Local mass matrices, shape (num_elements, dofs_per_element, dofs_per_element).

    Notes
    -----
    For time-dependent problems, the mass matrix appears in:
    M @ du/dt + K @ u = f

    The computation uses isoparametric mapping from reference to physical elements:
    - Reference to physical: x = sum(N_i * x_i) where x_i are element vertices
    - Jacobian: J = dx/dxi = sum(x_i @ grad_N_i.T)
    - Integration: M_ij = sum_q(w_q * det(J) * density * N_i * N_j)

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import dof_map, local_mass_matrices
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
    >>> dm = dof_map(mesh, order=1)
    >>> M_local = local_mass_matrices(mesh, dm)
    >>> M_local.shape
    torch.Size([8, 3, 3])
    """
    element_type = mesh.element_type.lower()
    # Handle case where order might be stored as tensor in tensorclass
    order = int(dof_map.order)
    num_elements = mesh.num_elements
    dofs_per_element = int(dof_map.dofs_per_element)
    dim = mesh.dim

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Determine quadrature order
    if quad_order is None:
        quad_order = 2 * order

    # Handle density parameter
    if isinstance(density, (int, float)):
        density_tensor = torch.full(
            (num_elements,), float(density), dtype=dtype, device=device
        )
    else:
        density_tensor = density.to(dtype=dtype, device=device)

    # Get quadrature points and weights for reference element
    quad_points, quad_weights = quadrature_points(
        element_type, quad_order, dtype=dtype, device=device
    )

    # Evaluate basis functions at quadrature points (reference coordinates)
    # Shape: (num_quad_points, dofs_per_element)
    basis_values = lagrange_basis(element_type, order, quad_points)

    # Get element vertex coordinates
    # Shape: (num_elements, nodes_per_element, dim)
    element_vertices = mesh.vertices[mesh.elements]

    # For triangles and tetrahedra with linear mapping, Jacobian is constant
    # For quads and hexahedra, Jacobian varies within element
    if element_type in ("triangle", "tetrahedron"):
        M_local = _compute_mass_simplex(
            element_vertices,
            basis_values,
            quad_weights,
            density_tensor,
            dim,
        )
    elif element_type in ("quad", "hexahedron"):
        M_local = _compute_mass_tensor_product(
            element_vertices,
            basis_values,
            quad_points,
            quad_weights,
            density_tensor,
            dim,
            element_type,
        )
    else:
        raise ValueError(f"Unsupported element type: {element_type}")

    return M_local


def _compute_mass_simplex(
    element_vertices: Tensor,
    basis_values: Tensor,
    quad_weights: Tensor,
    density: Tensor,
    dim: int,
) -> Tensor:
    """Compute mass matrix for simplex elements (triangle, tetrahedron).

    For simplex elements with linear geometric mapping, the Jacobian is constant
    within each element, so we can compute it once and apply to all quadrature points.
    """
    num_elements = element_vertices.shape[0]
    device = element_vertices.device
    dtype = element_vertices.dtype

    # Compute Jacobian for each element
    # For a simplex, the Jacobian maps from reference to physical space
    # J = [x1-x0, x2-x0, ...] for triangle/tet

    if dim == 2:
        # Triangle: J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
        # where vertices are v0=(x0,y0), v1=(x1,y1), v2=(x2,y2)
        v0 = element_vertices[:, 0, :]  # (num_elements, 2)
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]

        # Jacobian columns are edge vectors
        J = torch.stack([v1 - v0, v2 - v0], dim=-1)  # (num_elements, 2, 2)
    else:  # dim == 3
        # Tetrahedron
        v0 = element_vertices[:, 0, :]
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]
        v3 = element_vertices[:, 3, :]

        J = torch.stack(
            [v1 - v0, v2 - v0, v3 - v0], dim=-1
        )  # (num_elements, 3, 3)

    # Compute determinant of Jacobian
    det_J = torch.linalg.det(J)  # (num_elements,)
    abs_det_J = torch.abs(det_J)  # Use absolute value for integration measure

    # Weight factor: w_q * |J| * density for each element and quadrature point
    # quad_weights: (Q,)
    # abs_det_J: (E,)
    # density: (E,)
    # Shape: (E, Q)
    weight_factors = (
        quad_weights.unsqueeze(0)
        * abs_det_J.unsqueeze(-1)
        * density.unsqueeze(-1)
    )

    # Compute mass matrix using einsum
    # M[e,i,j] = sum_q weight_factors[e,q] * N[q,i] * N[q,j]
    # basis_values: (Q, D) where D = dofs_per_element
    M_local = torch.einsum(
        "eq,qi,qj->eij", weight_factors, basis_values, basis_values
    )

    return M_local


def _compute_mass_tensor_product(
    element_vertices: Tensor,
    basis_values: Tensor,
    quad_points: Tensor,
    quad_weights: Tensor,
    density: Tensor,
    dim: int,
    element_type: str,
) -> Tensor:
    """Compute mass matrix for tensor product elements (quad, hexahedron).

    For tensor product elements, the Jacobian varies within the element,
    so we compute it at each quadrature point.
    """
    num_elements = element_vertices.shape[0]
    num_quad_points = quad_points.shape[0]
    device = element_vertices.device
    dtype = element_vertices.dtype

    # Evaluate geometric basis functions and their gradients at quadrature points
    # For isoparametric elements, use linear (order 1) basis for geometry
    geom_order = 1  # Linear geometry mapping

    # Get geometric basis gradients
    # Shape: (num_quad_points, nodes_per_geom_element, dim)
    grad_geom = lagrange_basis_gradient(element_type, geom_order, quad_points)
    nodes_per_geom = grad_geom.shape[1]

    # Use only the corner vertices for geometry (first nodes_per_geom nodes)
    geom_vertices = element_vertices[:, :nodes_per_geom, :]

    # Compute Jacobian: J[e,q,i,j] = sum_a vertex[e,a,i] * grad_geom[q,a,j]
    J = torch.einsum("eai,qaj->eqij", geom_vertices, grad_geom)

    # Compute determinant
    det_J = torch.linalg.det(J)  # (E, Q)
    abs_det_J = torch.abs(det_J)

    # Weight factors: w_q * |J_eq| * density_e
    # quad_weights: (Q,)
    # abs_det_J: (E, Q)
    # density: (E,)
    weight_factors = (
        quad_weights.unsqueeze(0) * abs_det_J * density.unsqueeze(-1)
    )  # (E, Q)

    # Compute mass matrix
    # basis_values: (Q, D)
    M_local = torch.einsum(
        "eq,qi,qj->eij", weight_factors, basis_values, basis_values
    )

    return M_local
