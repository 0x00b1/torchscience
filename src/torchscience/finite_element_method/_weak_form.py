"""Weak form representation for variational formulations of PDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from torchscience.finite_element_method._basis import (
    lagrange_basis,
    lagrange_basis_gradient,
)
from torchscience.finite_element_method._quadrature import quadrature_points

if TYPE_CHECKING:
    from torchscience.finite_element_method._dof_map import DOFMap
    from torchscience.geometry.mesh import Mesh


@dataclass
class BasisValues:
    """Container for basis function values and gradients at quadrature points.

    This class holds the evaluated basis functions for use in weak form assembly.

    Attributes
    ----------
    value : Tensor
        Basis function values at quadrature points.
        Shape: (num_quad_points, num_dofs_per_element).
    grad : Tensor
        Basis function gradients at quadrature points.
        Shape: (num_quad_points, num_dofs_per_element, dim).

    Examples
    --------
    >>> import torch
    >>> from torchscience.finite_element_method import BasisValues
    >>> value = torch.tensor([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    >>> grad = torch.randn(2, 3, 2)
    >>> bv = BasisValues(value=value, grad=grad)
    >>> bv.value.shape
    torch.Size([2, 3])
    """

    value: Tensor
    grad: Tensor


@dataclass
class WeakForm:
    """Variational formulation of a partial differential equation.

    Represents the weak form of a PDE for finite element discretization.
    A weak form consists of:

    - A bilinear form a(u, v) integrated over the domain
    - A linear form L(v) integrated over the domain
    - Optionally, boundary integrals for Neumann or Robin conditions

    The finite element method seeks u in V such that:
        a(u, v) = L(v) for all v in V

    Attributes
    ----------
    bilinear_form : Callable
        The bilinear form a(u, v, x) -> Tensor.
        Arguments:
            - u: Trial function (with .value and/or .grad attributes)
            - v: Test function (with .value and/or .grad attributes)
            - x: Physical coordinates at quadrature points, shape (n_points, dim)
        Returns:
            - Tensor of shape (n_points,) to be integrated with quadrature weights

    linear_form : Callable
        The linear form L(v, x) -> Tensor.
        Arguments:
            - v: Test function (with .value and/or .grad attributes)
            - x: Physical coordinates at quadrature points, shape (n_points, dim)
        Returns:
            - Tensor of shape (n_points,) to be integrated with quadrature weights

    boundary_form : Callable | None
        Optional boundary integral for Neumann or Robin conditions.
        Arguments:
            - v: Test function values on boundary
            - x: Boundary quadrature points, shape (n_boundary_points, dim)
            - n: Outward unit normals, shape (n_boundary_points, dim)
        Returns:
            - Tensor of shape (n_boundary_points,) to be integrated

    Examples
    --------
    Poisson equation: -nabla^2 u = f

    >>> def poisson_bilinear(u, v, x):
    ...     # Stiffness form: integral of grad(u) dot grad(v)
    ...     return (u.grad * v.grad).sum(dim=-1)
    ...
    >>> def poisson_linear(v, x):
    ...     # Source term: f(x) * v where f = 1
    ...     return v.value
    ...
    >>> poisson = WeakForm(
    ...     bilinear_form=poisson_bilinear,
    ...     linear_form=poisson_linear,
    ... )

    Heat equation with Neumann BC:

    >>> def heat_bilinear(u, v, x):
    ...     return u.value * v.value + 0.1 * (u.grad * v.grad).sum(dim=-1)
    ...
    >>> def heat_linear(v, x):
    ...     return torch.zeros_like(v.value)
    ...
    >>> def neumann_bc(v, x, n):
    ...     g = 1.0  # prescribed flux
    ...     return g * v.value
    ...
    >>> heat = WeakForm(
    ...     bilinear_form=heat_bilinear,
    ...     linear_form=heat_linear,
    ...     boundary_form=neumann_bc,
    ... )

    Notes
    -----
    The trial function `u` and test function `v` passed to the forms typically
    have the following attributes:
    - `value`: Function values at quadrature points, shape (n_points,) or
               (n_points, n_components) for vector problems
    - `grad`: Function gradients at quadrature points, shape (n_points, dim)
              or (n_points, n_components, dim) for vector problems

    For scalar problems like Poisson, these are scalar-valued at each point.
    For vector problems like elasticity, `value` and `grad` have additional
    component dimensions.

    This class uses a standard dataclass rather than tensorclass because
    the fields are callables rather than tensors. The callables themselves
    operate on tensor inputs and return tensor outputs.
    """

    bilinear_form: Callable[[object, object, Tensor], Tensor]
    linear_form: Callable[[object, Tensor], Tensor]
    boundary_form: Callable[[object, Tensor, Tensor], Tensor] | None = None


def assemble_weak_form(
    mesh: Mesh,
    dof_map: DOFMap,
    weak_form: WeakForm,
    quad_order: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Assemble system matrix and vector from a weak form.

    This function assembles the global stiffness matrix and load vector from a
    variational weak form specification. It handles the following steps:

    1. Get quadrature points and weights for each element type
    2. Evaluate basis functions and their gradients at quadrature points
    3. Transform to physical coordinates using the Jacobian
    4. Evaluate the bilinear and linear forms at each quadrature point
    5. Integrate over each element using numerical quadrature
    6. Assemble into global matrix and vector

    Parameters
    ----------
    mesh : Mesh
        Input mesh containing vertices and element connectivity.
    dof_map : DOFMap
        DOF mapping from local to global indices.
    weak_form : WeakForm
        Weak form specification with bilinear_form and linear_form callables.
    quad_order : int, optional
        Quadrature order. Default is 2*order + 1, which is sufficient for
        most polynomial integrations with order p elements.

    Returns
    -------
    matrix : Tensor
        System matrix (sparse CSR format), shape (num_dofs, num_dofs).
    vector : Tensor
        Load vector (dense), shape (num_dofs,).

    Notes
    -----
    The bilinear form a(u, v) is evaluated with trial and test BasisValues objects
    that have `.value` and `.grad` attributes. For the matrix assembly, the bilinear
    form should return a tensor that, when multiplied element-wise with the outer
    product of trial and test functions and integrated, gives the local element matrix.

    For typical forms like stiffness (grad(u) dot grad(v)) or mass (u * v), the
    implementation handles the integration automatically.

    Examples
    --------
    >>> import torch
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import (
    ...     dof_map, WeakForm, assemble_weak_form
    ... )
    >>> mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> dm = dof_map(mesh, order=1)
    >>> # Poisson weak form: a(u,v) = integral(grad(u) dot grad(v))
    >>> def poisson_bilinear(u, v, x):
    ...     return (u.grad * v.grad).sum(dim=-1)
    >>> def poisson_linear(v, x):
    ...     return v.value  # f = 1
    >>> weak_form = WeakForm(poisson_bilinear, poisson_linear)
    >>> K, f = assemble_weak_form(mesh, dm, weak_form)
    >>> K.shape
    torch.Size([16, 16])
    >>> f.shape
    torch.Size([16,])

    See Also
    --------
    local_stiffness_matrices : Direct computation of stiffness matrices.
    local_mass_matrices : Direct computation of mass matrices.
    assemble_matrix : Assembly of local matrices to global sparse matrix.
    """
    element_type = mesh.element_type.lower()
    order = int(dof_map.order)
    num_elements = mesh.num_elements
    dofs_per_element = int(dof_map.dofs_per_element)
    dim = mesh.dim

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Determine quadrature order
    if quad_order is None:
        quad_order = 2 * order + 1

    # Get quadrature points and weights for reference element
    quad_points, quad_weights = quadrature_points(
        element_type, quad_order, dtype=dtype, device=device
    )
    num_quad_points = quad_points.shape[0]

    # Evaluate basis functions and gradients at quadrature points (reference coords)
    # Shape: basis_values (num_quad, dofs_per_element)
    # Shape: basis_grads (num_quad, dofs_per_element, dim)
    basis_values = lagrange_basis(element_type, order, quad_points)
    basis_grads_ref = lagrange_basis_gradient(element_type, order, quad_points)

    # Get element vertex coordinates
    # Shape: (num_elements, nodes_per_element, dim)
    element_vertices = mesh.vertices[mesh.elements]

    # Compute Jacobian and transformed gradients for each element
    if element_type in ("triangle", "tetrahedron"):
        J, det_J, basis_grads_phys, physical_coords = (
            _compute_jacobian_simplex(
                element_vertices, basis_grads_ref, quad_points, dim
            )
        )
    elif element_type in ("quad", "hexahedron"):
        J, det_J, basis_grads_phys, physical_coords = (
            _compute_jacobian_tensor_product(
                element_vertices,
                basis_grads_ref,
                quad_points,
                dim,
                element_type,
            )
        )
    else:
        raise ValueError(f"Unsupported element type: {element_type}")

    # Compute local matrices and vectors using weak form
    # For each element e:
    #   K_e[i, j] = sum_q w_q * |J_e| * a(phi_i, phi_j, x_q)
    #   f_e[i] = sum_q w_q * |J_e| * L(phi_i, x_q)

    abs_det_J = torch.abs(det_J)

    # Assemble local matrices
    local_matrices = _assemble_local_matrices(
        weak_form.bilinear_form,
        basis_values,
        basis_grads_phys,
        physical_coords,
        quad_weights,
        abs_det_J,
        num_elements,
        num_quad_points,
        dofs_per_element,
    )

    # Assemble local vectors
    local_vectors = _assemble_local_vectors(
        weak_form.linear_form,
        basis_values,
        basis_grads_phys,
        physical_coords,
        quad_weights,
        abs_det_J,
        num_elements,
        num_quad_points,
        dofs_per_element,
    )

    # Assemble global matrix and vector
    from torchscience.finite_element_method._assembly import (
        assemble_matrix,
        assemble_vector,
    )

    global_matrix = assemble_matrix(local_matrices, dof_map)
    global_vector = assemble_vector(local_vectors, dof_map)

    return global_matrix, global_vector


def _compute_jacobian_simplex(
    element_vertices: Tensor,
    basis_grads_ref: Tensor,
    quad_points: Tensor,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute Jacobian for simplex elements (triangle, tetrahedron).

    For simplex elements with linear geometric mapping, the Jacobian is constant
    within each element.

    Parameters
    ----------
    element_vertices : Tensor
        Element vertex coordinates, shape (num_elements, nodes_per_element, dim).
    basis_grads_ref : Tensor
        Basis gradients in reference coordinates, shape (num_quad, dofs_per_elem, dim).
    quad_points : Tensor
        Quadrature points, shape (num_quad, dim).
    dim : int
        Spatial dimension.

    Returns
    -------
    J : Tensor
        Jacobian matrix, shape (num_elements, dim, dim).
    det_J : Tensor
        Jacobian determinant, shape (num_elements,).
    basis_grads_phys : Tensor
        Basis gradients in physical coordinates, shape (num_elements, num_quad, dofs, dim).
    physical_coords : Tensor
        Physical coordinates at quadrature points, shape (num_elements, num_quad, dim).
    """
    num_elements = element_vertices.shape[0]
    num_quad = basis_grads_ref.shape[0]
    dofs_per_element = basis_grads_ref.shape[1]

    device = element_vertices.device
    dtype = element_vertices.dtype

    # Compute Jacobian: J = [v1-v0, v2-v0, ...] for simplex
    if dim == 2:
        v0 = element_vertices[:, 0, :]
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]
        J = torch.stack([v1 - v0, v2 - v0], dim=-1)  # (E, 2, 2)
    else:  # dim == 3
        v0 = element_vertices[:, 0, :]
        v1 = element_vertices[:, 1, :]
        v2 = element_vertices[:, 2, :]
        v3 = element_vertices[:, 3, :]
        J = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (E, 3, 3)

    det_J = torch.linalg.det(J)  # (E,)
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)  # (E, dim, dim)

    # Transform basis gradients: grad_phys = J^{-T} @ grad_ref
    # basis_grads_ref: (Q, D, dim)
    # J_inv_T: (E, dim, dim)
    # Result: (E, Q, D, dim)
    basis_grads_ref_expanded = basis_grads_ref.unsqueeze(0).unsqueeze(
        -1
    )  # (1, Q, D, dim, 1)
    J_inv_T_expanded = J_inv_T.unsqueeze(1).unsqueeze(1)  # (E, 1, 1, dim, dim)
    basis_grads_phys = (J_inv_T_expanded @ basis_grads_ref_expanded).squeeze(
        -1
    )  # (E, Q, D, dim)

    # Compute physical coordinates at quadrature points
    # For simplex: x = v0 + xi * (v1 - v0) + eta * (v2 - v0) + ...
    # Using basis functions: x = sum_i N_i(xi) * v_i (but only corner vertices for geometry)
    # Simplified for simplex: x = v0 + J @ xi
    if dim == 2:
        v0_expanded = element_vertices[:, 0, :].unsqueeze(1)  # (E, 1, dim)
        # quad_points: (Q, dim) -> (1, Q, dim)
        xi = quad_points.unsqueeze(0)  # (1, Q, dim)
        # J: (E, dim, dim), xi.T: (dim, Q) -> physical: (E, Q, dim)
        physical_coords = v0_expanded + torch.einsum(
            "eij,qj->eqi", J, quad_points
        )
    else:  # dim == 3
        v0_expanded = element_vertices[:, 0, :].unsqueeze(1)
        physical_coords = v0_expanded + torch.einsum(
            "eij,qj->eqi", J, quad_points
        )

    return J, det_J, basis_grads_phys, physical_coords


def _compute_jacobian_tensor_product(
    element_vertices: Tensor,
    basis_grads_ref: Tensor,
    quad_points: Tensor,
    dim: int,
    element_type: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute Jacobian for tensor product elements (quad, hexahedron).

    For tensor product elements, the Jacobian varies within the element.

    Parameters
    ----------
    element_vertices : Tensor
        Element vertex coordinates, shape (num_elements, nodes_per_element, dim).
    basis_grads_ref : Tensor
        Basis gradients in reference coordinates, shape (num_quad, dofs_per_elem, dim).
    quad_points : Tensor
        Quadrature points, shape (num_quad, dim).
    dim : int
        Spatial dimension.
    element_type : str
        Element type ("quad" or "hexahedron").

    Returns
    -------
    J : Tensor
        Jacobian matrix, shape (num_elements, num_quad, dim, dim).
    det_J : Tensor
        Jacobian determinant, shape (num_elements, num_quad).
    basis_grads_phys : Tensor
        Basis gradients in physical coordinates, shape (num_elements, num_quad, dofs, dim).
    physical_coords : Tensor
        Physical coordinates at quadrature points, shape (num_elements, num_quad, dim).
    """
    num_elements = element_vertices.shape[0]
    num_quad = quad_points.shape[0]
    dofs_per_element = basis_grads_ref.shape[1]

    # Use linear geometry mapping
    geom_order = 1
    grad_geom = lagrange_basis_gradient(element_type, geom_order, quad_points)
    basis_geom = lagrange_basis(element_type, geom_order, quad_points)
    nodes_per_geom = grad_geom.shape[1]

    # Use corner vertices for geometry
    geom_vertices = element_vertices[:, :nodes_per_geom, :]

    # Compute Jacobian: J[e,q,i,j] = sum_a vertex[e,a,i] * grad_geom[q,a,j]
    J = torch.einsum(
        "eai,qaj->eqij", geom_vertices, grad_geom
    )  # (E, Q, dim, dim)

    det_J = torch.linalg.det(J)  # (E, Q)
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)  # (E, Q, dim, dim)

    # Transform basis gradients
    # basis_grads_ref: (Q, D, dim) -> (1, Q, D, dim, 1)
    # J_inv_T: (E, Q, dim, dim) -> (E, Q, 1, dim, dim)
    basis_grads_ref_expanded = basis_grads_ref.unsqueeze(0).unsqueeze(-1)
    J_inv_T_expanded = J_inv_T.unsqueeze(2)
    basis_grads_phys = (J_inv_T_expanded @ basis_grads_ref_expanded).squeeze(
        -1
    )  # (E, Q, D, dim)

    # Compute physical coordinates: x = sum_a N_a(xi) * v_a
    # basis_geom: (Q, nodes_per_geom)
    # geom_vertices: (E, nodes_per_geom, dim)
    physical_coords = torch.einsum(
        "qa,eai->eqi", basis_geom, geom_vertices
    )  # (E, Q, dim)

    return J, det_J, basis_grads_phys, physical_coords


def _assemble_local_matrices(
    bilinear_form: Callable,
    basis_values: Tensor,
    basis_grads_phys: Tensor,
    physical_coords: Tensor,
    quad_weights: Tensor,
    abs_det_J: Tensor,
    num_elements: int,
    num_quad: int,
    dofs_per_element: int,
) -> Tensor:
    """Assemble local element matrices from bilinear form.

    Parameters
    ----------
    bilinear_form : Callable
        Bilinear form function a(u, v, x).
    basis_values : Tensor
        Basis function values, shape (num_quad, dofs_per_element).
    basis_grads_phys : Tensor
        Basis gradients in physical coords, shape (num_elements, num_quad, dofs, dim).
    physical_coords : Tensor
        Physical coordinates, shape (num_elements, num_quad, dim).
    quad_weights : Tensor
        Quadrature weights, shape (num_quad,).
    abs_det_J : Tensor
        Absolute value of Jacobian determinant.
    num_elements : int
        Number of elements.
    num_quad : int
        Number of quadrature points.
    dofs_per_element : int
        DOFs per element.

    Returns
    -------
    Tensor
        Local matrices, shape (num_elements, dofs_per_element, dofs_per_element).
    """
    device = basis_values.device
    dtype = basis_values.dtype

    # Initialize local matrices
    local_matrices = torch.zeros(
        (num_elements, dofs_per_element, dofs_per_element),
        dtype=dtype,
        device=device,
    )

    # Handle different shapes of abs_det_J
    # For simplex: abs_det_J has shape (E,)
    # For tensor product: abs_det_J has shape (E, Q)
    if abs_det_J.dim() == 1:
        # Simplex: expand to (E, Q)
        abs_det_J_expanded = abs_det_J.unsqueeze(-1).expand(-1, num_quad)
    else:
        abs_det_J_expanded = abs_det_J

    # weight_factors: (E, Q)
    weight_factors = quad_weights.unsqueeze(0) * abs_det_J_expanded

    # For each pair of trial (i) and test (j) basis functions, evaluate the bilinear form
    # We need to loop over i and j because the bilinear form expects full basis objects
    for i in range(dofs_per_element):
        for j in range(dofs_per_element):
            # Create BasisValues for trial function i
            # value: (E, Q) - basis function i evaluated at each quad point for each element
            # grad: (E, Q, dim) - gradient of basis function i
            u_value = (
                basis_values[:, i].unsqueeze(0).expand(num_elements, -1)
            )  # (E, Q)
            u_grad = basis_grads_phys[:, :, i, :]  # (E, Q, dim)

            # Create BasisValues for test function j
            v_value = (
                basis_values[:, j].unsqueeze(0).expand(num_elements, -1)
            )  # (E, Q)
            v_grad = basis_grads_phys[:, :, j, :]  # (E, Q, dim)

            u = BasisValues(value=u_value, grad=u_grad)
            v = BasisValues(value=v_value, grad=v_grad)

            # Evaluate bilinear form: returns (E, Q)
            integrand = bilinear_form(u, v, physical_coords)

            # Integrate: K[i,j] = sum_q w_q * |J| * a(phi_i, phi_j)
            local_matrices[:, i, j] = (weight_factors * integrand).sum(dim=-1)

    return local_matrices


def poisson_form(diffusivity: Tensor | float = 1.0) -> WeakForm:
    """Create weak form for Poisson/Laplace operator.

    Creates a WeakForm representing the bilinear form for the Poisson equation:
        -nabla cdot (kappa nabla u) = f

    The resulting bilinear form computes:
        a(u, v) = kappa * nabla u cdot nabla v

    This corresponds to the stiffness matrix in finite element terminology.

    Parameters
    ----------
    diffusivity : Tensor or float, optional
        The diffusivity coefficient kappa. Can be a scalar or tensor.
        Default is 1.0.

    Returns
    -------
    WeakForm
        A WeakForm instance with:
        - bilinear_form: kappa * grad(u) cdot grad(v)
        - linear_form: returns zeros (source term handled separately)
        - boundary_form: None

    Examples
    --------
    >>> from torchscience.finite_element_method import poisson_form, assemble_weak_form
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> dm = dof_map(mesh, order=1)
    >>> wf = poisson_form(diffusivity=1.0)
    >>> K, f = assemble_weak_form(mesh, dm, wf)

    See Also
    --------
    mass_form : Create weak form for mass operator.
    local_stiffness_matrices : Direct computation of stiffness matrices.
    """
    kappa = diffusivity

    def bilinear_form(u: BasisValues, v: BasisValues, x: Tensor) -> Tensor:
        """Compute kappa * grad(u) cdot grad(v)."""
        return kappa * (u.grad * v.grad).sum(dim=-1)

    def linear_form(v: BasisValues, x: Tensor) -> Tensor:
        """Return zeros (source term handled separately during assembly)."""
        return torch.zeros_like(v.value)

    return WeakForm(
        bilinear_form=bilinear_form,
        linear_form=linear_form,
    )


def mass_form(density: Tensor | float = 1.0) -> WeakForm:
    """Create weak form for mass operator.

    Creates a WeakForm representing the bilinear form for the mass matrix:
        M_ij = integral(rho * N_i * N_j) dV

    The resulting bilinear form computes:
        a(u, v) = rho * u * v

    This corresponds to the mass matrix in finite element terminology,
    commonly used in time-dependent problems.

    Parameters
    ----------
    density : Tensor or float, optional
        The density coefficient rho. Can be a scalar or tensor.
        Default is 1.0.

    Returns
    -------
    WeakForm
        A WeakForm instance with:
        - bilinear_form: rho * u * v
        - linear_form: returns zeros
        - boundary_form: None

    Examples
    --------
    >>> from torchscience.finite_element_method import mass_form, assemble_weak_form
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> dm = dof_map(mesh, order=1)
    >>> wf = mass_form(density=1.0)
    >>> M, f = assemble_weak_form(mesh, dm, wf)

    See Also
    --------
    poisson_form : Create weak form for Poisson/Laplace operator.
    local_mass_matrices : Direct computation of mass matrices.
    """
    rho = density

    def bilinear_form(u: BasisValues, v: BasisValues, x: Tensor) -> Tensor:
        """Compute rho * u * v."""
        return rho * u.value * v.value

    def linear_form(v: BasisValues, x: Tensor) -> Tensor:
        """Return zeros."""
        return torch.zeros_like(v.value)

    return WeakForm(
        bilinear_form=bilinear_form,
        linear_form=linear_form,
    )


def _assemble_local_vectors(
    linear_form: Callable,
    basis_values: Tensor,
    basis_grads_phys: Tensor,
    physical_coords: Tensor,
    quad_weights: Tensor,
    abs_det_J: Tensor,
    num_elements: int,
    num_quad: int,
    dofs_per_element: int,
) -> Tensor:
    """Assemble local element vectors from linear form.

    Parameters
    ----------
    linear_form : Callable
        Linear form function L(v, x).
    basis_values : Tensor
        Basis function values, shape (num_quad, dofs_per_element).
    basis_grads_phys : Tensor
        Basis gradients in physical coords, shape (num_elements, num_quad, dofs, dim).
    physical_coords : Tensor
        Physical coordinates, shape (num_elements, num_quad, dim).
    quad_weights : Tensor
        Quadrature weights, shape (num_quad,).
    abs_det_J : Tensor
        Absolute value of Jacobian determinant.
    num_elements : int
        Number of elements.
    num_quad : int
        Number of quadrature points.
    dofs_per_element : int
        DOFs per element.

    Returns
    -------
    Tensor
        Local vectors, shape (num_elements, dofs_per_element).
    """
    device = basis_values.device
    dtype = basis_values.dtype

    # Initialize local vectors
    local_vectors = torch.zeros(
        (num_elements, dofs_per_element),
        dtype=dtype,
        device=device,
    )

    # Handle different shapes of abs_det_J
    if abs_det_J.dim() == 1:
        abs_det_J_expanded = abs_det_J.unsqueeze(-1).expand(-1, num_quad)
    else:
        abs_det_J_expanded = abs_det_J

    # weight_factors: (E, Q)
    weight_factors = quad_weights.unsqueeze(0) * abs_det_J_expanded

    # For each test function j, evaluate the linear form
    for j in range(dofs_per_element):
        # Create BasisValues for test function j
        v_value = (
            basis_values[:, j].unsqueeze(0).expand(num_elements, -1)
        )  # (E, Q)
        v_grad = basis_grads_phys[:, :, j, :]  # (E, Q, dim)

        v = BasisValues(value=v_value, grad=v_grad)

        # Evaluate linear form: returns (E, Q)
        integrand = linear_form(v, physical_coords)

        # Integrate: f[j] = sum_q w_q * |J| * L(phi_j)
        local_vectors[:, j] = (weight_factors * integrand).sum(dim=-1)

    return local_vectors
