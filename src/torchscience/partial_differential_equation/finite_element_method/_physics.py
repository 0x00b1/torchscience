"""High-level physics solvers for finite element methods."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from torchscience.geometry.mesh import Mesh
from torchscience.partial_differential_equation.finite_element_method._assembly import (
    assemble_matrix,
)
from torchscience.partial_differential_equation.finite_element_method._basis import (
    lagrange_basis,
    lagrange_basis_gradient,
)
from torchscience.partial_differential_equation.finite_element_method._boundary import (
    apply_dirichlet_elimination,
    boundary_dofs,
)
from torchscience.partial_differential_equation.finite_element_method._dof_map import (
    DOFMap,
    dof_map,
)
from torchscience.partial_differential_equation.finite_element_method._local_matrices import (
    local_stiffness_matrices,
)
from torchscience.partial_differential_equation.finite_element_method._quadrature import (
    quadrature_points,
)
from torchscience.partial_differential_equation.finite_element_method._solve import (
    solve_cg,
    solve_direct,
)


def solve_poisson(
    mesh: Mesh,
    source: Tensor | Callable[[Tensor], Tensor] | float,
    diffusivity: Tensor | float = 1.0,
    dirichlet_dofs: Tensor | None = None,
    dirichlet_values: Tensor | float = 0.0,
    order: int = 1,
    solver: str = "direct",
) -> Tensor:
    """Solve the Poisson equation -nabla . (kappa nabla u) = f.

    This function provides a high-level interface for solving the Poisson equation
    with Dirichlet boundary conditions using the finite element method.

    Parameters
    ----------
    mesh : Mesh
        Input mesh defining the computational domain.
    source : Tensor, callable, or float
        Source term f. If callable, takes coordinates (n, dim) and returns values (n,).
        If Tensor, shape (num_vertices,) for nodal values or (num_elements,) for
        element-wise constant values.
        If float, constant source throughout the domain.
    diffusivity : Tensor or float, optional
        Diffusivity coefficient kappa. Can be a scalar or per-element tensor of
        shape (num_elements,). Default 1.0.
    dirichlet_dofs : Tensor, optional
        DOF indices for Dirichlet boundary conditions. If None, uses all boundary
        DOFs (from boundary_dofs function).
    dirichlet_values : Tensor or float, optional
        Prescribed values at Dirichlet DOFs. If float, the same value is used for
        all Dirichlet DOFs. If Tensor, must have shape (num_dirichlet_dofs,).
        Default 0.0 (homogeneous boundary conditions).
    order : int, optional
        Polynomial order for the finite element space. Default 1 (P1 elements).
        Supports orders 1-4.
    solver : str, optional
        Linear solver to use. Either "direct" for direct (LU) solver or "cg" for
        conjugate gradient. Default "direct".

    Returns
    -------
    Tensor
        Solution values at DOFs, shape (num_dofs,). For P1 elements, this equals
        (num_vertices,). For higher-order elements, includes additional DOFs at
        edges, faces, and element interiors.

    Raises
    ------
    ValueError
        If solver is not "direct" or "cg".

    Notes
    -----
    The Poisson equation in strong form is:

    .. math::

        -\\nabla \\cdot (\\kappa \\nabla u) = f \\quad \\text{in } \\Omega

        u = g \\quad \\text{on } \\partial\\Omega

    The finite element discretization uses the weak form:

    .. math::

        \\int_\\Omega \\kappa \\nabla u \\cdot \\nabla v \\, dV = \\int_\\Omega f v \\, dV

    for all test functions v in the finite element space.

    The implementation:
    1. Creates the DOF map for the specified polynomial order
    2. Computes local stiffness matrices using numerical quadrature
    3. Assembles the global stiffness matrix
    4. Computes the load vector from the source term
    5. Applies Dirichlet boundary conditions using elimination
    6. Solves the linear system

    Examples
    --------
    Solve -nabla^2 u = 1 with u = 0 on boundary:

    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.partial_differential_equation.finite_element_method import solve_poisson
    >>> mesh = rectangle_mesh(10, 10)
    >>> u = solve_poisson(mesh, source=1.0)

    With non-homogeneous boundary conditions:

    >>> from torchscience.partial_differential_equation.finite_element_method import boundary_dofs, dof_map
    >>> mesh = rectangle_mesh(10, 10)
    >>> dm = dof_map(mesh, order=1)
    >>> bc_dofs = boundary_dofs(mesh, dm)
    >>> bc_values = torch.ones(len(bc_dofs))
    >>> u = solve_poisson(mesh, source=0.0, dirichlet_dofs=bc_dofs, dirichlet_values=bc_values)

    With custom diffusivity:

    >>> u = solve_poisson(mesh, source=1.0, diffusivity=2.0)

    With callable source term:

    >>> def source_fn(coords):
    ...     x, y = coords[:, 0], coords[:, 1]
    ...     return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    >>> u = solve_poisson(mesh, source=source_fn)

    See Also
    --------
    local_stiffness_matrices : Compute element stiffness matrices.
    assemble_matrix : Assemble global matrix from local matrices.
    apply_dirichlet_elimination : Apply Dirichlet BCs via elimination.
    solve_direct : Direct linear solver.
    solve_cg : Conjugate gradient solver.
    """
    # Validate solver
    if solver not in ("direct", "cg"):
        raise ValueError(f"solver must be 'direct' or 'cg', got '{solver}'")

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Create DOF map
    dm = dof_map(mesh, order=order)
    num_dofs = dm.num_global_dofs

    # Compute local stiffness matrices
    K_local = local_stiffness_matrices(mesh, dm, material=diffusivity)

    # Assemble global stiffness matrix
    K = assemble_matrix(K_local, dm)

    # Compute load vector
    f = _compute_load_vector(mesh, dm, source)

    # Get boundary DOFs if not provided
    if dirichlet_dofs is None:
        dirichlet_dofs = boundary_dofs(mesh, dm)

    # Prepare Dirichlet values
    if isinstance(dirichlet_values, (int, float)):
        dirichlet_vals = torch.full(
            (dirichlet_dofs.shape[0],),
            float(dirichlet_values),
            dtype=dtype,
            device=device,
        )
    else:
        dirichlet_vals = dirichlet_values.to(dtype=dtype, device=device)

    # Apply Dirichlet boundary conditions
    K_bc, f_bc = apply_dirichlet_elimination(
        K, f, dirichlet_dofs, dirichlet_vals
    )

    # Solve the linear system
    if solver == "direct":
        u = solve_direct(K_bc, f_bc)
    else:  # solver == "cg"
        u = solve_cg(K_bc, f_bc, tol=1e-10, maxiter=num_dofs * 2)

    return u


def _compute_load_vector(
    mesh: Mesh,
    dm: DOFMap,
    source: Tensor | Callable[[Tensor], Tensor] | float,
) -> Tensor:
    """Compute the load vector from the source term.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    dm : DOFMap
        DOF mapping.
    source : Tensor, callable, or float
        Source term.

    Returns
    -------
    Tensor
        Load vector, shape (num_dofs,).
    """
    element_type = mesh.element_type.lower()
    order = int(dm.order)
    num_elements = mesh.num_elements
    dofs_per_element = int(dm.dofs_per_element)
    dim = mesh.dim

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Quadrature order (sufficient for integrating f * N_i)
    quad_order = 2 * order + 1

    # Get quadrature points and weights
    quad_points, quad_weights = quadrature_points(
        element_type, quad_order, dtype=dtype, device=device
    )
    num_quad = quad_points.shape[0]

    # Evaluate basis functions at quadrature points
    basis_values = lagrange_basis(element_type, order, quad_points)  # (Q, D)

    # Get element vertices
    element_vertices = mesh.vertices[mesh.elements]  # (E, nodes_per_elem, dim)

    # Compute physical coordinates at quadrature points and Jacobians
    if element_type in ("triangle", "tetrahedron"):
        physical_coords, abs_det_J = _compute_coords_simplex(
            element_vertices, quad_points, dim
        )
    elif element_type in ("quad", "hexahedron"):
        physical_coords, abs_det_J = _compute_coords_tensor_product(
            element_vertices, quad_points, dim, element_type
        )
    else:
        raise ValueError(f"Unsupported element type: {element_type}")

    # Evaluate source at quadrature points
    # physical_coords: (E, Q, dim)
    if callable(source):
        # Flatten for function call
        coords_flat = physical_coords.reshape(-1, dim)  # (E*Q, dim)
        source_flat = source(coords_flat)  # (E*Q,)
        source_at_quad = source_flat.reshape(num_elements, num_quad)  # (E, Q)
    elif isinstance(source, Tensor):
        if source.shape[0] == mesh.num_vertices:
            # Nodal values: interpolate to quadrature points
            # source_at_quad[e,q] = sum_i N_i(xi_q) * source[elements[e,i]]
            # For simplicity, use linear interpolation from element vertices
            elem_source = source[mesh.elements]  # (E, nodes_per_elem)
            # basis_values: (Q, D), elem_source: (E, nodes_per_elem)
            # For P1, D = nodes_per_elem = 3 (triangle) or 4 (quad)
            # Use basis values to interpolate
            basis_geom = lagrange_basis(
                element_type, 1, quad_points
            )  # (Q, nodes_geom)
            nodes_geom = basis_geom.shape[1]
            elem_source_geom = source[
                mesh.elements[:, :nodes_geom]
            ]  # (E, nodes_geom)
            source_at_quad = torch.einsum(
                "qn,en->eq", basis_geom, elem_source_geom
            )  # (E, Q)
        elif source.shape[0] == num_elements:
            # Element-wise constant: broadcast to all quadrature points
            source_at_quad = source.unsqueeze(-1).expand(
                -1, num_quad
            )  # (E, Q)
        else:
            raise ValueError(
                f"source tensor shape {source.shape} does not match "
                f"num_vertices={mesh.num_vertices} or num_elements={num_elements}"
            )
    else:
        # Constant source
        source_at_quad = torch.full(
            (num_elements, num_quad), float(source), dtype=dtype, device=device
        )

    # Compute local load vectors
    # f_e[i] = sum_q w_q * |J| * f(x_q) * N_i(xi_q)
    # Handle different shapes of abs_det_J
    if abs_det_J.dim() == 1:
        # Simplex: (E,) -> (E, Q)
        abs_det_J_expanded = abs_det_J.unsqueeze(-1).expand(-1, num_quad)
    else:
        abs_det_J_expanded = abs_det_J

    # weight_factors: (E, Q)
    weight_factors = quad_weights.unsqueeze(0) * abs_det_J_expanded

    # local_vectors[e, i] = sum_q weight_factors[e, q] * source_at_quad[e, q] * basis_values[q, i]
    local_vectors = torch.einsum(
        "eq,eq,qi->ei", weight_factors, source_at_quad, basis_values
    )  # (E, D)

    # Assemble global load vector
    from torchscience.partial_differential_equation.finite_element_method._assembly import (
        assemble_vector,
    )

    return assemble_vector(local_vectors, dm)


def solve_heat(
    mesh: Mesh,
    initial: Tensor | Callable[[Tensor], Tensor],
    source: Tensor | Callable[[Tensor], Tensor] | float = 0.0,
    diffusivity: Tensor | float = 1.0,
    density: Tensor | float = 1.0,
    dt: float = 0.01,
    num_steps: int = 100,
    dirichlet_dofs: Tensor | None = None,
    dirichlet_values: Tensor | float = 0.0,
    order: int = 1,
    return_all: bool = False,
) -> Tensor:
    """Solve the heat equation rho*c du/dt - nabla . (kappa nabla u) = f.

    This function provides a high-level interface for solving the heat equation
    using the finite element method with implicit Euler time stepping (backward
    Euler) for unconditional stability.

    Parameters
    ----------
    mesh : Mesh
        Input mesh defining the computational domain.
    initial : Tensor or callable
        Initial condition u_0. If callable, takes coordinates (n, dim) and returns
        values (n,). If Tensor, shape (num_dofs,) for nodal values.
    source : Tensor, callable, or float, optional
        Heat source f. If callable, takes coordinates (n, dim) and returns values (n,).
        If Tensor, shape (num_vertices,) for nodal values or (num_elements,) for
        element-wise constant values. If float, constant source throughout.
        Default 0.0.
    diffusivity : Tensor or float, optional
        Thermal diffusivity kappa. Can be a scalar or per-element tensor of
        shape (num_elements,). Default 1.0.
    density : Tensor or float, optional
        Product rho*c (density times specific heat capacity). Can be a scalar
        or per-element tensor of shape (num_elements,). Default 1.0.
    dt : float, optional
        Time step size. Default 0.01.
    num_steps : int, optional
        Number of time steps to perform. Default 100.
    dirichlet_dofs : Tensor, optional
        DOF indices for Dirichlet boundary conditions. If None, uses all boundary
        DOFs (from boundary_dofs function).
    dirichlet_values : Tensor or float, optional
        Prescribed values at Dirichlet DOFs. If float, the same value is used for
        all Dirichlet DOFs. If Tensor, must have shape (num_dirichlet_dofs,).
        Default 0.0 (homogeneous boundary conditions).
    order : int, optional
        Polynomial order for the finite element space. Default 1 (P1 elements).
    return_all : bool, optional
        If True, return solutions at all time steps including t=0.
        If False (default), return only the final solution.

    Returns
    -------
    Tensor
        Solution values at DOFs. Shape (num_dofs,) if return_all=False,
        or (num_steps+1, num_dofs) if return_all=True.

    Notes
    -----
    The heat equation in strong form is:

    .. math::

        \\rho c \\frac{\\partial u}{\\partial t} - \\nabla \\cdot (\\kappa \\nabla u) = f
        \\quad \\text{in } \\Omega

        u = g \\quad \\text{on } \\partial\\Omega

        u(t=0) = u_0

    Using implicit Euler (backward Euler) time discretization:

    .. math::

        \\rho c \\frac{u^{n+1} - u^n}{\\Delta t} - \\nabla \\cdot (\\kappa \\nabla u^{n+1}) = f^{n+1}

    The finite element discretization leads to the linear system:

    .. math::

        \\left(\\frac{M}{\\Delta t} + K\\right) u^{n+1} = \\frac{M}{\\Delta t} u^n + f^{n+1}

    where M is the mass matrix and K is the stiffness matrix.

    The implicit Euler method is unconditionally stable, meaning there are no
    restrictions on the time step size for stability (though accuracy may require
    smaller time steps).

    Examples
    --------
    Solve heat equation starting from constant initial condition:

    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.partial_differential_equation.finite_element_method import solve_heat
    >>> mesh = rectangle_mesh(10, 10)
    >>> initial = torch.ones(mesh.num_vertices, dtype=torch.float64)
    >>> u = solve_heat(mesh, initial=initial, dt=0.01, num_steps=100)

    With non-zero source term:

    >>> u = solve_heat(mesh, initial=initial, source=1.0, dt=0.01, num_steps=100)

    Return all time steps:

    >>> u_all = solve_heat(mesh, initial=initial, dt=0.01, num_steps=100, return_all=True)
    >>> u_all.shape
    torch.Size([101, ...])  # num_steps + 1 (includes t=0)

    See Also
    --------
    solve_poisson : Solve the steady-state Poisson equation.
    local_mass_matrices : Compute element mass matrices.
    local_stiffness_matrices : Compute element stiffness matrices.
    """
    from torchscience.partial_differential_equation.finite_element_method._local_matrices import (
        local_mass_matrices,
    )

    device = mesh.vertices.device
    dtype = mesh.vertices.dtype

    # Create DOF map
    dm = dof_map(mesh, order=order)
    num_dofs = dm.num_global_dofs

    # Compute local stiffness matrices (for diffusion term)
    K_local = local_stiffness_matrices(mesh, dm, material=diffusivity)

    # Compute local mass matrices (for time derivative term)
    M_local = local_mass_matrices(mesh, dm, density=density)

    # Assemble global matrices
    K = assemble_matrix(K_local, dm)  # Stiffness matrix
    M = assemble_matrix(M_local, dm)  # Mass matrix

    # Compute load vector from source term
    f = _compute_load_vector(mesh, dm, source)

    # Get boundary DOFs if not provided
    if dirichlet_dofs is None:
        dirichlet_dofs = boundary_dofs(mesh, dm)

    # Prepare Dirichlet values
    if isinstance(dirichlet_values, (int, float)):
        dirichlet_vals = torch.full(
            (dirichlet_dofs.shape[0],),
            float(dirichlet_values),
            dtype=dtype,
            device=device,
        )
    else:
        dirichlet_vals = dirichlet_values.to(dtype=dtype, device=device)

    # Initialize solution from initial condition
    if callable(initial):
        # Evaluate at DOF coordinates
        # For P1 elements, DOFs are at vertices
        if order == 1:
            coords = mesh.vertices
        else:
            # For higher-order elements, we need DOF coordinates
            # For now, evaluate at vertices and use those for vertex DOFs
            # Edge/face DOFs would need interpolation
            coords = mesh.vertices
            # TODO: Handle higher-order DOF coordinates properly

        u = initial(coords)
        if u.shape[0] != num_dofs:
            # If initial returns values only at vertices, pad for higher-order DOFs
            if (
                u.shape[0] == mesh.num_vertices
                and num_dofs > mesh.num_vertices
            ):
                u_full = torch.zeros(num_dofs, dtype=dtype, device=device)
                u_full[: mesh.num_vertices] = u
                u = u_full
    else:
        u = initial.clone().to(dtype=dtype, device=device)
        if u.shape[0] != num_dofs:
            if (
                u.shape[0] == mesh.num_vertices
                and num_dofs > mesh.num_vertices
            ):
                u_full = torch.zeros(num_dofs, dtype=dtype, device=device)
                u_full[: mesh.num_vertices] = u
                u = u_full

    # Apply initial Dirichlet BC values
    u[dirichlet_dofs] = dirichlet_vals

    # Storage for all time steps if requested
    if return_all:
        u_history = torch.zeros(
            num_steps + 1, num_dofs, dtype=dtype, device=device
        )
        u_history[0] = u.clone()

    # Build system matrix: A = M/dt + K
    # We need to convert sparse matrices to dense for now
    # TODO: Implement sparse arithmetic or use sparse solvers
    M_dense = M.to_dense()
    K_dense = K.to_dense()
    A = M_dense / dt + K_dense

    # Time stepping loop (implicit Euler)
    for step in range(num_steps):
        # Build RHS: b = M/dt @ u^n + f
        rhs = (M_dense / dt) @ u + f

        # Apply Dirichlet boundary conditions to the system A @ u = rhs
        # Use elimination method for accuracy
        A_bc, rhs_bc = _apply_dirichlet_to_dense(
            A, rhs, dirichlet_dofs, dirichlet_vals
        )

        # Solve the linear system
        u = torch.linalg.solve(A_bc, rhs_bc)

        # Store if returning all time steps
        if return_all:
            u_history[step + 1] = u.clone()

    if return_all:
        return u_history
    else:
        return u


def _apply_dirichlet_to_dense(
    matrix: Tensor,
    vector: Tensor,
    dofs: Tensor,
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Dirichlet BCs to a dense system using elimination.

    Parameters
    ----------
    matrix : Tensor
        Dense system matrix, shape (n, n).
    vector : Tensor
        RHS vector, shape (n,).
    dofs : Tensor
        DOF indices to constrain.
    values : Tensor
        Prescribed values.

    Returns
    -------
    tuple[Tensor, Tensor]
        Modified matrix and vector.
    """
    if dofs.numel() == 0:
        return matrix, vector

    A = matrix.clone()
    b = vector.clone()

    # Modify RHS for elimination
    b = b - A[:, dofs] @ values

    # Zero out rows and columns
    A[dofs, :] = 0.0
    A[:, dofs] = 0.0

    # Set diagonal to 1
    A[dofs, dofs] = 1.0

    # Set RHS to prescribed values
    b[dofs] = values

    return A, b


def _compute_coords_simplex(
    element_vertices: Tensor,
    quad_points: Tensor,
    dim: int,
) -> tuple[Tensor, Tensor]:
    """Compute physical coordinates and Jacobian determinant for simplex elements.

    Parameters
    ----------
    element_vertices : Tensor
        Element vertex coordinates, shape (num_elements, nodes_per_element, dim).
    quad_points : Tensor
        Quadrature points in reference coordinates, shape (num_quad, dim).
    dim : int
        Spatial dimension.

    Returns
    -------
    physical_coords : Tensor
        Physical coordinates at quadrature points, shape (num_elements, num_quad, dim).
    abs_det_J : Tensor
        Absolute value of Jacobian determinant, shape (num_elements,).
    """
    # For simplex, Jacobian is constant within element
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

    det_J = torch.linalg.det(J)
    abs_det_J = torch.abs(det_J)

    # Physical coordinates: x = v0 + J @ xi
    v0_expanded = element_vertices[:, 0, :].unsqueeze(1)  # (E, 1, dim)
    physical_coords = v0_expanded + torch.einsum("eij,qj->eqi", J, quad_points)

    return physical_coords, abs_det_J


def _compute_coords_tensor_product(
    element_vertices: Tensor,
    quad_points: Tensor,
    dim: int,
    element_type: str,
) -> tuple[Tensor, Tensor]:
    """Compute physical coordinates and Jacobian determinant for tensor product elements.

    Parameters
    ----------
    element_vertices : Tensor
        Element vertex coordinates, shape (num_elements, nodes_per_element, dim).
    quad_points : Tensor
        Quadrature points in reference coordinates, shape (num_quad, dim).
    dim : int
        Spatial dimension.
    element_type : str
        Element type ("quad" or "hexahedron").

    Returns
    -------
    physical_coords : Tensor
        Physical coordinates at quadrature points, shape (num_elements, num_quad, dim).
    abs_det_J : Tensor
        Absolute value of Jacobian determinant, shape (num_elements, num_quad).
    """
    # Use linear geometry mapping
    geom_order = 1
    basis_geom = lagrange_basis(
        element_type, geom_order, quad_points
    )  # (Q, nodes_geom)
    grad_geom = lagrange_basis_gradient(element_type, geom_order, quad_points)
    nodes_per_geom = grad_geom.shape[1]

    # Use corner vertices for geometry
    geom_vertices = element_vertices[
        :, :nodes_per_geom, :
    ]  # (E, nodes_geom, dim)

    # Compute physical coordinates: x = sum_a N_a(xi) * v_a
    physical_coords = torch.einsum(
        "qa,eai->eqi", basis_geom, geom_vertices
    )  # (E, Q, dim)

    # Compute Jacobian: J[e,q,i,j] = sum_a vertex[e,a,i] * grad_geom[q,a,j]
    J = torch.einsum(
        "eai,qaj->eqij", geom_vertices, grad_geom
    )  # (E, Q, dim, dim)

    det_J = torch.linalg.det(J)  # (E, Q)
    abs_det_J = torch.abs(det_J)

    return physical_coords, abs_det_J
