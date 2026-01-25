"""Tests for physics solvers in finite element method module."""

import pytest
import torch
from torch import Tensor

from torchscience.geometry.mesh import rectangle_mesh


class TestSolvePoisson:
    """Tests for solve_poisson function."""

    def test_constant_source_homogeneous_bc(self) -> None:
        """Test solving -nabla^2 u = 1 with u = 0 on boundary.

        The solution should be positive in the interior and zero on the boundary.
        For a unit square, the maximum value is approximately 0.073 (at the center).
        """
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])
        u = solve_poisson(mesh, source=1.0)

        # Check shape
        assert u.shape == (mesh.num_vertices,)

        # Solution should be non-negative (minimum principle)
        assert u.min() >= -1e-10

        # Solution should be strictly positive in interior
        assert u.max() > 0

        # For unit square with f=1, max value is about 0.073
        # Allow some tolerance for numerical approximation
        assert 0.05 < u.max() < 0.1

    def test_custom_diffusivity(self) -> None:
        """Test with non-unit diffusivity coefficient.

        For -nabla . (kappa nabla u) = f, scaling kappa by factor c
        scales the solution by 1/c.
        """
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Solve with kappa = 1
        u1 = solve_poisson(mesh, source=1.0, diffusivity=1.0)

        # Solve with kappa = 2
        u2 = solve_poisson(mesh, source=1.0, diffusivity=2.0)

        # u2 should be approximately u1 / 2
        assert torch.allclose(u2, u1 / 2, rtol=1e-5, atol=1e-10)

    def test_nonhomogeneous_dirichlet_bc(self) -> None:
        """Test with non-zero Dirichlet boundary conditions.

        Solve -nabla^2 u = 0 with u = 1 on boundary.
        The solution should be constant u = 1 everywhere.
        """
        from torchscience.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_poisson,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        bc_dofs = boundary_dofs(mesh, dm)

        u = solve_poisson(
            mesh,
            source=0.0,
            dirichlet_dofs=bc_dofs,
            dirichlet_values=1.0,
        )

        # Solution should be constant 1 everywhere
        assert torch.allclose(u, torch.ones_like(u), rtol=1e-5, atol=1e-10)

    def test_callable_source(self) -> None:
        """Test with a callable source term.

        Use a manufactured solution approach:
        Let u_exact = sin(pi*x) * sin(pi*y)
        Then f = -nabla^2 u = 2*pi^2 * sin(pi*x) * sin(pi*y)

        With u = 0 on boundary (which sin(pi*x)*sin(pi*y) satisfies),
        the numerical solution should approximate u_exact.
        """
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(20, 20, bounds=[[0.0, 1.0], [0.0, 1.0]])

        def source_fn(coords: Tensor) -> Tensor:
            """Source term: f = 2*pi^2 * sin(pi*x) * sin(pi*y)."""
            x, y = coords[:, 0], coords[:, 1]
            return (
                2
                * torch.pi**2
                * torch.sin(torch.pi * x)
                * torch.sin(torch.pi * y)
            )

        u = solve_poisson(mesh, source=source_fn)

        # Compute exact solution at vertices
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        u_exact = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        # Check L2 relative error is small (should be O(h^2) for P1 elements)
        l2_error = torch.sqrt(torch.sum((u - u_exact) ** 2))
        l2_norm = torch.sqrt(torch.sum(u_exact**2))
        relative_error = l2_error / l2_norm

        # For a 20x20 mesh, h ~= 0.05, so h^2 ~= 0.0025
        # Allow some margin for the discrete approximation
        assert relative_error < 0.05

    def test_manufactured_polynomial_solution(self) -> None:
        """Test with manufactured polynomial solution.

        For P1 elements, quadratic solutions are not exactly representable,
        but linear solutions should be exact.

        Let u_exact = x + y (linear function)
        Then -nabla^2 u = 0

        With u = x + y on boundary, the solution should be exact.
        """
        from torchscience.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_poisson,
        )

        mesh = rectangle_mesh(5, 5, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        bc_dofs = boundary_dofs(mesh, dm)

        # Compute exact values at boundary DOFs
        bc_coords = mesh.vertices[bc_dofs]
        bc_values = bc_coords[:, 0] + bc_coords[:, 1]

        u = solve_poisson(
            mesh,
            source=0.0,
            dirichlet_dofs=bc_dofs,
            dirichlet_values=bc_values,
        )

        # Compute exact solution at all vertices
        u_exact = mesh.vertices[:, 0] + mesh.vertices[:, 1]

        # Linear solution should be exactly representable
        assert torch.allclose(u, u_exact, rtol=1e-5, atol=1e-10)

    def test_cg_solver(self) -> None:
        """Test using conjugate gradient solver."""
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Solve with direct solver
        u_direct = solve_poisson(mesh, source=1.0, solver="direct")

        # Solve with CG solver
        u_cg = solve_poisson(mesh, source=1.0, solver="cg")

        # Both should give same result
        assert torch.allclose(u_direct, u_cg, rtol=1e-4, atol=1e-8)

    def test_p2_elements(self) -> None:
        """Test with P2 (quadratic) elements.

        Higher-order elements should give better accuracy for the same mesh.
        """
        from torchscience.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_poisson,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        def source_fn(coords: Tensor) -> Tensor:
            """Source term: f = 2*pi^2 * sin(pi*x) * sin(pi*y)."""
            x, y = coords[:, 0], coords[:, 1]
            return (
                2
                * torch.pi**2
                * torch.sin(torch.pi * x)
                * torch.sin(torch.pi * y)
            )

        # Solve with P1 elements
        u_p1 = solve_poisson(mesh, source=source_fn, order=1)

        # Solve with P2 elements
        dm_p2 = dof_map(mesh, order=2)
        bc_dofs_p2 = boundary_dofs(mesh, dm_p2)
        u_p2 = solve_poisson(
            mesh,
            source=source_fn,
            order=2,
            dirichlet_dofs=bc_dofs_p2,
            dirichlet_values=0.0,
        )

        # Compute exact solution at vertices (P1 DOFs)
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        u_exact_vertices = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        # P1 error at vertices
        l2_error_p1 = torch.sqrt(torch.sum((u_p1 - u_exact_vertices) ** 2))
        l2_norm = torch.sqrt(torch.sum(u_exact_vertices**2))

        # P2 error at vertices (first num_vertices DOFs)
        u_p2_vertices = u_p2[: mesh.num_vertices]
        l2_error_p2 = torch.sqrt(
            torch.sum((u_p2_vertices - u_exact_vertices) ** 2)
        )

        # P2 should have smaller error than P1 on same mesh
        assert l2_error_p2 < l2_error_p1

    def test_tensor_source(self) -> None:
        """Test with tensor source term (nodal values)."""
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Create nodal source values
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        source_tensor = (
            2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        )

        u = solve_poisson(mesh, source=source_tensor)

        # Solution should be non-negative and bounded
        assert u.min() >= -0.1
        assert u.max() < 1.0

    def test_invalid_solver(self) -> None:
        """Test that invalid solver raises an error."""
        from torchscience.finite_element_method import solve_poisson

        mesh = rectangle_mesh(5, 5, bounds=[[0.0, 1.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="solver"):
            solve_poisson(mesh, source=1.0, solver="invalid")

    def test_default_boundary_conditions(self) -> None:
        """Test that default uses all boundary DOFs with zero values."""
        from torchscience.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_poisson,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        bc_dofs = boundary_dofs(mesh, dm)

        # Solve with default BCs
        u_default = solve_poisson(mesh, source=1.0)

        # Solve with explicit homogeneous BCs on all boundary
        u_explicit = solve_poisson(
            mesh,
            source=1.0,
            dirichlet_dofs=bc_dofs,
            dirichlet_values=0.0,
        )

        # Should be the same
        assert torch.allclose(u_default, u_explicit, rtol=1e-10, atol=1e-10)
