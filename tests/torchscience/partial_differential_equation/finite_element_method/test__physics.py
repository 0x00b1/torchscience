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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

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
        from torchscience.partial_differential_equation.finite_element_method import (
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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

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
        from torchscience.partial_differential_equation.finite_element_method import (
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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

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
        from torchscience.partial_differential_equation.finite_element_method import (
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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

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
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_poisson,
        )

        mesh = rectangle_mesh(5, 5, bounds=[[0.0, 1.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="solver"):
            solve_poisson(mesh, source=1.0, solver="invalid")

    def test_default_boundary_conditions(self) -> None:
        """Test that default uses all boundary DOFs with zero values."""
        from torchscience.partial_differential_equation.finite_element_method import (
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


class TestSolveHeat:
    """Tests for solve_heat function."""

    def test_constant_initial_no_source(self) -> None:
        """Test constant initial condition with no source.

        With u(t=0) = 1 everywhere (interior only, BC = 0 on boundary),
        the solution should decay over time towards the steady state (u = 0).
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Constant initial condition of 1.0 at all vertices
        initial = torch.ones(mesh.num_vertices, dtype=torch.float64)

        u = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            dt=0.01,
            num_steps=10,
        )

        # Final solution should be smaller than initial (decay to steady state)
        # Since BC = 0 on boundary, interior values should decrease
        assert u.shape == (mesh.num_vertices,)
        assert u.max() < initial.max()

    def test_steady_state_initial(self) -> None:
        """Test that steady-state initial condition remains unchanged.

        If the initial condition is already the steady-state solution
        (satisfies the elliptic problem), it should not change over time.
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
            solve_poisson,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Compute steady-state solution: -nabla^2 u = 1 with u = 0 on boundary
        u_steady = solve_poisson(mesh, source=1.0)

        # Use steady-state as initial condition with same source
        u = solve_heat(
            mesh,
            initial=u_steady,
            source=1.0,
            diffusivity=1.0,
            dt=0.01,
            num_steps=50,
        )

        # Solution should remain approximately at steady state
        assert torch.allclose(u, u_steady, rtol=1e-3, atol=1e-6)

    def test_decay_to_zero(self) -> None:
        """Test decay to zero with homogeneous BCs and no source.

        Starting from a non-zero initial condition, the solution should
        decay exponentially towards zero (the steady state).
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Initial condition: sin(pi*x) * sin(pi*y) (first eigenmode)
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        initial = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        u = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            dt=0.01,
            num_steps=100,
            return_all=True,
        )

        # Should return all time steps
        assert u.shape == (
            101,
            mesh.num_vertices,
        )  # num_steps + 1 (includes t=0)

        # Solution should decay: L2 norm at final time < L2 norm at initial time
        l2_initial = torch.sqrt(torch.sum(u[0] ** 2))
        l2_final = torch.sqrt(torch.sum(u[-1] ** 2))
        assert l2_final < 0.5 * l2_initial  # Should decay significantly

    def test_heating_with_source(self) -> None:
        """Test heating with a positive source term.

        Starting from zero, the solution should increase over time
        due to the positive heat source.
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Start from zero
        initial = torch.zeros(mesh.num_vertices, dtype=torch.float64)

        u = solve_heat(
            mesh,
            initial=initial,
            source=1.0,  # Positive source
            diffusivity=1.0,
            dt=0.01,
            num_steps=50,
        )

        # Solution should be positive in the interior (heating up)
        assert u.max() > 0.0

    def test_callable_initial_condition(self) -> None:
        """Test with callable initial condition."""
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        def initial_fn(coords: Tensor) -> Tensor:
            """Initial condition: sin(pi*x) * sin(pi*y)."""
            x, y = coords[:, 0], coords[:, 1]
            return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        u = solve_heat(
            mesh,
            initial=initial_fn,
            source=0.0,
            diffusivity=1.0,
            dt=0.01,
            num_steps=10,
        )

        assert u.shape == (mesh.num_vertices,)
        # Should be similar to but smaller than initial (due to diffusion)
        initial_values = initial_fn(mesh.vertices)
        assert u.max() < initial_values.max()

    def test_return_all_time_steps(self) -> None:
        """Test returning solutions at all time steps."""
        from torchscience.partial_differential_equation.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_heat,
        )

        mesh = rectangle_mesh(5, 5, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        bc_dofs = boundary_dofs(mesh, dm)

        initial = torch.ones(mesh.num_vertices, dtype=torch.float64)

        u_all = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            dt=0.01,
            num_steps=20,
            return_all=True,
        )

        # Should have num_steps + 1 time steps (including initial)
        assert u_all.shape == (21, mesh.num_vertices)

        # First row should be the initial condition with BCs applied
        # (boundary values should be 0, interior values should be 1)
        expected_initial = initial.clone()
        expected_initial[bc_dofs] = 0.0  # Dirichlet BC
        assert torch.allclose(u_all[0], expected_initial, rtol=1e-10)

    def test_energy_decay(self) -> None:
        """Test that energy (L2 norm squared) decays monotonically without source.

        For the heat equation with homogeneous BCs and no source,
        the total energy should decrease monotonically.
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Initial condition
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        initial = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        u_all = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            dt=0.001,  # Small time step for stability
            num_steps=50,
            return_all=True,
        )

        # Compute energy (L2 norm squared) at each time step
        energy = torch.sum(u_all**2, dim=1)

        # Energy should be monotonically decreasing
        energy_diff = energy[1:] - energy[:-1]
        assert torch.all(
            energy_diff <= 1e-10
        )  # Allow small numerical tolerance

    def test_custom_density(self) -> None:
        """Test with non-unit density coefficient.

        Higher density (rho*c) means more thermal inertia,
        so the solution should change more slowly.
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Initial condition
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        initial = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        # Solve with density = 1
        u1 = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            density=1.0,
            dt=0.01,
            num_steps=20,
        )

        # Solve with density = 2 (more thermal inertia)
        u2 = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            density=2.0,
            dt=0.01,
            num_steps=20,
        )

        # With higher density, solution should decay slower
        # (be closer to initial condition)
        l2_diff_1 = torch.sqrt(torch.sum((u1 - initial) ** 2))
        l2_diff_2 = torch.sqrt(torch.sum((u2 - initial) ** 2))
        assert l2_diff_2 < l2_diff_1

    def test_nonhomogeneous_dirichlet_bc(self) -> None:
        """Test with non-zero Dirichlet boundary conditions."""
        from torchscience.partial_differential_equation.finite_element_method import (
            boundary_dofs,
            dof_map,
            solve_heat,
        )

        mesh = rectangle_mesh(10, 10, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        bc_dofs = boundary_dofs(mesh, dm)

        # Start from zero, but boundary is held at 1.0
        initial = torch.zeros(mesh.num_vertices, dtype=torch.float64)

        u = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            dirichlet_dofs=bc_dofs,
            dirichlet_values=1.0,
            dt=0.01,
            num_steps=100,
        )

        # After many steps, interior should approach boundary value
        # (steady state with zero source and constant boundary is constant)
        # Interior values should be increasing towards 1.0
        assert u.min() > initial.min()

    def test_analytical_decay_rate(self) -> None:
        """Test decay rate matches analytical solution for first eigenmode.

        For u(x,y,t) = exp(-lambda*t) * sin(pi*x) * sin(pi*y)
        where lambda = 2*pi^2 (for unit square with kappa=1),
        the solution should decay exponentially.
        """
        from torchscience.partial_differential_equation.finite_element_method import (
            solve_heat,
        )

        mesh = rectangle_mesh(20, 20, bounds=[[0.0, 1.0], [0.0, 1.0]])

        # Initial condition: first eigenmode
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        initial = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        # Time parameters
        dt = 0.001
        num_steps = 100
        t_final = dt * num_steps

        u = solve_heat(
            mesh,
            initial=initial,
            source=0.0,
            diffusivity=1.0,
            dt=dt,
            num_steps=num_steps,
        )

        # Analytical decay factor: exp(-2*pi^2 * t)
        lambda_1 = 2 * torch.pi**2
        decay_factor = torch.exp(torch.tensor(-lambda_1 * t_final))

        # Numerical decay (compare L2 norms since boundary values are zero)
        l2_initial = torch.sqrt(torch.sum(initial**2))
        l2_final = torch.sqrt(torch.sum(u**2))
        numerical_decay = l2_final / l2_initial

        # Should match analytical decay rate within tolerance
        # (finite element discretization introduces some error)
        assert torch.abs(numerical_decay - decay_factor) < 0.1
