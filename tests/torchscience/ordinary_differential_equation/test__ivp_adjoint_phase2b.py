"""Phase 2b tests for key differentiator features."""

import torch

from torchscience.ordinary_differential_equation import (
    adjoint,
    dormand_prince_5,
)


class TestImplicitAdjoint:
    """Test implicit adjoint for stiff systems."""

    def test_implicit_adjoint_basic(self):
        """Test implicit adjoint option is accepted."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def stiff_decay(t, y):
            return -1000 * theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "implicit"},
        )

        y_final, _ = solver(stiff_decay, y0, t_span=(0.0, 0.01))
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_implicit_adjoint_gradient_accuracy(self):
        """Test implicit adjoint gradient has correct sign and order of magnitude."""
        theta = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Implicit adjoint
        solver_implicit = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "implicit", "n_steps": 200},
        )
        y_implicit, _ = solver_implicit(decay, y0.clone(), t_span=(0.0, 1.0))
        y_implicit.sum().backward()

        # For y' = -theta * y with y0=1, the solution is y = exp(-theta*t)
        # At t=1: y_final = exp(-2) ≈ 0.135
        # The gradient dL/dtheta where L = y_final = exp(-theta)
        # dL/dtheta = -t * exp(-theta*t) = -1 * exp(-2) ≈ -0.135

        # Check gradient is finite and has correct sign (negative)
        assert torch.isfinite(theta.grad)
        assert theta.grad < 0  # Should be negative

        # Check order of magnitude - should be around -0.135
        # Allow fairly loose tolerance since implicit method is approximate
        assert theta.grad.abs() > 0.01
        assert theta.grad.abs() < 1.0

    def test_implicit_stiff_system(self):
        """Test implicit adjoint on stiff system."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        # Stiff system with eigenvalues -1 and -100
        def stiff(t, y):
            return theta * torch.stack([-y[0], -100 * y[1]])

        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "implicit", "n_steps": 50},
        )

        y_final, _ = solver(stiff, y0, t_span=(0.0, 0.1))
        y_final.sum().backward()

        assert torch.isfinite(theta.grad)
        assert theta.grad.abs() > 1e-10  # Not vanishing


class TestSparseJacobian:
    """Test sparse Jacobian exploitation."""

    def test_sparse_jacobian_basic(self):
        """Test sparse Jacobian option is accepted."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        # Tridiagonal system (sparse Jacobian)
        def tridiag_system(t, y):
            n = y.shape[0]
            dy = torch.zeros_like(y)
            dy[0] = -2 * y[0] + y[1]
            for i in range(1, n - 1):
                dy[i] = y[i - 1] - 2 * y[i] + y[i + 1]
            dy[-1] = y[-2] - 2 * y[-1]
            return theta * dy

        y0 = torch.randn(20, dtype=torch.float64)

        # Define sparsity pattern (tridiagonal)
        n = 20
        sparsity = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            sparsity[i, i] = True
            if i > 0:
                sparsity[i, i - 1] = True
            if i < n - 1:
                sparsity[i, i + 1] = True

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"sparsity_pattern": sparsity, "n_steps": 20},
        )

        y_final, _ = solver(tridiag_system, y0, t_span=(0.0, 0.1))
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_sparse_coloring(self):
        """Test graph coloring for sparse patterns."""
        from torchscience.ordinary_differential_equation._sparse_jacobian import (
            compute_coloring,
        )

        # Tridiagonal pattern needs only 3 colors
        n = 10
        sparsity = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            sparsity[i, i] = True
            if i > 0:
                sparsity[i, i - 1] = True
            if i < n - 1:
                sparsity[i, i + 1] = True

        colors = compute_coloring(sparsity)
        n_colors = colors.max().item() + 1

        # Tridiagonal should need at most 3 colors
        assert n_colors <= 3


class TestMixedPrecisionAdjoint:
    """Test mixed precision adjoint (forward float32, accumulation float64)."""

    def test_mixed_precision_basic(self):
        """Test mixed precision option is accepted."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float32)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"mixed_precision": True},
        )

        y_final, _ = solver(decay, y0, t_span=(0.0, 1.0))
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)
        # Gradient should still be float32
        assert theta.grad.dtype == torch.float32

    def test_mixed_precision_gradient_accuracy(self):
        """Test that mixed precision produces reasonable gradients."""
        theta = torch.tensor([2.0], requires_grad=True, dtype=torch.float32)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float32)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"mixed_precision": True, "n_steps": 100},
        )

        y_final, _ = solver(decay, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        # Gradient should be negative (more decay = lower final value)
        assert theta.grad < 0
        assert torch.isfinite(theta.grad)


class TestCNFUtilities:
    """Test Continuous Normalizing Flow utilities."""

    def test_trace_estimator_hutchinson(self):
        """Test Hutchinson trace estimator."""
        from torchscience.ordinary_differential_equation import (
            hutchinson_trace,
        )

        # Simple linear function: f(x) = Ax, trace(A) = sum of eigenvalues
        A = torch.randn(10, 10, dtype=torch.float64)
        expected_trace = A.trace()

        def f(x):
            return A @ x

        x = torch.randn(10, dtype=torch.float64, requires_grad=True)

        # Estimate trace
        trace_est = hutchinson_trace(f, x, n_samples=1000)

        # Should be close to true trace
        assert (
            abs(trace_est - expected_trace) < 0.5 * abs(expected_trace) + 1.0
        )

    def test_exact_trace(self):
        """Test exact trace computation."""
        from torchscience.ordinary_differential_equation import exact_trace

        # Simple linear function: f(x) = Ax
        A = torch.randn(5, 5, dtype=torch.float64)
        expected_trace = A.trace()

        def f(x):
            return A @ x

        x = torch.randn(5, dtype=torch.float64, requires_grad=True)

        # Compute exact trace
        trace_exact = exact_trace(f, x)

        # Should be very close to true trace
        assert torch.allclose(trace_exact, expected_trace, rtol=1e-4)

    def test_cnf_dynamics_wrapper(self):
        """Test CNF dynamics wrapper that computes divergence."""
        from torchscience.ordinary_differential_equation import cnf_dynamics

        # Simple dynamics: dz/dt = -z (contraction)
        def velocity(t, z):
            return -z

        z0 = torch.randn(5, dtype=torch.float64)
        log_p0 = torch.tensor(0.0, dtype=torch.float64)

        # Wrap in CNF dynamics
        cnf_f = cnf_dynamics(velocity, trace_method="exact")

        # State is [z, log_p]
        state = torch.cat([z0, log_p0.unsqueeze(0)])
        dstate = cnf_f(0.0, state)

        # Should have velocity for z and divergence for log_p
        assert dstate.shape == state.shape
        # For f(z) = -z, Jacobian J = -I, trace(J) = -n
        # d(log_p)/dt = -div(f) = -trace(J) = -(-n) = +n = 5
        # (Contraction increases density, so log_p increases)
        expected_d_log_p = 5.0
        assert abs(dstate[-1] - expected_d_log_p) < 0.1

    def test_cnf_solve_with_log_prob(self):
        """Test full CNF solve that tracks log probability."""
        from torchscience.ordinary_differential_equation import (
            cnf_dynamics,
            solve_ivp,
        )

        # Linear contraction: z(t) = z0 * exp(-t)
        def velocity(t, z):
            return -z

        z0 = torch.randn(3, dtype=torch.float64)
        log_p0 = torch.tensor(0.0, dtype=torch.float64)

        cnf_f = cnf_dynamics(velocity, trace_method="exact")
        state0 = torch.cat([z0, log_p0.unsqueeze(0)])

        result = solve_ivp(cnf_f, state0, t_span=(0.0, 1.0))

        z_final = result.y_final[:-1]
        log_p_final = result.y_final[-1]

        # z should decay: z_final = z0 * exp(-1)
        expected_z = z0 * torch.exp(torch.tensor(-1.0))
        assert torch.allclose(z_final, expected_z, rtol=1e-3)

        # log_p change = integral of -div(f) = integral of -(-3) = +3
        # For f(z) = -z in 3D: div(f) = trace(J) = trace(-I) = -3
        # d(log_p)/dt = -div(f) = +3, so over t=0 to t=1: delta_log_p = +3
        # (Contraction increases density)
        expected_log_p_change = 3.0
        assert abs(log_p_final - log_p0 - expected_log_p_change) < 0.1
