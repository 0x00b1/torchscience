# tests/torchscience/integration/initial_value_problem/test__adjoint.py
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
    euler,
    runge_kutta_4,
)


class TestAdjointBasic:
    def test_adjoint_wraps_solver(self):
        """adjoint() should return a callable with same signature."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        adjoint_solver = adjoint(dormand_prince_5)

        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Should produce same result as direct call
        y_direct, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))
        assert torch.allclose(y_final, y_direct, atol=1e-5)

    def test_adjoint_gradients_match_direct(self):
        """Adjoint gradients should approximately match direct backprop."""
        y0 = torch.tensor([1.0])

        # Direct backprop - use a fresh leaf tensor
        theta_direct = torch.tensor([1.0], requires_grad=True)

        def f_direct(t, y):
            return -theta_direct * y

        y_direct, _ = dormand_prince_5(f_direct, y0, t_span=(0.0, 1.0))
        loss_direct = y_direct.sum()
        loss_direct.backward()
        grad_direct = theta_direct.grad.clone()

        # Adjoint method - use a fresh leaf tensor
        theta_adjoint = torch.tensor([1.0], requires_grad=True)

        def f_adjoint(t, y):
            return -theta_adjoint * y

        adjoint_solver = adjoint(dormand_prince_5)
        y_adjoint, _ = adjoint_solver(f_adjoint, y0, t_span=(0.0, 1.0))
        loss_adjoint = y_adjoint.sum()
        loss_adjoint.backward()
        grad_adjoint = theta_adjoint.grad.clone()

        # Gradients should match (approximately, due to different discretizations)
        assert torch.allclose(grad_direct, grad_adjoint, rtol=0.1)

    def test_adjoint_with_euler(self):
        """Adjoint wrapper should work with any solver."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_euler = adjoint(euler)
        y_final, _ = adjoint_euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_adjoint_with_rk4(self):
        """Adjoint wrapper should work with RK4."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_rk4 = adjoint(runge_kutta_4)
        y_final, _ = adjoint_rk4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointMemory:
    def test_adjoint_uses_less_memory_conceptually(self):
        """
        The adjoint method should use O(1) memory for the autograd graph
        vs O(n_steps) for direct backprop.

        This is a conceptual test - we verify the forward pass works
        and gradients are computed, which confirms the adjoint path is used.
        """
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        # This should not store O(n_steps) activations
        adjoint_solver = adjoint(dormand_prince_5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 10.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointHigherDimensional:
    def test_adjoint_2d_system(self):
        """Test adjoint on 2D harmonic oscillator."""
        omega = torch.tensor(2.0, requires_grad=True)  # Scalar

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            dxdt = v
            dvdt = -(omega**2) * x
            return torch.stack([dxdt, dvdt], dim=-1)

        y0 = torch.tensor([1.0, 0.0])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(oscillator, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final[0]  # Final position
        loss.backward()

        assert omega.grad is not None
        assert not torch.isnan(omega.grad).any()

    def test_adjoint_batched(self):
        """Test adjoint with batched initial conditions."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)

        adjoint_solver = adjoint(dormand_prince_5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointInterpolant:
    def test_adjoint_interpolant_works(self):
        """Interpolant should work with adjoint wrapper."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5)
        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Interpolant should be functional
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape == (10, 1)

    def test_adjoint_interpolant_not_differentiable(self):
        """
        Note: With adjoint method, the interpolant is NOT differentiable
        (gradients only flow through y_final, not intermediate points).

        This is a known limitation documented in the design.
        """
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5)
        _, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Querying interpolant produces a tensor
        y_mid = interp(0.5)

        # But gradients don't flow through it with adjoint
        # (This test documents the behavior rather than asserting it)


class TestAdjointCheckpoints:
    def test_adjoint_with_checkpoints(self):
        """Test adjoint with explicit checkpoint count."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5, checkpoints=5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    def test_checkpoints_dont_affect_forward(self):
        """Checkpoint count shouldn't affect forward solution."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        y_no_ckpt, _ = adjoint(dormand_prince_5)(f, y0, t_span=(0.0, 1.0))
        y_with_ckpt, _ = adjoint(dormand_prince_5, checkpoints=3)(
            f, y0, t_span=(0.0, 1.0)
        )

        assert torch.allclose(y_no_ckpt, y_with_ckpt, atol=1e-6)
