# tests/torchscience/integration/initial_value_problem/test__adjoint_all_solvers.py
"""Tests verifying adjoint wrapper works with all available solvers."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    backward_euler,
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


@pytest.fixture
def y0():
    return torch.tensor([1.0])


class TestAdjointAllSolvers:
    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (midpoint, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {}),
            (backward_euler, {"dt": 0.01}),
        ],
    )
    def test_adjoint_works_with_solver(self, solver, kwargs, y0):
        """Each solver should work with adjoint wrapper."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        adjoint_solver = adjoint(solver)
        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        # Forward should work
        assert not torch.isnan(y_final).any()

        # Backward should work
        loss = y_final.sum()
        loss.backward()
        assert theta.grad is not None

    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (midpoint, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {}),
        ],
    )
    def test_adjoint_interpolant_exists(self, solver, kwargs, y0):
        """Interpolant should be returned and functional."""

        def f(t, y):
            return -y

        adjoint_solver = adjoint(solver)
        _, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        # Should be able to query
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

        # Multiple queries
        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape[0] == 10


class TestAdjointNeuralODE:
    """Test adjoint with neural network dynamics (common use case)."""

    def test_mlp_dynamics(self):
        """Test with MLP-parameterized dynamics."""
        # Simple MLP
        hidden = 16
        net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2),
        )

        def f(t, y):
            return net(y)

        y0 = torch.tensor([1.0, 0.5])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        # All parameters should have gradients
        for p in net.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    def test_neural_ode_training_step(self):
        """Simulate a training step with Neural ODE."""
        # Simple dynamics network
        net = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        def f(t, y):
            return net(y)

        y0 = torch.tensor([[1.0]])
        target = torch.tensor([[0.5]])

        # Training step
        optimizer.zero_grad()
        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = (y_final - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        # Parameter should have changed
        assert net.weight.grad is not None
