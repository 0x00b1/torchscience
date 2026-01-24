# tests/torchscience/integration/initial_value_problem/test__full_integration.py
"""Full integration tests covering all features together."""

import pytest
import torch
from tensordict import TensorDict

from torchscience.integration import (
    adjoint,
    backward_euler,
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


class TestFullPipeline:
    """Test complete workflows combining multiple features."""

    def test_neural_ode_training_loop(self):
        """Simulate a Neural ODE training loop."""
        torch.manual_seed(42)

        # Network
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        def dynamics(t, y):
            return net(y)

        # Training data
        y0 = torch.randn(10, 2)  # Batch of 10
        targets = y0 * 0.5  # Shrink by half

        # Training loop
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()

            adjoint_solver = adjoint(runge_kutta_4)
            y_final, _ = adjoint_solver(
                dynamics, y0, t_span=(0.0, 1.0), dt=0.1
            )

            loss = (y_final - targets).pow(2).mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_tensordict_with_adjoint(self):
        """TensorDict state with adjoint method."""
        omega = torch.tensor([2.0], requires_grad=True)

        def dynamics(t, state):
            return TensorDict({"x": state["v"], "v": -(omega**2) * state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )

        adjoint_solver = adjoint(dormand_prince_5)
        state_final, _ = adjoint_solver(dynamics, state0, t_span=(0.0, 1.0))

        loss = state_final["x"].sum()
        loss.backward()

        assert omega.grad is not None

    @pytest.mark.xfail(
        reason="Interpolant bounds check uses t.min() which doesn't support complex dtype"
    )
    def test_complex_with_adjoint(self):
        """Complex-valued ODE with adjoint method."""
        theta = torch.tensor([1.0], requires_grad=True)

        def dynamics(t, y):
            return -theta.to(torch.complex128) * 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(dynamics, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.abs().sum()
        loss.backward()

        assert theta.grad is not None

    @pytest.mark.xfail(
        reason="Interpolant bounds check assumes t_min < t_max, fails for backward integration"
    )
    def test_backward_integration_with_adjoint(self):
        """Backward time integration with adjoint."""
        theta = torch.tensor([1.0], requires_grad=True)

        def dynamics(t, y):
            return -theta * y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])

        adjoint_solver = adjoint(runge_kutta_4)
        y0_recovered, _ = adjoint_solver(
            dynamics, y1, t_span=(1.0, 0.0), dt=0.01
        )

        loss = (y0_recovered - 1.0).pow(2)
        loss.backward()

        assert theta.grad is not None


class TestAllSolversAllFeatures:
    """Ensure all solvers support all features consistently."""

    SOLVERS = [
        ("euler", euler, {"dt": 0.01}),
        ("midpoint", midpoint, {"dt": 0.01}),
        ("runge_kutta_4", runge_kutta_4, {"dt": 0.01}),
        ("dormand_prince_5", dormand_prince_5, {}),
        ("backward_euler", backward_euler, {"dt": 0.01}),
    ]

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_tensor_state(self, name, solver, kwargs):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        assert y_final.shape == y0.shape
        assert interp(0.5).shape == y0.shape

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_tensordict_state(self, name, solver, kwargs):
        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0])})
        state_final, interp = solver(f, state0, t_span=(0.0, 1.0), **kwargs)

        assert isinstance(state_final, TensorDict)

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_batched_state(self, name, solver, kwargs):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        assert y_final.shape == (3, 1)

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_gradient_support(self, name, solver, kwargs):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_adjoint_wrapper(self, name, solver, kwargs):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(solver)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
