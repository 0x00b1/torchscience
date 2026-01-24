# tests/torchscience/integration/initial_value_problem/test__runge_kutta_4.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration import runge_kutta_4


class TestRungeKutta4Basic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = runge_kutta_4(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0]))
        # RK4 is 4th order, very accurate
        assert torch.allclose(y_final, expected, rtol=1e-5)

    def test_more_accurate_than_midpoint(self):
        """RK4 should be more accurate than midpoint for same step size"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        expected = torch.exp(torch.tensor([-1.0]))

        from torchscience.integration import midpoint

        y_midpoint, _ = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.1)
        y_rk4, _ = runge_kutta_4(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        error_midpoint = (y_midpoint - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()

        assert error_rk4 < error_midpoint

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0])
        y_final, interp = runge_kutta_4(
            oscillator, y0, t_span=(0.0, 2 * torch.pi), dt=0.1
        )

        # After one period, should return close to initial state
        assert torch.allclose(y_final, y0, atol=1e-3)

    def test_interpolant_trajectory(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        t_query = torch.linspace(0, 1, 20)
        trajectory = interp(t_query)
        assert trajectory.shape == (20, 1)

        # Should be monotonically decreasing
        for i in range(19):
            assert trajectory[i, 0] > trajectory[i + 1, 0]


class TestRungeKutta4Autograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    def test_gradcheck(self):
        """Verify gradients are correct"""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def solve(y0):
            y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 0.5), dt=0.1)
            return y_final

        assert torch.autograd.gradcheck(solve, (y0,), raise_exception=True)


class TestRungeKutta4TensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )
        state_final, _ = runge_kutta_4(f, state0, t_span=(0.0, 1.0), dt=0.1)

        assert isinstance(state_final, TensorDict)


class TestRungeKutta4Complex:
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(-1j * torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-5)


class TestRungeKutta4SciPy:
    def test_matches_scipy(self):
        scipy = pytest.importorskip("scipy")
        from scipy.integrate import solve_ivp

        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 2.0)

        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = runge_kutta_4(f_torch, y0_torch, t_span, dt=0.01)

        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="RK45", max_step=0.01
        )

        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=1e-3,
        )
