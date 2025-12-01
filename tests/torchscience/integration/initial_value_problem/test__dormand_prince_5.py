import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import (
    MaxStepsExceeded,
    StepSizeError,
    dormand_prince_5,
)


class TestDormandPrince5Basic:
    def test_exponential_decay(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))

        expected = torch.exp(torch.tensor([-5.0]))
        assert torch.allclose(y_final, expected, rtol=1e-4)

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)  # dx/dt = v, dv/dt = -x

        y0 = torch.tensor([1.0, 0.0])  # x=1, v=0
        y_final, interp = dormand_prince_5(
            oscillator, y0, t_span=(0.0, 2 * torch.pi)
        )

        # After one period, should return to initial state
        assert torch.allclose(y_final, y0, atol=1e-3)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Interpolant should be callable
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # At t=0, should match y0
        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        # At t=1, should match y_final
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)

    def test_multiple_time_queries(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape == (10, 1)

        # Should be monotonically decreasing
        for i in range(9):
            assert trajectory[i, 0] > trajectory[i + 1, 0]


class TestDormandPrince5Complex:
    def test_complex_exponential_decay(self):
        """Test complex ODE: dy/dt = -i*y, y(0) = 1 => y(t) = exp(-i*t)"""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, torch.pi))

        expected = torch.exp(-1j * torch.tensor(torch.pi, dtype=torch.float64))
        assert torch.allclose(y_final, expected.unsqueeze(0), atol=1e-4)

    def test_schrodinger_like(self):
        """Test Schrodinger-like equation: dy/dt = -i*H*y"""
        H = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

        def f(t, y):
            return -1j * H @ y

        psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, psi0, t_span=(0.0, 1.0))

        # Check normalization preserved
        norm = torch.abs(y_final).pow(2).sum()
        assert torch.allclose(
            norm, torch.tensor(1.0, dtype=torch.float64), atol=1e-5
        )

    def test_complex_gradcheck(self):
        """Verify gradients work for complex states"""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta.to(y.dtype) * 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Can compute gradients through complex operations
        loss = y_final.abs().sum()
        loss.backward()

        assert theta.grad is not None


class TestDormandPrince5TensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )
        state_final, interp = dormand_prince_5(
            f, state0, t_span=(0.0, 2 * torch.pi)
        )

        # After one period, should return to initial state
        assert isinstance(state_final, TensorDict)
        assert torch.allclose(state_final["x"], state0["x"], atol=1e-3)
        assert torch.allclose(state_final["v"], state0["v"], atol=1e-3)

    def test_nested_tensordict(self):
        def f(t, state):
            return TensorDict(
                {
                    "robot": TensorDict(
                        {
                            "q": state["robot"]["dq"],
                            "dq": -state["robot"]["q"],
                        }
                    )
                }
            )

        state0 = TensorDict(
            {
                "robot": TensorDict(
                    {
                        "q": torch.tensor([1.0, 0.0]),
                        "dq": torch.tensor([0.0, 1.0]),
                    }
                )
            }
        )

        state_final, interp = dormand_prince_5(f, state0, t_span=(0.0, 1.0))

        assert isinstance(state_final, TensorDict)
        assert "robot" in state_final.keys()
        assert state_final["robot", "q"].shape == (2,)

    def test_tensordict_interpolant(self):
        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0])})
        _, interp = dormand_prince_5(f, state0, t_span=(0.0, 1.0))

        state_mid = interp(0.5)
        assert isinstance(state_mid, TensorDict)
        assert "x" in state_mid.keys()


class TestDormandPrince5ErrorHandling:
    def test_max_steps_exceeded_throws(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        with pytest.raises(MaxStepsExceeded):
            dormand_prince_5(f, y0, t_span=(0.0, 1000.0), max_steps=5)

    def test_max_steps_exceeded_no_throw(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        y_final, interp = dormand_prince_5(
            f, y0, t_span=(0.0, 1000.0), max_steps=5, throw=False
        )

        assert torch.isnan(y_final).all()
        assert interp.success is not None
        assert not interp.success.all()

    def test_step_size_error_throws(self):
        # Stiff problem that requires tiny steps
        def stiff(t, y):
            return -1000 * y

        y0 = torch.tensor([1.0])

        with pytest.raises(StepSizeError):
            dormand_prince_5(stiff, y0, t_span=(0.0, 1.0), dt_min=0.1)

    def test_interpolant_out_of_bounds(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)


class TestDormandPrince5BackwardIntegration:
    def test_backward_exponential(self):
        """Integrate backwards: y(1) = e^-1 => y(0) = 1"""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, interp = dormand_prince_5(f, y1, t_span=(1.0, 0.0))

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=1e-4)

    def test_backward_interpolant_range(self):
        """Interpolant should cover [0, 1] regardless of direction"""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = dormand_prince_5(f, y0, t_span=(1.0, 0.0))

        # Should be able to query anywhere in [0, 1]
        y_mid = interp(0.5)
        assert not torch.isnan(y_mid).any()


scipy = pytest.importorskip("scipy")
from scipy.integrate import solve_ivp


class TestDormandPrince5SciPyComparison:
    def test_exponential_decay_matches_scipy(self):
        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 5.0)

        # Solve with torchscience
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = dormand_prince_5(
            f_torch, y0_torch, t_span, rtol=1e-8, atol=1e-10
        )

        # Solve with scipy
        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="DOP853", rtol=1e-8, atol=1e-10
        )

        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=1e-5,
        )

    def test_lotka_volterra_matches_scipy(self):
        """Lotka-Volterra predator-prey model"""
        alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

        def f_torch(t, y):
            x, p = y[..., 0], y[..., 1]
            dx = alpha * x - beta * x * p
            dp = delta * x * p - gamma * p
            return torch.stack([dx, dp], dim=-1)

        def f_scipy(t, y):
            x, p = y
            dx = alpha * x - beta * x * p
            dp = delta * x * p - gamma * p
            return [dx, dp]

        y0_val = [10.0, 5.0]
        t_span = (0.0, 10.0)

        y0_torch = torch.tensor(y0_val, dtype=torch.float64)
        y_torch, _ = dormand_prince_5(
            f_torch, y0_torch, t_span, rtol=1e-8, atol=1e-10
        )

        sol_scipy = solve_ivp(
            f_scipy, t_span, y0_val, method="DOP853", rtol=1e-8, atol=1e-10
        )

        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=1e-4,
        )
