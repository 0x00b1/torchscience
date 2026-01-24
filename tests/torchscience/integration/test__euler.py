# tests/torchscience/integration/initial_value_problem/test__euler.py
import torch
from tensordict import TensorDict

from torchscience.integration import euler


class TestEulerBasic:
    def test_exponential_decay(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(decay, y0, t_span=(0.0, 1.0), dt=0.001)

        expected = torch.exp(torch.tensor([-1.0]))
        # Euler is 1st order, so we need small dt for good accuracy
        assert torch.allclose(y_final, expected, rtol=1e-2)

    def test_linear_ode(self):
        """dy/dt = 1 => y(t) = y0 + t (exact for Euler)"""

        def constant(t, y):
            return torch.ones_like(y)

        y0 = torch.tensor([0.0])
        y_final, interp = euler(constant, y0, t_span=(0.0, 1.0), dt=0.1)

        # Euler is exact for linear ODEs
        assert torch.allclose(y_final, torch.tensor([1.0]), atol=1e-6)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)


class TestEulerAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        _, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None


class TestEulerBatched:
    def test_batched_initial_conditions(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        assert y_final.shape == (3, 1)

    def test_batched_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0]])  # (2, 1)
        _, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        t_query = torch.tensor([0.25, 0.5, 0.75])
        trajectory = interp(t_query)
        assert trajectory.shape == (3, 2, 1)  # (T, B, D)


class TestEulerTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )
        state_final, interp = euler(f, state0, t_span=(0.0, 1.0), dt=0.01)

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()


class TestEulerComplex:
    def test_complex_exponential(self):
        """dy/dt = -i*y => y(t) = exp(-i*t)"""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.001)

        expected = torch.exp(-1j * torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-2)


class TestEulerBackward:
    def test_backward_integration(self):
        """Integrate backwards"""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, _ = euler(f, y1, t_span=(1.0, 0.0), dt=0.01)

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=0.1)
