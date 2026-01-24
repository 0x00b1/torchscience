# tests/torchscience/integration/initial_value_problem/test__midpoint.py
import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation import midpoint


class TestMidpointBasic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(torch.tensor([-1.0]))
        # Midpoint is 2nd order, more accurate than Euler
        assert torch.allclose(y_final, expected, rtol=1e-3)

    def test_more_accurate_than_euler(self):
        """Midpoint should be more accurate than Euler for same step size"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        expected = torch.exp(torch.tensor([-1.0]))

        # Import euler for comparison
        from torchscience.ordinary_differential_equation import euler

        y_euler, _ = euler(decay, y0, t_span=(0.0, 1.0), dt=0.1)
        y_midpoint, _ = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_midpoint = (y_midpoint - expected).abs().item()

        assert error_midpoint < error_euler

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0])
        y_final, interp = midpoint(
            oscillator, y0, t_span=(0.0, 2 * torch.pi), dt=0.01
        )

        # After one period, should return near initial state
        assert torch.allclose(y_final, y0, atol=0.1)


class TestMidpointAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestMidpointTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )
        state_final, interp = midpoint(f, state0, t_span=(0.0, 1.0), dt=0.01)

        assert isinstance(state_final, TensorDict)


class TestMidpointComplex:
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-3)
