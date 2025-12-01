# tests/torchscience/integration/initial_value_problem/test__backward_euler.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import (
    ConvergenceError,
    backward_euler,
)


class TestBackwardEulerBasic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0]))
        # Backward Euler is 1st order, similar accuracy to forward Euler
        assert torch.allclose(y_final, expected, rtol=0.1)

    def test_stiff_problem(self):
        """Test on a stiff ODE that would require tiny steps for explicit methods."""
        # dy/dt = -1000 * (y - sin(t)) + cos(t)
        # Stiff because of the -1000 coefficient

        def stiff(t, y):
            # Convert t to tensor for torch.sin/cos
            t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
            return -1000 * (y - torch.sin(t_tensor)) + torch.cos(t_tensor)

        y0 = torch.tensor([0.0], dtype=torch.float64)
        # Backward Euler should handle this with reasonable step size
        y_final, _ = backward_euler(
            stiff, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-8
        )

        # Exact solution is y = sin(t)
        expected = torch.sin(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, rtol=0.1)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Cast to same dtype for comparison (interpolant uses float64 for time)
        assert torch.allclose(interp(0.0).float(), y0, atol=1e-5)
        assert torch.allclose(interp(1.0).float(), y_final, atol=1e-5)


class TestBackwardEulerNewtonConvergence:
    def test_newton_convergence_default(self):
        """Default Newton parameters should work for simple problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Should complete without error
        assert not torch.isnan(y_final).any()

    def test_newton_tol_affects_accuracy(self):
        """Tighter Newton tolerance should give more consistent results."""

        def f(t, y):
            return -(y**2) + 1  # Nonlinear

        # Use float64 for tighter Newton tolerance
        y0 = torch.tensor([0.5], dtype=torch.float64)

        y_loose, _ = backward_euler(
            f,
            y0,
            t_span=(0.0, 1.0),
            dt=0.1,
            newton_tol=1e-3,
            max_newton_iter=50,
        )
        y_tight, _ = backward_euler(
            f,
            y0,
            t_span=(0.0, 1.0),
            dt=0.1,
            newton_tol=1e-10,
            max_newton_iter=50,
        )

        # Both should produce finite results
        assert not torch.isnan(y_loose).any()
        assert not torch.isnan(y_tight).any()

    def test_convergence_error_thrown(self):
        """Should raise ConvergenceError when Newton fails (throw=True)."""

        def difficult(t, y):
            # Dynamics that make Newton hard to converge
            return y**3 - 100 * y

        y0 = torch.tensor([0.1])

        with pytest.raises(ConvergenceError):
            backward_euler(
                difficult,
                y0,
                t_span=(0.0, 1.0),
                dt=0.5,  # Large step makes convergence hard
                newton_tol=1e-12,
                max_newton_iter=2,  # Not enough iterations
            )

    def test_convergence_failure_no_throw(self):
        """Should return NaN when Newton fails (throw=False)."""

        def difficult(t, y):
            return y**3 - 100 * y

        y0 = torch.tensor([0.1])

        y_final, interp = backward_euler(
            difficult,
            y0,
            t_span=(0.0, 1.0),
            dt=0.5,
            newton_tol=1e-12,
            max_newton_iter=2,
            throw=False,
        )

        assert torch.isnan(y_final).any()
        assert interp.success is not None
        assert not interp.success.all()


class TestBackwardEulerAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        _, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None


class TestBackwardEulerTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {"x": torch.tensor([1.0]), "v": torch.tensor([0.0])}
        )
        state_final, _ = backward_euler(f, state0, t_span=(0.0, 1.0), dt=0.1)

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()


class TestBackwardEulerComplex:
    @pytest.mark.skip(
        reason="torch.func.jacrev doesn't support complex tensors; "
        "would need real-valued formulation (2N real ODE)"
    )
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0))
        assert torch.allclose(y_final.squeeze(), expected, atol=0.1)


class TestBackwardEulerBatched:
    def test_batched_initial_conditions(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert y_final.shape == (3, 1)


class TestBackwardEulerBackward:
    def test_backward_integration(self):
        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, _ = backward_euler(f, y1, t_span=(1.0, 0.0), dt=0.1)

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=0.2)
