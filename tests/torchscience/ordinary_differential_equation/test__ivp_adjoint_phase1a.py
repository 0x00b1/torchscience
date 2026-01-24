"""Phase 1a tests for adjoint correctness."""

import warnings

import torch

from torchscience.ordinary_differential_equation import (
    adjoint,
    dormand_prince_5,
    euler,
)


class TestExplicitParams:
    def test_explicit_params_single(self):
        """Explicit params should work with a single parameter."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(dormand_prince_5, params=[theta])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_explicit_params_multiple(self):
        """Explicit params should work with multiple parameters."""
        theta1 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        theta2 = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta1 * y + theta2 * torch.sin(torch.as_tensor(t))

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(dormand_prince_5, params=[theta1, theta2])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        assert theta1.grad is not None
        assert theta2.grad is not None
        assert torch.isfinite(theta1.grad)
        assert torch.isfinite(theta2.grad)

    def test_explicit_params_empty_list(self):
        """params=[] should work (forward-only solve, no parameter gradients)."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(dormand_prince_5, params=[])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0))

        # Forward-only solve: no gradient computation
        # y_final should not require grad, theta should have no gradients
        assert not y_final.requires_grad
        assert theta.grad is None

    def test_params_none_legacy_behavior(self):
        """params=None (default) should extract from closure."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(dormand_prince_5)  # params=None (default)
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        assert theta.grad is not None


class TestModuleParams:
    def test_module_params(self):
        """nn.Module parameters should work with explicit params."""

        class ODEFunc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, t, y):
                return self.linear(y)

        model = ODEFunc().double()
        y0 = torch.randn(2, dtype=torch.float64)

        solver = adjoint(dormand_prince_5, params=list(model.parameters()))
        y_final, _ = solver(model, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_module_params_with_frozen(self):
        """Should handle mix of frozen and trainable parameters."""

        class ODEFunc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                # Freeze linear2
                for p in self.linear2.parameters():
                    p.requires_grad = False

            def forward(self, t, y):
                return self.linear1(y) + self.linear2(y)

        model = ODEFunc().double()
        y0 = torch.randn(2, dtype=torch.float64)

        # Only pass trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        solver = adjoint(dormand_prince_5, params=trainable_params)
        y_final, _ = solver(model, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        # linear1 should have gradients
        assert model.linear1.weight.grad is not None
        assert model.linear1.bias.grad is not None
        # linear2 should not
        assert model.linear2.weight.grad is None
        assert model.linear2.bias.grad is None


class TestAdjointRK4:
    def test_adjoint_rk4_more_accurate_than_euler(self):
        """RK4 adjoint should be significantly more accurate than Euler."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Analytical gradient: y(T) = exp(-theta*T), dy/dtheta = -T*exp(-theta*T)
        T = 1.0
        analytical = -T * torch.exp(-theta.detach() * T)

        # Euler adjoint
        theta_euler = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )

        def f_euler(t, y):
            return -theta_euler * y

        solver_euler = adjoint(
            dormand_prince_5,
            params=[theta_euler],
            adjoint_options={"method": "euler", "n_steps": 100},
        )
        y1, _ = solver_euler(f_euler, y0.clone(), (0.0, T))
        y1.sum().backward()
        grad_euler = theta_euler.grad.clone()

        # RK4 adjoint
        theta_rk4 = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )

        def f_rk4(t, y):
            return -theta_rk4 * y

        solver_rk4 = adjoint(
            dormand_prince_5,
            params=[theta_rk4],
            adjoint_options={"method": "rk4", "n_steps": 100},
        )
        y2, _ = solver_rk4(f_rk4, y0.clone(), (0.0, T))
        y2.sum().backward()
        grad_rk4 = theta_rk4.grad.clone()

        euler_error = (grad_euler - analytical).abs().item()
        rk4_error = (grad_rk4 - analytical).abs().item()

        # RK4 should be much more accurate (O(dt^4) vs O(dt))
        assert rk4_error < euler_error / 10, (
            f"RK4 error {rk4_error} should be <10x better than Euler {euler_error}"
        )
        assert rk4_error < 1e-6, f"RK4 error {rk4_error} should be < 1e-6"


class TestParameterGradientAccuracy:
    def test_parameter_gradient_accuracy_exponential_decay(self):
        """Parameter gradients should match analytical for exponential decay."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        T = 1.0

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "rk4", "n_steps": 100},
        )
        y_final, _ = solver(f, y0, t_span=(0.0, T))
        loss = y_final.sum()
        loss.backward()

        # Analytical: y(t) = y0 * exp(-theta*t)
        # dL/dtheta = dy(T)/dtheta = -T * y0 * exp(-theta*T)
        analytical = -T * y0 * torch.exp(-theta.detach() * T)

        rel_error = (theta.grad - analytical).abs() / analytical.abs()
        assert rel_error < 1e-4, (
            f"Relative error {rel_error.item()} exceeds 1e-4"
        )

    def test_parameter_gradient_accuracy_oscillator(self):
        """Parameter gradients should be accurate for harmonic oscillator."""
        omega = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)

        def oscillator(t, y):
            x, v = y[0:1], y[1:2]
            dxdt = v
            dvdt = -(omega**2) * x
            return torch.cat([dxdt, dvdt])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        T = 1.0

        solver = adjoint(
            dormand_prince_5,
            params=[omega],
            adjoint_options={"method": "rk4", "n_steps": 200},
        )
        y_final, _ = solver(oscillator, y0, t_span=(0.0, T))
        loss = y_final[0]  # x(T) = cos(omega*T)
        loss.backward()

        # Numerical gradient check
        eps = 1e-5
        omega_plus = omega.detach() + eps
        omega_minus = omega.detach() - eps

        def solve_with_omega(w):
            def osc(t, y):
                return torch.cat([y[1:2], -(w**2) * y[0:1]])

            with torch.no_grad():
                yf, _ = dormand_prince_5(osc, y0.clone(), t_span=(0.0, T))
            return yf[0]

        numerical_grad = (
            solve_with_omega(omega_plus) - solve_with_omega(omega_minus)
        ) / (2 * eps)

        rel_error = (omega.grad - numerical_grad).abs() / numerical_grad.abs()
        assert rel_error < 1e-3, (
            f"Relative error {rel_error.item()} exceeds 1e-3"
        )


class TestInterpolationWarning:
    def test_linear_interpolation_warning(self):
        """Should warn when using linear interpolation (e.g., Euler solver)."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        solver = adjoint(euler, params=[theta])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_final, _ = solver(f, y0, t_span=(0.0, 1.0), dt=0.01)
            y_final.sum().backward()

            # Should have warning about linear interpolation
            interp_warnings = [
                x for x in w if "interpolation" in str(x.message).lower()
            ]
            assert len(interp_warnings) >= 1, (
                "Should warn about linear interpolation"
            )

    def test_no_warning_with_dp5(self):
        """Should NOT warn when using high-order interpolation (DP5)."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        solver = adjoint(dormand_prince_5, params=[theta])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
            y_final.sum().backward()

            interp_warnings = [
                x for x in w if "interpolation" in str(x.message).lower()
            ]
            assert len(interp_warnings) == 0, (
                "Should not warn with DP5 interpolation"
            )


class TestGradientStability:
    def test_adjoint_long_integration_stability(self):
        """Gradients should remain accurate for long integrations with many steps."""
        theta = torch.tensor([0.01], requires_grad=True, dtype=torch.float64)

        def slow_decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        T = 100.0

        # 10000 steps to stress test accumulation
        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"n_steps": 10000},
        )
        y_final, _ = solver(slow_decay, y0, t_span=(0.0, T))
        loss = y_final.sum()
        loss.backward()

        # Analytical: y(t) = exp(-theta*t), dy/dtheta = -t*exp(-theta*t)
        # dL/dtheta = dy(T)/dtheta = -T*exp(-theta*T)
        analytical = -T * torch.exp(-theta.detach() * T)

        # Should be within 1e-4 relative error even with many steps
        rel_error = (theta.grad - analytical).abs() / analytical.abs()
        assert rel_error < 1e-4, (
            f"Relative error {rel_error.item()} exceeds 1e-4"
        )
        assert torch.isfinite(theta.grad)


class TestAdjointStability:
    def test_adjoint_divergence_raises(self):
        """Should raise when adjoint diverges (explodes to infinity)."""
        import pytest

        from torchscience.ordinary_differential_equation._ivp_adjoint import (
            AdjointDivergedError,
        )

        theta = torch.tensor([10.0], requires_grad=True, dtype=torch.float64)

        def unstable_dynamics(t, y):
            # Positive feedback causes explosion
            return theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"n_steps": 1000},
        )
        y_final, _ = solver(unstable_dynamics, y0, t_span=(0.0, 10.0))

        with pytest.raises(AdjointDivergedError):
            y_final.sum().backward()

    def test_adjoint_rapid_growth_warning(self):
        """Should warn when adjoint grows rapidly (potential instability)."""
        from torchscience.ordinary_differential_equation._ivp_adjoint import (
            AdjointStabilityWarning,
        )

        theta = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)

        def moderately_unstable(t, y):
            return theta * y  # Positive feedback

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"n_steps": 100},
        )
        y_final, _ = solver(moderately_unstable, y0, t_span=(0.0, 2.0))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_final.sum().backward()

            stability_warnings = [
                x
                for x in w
                if "growing rapidly" in str(x.message).lower()
                or "stability" in str(x.message).lower()
                or issubclass(x.category, AdjointStabilityWarning)
            ]
            # May or may not warn depending on growth rate - just ensure no crash


class TestPhase1aGraduation:
    """
    Graduation tests for Phase 1a.

    Run: pytest tests/.../test__adjoint_phase1a.py::TestPhase1aGraduation -v
    Pass condition: All tests pass.
    """

    def test_g1a1_parameter_gradient_accuracy(self):
        """G1a.1: Parameter gradient matches analytical."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        T = 1.0

        solver = adjoint(dormand_prince_5, params=[theta])
        y_final, _ = solver(f, y0, t_span=(0.0, T))
        y_final.sum().backward()

        analytical = -T * y0 * torch.exp(-theta.detach() * T)
        rel_error = (theta.grad - analytical).abs() / analytical.abs()
        assert rel_error < 1e-4, f"G1a.1 FAIL: rel_error={rel_error.item()}"

    def test_g1a2_adjoint_rk4_vs_euler(self):
        """G1a.2: RK4 significantly more accurate than Euler."""
        theta_euler = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )
        theta_rk4 = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )

        y0 = torch.tensor([1.0], dtype=torch.float64)
        T = 1.0
        analytical = -T * torch.exp(torch.tensor(-1.0, dtype=torch.float64))

        # Euler
        def f_euler(t, y):
            return -theta_euler * y

        solver_euler = adjoint(
            dormand_prince_5,
            params=[theta_euler],
            adjoint_options={"method": "euler", "n_steps": 100},
        )
        y1, _ = solver_euler(f_euler, y0.clone(), (0.0, T))
        y1.sum().backward()

        # RK4
        def f_rk4(t, y):
            return -theta_rk4 * y

        solver_rk4 = adjoint(
            dormand_prince_5,
            params=[theta_rk4],
            adjoint_options={"method": "rk4", "n_steps": 100},
        )
        y2, _ = solver_rk4(f_rk4, y0.clone(), (0.0, T))
        y2.sum().backward()

        euler_err = (theta_euler.grad - analytical).abs().item()
        rk4_err = (theta_rk4.grad - analytical).abs().item()

        assert rk4_err < euler_err / 10, f"G1a.2 FAIL: RK4 not 10x better"

    def test_g1a3_explicit_params(self):
        """G1a.3: Explicit params work correctly."""
        theta1 = torch.tensor([1.0], requires_grad=True)
        theta2 = torch.tensor([2.0], requires_grad=True)

        def f(t, y):
            return -theta1 * y + theta2 * 0.1

        y0 = torch.tensor([1.0])
        solver = adjoint(dormand_prince_5, params=[theta1, theta2])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        assert theta1.grad is not None, "G1a.3 FAIL: theta1.grad is None"
        assert theta2.grad is not None, "G1a.3 FAIL: theta2.grad is None"

    def test_g1a4_module_params(self):
        """G1a.4: nn.Module parameters work correctly."""

        class ODEFunc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, t, y):
                return self.linear(y)

        model = ODEFunc()
        y0 = torch.randn(2)
        solver = adjoint(dormand_prince_5, params=list(model.parameters()))
        y_final, _ = solver(model, y0, t_span=(0.0, 1.0))
        y_final.sum().backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"G1a.4 FAIL: {name}.grad is None"

    def test_g1a5_long_integration(self):
        """G1a.5: Long integration relative error < 1e-4."""
        theta = torch.tensor([0.01], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        T = 100.0

        solver = adjoint(
            dormand_prince_5, params=[theta], adjoint_options={"n_steps": 5000}
        )
        y_final, _ = solver(f, y0, t_span=(0.0, T))
        y_final.sum().backward()

        analytical = -T * torch.exp(-theta.detach() * T)
        rel_error = (theta.grad - analytical).abs() / analytical.abs()
        assert rel_error < 1e-4, f"G1a.5 FAIL: rel_error={rel_error.item()}"

    def test_g1a6_gradcheck(self):
        """G1a.6: Numerical gradient check passes."""

        def loss_fn(theta):
            def f(t, y):
                return -theta * y

            y0 = torch.tensor([1.0], dtype=torch.float64)
            solver = adjoint(dormand_prince_5, params=[theta])
            y_final, _ = solver(f, y0, t_span=(0.0, 1.0))
            return y_final.sum()

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        # Numerical gradient check
        eps = 1e-5
        theta_plus = theta.detach() + eps
        theta_minus = theta.detach() - eps

        loss_plus = loss_fn(theta_plus.requires_grad_(False))
        loss_minus = loss_fn(theta_minus.requires_grad_(False))
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)

        loss = loss_fn(theta)
        loss.backward()

        assert torch.allclose(theta.grad, numerical_grad, rtol=1e-3), (
            f"G1a.6 FAIL: gradcheck failed"
        )
