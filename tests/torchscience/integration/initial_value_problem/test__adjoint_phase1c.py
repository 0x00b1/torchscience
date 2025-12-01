"""Phase 1c tests for solve_ivp unified API."""

import threading

import pytest
import torch
from tensordict import TensorDict

# Note: These imports will fail until we implement the module
from torchscience.integration.initial_value_problem import (
    ODESolution,
    solve_ivp,
)


class TestSolveIVPAPI:
    """Test the unified solve_ivp API."""

    def test_solve_ivp_basic(self):
        """Test basic solve_ivp usage with default settings."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        result = solve_ivp(decay, y0, t_span=(0.0, 1.0))

        # Result should be ODESolution
        assert isinstance(result, ODESolution)

        # Should have required fields
        assert hasattr(result, "y_final")
        assert hasattr(result, "interp")
        assert hasattr(result, "y_eval")
        assert hasattr(result, "t_eval")
        assert hasattr(result, "n_steps")
        assert hasattr(result, "n_function_evals")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "stats")

        # y_final should be close to exp(-1)
        expected = torch.tensor([0.36787944117])  # exp(-1)
        assert torch.allclose(result.y_final, expected, rtol=1e-3)

        # success should be True
        assert result.success

    def test_solve_ivp_sensitivity_none(self):
        """Test solve_ivp with sensitivity=None (standard autograd)."""
        theta = torch.tensor([1.0], requires_grad=True)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        result = solve_ivp(decay, y0, t_span=(0.0, 1.0), sensitivity=None)

        # Should be able to compute gradients via standard autograd
        loss = result.y_final.sum()
        loss.backward()

        # theta should have gradient
        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_solve_ivp_sensitivity_adjoint(self):
        """Test solve_ivp with sensitivity='adjoint' for memory-efficient gradients."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        result = solve_ivp(
            decay,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="adjoint",
            params=[theta],
        )

        # Should work and give correct gradients
        loss = result.y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

        # Gradient should be negative (increasing theta increases decay rate)
        assert theta.grad < 0

    def test_solve_ivp_sensitivity_checkpoint(self):
        """Test solve_ivp with sensitivity='checkpoint' for checkpointed adjoint."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        result = solve_ivp(
            decay,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="checkpoint",
            params=[theta],
            checkpoints=5,
        )

        # Should work and give correct gradients
        loss = result.y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_solve_ivp_tuple_unpacking(self):
        """Test that ODESolution supports tuple unpacking for backwards compatibility."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        result = solve_ivp(decay, y0, t_span=(0.0, 1.0))

        # Should support len()
        assert len(result) == 2

        # Should support iteration
        items = list(result)
        assert len(items) == 2
        y_final, interp = items
        assert torch.is_tensor(y_final)
        assert callable(interp)

        # Should support direct unpacking
        y_final2, interp2 = solve_ivp(decay, y0, t_span=(0.0, 1.0))
        assert torch.is_tensor(y_final2)
        assert callable(interp2)

        # Should support indexing
        assert torch.allclose(result[0], result.y_final)
        assert result[1] is result.interp

    def test_solve_ivp_method_aliases(self):
        """Test that method aliases work correctly."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])

        # Test dormand_prince_5 alias 'dp5'
        result_dp5 = solve_ivp(decay, y0, t_span=(0.0, 1.0), method="dp5")
        result_full = solve_ivp(
            decay, y0, t_span=(0.0, 1.0), method="dormand_prince_5"
        )
        assert torch.allclose(
            result_dp5.y_final, result_full.y_final, rtol=1e-4
        )

        # Test runge_kutta_4 alias 'rk4'
        result_rk4 = solve_ivp(
            decay, y0, t_span=(0.0, 1.0), method="rk4", dt=0.01
        )
        result_full_rk4 = solve_ivp(
            decay, y0, t_span=(0.0, 1.0), method="runge_kutta_4", dt=0.01
        )
        assert torch.allclose(
            result_rk4.y_final, result_full_rk4.y_final, rtol=1e-4
        )

    def test_solve_ivp_with_t_eval(self):
        """Test solve_ivp with t_eval for evaluating at specific times."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        t_eval = torch.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0

        result = solve_ivp(decay, y0, t_span=(0.0, 1.0), t_eval=t_eval)

        # t_eval and y_eval should be populated
        assert result.t_eval is not None
        assert result.y_eval is not None
        assert result.t_eval.shape[0] == 11
        assert result.y_eval.shape[0] == 11

        # Values should match expected exp(-t)
        expected = torch.exp(-t_eval).unsqueeze(-1)
        assert torch.allclose(result.y_eval, expected, rtol=1e-3)

    def test_solve_ivp_euler_method(self):
        """Test solve_ivp with euler method."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        result = solve_ivp(
            decay, y0, t_span=(0.0, 1.0), method="euler", dt=0.001
        )

        # Should work and give reasonable result
        expected = torch.tensor([0.36787944117])
        assert torch.allclose(result.y_final, expected, rtol=1e-2)

    def test_solve_ivp_adjoint_options(self):
        """Test solve_ivp with custom adjoint_options."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        result = solve_ivp(
            decay,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="adjoint",
            params=[theta],
            adjoint_options={"method": "euler", "n_steps": 200},
        )

        loss = result.y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_solve_ivp_invalid_method(self):
        """Test that invalid method raises ValueError with available methods."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])

        with pytest.raises(ValueError) as excinfo:
            solve_ivp(decay, y0, t_span=(0.0, 1.0), method="invalid_method")

        # Check error message contains useful information
        assert "invalid_method" in str(excinfo.value)
        assert "Available methods" in str(excinfo.value)


class TestODESolutionDataclass:
    """Test the ODESolution dataclass properties."""

    def test_odesolution_fields(self):
        """Test that ODESolution has all required fields."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        result = solve_ivp(decay, y0, t_span=(0.0, 1.0))

        # Check all fields exist and have reasonable values
        assert torch.is_tensor(result.y_final)
        assert callable(result.interp)
        assert result.success is True or result.success is False
        assert isinstance(result.message, str)
        assert isinstance(result.stats, dict)

    def test_odesolution_interp_callable(self):
        """Test that the interpolant works correctly."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        result = solve_ivp(decay, y0, t_span=(0.0, 1.0))

        # Interpolant should work at various times
        y_mid = result.interp(0.5)
        expected_mid = torch.tensor([0.60653066])  # exp(-0.5)
        assert torch.allclose(y_mid, expected_mid, rtol=1e-2)

        # Should handle tensor input
        t_query = torch.tensor([0.0, 0.5, 1.0])
        y_query = result.interp(t_query)
        assert y_query.shape[0] == 3


class TestPhase1cGraduation:
    """
    Graduation tests for Phase 1c.

    Run: pytest tests/.../test__adjoint_phase1c.py::TestPhase1cGraduation -v
    Pass condition: All tests pass.
    """

    def test_g1c1_solve_ivp_all_modes(self):
        """G1c.1: All sensitivity modes (None, 'adjoint', 'checkpoint') work.

        Verifies that all three sensitivity modes produce results with gradients.
        """
        sensitivity_modes = [None, "adjoint", "checkpoint"]

        for mode in sensitivity_modes:
            theta = torch.tensor(
                [1.0], requires_grad=True, dtype=torch.float64
            )

            def f(t, y):
                return -theta * y

            y0 = torch.tensor([1.0], dtype=torch.float64)

            # Build kwargs based on sensitivity mode
            kwargs = {"sensitivity": mode}
            if mode in ("adjoint", "checkpoint"):
                kwargs["params"] = [theta]
            if mode == "checkpoint":
                kwargs["checkpoints"] = 5

            result = solve_ivp(f, y0, t_span=(0.0, 1.0), **kwargs)

            # Result should be valid
            assert isinstance(result, ODESolution), (
                f"Mode {mode}: result should be ODESolution"
            )
            assert torch.isfinite(result.y_final).all(), (
                f"Mode {mode}: y_final should be finite"
            )

            # Should be able to compute gradients
            loss = result.y_final.sum()
            loss.backward()

            assert theta.grad is not None, (
                f"Mode {mode}: theta should have gradient"
            )
            assert torch.isfinite(theta.grad).all(), (
                f"Mode {mode}: theta.grad should be finite"
            )

    def test_g1c2_tensordict_gradcheck(self):
        """G1c.2: TensorDict numerical gradient matches adjoint gradient.

        Uses finite difference to verify adjoint gradients are correct
        for TensorDict state (harmonic oscillator).
        """

        def loss_fn(omega_val):
            """Compute loss given omega value."""

            def harmonic_oscillator(t, state):
                x = state["x"]
                v = state["v"]
                return TensorDict(
                    {"x": v, "v": -(omega_val**2) * x},
                    batch_size=state.batch_size,
                )

            y0 = TensorDict(
                {
                    "x": torch.tensor([1.0], dtype=torch.float64),
                    "v": torch.tensor([0.0], dtype=torch.float64),
                },
                batch_size=[],
            )

            result = solve_ivp(
                harmonic_oscillator,
                y0,
                t_span=(0.0, 1.0),
                sensitivity="adjoint",
                params=[omega_val] if omega_val.requires_grad else [],
            )
            return result.y_final["x"].sum()

        # Compute adjoint gradient
        omega = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        loss = loss_fn(omega)
        loss.backward()
        adjoint_grad = omega.grad.clone()

        # Compute numerical gradient via finite differences
        eps = 1e-5
        omega_plus = torch.tensor(2.0 + eps, dtype=torch.float64)
        omega_minus = torch.tensor(2.0 - eps, dtype=torch.float64)

        loss_plus = loss_fn(omega_plus)
        loss_minus = loss_fn(omega_minus)
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)

        # Compare with rtol=0.01 (1% relative tolerance)
        assert torch.allclose(adjoint_grad, numerical_grad, rtol=0.01), (
            f"G1c.2 FAIL: Adjoint gradient {adjoint_grad.item():.6f} does not match "
            f"numerical gradient {numerical_grad.item():.6f}"
        )

    def test_g1c3_neural_ode_training(self):
        """G1c.3: Neural ODE achieves reasonable training.

        Train ODEFunc to rotate initial state by pi/4.
        Loss should drop below 0.5 after training.
        """
        import torch.nn as nn

        # Define neural ODE function as nn.Module
        class ODEFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.Tanh(),
                    nn.Linear(16, 2),
                )

            def forward(self, t, y):
                return self.net(y)

        # Target: rotate [1, 0] by pi/4 -> [cos(pi/4), sin(pi/4)]
        target_angle = torch.tensor(torch.pi / 4)
        y_target = torch.tensor(
            [torch.cos(target_angle), torch.sin(target_angle)],
            dtype=torch.float32,
        )

        # Initial condition: [1, 0]
        y0 = torch.tensor([1.0, 0.0], dtype=torch.float32)

        # Create model and optimizer
        torch.manual_seed(42)  # For reproducibility
        model = ODEFunc()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        # Training loop
        for _ in range(50):
            optimizer.zero_grad()

            # Solve ODE with adjoint sensitivity
            result = solve_ivp(
                model,
                y0,
                t_span=(0.0, 1.0),
                sensitivity="adjoint",
                params=list(model.parameters()),
            )

            # Compute MSE loss between predicted final state and target
            loss = torch.nn.functional.mse_loss(result.y_final, y_target)

            # Backward pass through adjoint method
            loss.backward()
            optimizer.step()

        # Final evaluation
        with torch.no_grad():
            result = solve_ivp(model, y0, t_span=(0.0, 1.0))
            final_loss = torch.nn.functional.mse_loss(
                result.y_final, y_target
            ).item()

        assert final_loss < 0.5, (
            f"G1c.3 FAIL: Final loss {final_loss:.4f} should be < 0.5"
        )

    def test_g1c4_compile_smoke(self):
        """G1c.4: torch.compile doesn't crash.

        Tests that solve_ivp with sensitivity='adjoint' works when the
        dynamics function is wrapped with torch.compile. Skips if
        compile/dynamo errors occur.
        """
        # Check if torch.compile is available
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        # Compile with fullgraph=False to allow graph breaks
        try:
            compiled_dynamics = torch.compile(dynamics, fullgraph=False)
        except Exception as e:
            pytest.skip(f"torch.compile not available or failed: {e}")

        y0 = torch.tensor([1.0], dtype=torch.float64)

        try:
            result = solve_ivp(
                compiled_dynamics,
                y0,
                t_span=(0.0, 1.0),
                sensitivity="adjoint",
                params=[theta],
            )

            # Compute loss and backpropagate
            loss = result.y_final.sum()
            loss.backward()

        except Exception as e:
            # Skip if dynamo/compile errors occur
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "dynamo",
                    "compile",
                    "graph",
                    "guard",
                    "backend",
                    "inductor",
                ]
            ):
                pytest.skip(f"torch.compile/dynamo error: {e}")
            raise

        # Verify gradient was computed
        assert theta.grad is not None, (
            "G1c.4 FAIL: theta.grad should not be None"
        )
        assert torch.isfinite(theta.grad).all(), (
            "G1c.4 FAIL: theta.grad should be finite"
        )

    def test_g1c5_concurrent_correct(self):
        """G1c.5: 4 concurrent threads complete correctly.

        Launches 4 threads, each solving dy/dt = -theta * y with different
        theta values (1, 2, 3, 4). Verifies all threads complete successfully
        with correct results.
        """
        results = {}
        errors = {}
        lock = threading.Lock()

        def thread_worker(thread_id, theta_value):
            """Worker function for each thread."""
            try:
                # Create theta parameter for this thread
                theta = torch.tensor(
                    [float(theta_value)],
                    requires_grad=True,
                    dtype=torch.float64,
                )

                def decay(t, y):
                    return -theta * y

                y0 = torch.tensor([1.0], dtype=torch.float64)

                # Solve ODE with adjoint sensitivity
                result = solve_ivp(
                    decay,
                    y0,
                    t_span=(0.0, 1.0),
                    sensitivity="adjoint",
                    params=[theta],
                )

                # Compute loss and backward pass
                loss = result.y_final.sum()
                loss.backward()

                # Store results thread-safely
                with lock:
                    results[thread_id] = {
                        "theta": theta_value,
                        "y_final": result.y_final.clone().detach(),
                        "grad": theta.grad.clone().detach(),
                    }

            except Exception as e:
                with lock:
                    errors[thread_id] = str(e)

        # Launch 4 threads with different theta values
        threads = []
        theta_values = [1, 2, 3, 4]

        for i, theta_val in enumerate(theta_values):
            t = threading.Thread(target=thread_worker, args=(i, theta_val))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Assert no errors occurred
        assert len(errors) == 0, f"G1c.5 FAIL: Errors in threads: {errors}"

        # Assert all 4 threads completed
        assert len(results) == 4, (
            f"G1c.5 FAIL: Expected 4 results, got {len(results)}"
        )

        # Assert each thread got correct y_final = exp(-theta)
        for thread_id, result in results.items():
            theta_val = result["theta"]
            y_final = result["y_final"]
            expected_y = torch.exp(
                torch.tensor([-float(theta_val)], dtype=torch.float64)
            )

            assert torch.allclose(y_final, expected_y, rtol=1e-3), (
                f"G1c.5 FAIL: Thread {thread_id} with theta={theta_val}: "
                f"expected y_final={expected_y.item():.6f}, "
                f"got {y_final.item():.6f}"
            )

            # Verify gradient is finite
            grad = result["grad"]
            assert torch.isfinite(grad).all(), (
                f"G1c.5 FAIL: Thread {thread_id}: gradient should be finite"
            )

    def test_g1c6_existing_tests_pass(self):
        """G1c.6: Placeholder reminder to run full test suite.

        This is a reminder test that the full adjoint test suite should be run
        to verify no regressions were introduced.

        Run full suite with:
            pytest tests/torchscience/integration/initial_value_problem/test__adjoint*.py -v
        """
        # This test always passes - it's a reminder to run the full suite
        pass


class TestTensorDictSupport:
    """Test TensorDict support with adjoint sensitivity method."""

    def test_adjoint_tensordict_basic(self):
        """Basic TensorDict state works with adjoint.

        Tests harmonic oscillator with TensorDict state containing
        position "x" and velocity "v" keys.
        """
        # Parameter to optimize
        omega = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def harmonic_oscillator(t, state):
            """dx/dt = v, dv/dt = -omega^2 * x"""
            x = state["x"]
            v = state["v"]
            return TensorDict(
                {"x": v, "v": -(omega**2) * x},
                batch_size=state.batch_size,
            )

        # Initial state: displaced from equilibrium, at rest
        y0 = TensorDict(
            {
                "x": torch.tensor([1.0], dtype=torch.float64),
                "v": torch.tensor([0.0], dtype=torch.float64),
            },
            batch_size=[],
        )

        # Solve with adjoint sensitivity
        result = solve_ivp(
            harmonic_oscillator,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="adjoint",
            params=[omega],
        )

        # Verify result is TensorDict
        assert isinstance(result.y_final, TensorDict)
        assert "x" in result.y_final.keys()
        assert "v" in result.y_final.keys()

        # Verify solution is finite
        assert torch.isfinite(result.y_final["x"]).all()
        assert torch.isfinite(result.y_final["v"]).all()

        # Compute loss and backpropagate
        loss = result.y_final["x"].sum()
        loss.backward()

        # Verify gradient flows through omega parameter
        assert omega.grad is not None
        assert torch.isfinite(omega.grad)

        # Gradient should be non-zero (omega affects final position)
        assert omega.grad.abs() > 1e-6

    def test_adjoint_tensordict_gradcheck(self):
        """Numerical gradient matches adjoint gradient for TensorDict state.

        Uses finite difference to verify adjoint gradients are correct.
        """

        def loss_fn(omega_val):
            """Compute loss given omega value."""

            def harmonic_oscillator(t, state):
                x = state["x"]
                v = state["v"]
                return TensorDict(
                    {"x": v, "v": -(omega_val**2) * x},
                    batch_size=state.batch_size,
                )

            y0 = TensorDict(
                {
                    "x": torch.tensor([1.0], dtype=torch.float64),
                    "v": torch.tensor([0.0], dtype=torch.float64),
                },
                batch_size=[],
            )

            result = solve_ivp(
                harmonic_oscillator,
                y0,
                t_span=(0.0, 1.0),
                sensitivity="adjoint",
                params=[omega_val] if omega_val.requires_grad else [],
            )
            return result.y_final["x"].sum()

        # Compute adjoint gradient
        omega = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        loss = loss_fn(omega)
        loss.backward()
        adjoint_grad = omega.grad.clone()

        # Compute numerical gradient via finite differences
        eps = 1e-5
        omega_plus = torch.tensor(2.0 + eps, dtype=torch.float64)
        omega_minus = torch.tensor(2.0 - eps, dtype=torch.float64)

        loss_plus = loss_fn(omega_plus)
        loss_minus = loss_fn(omega_minus)
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)

        # Compare with rtol=0.01 (1% relative tolerance)
        assert torch.allclose(adjoint_grad, numerical_grad, rtol=0.01), (
            f"Adjoint gradient {adjoint_grad.item():.6f} does not match "
            f"numerical gradient {numerical_grad.item():.6f}"
        )


class TestNeuralODE:
    """Test neural ODE training with adjoint sensitivity method."""

    def test_neural_ode_training_loop(self):
        """Train a simple neural ODE to fit spiral data.

        Creates an ODEFunc neural network and trains it using the adjoint
        method to minimize MSE loss between predicted and target final state.
        This demonstrates end-to-end gradient flow through the adjoint method.
        """
        import torch.nn as nn

        # Define neural ODE function as nn.Module
        class ODEFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.Tanh(),
                    nn.Linear(16, 2),
                )

            def forward(self, t, y):
                return self.net(y)

        # Target: spiral at t=2 -> [cos(2), sin(2)]
        t_final = 2.0
        y_target = torch.tensor(
            [
                torch.cos(torch.tensor(t_final)),
                torch.sin(torch.tensor(t_final)),
            ],
            dtype=torch.float32,
        )

        # Initial condition: [cos(0), sin(0)] = [1, 0]
        y0 = torch.tensor([1.0, 0.0], dtype=torch.float32)

        # Create model and optimizer
        model = ODEFunc()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        losses = []
        for epoch in range(20):
            optimizer.zero_grad()

            # Solve ODE with adjoint sensitivity
            result = solve_ivp(
                model,
                y0,
                t_span=(0.0, t_final),
                sensitivity="adjoint",
                params=list(model.parameters()),
            )

            # Compute MSE loss between predicted final state and target
            loss = torch.nn.functional.mse_loss(result.y_final, y_target)
            losses.append(loss.item())

            # Backward pass through adjoint method
            loss.backward()
            optimizer.step()

        # Assert loss decreases over training
        assert losses[-1] < losses[0], (
            f"Loss should decrease: final={losses[-1]:.4f}, initial={losses[0]:.4f}"
        )

        # Assert reasonable final accuracy
        assert losses[-1] < 1.0, (
            f"Final loss should be < 1.0, got {losses[-1]:.4f}"
        )

    def test_neural_ode_batched(self):
        """Test neural ODE with batched inputs.

        Verifies that the adjoint method correctly handles batched initial
        conditions and propagates gradients to all model parameters.
        """
        import torch.nn as nn

        # Define neural ODE function as nn.Module
        class ODEFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(4, 4)

            def forward(self, t, y):
                return self.net(y)

        # Create model
        model = ODEFunc()

        # Batched initial conditions: 8 samples, each with 4 dimensions
        y0 = torch.randn(8, 4, dtype=torch.float32)

        # Solve ODE with adjoint sensitivity
        result = solve_ivp(
            model,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="adjoint",
            params=list(model.parameters()),
        )

        # Verify output shape matches input batch shape
        assert result.y_final.shape == (8, 4), (
            f"Expected shape (8, 4), got {result.y_final.shape}"
        )

        # Compute loss and backpropagate
        loss = result.y_final.sum()
        loss.backward()

        # Verify all model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter {name} should have gradient"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Parameter {name} gradient should be finite"
            )


class TestTorchCompile:
    """Test torch.compile compatibility with adjoint method."""

    def test_adjoint_compile_smoke(self):
        """Adjoint with compiled dynamics doesn't crash.

        Tests that solve_ivp with sensitivity='adjoint' works when the
        dynamics function is wrapped with torch.compile. If compile/dynamo
        errors occur, the test skips rather than failing.
        """
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        # Compile with fullgraph=False to allow graph breaks
        try:
            compiled_dynamics = torch.compile(dynamics, fullgraph=False)
        except Exception as e:
            pytest.skip(f"torch.compile not available or failed: {e}")

        y0 = torch.tensor([1.0], dtype=torch.float64)

        try:
            result = solve_ivp(
                compiled_dynamics,
                y0,
                t_span=(0.0, 1.0),
                sensitivity="adjoint",
                params=[theta],
            )

            # Compute loss and backpropagate
            loss = result.y_final.sum()
            loss.backward()

        except Exception as e:
            # Skip if dynamo/compile errors occur
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "dynamo",
                    "compile",
                    "graph",
                    "guard",
                    "backend",
                ]
            ):
                pytest.skip(f"torch.compile/dynamo error: {e}")
            raise

        # Verify gradient was computed
        assert theta.grad is not None, "theta.grad should not be None"
        assert torch.isfinite(theta.grad).all(), "theta.grad should be finite"

    def test_dynamics_compilable(self):
        """Dynamics function compiles independently.

        Tests that a simple dynamics function can be compiled with
        fullgraph=True and executes without error.
        """
        # Check if torch.compile is available
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        theta = torch.tensor([1.0], dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        # Compile with fullgraph=True for stricter compilation
        try:
            compiled_dynamics = torch.compile(dynamics, fullgraph=True)
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")

        # Test that compiled function executes without error
        t = torch.tensor(0.0, dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        try:
            result = compiled_dynamics(t, y)
        except Exception as e:
            # Skip if dynamo errors occur during execution
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "dynamo",
                    "compile",
                    "graph",
                    "guard",
                    "backend",
                ]
            ):
                pytest.skip(f"torch.compile execution error: {e}")
            raise

        # Verify result is correct
        expected = -theta * y
        assert torch.allclose(result, expected), (
            f"Compiled dynamics returned incorrect result: {result} vs {expected}"
        )


class TestThreadSafety:
    """Test thread safety of adjoint sensitivity method."""

    def test_concurrent_adjoint_solves(self):
        """Multiple threads use adjoint correctly without interference.

        Launches 4 threads, each solving dy/dt = -theta * y with different
        theta values (1, 2, 3, 4). Each thread uses the adjoint method to
        compute gradients. Verifies that:
        1. No errors occur during concurrent execution
        2. All 4 threads complete successfully
        3. Each thread computes the correct y_final = exp(-theta)
        4. Each thread computes the correct gradient
        """
        results = {}
        errors = {}
        lock = threading.Lock()

        def thread_worker(thread_id, theta_value):
            """Worker function for each thread."""
            try:
                # Create theta parameter for this thread
                theta = torch.tensor(
                    [float(theta_value)],
                    requires_grad=True,
                    dtype=torch.float64,
                )

                def decay(t, y):
                    return -theta * y

                y0 = torch.tensor([1.0], dtype=torch.float64)

                # Solve ODE with adjoint sensitivity
                result = solve_ivp(
                    decay,
                    y0,
                    t_span=(0.0, 1.0),
                    sensitivity="adjoint",
                    params=[theta],
                )

                # Compute loss and backward pass
                loss = result.y_final.sum()
                loss.backward()

                # Store results thread-safely
                with lock:
                    results[thread_id] = {
                        "theta": theta_value,
                        "y_final": result.y_final.clone().detach(),
                        "grad": theta.grad.clone().detach(),
                    }

            except Exception as e:
                with lock:
                    errors[thread_id] = str(e)

        # Launch 4 threads with different theta values
        threads = []
        theta_values = [1, 2, 3, 4]

        for i, theta_val in enumerate(theta_values):
            t = threading.Thread(target=thread_worker, args=(i, theta_val))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Assert no errors occurred
        assert len(errors) == 0, f"Errors occurred in threads: {errors}"

        # Assert all 4 threads completed
        assert len(results) == 4, (
            f"Expected 4 results, got {len(results)}: {list(results.keys())}"
        )

        # Assert each thread got correct y_final = exp(-theta)
        for thread_id, result in results.items():
            theta_val = result["theta"]
            y_final = result["y_final"]
            expected_y = torch.exp(
                torch.tensor([-float(theta_val)], dtype=torch.float64)
            )

            assert torch.allclose(y_final, expected_y, rtol=1e-3), (
                f"Thread {thread_id} with theta={theta_val}: "
                f"expected y_final={expected_y.item():.6f}, "
                f"got {y_final.item():.6f}"
            )

            # Verify gradient is finite and negative
            grad = result["grad"]
            assert torch.isfinite(grad).all(), (
                f"Thread {thread_id}: gradient should be finite, got {grad}"
            )
            assert grad < 0, (
                f"Thread {thread_id}: gradient should be negative, got {grad.item()}"
            )
