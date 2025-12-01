"""Phase 1b tests for adjoint performance."""

import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
)


class TestSingleForwardPass:
    def test_single_forward_pass(self):
        """Forward solve should only run once, not twice."""
        call_count = [0]

        original_dp5 = dormand_prince_5

        def counting_dp5(f, y0, t_span, **kwargs):
            call_count[0] += 1
            return original_dp5(f, y0, t_span, **kwargs)

        theta = torch.tensor([1.0], requires_grad=True)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        solver = adjoint(counting_dp5, params=[theta])
        y_final, interp = solver(dynamics, y0, t_span=(0.0, 1.0))

        # Should only call solver once for forward pass
        assert call_count[0] == 1, (
            f"Expected 1 forward solve, got {call_count[0]}"
        )

        # Backward should not call solver again
        y_final.sum().backward()
        assert call_count[0] == 1, (
            f"Expected still 1 solve after backward, got {call_count[0]}"
        )


class TestCheckpointing:
    def test_checkpointed_correctness(self):
        """Checkpointed adjoint should give same gradients as non-checkpointed."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Non-checkpointed
        solver1 = adjoint(dormand_prince_5, params=[theta])
        y1, _ = solver1(f, y0.clone(), (0.0, 10.0))
        y1.sum().backward()
        grad1 = theta.grad.clone()
        theta.grad = None

        # Checkpointed with 5 checkpoints
        theta2 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f2(t, y):
            return -theta2 * y

        solver2 = adjoint(dormand_prince_5, params=[theta2], checkpoints=5)
        y2, _ = solver2(f2, y0.clone(), (0.0, 10.0))
        y2.sum().backward()
        grad2 = theta2.grad.clone()

        assert torch.allclose(y1, y2, atol=1e-10), (
            "Forward solutions should match"
        )
        assert torch.allclose(grad1, grad2, atol=1e-4), (
            f"Gradients should match: {grad1.item()} vs {grad2.item()}"
        )

    def test_checkpointed_uses_less_memory_conceptually(self):
        """Checkpointed adjoint should use checkpoints, not full trajectory."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # With checkpoints - should work and give correct gradients
        solver = adjoint(dormand_prince_5, params=[theta], checkpoints=3)
        y_final, _ = solver(f, y0.clone(), (0.0, 10.0))
        y_final.sum().backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_checkpointed_with_varying_checkpoint_counts(self):
        """Test checkpointing with different numbers of checkpoints."""
        for n_checkpoints in [2, 5, 10]:
            theta = torch.tensor(
                [1.0], requires_grad=True, dtype=torch.float64
            )

            def f(t, y):
                return -theta * y

            y0 = torch.tensor([1.0], dtype=torch.float64)

            solver = adjoint(
                dormand_prince_5, params=[theta], checkpoints=n_checkpoints
            )
            y_final, _ = solver(f, y0.clone(), (0.0, 10.0))
            y_final.sum().backward()

            assert theta.grad is not None, (
                f"Failed with {n_checkpoints} checkpoints"
            )
            assert torch.isfinite(theta.grad), (
                f"Non-finite grad with {n_checkpoints} checkpoints"
            )

    def test_checkpointed_recomputes_during_backward(self):
        """Verify checkpointed adjoint calls solver during backward."""
        call_count = [0]
        original_dp5 = dormand_prince_5

        def counting_dp5(f, y0, t_span, **kwargs):
            call_count[0] += 1
            return original_dp5(f, y0, t_span, **kwargs)

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(counting_dp5, params=[theta], checkpoints=3)
        y_final, _ = solver(dynamics, y0, t_span=(0.0, 10.0))

        # Should call solver once for forward pass
        assert call_count[0] == 1, (
            f"Expected 1 forward solve, got {call_count[0]}"
        )

        # Backward should call solver again for recomputing segments
        y_final.sum().backward()
        # With 3 checkpoints, we have 3 segments, so solver should be called 3 more times
        assert call_count[0] == 4, (
            f"Expected 4 total solves (1 forward + 3 segment recomputes), got {call_count[0]}"
        )


class TestMemoryScaling:
    def test_memory_scaling_regression(self):
        """
        Verify O(1) memory for adjoint vs O(n) for direct backprop.

        This is the key benefit of the adjoint method.
        """
        import tracemalloc

        def measure_memory(use_adjoint, t_end):
            """Measure peak memory for solving and backprop."""
            theta = torch.tensor([1.0], requires_grad=True)

            def f(t, y):
                return -theta * y

            y0 = torch.randn(100)  # Moderate state size

            tracemalloc.start()

            if use_adjoint:
                solver = adjoint(dormand_prince_5, params=[theta])
                y_final, _ = solver(f, y0, t_span=(0.0, t_end))
            else:
                # Direct backprop through solver
                y_final, _ = dormand_prince_5(f, y0, t_span=(0.0, t_end))

            y_final.sum().backward()

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        # Measure at different integration lengths
        mem_adjoint_short = measure_memory(use_adjoint=True, t_end=1.0)
        mem_adjoint_long = measure_memory(use_adjoint=True, t_end=10.0)
        mem_direct_short = measure_memory(use_adjoint=False, t_end=1.0)
        mem_direct_long = measure_memory(use_adjoint=False, t_end=10.0)

        # Adjoint memory should scale sub-linearly
        adjoint_ratio = mem_adjoint_long / mem_adjoint_short
        direct_ratio = mem_direct_long / mem_direct_short

        # Adjoint ratio should be much smaller than direct ratio
        # Direct should grow ~10x, adjoint should grow much less
        assert adjoint_ratio < direct_ratio * 0.7, (
            f"Adjoint memory ratio {adjoint_ratio:.1f} should be less than "
            f"70% of direct {direct_ratio:.1f}"
        )

        # Absolute check: adjoint_long shouldn't be 10x adjoint_short
        assert adjoint_ratio < 5, (
            f"Adjoint memory grew too much: {adjoint_ratio:.1f}x"
        )


class TestPhase1bGraduation:
    """
    Graduation tests for Phase 1b.

    Run: pytest tests/.../test__adjoint_phase1b.py::TestPhase1bGraduation -v
    Pass condition: All YES-blocker tests pass.
    """

    def test_g1b1_single_forward_pass(self):
        """G1b.1 [BLOCKER]: Forward solve called exactly once."""
        call_count = [0]
        original = dormand_prince_5

        def counting_solver(f, y0, t_span, **kwargs):
            call_count[0] += 1
            return original(f, y0, t_span, **kwargs)

        theta = torch.tensor([1.0], requires_grad=True)
        solver = adjoint(counting_solver, params=[theta])
        y_final, _ = solver(
            lambda t, y: -theta * y, torch.tensor([1.0]), (0.0, 1.0)
        )
        y_final.sum().backward()

        assert call_count[0] == 1, (
            f"G1b.1 FAIL: {call_count[0]} calls instead of 1"
        )

    def test_g1b2_memory_scaling_regression(self):
        """G1b.2 [BLOCKER]: Memory ratio (long/short) < 10x for adjoint."""
        import tracemalloc

        def measure(t_end):
            theta = torch.tensor([1.0], requires_grad=True)
            tracemalloc.start()
            solver = adjoint(dormand_prince_5, params=[theta])
            y, _ = solver(
                lambda t, y: -theta * y, torch.randn(100), (0.0, t_end)
            )
            y.sum().backward()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        mem_short = measure(1.0)
        mem_long = measure(10.0)
        ratio = mem_long / mem_short

        assert ratio < 10, f"G1b.2 FAIL: memory ratio {ratio:.1f}x >= 10x"

    def test_g1b3_checkpointed_correctness(self):
        """G1b.3 [BLOCKER]: Checkpointed gradients match non-checkpointed."""
        theta1 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        theta2 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        y0 = torch.tensor([1.0], dtype=torch.float64)

        def f1(t, y):
            return -theta1 * y

        def f2(t, y):
            return -theta2 * y

        solver1 = adjoint(dormand_prince_5, params=[theta1])
        y1, _ = solver1(f1, y0.clone(), (0.0, 10.0))
        y1.sum().backward()

        solver2 = adjoint(dormand_prince_5, params=[theta2], checkpoints=5)
        y2, _ = solver2(f2, y0.clone(), (0.0, 10.0))
        y2.sum().backward()

        assert torch.allclose(theta1.grad, theta2.grad, atol=1e-4), (
            f"G1b.3 FAIL: {theta1.grad.item()} vs {theta2.grad.item()}"
        )
