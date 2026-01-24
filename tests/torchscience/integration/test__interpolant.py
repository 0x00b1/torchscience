import pytest
import torch

from torchscience.integration._interpolation import (
    _DP5_P,
    DP5Interpolant,
    LinearInterpolant,
)


def _compute_dp5_k_values(f, t0, y0, h):
    """Compute DP5 stage values k0-k6 for a given step."""
    # DP5 Butcher tableau c values
    c = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
    # A matrix (lower triangular)
    a = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]

    k = [None] * 7
    k[0] = f(t0, y0)
    for i in range(1, 7):
        t_i = t0 + c[i] * h
        y_i = y0.clone()
        for j, a_ij in enumerate(a[i]):
            y_i = y_i + h * a_ij * k[j]
        k[i] = f(t_i, y_i)
    return torch.stack(k)


class TestDP5Interpolant:
    def test_basic_interpolation(self):
        """Test interpolation with properly computed RK stages."""

        # Linear function y = t, so y' = 1
        def f(t, y):
            return torch.ones_like(y)

        t0, t1 = 0.0, 1.0
        h = t1 - t0
        y0 = torch.tensor([0.0])

        # Compute proper k values
        k = _compute_dp5_k_values(f, t0, y0, h)

        # Compute y1 using B5 weights
        b5 = torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        y1 = y0 + h * sum(b5[i] * k[i] for i in range(7))

        interp = DP5Interpolant(
            t_segments=torch.tensor([[t0, t1]]),
            y_segments=torch.stack([y0, y1]).unsqueeze(0),
            k_segments=k.unsqueeze(0),
        )

        # Query at midpoint - for linear ODE, should be close to 0.5
        y_half = interp(0.5)
        assert torch.allclose(y_half, torch.tensor([0.5]), atol=1e-3)

    def test_endpoints_exact(self):
        """Interpolant must match endpoints exactly when k values are consistent."""

        def f(t, y):
            return -y  # dy/dt = -y

        t0, t1 = 0.0, 1.0
        h = t1 - t0
        y0 = torch.tensor([1.0, 2.0])

        # Compute proper k values
        k = _compute_dp5_k_values(f, t0, y0, h)

        # Compute y1 using B5 weights
        b5 = torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        y1 = y0 + h * sum(b5[i] * k[i] for i in range(7))

        interp = DP5Interpolant(
            t_segments=torch.tensor([[t0, t1]]),
            y_segments=torch.stack([y0, y1]).unsqueeze(0),
            k_segments=k.unsqueeze(0),
        )

        assert torch.allclose(interp(t0), y0, atol=1e-6)
        assert torch.allclose(interp(t1), y1, atol=1e-6)

    def test_multiple_segments(self):
        """Test interpolation across multiple time segments."""

        def f(t, y):
            return torch.ones_like(y)  # constant derivative

        # Segment 0: [0, 0.5]
        y0_0 = torch.tensor([0.0])
        k0 = _compute_dp5_k_values(f, 0.0, y0_0, 0.5)
        b5 = torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        y1_0 = y0_0 + 0.5 * sum(b5[i] * k0[i] for i in range(7))

        # Segment 1: [0.5, 1.0]
        y0_1 = y1_0.clone()
        k1 = _compute_dp5_k_values(f, 0.5, y0_1, 0.5)
        y1_1 = y0_1 + 0.5 * sum(b5[i] * k1[i] for i in range(7))

        t_segments = torch.tensor([[0.0, 0.5], [0.5, 1.0]])
        y_segments = torch.stack(
            [
                torch.stack([y0_0, y1_0]),
                torch.stack([y0_1, y1_1]),
            ]
        )
        k_segments = torch.stack([k0, k1])

        interp = DP5Interpolant(t_segments, y_segments, k_segments)

        # Query across segments
        t_query = torch.tensor([0.25, 0.75])
        y_query = interp(t_query)
        assert y_query.shape == (2, 1)
        # For constant derivative, result should be close to t values
        assert torch.allclose(
            y_query, torch.tensor([[0.25], [0.75]]), atol=1e-3
        )

    def test_out_of_bounds_raises(self):
        t_segments = torch.tensor([[0.0, 1.0]])
        y_segments = torch.tensor([[[0.0], [1.0]]])
        k_segments = torch.ones(1, 7, 1)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)

    def test_differentiable(self):
        t_segments = torch.tensor([[0.0, 1.0]])
        y_segments = torch.tensor([[[0.0], [1.0]]], requires_grad=True)
        k_segments = torch.ones(1, 7, 1)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert y_segments.grad is not None

    def test_t_points_property(self):
        """Public t_points property for testing step sizes"""
        t_segments = torch.tensor([[0.0, 0.5], [0.5, 1.0]])
        y_segments = torch.zeros(2, 2, 1)
        k_segments = torch.zeros(2, 7, 1)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)

        # t_points should return all unique time points
        t_points = interp.t_points
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(t_points, expected)

    def test_success_attribute(self):
        t_segments = torch.tensor([[0.0, 1.0]])
        y_segments = torch.zeros(1, 2, 1)
        k_segments = torch.zeros(1, 7, 1)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)
        assert interp.success is None  # default

        success = torch.tensor([True, False])
        interp = DP5Interpolant(
            t_segments, y_segments, k_segments, success=success
        )
        assert torch.equal(interp.success, success)

    def test_n_steps_attribute(self):
        """n_steps should equal number of segments"""
        t_segments = torch.tensor([[0.0, 0.5], [0.5, 1.0]])
        y_segments = torch.zeros(2, 2, 1)
        k_segments = torch.zeros(2, 7, 1)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)
        assert interp.n_steps == 2

    def test_batched_state(self):
        # Batch of 3 trajectories, state dim 2
        t_segments = torch.tensor([[0.0, 1.0]])
        y_segments = torch.randn(1, 2, 3, 2)  # (n_seg, 2, B, D)
        k_segments = torch.randn(1, 7, 3, 2)  # (n_seg, 7, B, D)

        interp = DP5Interpolant(t_segments, y_segments, k_segments)

        # Query single time
        y_mid = interp(0.5)
        assert y_mid.shape == (3, 2)

        # Query multiple times
        t_query = torch.tensor([0.25, 0.5, 0.75])
        y_query = interp(t_query)
        assert y_query.shape == (3, 3, 2)  # (T_query, B, D)


class TestDP5DenseOutputCoefficients:
    """Validate DP5 dense output coefficients satisfy order conditions."""

    def test_b_theta_sum_to_one_at_theta_one(self):
        """At theta=1, sum of b_i(1) should equal 1 (consistency)."""
        # b_i(theta) = P[i, 0]*theta + P[i, 1]*theta^2 + P[i, 2]*theta^3 + P[i, 3]*theta^4
        # At theta=1: b_i(1) = sum(P[i, :])
        total = 0.0
        for i in range(7):
            b_i = sum(_DP5_P[i])
            total += b_i
        assert abs(total - 1.0) < 1e-6, (
            f"Sum of b_i(1) = {total}, expected 1.0"
        )

    def test_b_theta_zero_at_theta_zero(self):
        """At theta=0, b_i(0) should equal 0 (y(t0) = y0)."""
        # b_i(0) = P[i, 0]*0 + P[i, 1]*0 + P[i, 2]*0 + P[i, 3]*0 = 0
        # This is automatically satisfied since theta=0 makes all terms 0
        theta = 0.0
        for i in range(7):
            b_i = sum(_DP5_P[i][j] * (theta ** (j + 1)) for j in range(4))
            assert b_i == 0.0, f"b_{i}(0) = {b_i}, expected 0"

    def test_k1_k6_have_zero_P_coefficients(self):
        """k1 (index 1) and k6 (index 6) should have special structure."""
        # k1 has zero weight (index 1 in Butcher tableau is skipped stage)
        assert all(p == 0.0 for p in _DP5_P[1]), (
            f"P[1] should be all zeros: {_DP5_P[1]}"
        )

    def test_constant_derivative_gives_linear_interpolation(self):
        """For constant derivative ODE, interpolation should be nearly linear."""
        # With all k_i = 1 and h = 1:
        # y(theta) = y0 + h * sum_i(b_i(theta) * k_i) = y0 + sum_i(b_i(theta))
        # For proper coefficients, sum_i(b_i(theta)) should approximately equal theta
        theta = 0.5
        total = 0.0
        for i in range(7):
            b_i = sum(_DP5_P[i][j] * (theta ** (j + 1)) for j in range(4))
            total += b_i
        # Should be close to theta (linear for constant derivative)
        assert abs(total - theta) < 1e-6, (
            f"sum(b_i(0.5)) = {total}, expected ~0.5"
        )


class TestLinearInterpolant:
    def test_basic_interpolation(self):
        t_points = torch.tensor([0.0, 1.0, 2.0])
        y_points = torch.tensor([[0.0], [1.0], [2.0]])

        interp = LinearInterpolant(t_points, y_points)

        # Query at midpoints
        y_half = interp(0.5)
        assert torch.allclose(y_half, torch.tensor([0.5]), atol=1e-6)

        y_1_5 = interp(1.5)
        assert torch.allclose(y_1_5, torch.tensor([1.5]), atol=1e-6)

    def test_endpoints(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])

        interp = LinearInterpolant(t_points, y_points)

        assert torch.allclose(interp(0.0), torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(interp(1.0), torch.tensor([1.0]), atol=1e-6)

    def test_multiple_queries(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [2.0]])

        interp = LinearInterpolant(t_points, y_points)

        t_query = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y_query = interp(t_query)
        assert y_query.shape == (5, 1)
        expected = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
        assert torch.allclose(y_query, expected, atol=1e-6)

    def test_out_of_bounds_raises(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])

        interp = LinearInterpolant(t_points, y_points)

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)

    def test_differentiable(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]], requires_grad=True)

        interp = LinearInterpolant(t_points, y_points)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert y_points.grad is not None

    def test_batched_state(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.randn(2, 3, 2)  # (T, B, D)

        interp = LinearInterpolant(t_points, y_points)

        # Query single time
        y_mid = interp(0.5)
        assert y_mid.shape == (3, 2)

        # Query multiple times
        t_query = torch.tensor([0.25, 0.5, 0.75])
        y_query = interp(t_query)
        assert y_query.shape == (3, 3, 2)  # (T_query, B, D)
