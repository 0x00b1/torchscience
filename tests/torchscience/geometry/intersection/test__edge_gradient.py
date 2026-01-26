"""Tests for edge_gradient_contribution."""

import pytest
import torch

from torchscience.geometry.intersection._edge_gradient import (
    edge_gradient_contribution,
)
from torchscience.geometry.intersection._edge_sampling_result import (
    EdgeSamples,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def edge_samples_4():
    """Four edge samples with known tangent and length."""
    return EdgeSamples(
        positions=torch.randn(4, 3),
        edge_indices=torch.tensor([0, 0, 1, 1]),
        parametric_t=torch.tensor([0.25, 0.75, 0.25, 0.75]),
        edge_tangent=torch.tensor(
            [[1.0, 0.0, 0.0]] * 4
        ),  # unit tangent along x
        edge_length=torch.tensor([2.0, 2.0, 3.0, 3.0]),
        batch_size=[4],
    )


@pytest.fixture()
def ray_dirs_4():
    """Ray directions for 4 samples, pointing roughly along +z."""
    return torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestEdgeGradientShapes:
    """Output shape validation."""

    def test_scalar_output_shape(self, edge_samples_4, ray_dirs_4):
        """Scalar delta_intensity (N,) produces (N,) output."""
        delta = torch.ones(4)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        assert result.shape == (4,)

    def test_multichannel_output_shape(self, edge_samples_4, ray_dirs_4):
        """Multi-channel delta_intensity (N, C) produces (N, C) output."""
        delta = torch.ones(4, 3)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        assert result.shape == (4, 3)

    def test_multichannel_5_channels(self, edge_samples_4, ray_dirs_4):
        """Works with arbitrary channel count C=5."""
        delta = torch.ones(4, 5)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        assert result.shape == (4, 5)


# ---------------------------------------------------------------------------
# Zero delta tests
# ---------------------------------------------------------------------------


class TestEdgeGradientZeroDelta:
    """Zero intensity difference yields zero contribution."""

    def test_zero_delta_scalar(self, edge_samples_4, ray_dirs_4):
        """Zero scalar delta gives zero output."""
        delta = torch.zeros(4)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        assert torch.allclose(result, torch.zeros(4), atol=1e-7)

    def test_zero_delta_multichannel(self, edge_samples_4, ray_dirs_4):
        """Zero multi-channel delta gives zero output."""
        delta = torch.zeros(4, 3)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        assert torch.allclose(result, torch.zeros(4, 3), atol=1e-7)


# ---------------------------------------------------------------------------
# Linearity tests
# ---------------------------------------------------------------------------


class TestEdgeGradientLinearity:
    """Contribution scales linearly with intensity difference."""

    def test_scalar_linearity(self, edge_samples_4, ray_dirs_4):
        """Doubling delta doubles the contribution."""
        delta = torch.randn(4)
        result_1x = edge_gradient_contribution(
            edge_samples_4, delta, ray_dirs_4
        )
        result_2x = edge_gradient_contribution(
            edge_samples_4, 2.0 * delta, ray_dirs_4
        )
        assert torch.allclose(result_2x, 2.0 * result_1x, atol=1e-6)

    def test_multichannel_linearity(self, edge_samples_4, ray_dirs_4):
        """Scaling multi-channel delta scales output proportionally."""
        delta = torch.randn(4, 3)
        scale = 3.14
        result_1x = edge_gradient_contribution(
            edge_samples_4, delta, ray_dirs_4
        )
        result_sx = edge_gradient_contribution(
            edge_samples_4, scale * delta, ray_dirs_4
        )
        assert torch.allclose(result_sx, scale * result_1x, atol=1e-5)


# ---------------------------------------------------------------------------
# Screen normals override tests
# ---------------------------------------------------------------------------


class TestEdgeGradientScreenNormals:
    """Custom screen_normals override default computation."""

    def test_screen_normals_used(self, edge_samples_4, ray_dirs_4):
        """Providing screen_normals changes the result vs default."""
        delta = torch.ones(4)

        result_default = edge_gradient_contribution(
            edge_samples_4, delta, ray_dirs_4
        )

        # Custom normals different from computed normals
        custom_normals = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        result_custom = edge_gradient_contribution(
            edge_samples_4, delta, ray_dirs_4, screen_normals=custom_normals
        )

        # With custom normals [0,1,0] and ray dir [0,0,1], dot product is 0
        # which gets clamped to 1e-8, producing very large weights.
        # This should differ from the default computation.
        assert not torch.allclose(result_default, result_custom)

    def test_screen_normals_value(self, edge_samples_4, ray_dirs_4):
        """Verify screen_normals produce the expected weight."""
        delta = torch.ones(4)

        # Custom normals aligned with ray direction
        custom_normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = edge_gradient_contribution(
            edge_samples_4, delta, ray_dirs_4, screen_normals=custom_normals
        )

        # n_e . d = 1.0 for all, so weight = edge_length / 1.0 = edge_length
        expected = edge_samples_4.edge_length
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Weight computation tests
# ---------------------------------------------------------------------------


class TestEdgeGradientWeight:
    """Verify the geometric weight computation."""

    def test_weight_equals_edge_length_over_dot(self):
        """Weight = edge_length / (n_e . d) for unit delta."""
        samples = EdgeSamples(
            positions=torch.zeros(2, 3),
            edge_indices=torch.tensor([0, 1]),
            parametric_t=torch.tensor([0.5, 0.5]),
            edge_tangent=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            edge_length=torch.tensor([4.0, 6.0]),
            batch_size=[2],
        )

        # Ray direction along z for both
        ray_dirs = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        delta = torch.ones(2)
        result = edge_gradient_contribution(samples, delta, ray_dirs)

        # For tangent [1,0,0] and d [0,0,1]:
        #   rejection = [0,0,1] - 0*[1,0,0] = [0,0,1], normalized = [0,0,1]
        #   n_e . d = 1.0, weight = edge_length / 1.0
        # For tangent [0,1,0] and d [0,0,1]:
        #   rejection = [0,0,1] - 0*[0,1,0] = [0,0,1], normalized = [0,0,1]
        #   n_e . d = 1.0, weight = edge_length / 1.0
        expected = torch.tensor([4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestEdgeGradientAutograd:
    """Autograd gradient flow."""

    def test_gradient_flows_to_delta_intensity(
        self, edge_samples_4, ray_dirs_4
    ):
        """Gradient flows through delta_intensity."""
        delta = torch.randn(4, requires_grad=True)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        loss = result.sum()
        loss.backward()
        assert delta.grad is not None
        assert (delta.grad.abs() > 0).any()

    def test_gradient_flows_to_positions(self):
        """Gradient flows to vertices via edge_samples positions.

        Here we create edge_samples with positions that require grad and
        verify that gradients propagate.  Although the current algorithm
        does not directly use ``positions``, the tangent and length are
        often derived from positions upstream. We test that the overall
        computation graph remains differentiable by requiring grad on a
        tensor that feeds into EdgeSamples fields used by the function
        (edge_tangent via positions).
        """
        # Build edge tangent from positions to establish grad dependency
        v0 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v1 = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)

        diff = v1 - v0
        length = torch.linalg.norm(diff).unsqueeze(0)
        tangent = (diff / length).unsqueeze(0)  # (1, 3)

        samples = EdgeSamples(
            positions=(v0 + v1).unsqueeze(0) / 2.0,  # midpoint, (1, 3)
            edge_indices=torch.tensor([0]),
            parametric_t=torch.tensor([0.5]),
            edge_tangent=tangent,
            edge_length=length,
            batch_size=[1],
        )

        ray_dirs = torch.tensor([[0.0, 0.0, 1.0]])
        delta = torch.tensor([1.0])

        result = edge_gradient_contribution(samples, delta, ray_dirs)
        result.sum().backward()

        # v1 participates through tangent and length
        assert v1.grad is not None
        assert (v1.grad.abs() > 0).any()

    def test_gradient_flows_multichannel(self, edge_samples_4, ray_dirs_4):
        """Gradient flows for multi-channel delta."""
        delta = torch.randn(4, 3, requires_grad=True)
        result = edge_gradient_contribution(edge_samples_4, delta, ray_dirs_4)
        loss = result.sum()
        loss.backward()
        assert delta.grad is not None
        assert (delta.grad.abs() > 0).any()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestEdgeGradientValidation:
    """Input validation."""

    def test_invalid_ray_directions_shape(self, edge_samples_4):
        """ray_directions with wrong shape raises ValueError."""
        delta = torch.ones(4)
        with pytest.raises(ValueError, match="ray_directions"):
            edge_gradient_contribution(
                edge_samples_4, delta, torch.randn(4, 2)
            )

    def test_mismatched_ray_count(self, edge_samples_4):
        """Mismatched sample count in ray_directions raises ValueError."""
        delta = torch.ones(4)
        with pytest.raises(ValueError, match="ray_directions"):
            edge_gradient_contribution(
                edge_samples_4, delta, torch.randn(3, 3)
            )

    def test_mismatched_delta_count(self, edge_samples_4):
        """Mismatched sample count in delta_intensity raises ValueError."""
        with pytest.raises(ValueError, match="delta_intensity"):
            edge_gradient_contribution(
                edge_samples_4, torch.ones(3), torch.randn(4, 3)
            )

    def test_invalid_screen_normals_shape(self, edge_samples_4):
        """screen_normals with wrong shape raises ValueError."""
        delta = torch.ones(4)
        with pytest.raises(ValueError, match="screen_normals"):
            edge_gradient_contribution(
                edge_samples_4,
                delta,
                torch.randn(4, 3),
                screen_normals=torch.randn(4, 2),
            )

    def test_invalid_screen_normals_count(self, edge_samples_4):
        """screen_normals with wrong count raises ValueError."""
        delta = torch.ones(4)
        with pytest.raises(ValueError, match="screen_normals"):
            edge_gradient_contribution(
                edge_samples_4,
                delta,
                torch.randn(4, 3),
                screen_normals=torch.randn(3, 3),
            )
