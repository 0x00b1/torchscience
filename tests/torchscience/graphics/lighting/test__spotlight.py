"""Tests for spotlight light source."""

import math

import pytest
import torch
from torch.autograd import gradcheck


class TestSpotlightBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output has correct shape for single sample."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]])
        surface_pos = torch.tensor([[0.0, 0.0, 0.0]])
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]])
        intensity = torch.tensor([100.0])
        inner_angle = torch.tensor([math.radians(15)])
        outer_angle = torch.tensor([math.radians(30)])

        irradiance, light_dir = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert irradiance.shape == (1,)
        assert light_dir.shape == (1, 3)

    def test_output_shape_batch(self):
        """Output has correct shape for batched input."""
        from torchscience.graphics.lighting import spotlight

        batch = 10
        light_pos = torch.randn(batch, 3)
        surface_pos = torch.randn(batch, 3)
        spot_direction = torch.randn(batch, 3)
        spot_direction = spot_direction / spot_direction.norm(
            dim=-1, keepdim=True
        )
        intensity = torch.full((batch,), 100.0)
        inner_angle = torch.full((batch,), math.radians(15))
        outer_angle = torch.full((batch,), math.radians(30))

        irradiance, light_dir = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert irradiance.shape == (batch,)
        assert light_dir.shape == (batch, 3)

    def test_irradiance_non_negative(self):
        """Irradiance is always non-negative."""
        from torchscience.graphics.lighting import spotlight

        torch.manual_seed(42)
        batch = 100
        light_pos = torch.randn(batch, 3)
        surface_pos = torch.randn(batch, 3)
        spot_direction = torch.randn(batch, 3)
        spot_direction = spot_direction / spot_direction.norm(
            dim=-1, keepdim=True
        )
        intensity = torch.full((batch,), 100.0)
        inner_angle = torch.full((batch,), math.radians(15))
        outer_angle = torch.full((batch,), math.radians(30))

        irradiance, _ = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert (irradiance >= 0).all()


class TestSpotlightCorrectness:
    """Tests for numerical correctness."""

    def test_inside_inner_cone_full_intensity(self):
        """Point inside inner cone receives full intensity (before distance falloff)."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        surface_pos = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float64
        )  # Directly below
        spot_direction = torch.tensor(
            [[0.0, -1.0, 0.0]], dtype=torch.float64
        )  # Pointing down
        intensity = torch.tensor([100.0], dtype=torch.float64)
        inner_angle = torch.tensor([math.radians(30)], dtype=torch.float64)
        outer_angle = torch.tensor([math.radians(45)], dtype=torch.float64)

        irradiance, _ = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        # Distance is 5, so irradiance = 100 / 25 = 4.0 (full intensity, no angular falloff)
        expected = 100.0 / 25.0
        torch.testing.assert_close(
            irradiance,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_outside_outer_cone_zero(self):
        """Point outside outer cone receives zero irradiance."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        # Surface at 60 degrees from spotlight direction (beyond outer cone)
        surface_pos = torch.tensor([[8.66, 0.0, 0.0]], dtype=torch.float64)
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        intensity = torch.tensor([100.0], dtype=torch.float64)
        inner_angle = torch.tensor([math.radians(15)], dtype=torch.float64)
        outer_angle = torch.tensor([math.radians(30)], dtype=torch.float64)

        irradiance, _ = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert irradiance.item() == pytest.approx(0.0, abs=1e-6)

    def test_light_direction_normalized(self):
        """Returned light direction is normalized."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]])
        surface_pos = torch.tensor([[3.0, 2.0, 4.0]])
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]])
        intensity = torch.tensor([100.0])
        inner_angle = torch.tensor([math.radians(45)])
        outer_angle = torch.tensor([math.radians(60)])

        _, light_dir = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        norm = light_dir.norm(dim=-1)
        torch.testing.assert_close(
            norm, torch.ones_like(norm), rtol=1e-5, atol=1e-7
        )


class TestSpotlightGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor(
            [[0.0, 5.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        surface_pos = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        spot_direction = torch.tensor(
            [[0.0, -1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        intensity = torch.tensor(
            [100.0], dtype=torch.float64, requires_grad=True
        )
        inner_angle = torch.tensor(
            [math.radians(30)], dtype=torch.float64, requires_grad=True
        )
        outer_angle = torch.tensor(
            [math.radians(45)], dtype=torch.float64, requires_grad=True
        )

        def func(lp, sp, sd, i, ia, oa):
            irr, _ = spotlight(
                lp, sp, sd, intensity=i, inner_angle=ia, outer_angle=oa
            )
            return irr

        assert gradcheck(
            func,
            (
                light_pos,
                surface_pos,
                spot_direction,
                intensity,
                inner_angle,
                outer_angle,
            ),
            raise_exception=True,
        )


class TestSpotlightDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.randn(5, 3, dtype=torch.float32)
        surface_pos = torch.randn(5, 3, dtype=torch.float32)
        spot_direction = torch.randn(5, 3, dtype=torch.float32)
        spot_direction = spot_direction / spot_direction.norm(
            dim=-1, keepdim=True
        )
        intensity = torch.full((5,), 100.0, dtype=torch.float32)
        inner_angle = torch.full((5,), 0.5, dtype=torch.float32)
        outer_angle = torch.full((5,), 0.8, dtype=torch.float32)

        irradiance, light_dir = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert irradiance.dtype == torch.float32
        assert light_dir.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.randn(5, 3, dtype=torch.float64)
        surface_pos = torch.randn(5, 3, dtype=torch.float64)
        spot_direction = torch.randn(5, 3, dtype=torch.float64)
        spot_direction = spot_direction / spot_direction.norm(
            dim=-1, keepdim=True
        )
        intensity = torch.full((5,), 100.0, dtype=torch.float64)
        inner_angle = torch.full((5,), 0.5, dtype=torch.float64)
        outer_angle = torch.full((5,), 0.8, dtype=torch.float64)

        irradiance, light_dir = spotlight(
            light_pos,
            surface_pos,
            spot_direction,
            intensity=intensity,
            inner_angle=inner_angle,
            outer_angle=outer_angle,
        )

        assert irradiance.dtype == torch.float64
        assert light_dir.dtype == torch.float64
