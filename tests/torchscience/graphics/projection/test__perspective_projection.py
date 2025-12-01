"""Tests for perspective projection matrix operator."""

import math

import torch
from torch.autograd import gradcheck


class TestPerspectiveProjectionBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_scalar(self):
        """Output shape is (4, 4) for scalar inputs."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 4)  # 45 degrees
        aspect = torch.tensor(16.0 / 9.0)
        near = torch.tensor(0.1)
        far = torch.tensor(100.0)

        result = perspective_projection(fov, aspect, near, far)

        assert result.shape == (4, 4)

    def test_output_shape_batched(self):
        """Output shape is (B, 4, 4) for batched inputs."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor([math.pi / 4, math.pi / 3])
        aspect = torch.tensor([16.0 / 9.0, 4.0 / 3.0])
        near = torch.tensor([0.1, 0.1])
        far = torch.tensor([100.0, 1000.0])

        result = perspective_projection(fov, aspect, near, far)

        assert result.shape == (2, 4, 4)


class TestPerspectiveProjectionCorrectness:
    """Tests for numerical correctness."""

    def test_45_degree_fov(self):
        """Matches expected matrix for 45 degree FOV."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 4, dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(1.0, dtype=torch.float64)
        far = torch.tensor(10.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        # f = 1 / tan(fov/2) = 1 / tan(pi/8) ≈ 2.414
        f = 1.0 / math.tan(math.pi / 8)

        # Expected matrix (row-major)
        expected = torch.tensor(
            [
                [f, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    (far + near) / (near - far),
                    2 * far * near / (near - far),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=torch.float64,
        )

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_aspect_ratio_affects_x_scale(self):
        """Aspect ratio scales the x component."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 4, dtype=torch.float64)
        aspect = torch.tensor(2.0, dtype=torch.float64)
        near = torch.tensor(0.1, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        # Check that M[0,0] = f/aspect and M[1,1] = f
        f = 1.0 / math.tan(math.pi / 8)
        torch.testing.assert_close(
            result[0, 0], torch.tensor(f / 2.0, dtype=torch.float64)
        )
        torch.testing.assert_close(
            result[1, 1], torch.tensor(f, dtype=torch.float64)
        )

    def test_near_plane_maps_to_negative_one(self):
        """Points at near plane map to z = -1 in NDC."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 3, dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(1.0, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        M = perspective_projection(fov, aspect, near, far)

        # Point at near plane: [0, 0, -near, 1]
        point = torch.tensor(
            [0.0, 0.0, -near.item(), 1.0], dtype=torch.float64
        )
        clip = M @ point

        # After perspective divide: z_ndc = clip.z / clip.w
        z_ndc = clip[2] / clip[3]
        torch.testing.assert_close(
            z_ndc, torch.tensor(-1.0, dtype=torch.float64)
        )

    def test_far_plane_maps_to_positive_one(self):
        """Points at far plane map to z = +1 in NDC."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 3, dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(1.0, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        M = perspective_projection(fov, aspect, near, far)

        # Point at far plane: [0, 0, -far, 1]
        point = torch.tensor([0.0, 0.0, -far.item(), 1.0], dtype=torch.float64)
        clip = M @ point

        # After perspective divide: z_ndc = clip.z / clip.w
        z_ndc = clip[2] / clip[3]
        torch.testing.assert_close(
            z_ndc, torch.tensor(1.0, dtype=torch.float64)
        )


class TestPerspectiveProjectionGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(
            math.pi / 4, dtype=torch.float64, requires_grad=True
        )
        aspect = torch.tensor(
            16.0 / 9.0, dtype=torch.float64, requires_grad=True
        )
        near = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        far = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)

        def func(fov, aspect, near, far):
            return perspective_projection(fov, aspect, near, far)

        assert gradcheck(func, (fov, aspect, near, far), raise_exception=True)


class TestPerspectiveProjectionDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 4, dtype=torch.float32)
        aspect = torch.tensor(1.0, dtype=torch.float32)
        near = torch.tensor(0.1, dtype=torch.float32)
        far = torch.tensor(100.0, dtype=torch.float32)

        result = perspective_projection(fov, aspect, near, far)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.pi / 4, dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(0.1, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        assert result.dtype == torch.float64
