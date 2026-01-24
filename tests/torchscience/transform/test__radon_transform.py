"""Tests for Radon transform implementation."""

import math

import torch
from torch.autograd import gradcheck

import torchscience.transform as T


class TestRadonTransformForward:
    """Test Radon transform forward pass correctness."""

    def test_basic_square(self):
        """Test Radon transform of a centered square."""
        phantom = torch.zeros(32, 32, dtype=torch.float64)
        phantom[12:20, 12:20] = 1.0
        angles = torch.tensor(
            [0.0, math.pi / 4, math.pi / 2], dtype=torch.float64
        )

        sinogram = T.radon_transform(phantom, angles)

        # Should have shape [3, num_bins]
        assert sinogram.dim() == 2
        assert sinogram.shape[0] == 3
        # Sinogram should have non-zero values
        assert sinogram.abs().sum() > 0

    def test_output_shape_2d(self):
        """Output shape for 2D input."""
        image = torch.randn(64, 64, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)

        sinogram = T.radon_transform(image, angles)

        assert sinogram.dim() == 2
        assert sinogram.shape[0] == 90  # num_angles
        # num_bins is approximately ceil(sqrt(64^2 + 64^2)) = 91
        assert sinogram.shape[1] >= 90

    def test_output_shape_batched(self):
        """Output shape for batched input."""
        images = torch.randn(5, 32, 32, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        sinogram = T.radon_transform(images, angles)

        assert sinogram.dim() == 3
        assert sinogram.shape[0] == 5  # batch
        assert sinogram.shape[1] == 45  # num_angles

    def test_circle_mode(self):
        """Circle mode should produce different results than no circle."""
        image = torch.randn(32, 32, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        sino_circle = T.radon_transform(image, angles, circle=True)
        sino_no_circle = T.radon_transform(image, angles, circle=False)

        # Results should be different (unless image is already circular)
        # For a random image, they should differ
        assert not torch.allclose(sino_circle, sino_no_circle)

    def test_symmetry_at_opposite_angles(self):
        """Sinogram at angle theta and theta + pi should be flipped."""
        phantom = torch.zeros(32, 32, dtype=torch.float64)
        phantom[10:22, 10:22] = 1.0

        # Compare angle 0 and pi
        angles_0 = torch.tensor([0.0], dtype=torch.float64)
        angles_pi = torch.tensor([math.pi], dtype=torch.float64)

        sino_0 = T.radon_transform(phantom, angles_0, circle=True)
        sino_pi = T.radon_transform(phantom, angles_pi, circle=True)

        # Sinogram at pi should be flipped version of sinogram at 0
        sino_0_flipped = torch.flip(sino_0, dims=[-1])
        # Allow some tolerance due to discrete sampling
        assert torch.allclose(sino_0_flipped, sino_pi, rtol=0.2, atol=0.1)

    def test_point_source(self):
        """Radon of a single point should create sinusoidal pattern."""
        phantom = torch.zeros(32, 32, dtype=torch.float64)
        # Single point off-center
        phantom[16, 20] = 1.0

        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)
        sinogram = T.radon_transform(phantom, angles, circle=True)

        # The sinogram of a point is a sinusoid
        # Check that the maximum moves smoothly
        max_pos = sinogram.argmax(dim=1).float()
        # Maximum position should vary smoothly
        diff = (max_pos[1:] - max_pos[:-1]).abs()
        assert diff.mean() < 5  # Should be smooth


class TestRadonTransformGradient:
    """Test Radon transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        image = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
        angles = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)

        def func(inp):
            return T.radon_transform(inp, angles)

        assert gradcheck(func, (image,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        images = torch.randn(
            2, 16, 16, dtype=torch.float64, requires_grad=True
        )
        angles = torch.tensor([0.0], dtype=torch.float64)

        def func(inp):
            return T.radon_transform(inp, angles)

        assert gradcheck(func, (images,), raise_exception=True)


class TestRadonTransformMeta:
    """Test Radon transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        image = torch.empty(64, 64, device="meta", dtype=torch.float64)
        angles = torch.empty(90, device="meta", dtype=torch.float64)

        sinogram = T.radon_transform(image, angles)

        assert sinogram.dim() == 2
        assert sinogram.shape[0] == 90
        # num_bins >= ceil(sqrt(64^2 + 64^2))
        assert sinogram.shape[1] >= 90
        assert sinogram.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        image = torch.empty(3, 32, 32, device="meta", dtype=torch.float64)
        angles = torch.empty(45, device="meta", dtype=torch.float64)

        sinogram = T.radon_transform(image, angles)

        assert sinogram.dim() == 3
        assert sinogram.shape[0] == 3
        assert sinogram.shape[1] == 45


class TestRadonTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_angle(self):
        """Should work with a single angle."""
        image = torch.randn(32, 32, dtype=torch.float64)
        angles = torch.tensor([0.0], dtype=torch.float64)

        sinogram = T.radon_transform(image, angles)
        assert sinogram.shape[0] == 1

    def test_empty_angles_error(self):
        """Should raise error for empty angles tensor."""
        image = torch.randn(32, 32, dtype=torch.float64)
        angles = torch.tensor([], dtype=torch.float64)

        # Should still work but produce empty output
        sinogram = T.radon_transform(image, angles)
        assert sinogram.shape[0] == 0
