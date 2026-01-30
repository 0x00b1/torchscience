"""Tests for Radon transform implementation."""

import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T

# Check if skimage is available for reference tests
try:
    from skimage.transform import radon as skimage_radon

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        image = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
        angles = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)

        def func(inp):
            return T.radon_transform(inp, angles)

        assert gradgradcheck(func, (image,), raise_exception=True)


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


class TestRadonTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that radon transform works with vmap."""
        images = torch.randn(4, 16, 16, dtype=torch.float64)
        angles = torch.linspace(0, torch.pi, 10, dtype=torch.float64)

        # Direct batched call
        sinogram_batched = T.radon_transform(images, angles)

        # vmap version
        def radon_single(img):
            return T.radon_transform(img, angles)

        sinogram_vmap = torch.vmap(radon_single)(images)

        assert torch.allclose(sinogram_batched, sinogram_vmap, atol=1e-6)


class TestRadonTransformCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that radon transform works with torch.compile."""
        image = torch.randn(16, 16, dtype=torch.float64)
        angles = torch.linspace(0, torch.pi, 10, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_radon(img, angles):
            return T.radon_transform(img, angles)

        sinogram_compiled = compiled_radon(image, angles)
        sinogram_eager = T.radon_transform(image, angles)

        assert torch.allclose(sinogram_compiled, sinogram_eager, atol=1e-10)


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not available")
class TestRadonTransformSkimageReference:
    """Tests comparing against scikit-image's radon implementation."""

    def test_matches_skimage_basic(self):
        """Test basic case matches scikit-image."""
        np.random.seed(42)
        image_np = np.random.randn(32, 32)
        image_torch = torch.from_numpy(image_np)

        # Angles in degrees for skimage, radians for torchscience
        angles_deg = np.linspace(0, 180, 45, endpoint=False)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        skimage_result = skimage_radon(image_np, theta=angles_deg, circle=True)
        torch_result = T.radon_transform(
            image_torch, angles_rad, circle=True
        ).numpy()

        # Note: skimage returns (num_bins, num_angles), we return (num_angles, num_bins)
        skimage_result = skimage_result.T

        # Results should be close (implementation differences may cause some variation)
        # Use correlation to verify similar structure
        correlation = np.corrcoef(
            skimage_result.flatten(), torch_result.flatten()
        )[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"

    def test_matches_skimage_shepp_logan(self):
        """Test with Shepp-Logan phantom-like structure."""
        # Create a simple disk phantom
        size = 64
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 4
        disk = ((x - center) ** 2 + (y - center) ** 2 <= radius**2).astype(
            np.float64
        )

        image_torch = torch.from_numpy(disk)

        angles_deg = np.linspace(0, 180, 90, endpoint=False)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        skimage_result = skimage_radon(disk, theta=angles_deg, circle=True)
        torch_result = T.radon_transform(
            image_torch, angles_rad, circle=True
        ).numpy()

        # skimage returns (num_bins, num_angles)
        skimage_result = skimage_result.T

        # For simple geometric phantoms, results should be very similar
        correlation = np.corrcoef(
            skimage_result.flatten(), torch_result.flatten()
        )[0, 1]
        assert correlation > 0.95, f"Correlation too low: {correlation}"

    def test_sinogram_structure_matches(self):
        """Test that sinogram has similar structure (max positions)."""
        # Point source creates sinusoidal pattern
        size = 32
        phantom = np.zeros((size, size), dtype=np.float64)
        phantom[size // 2, size // 2 + 5] = 1.0  # Off-center point

        image_torch = torch.from_numpy(phantom)

        angles_deg = np.linspace(0, 180, 45, endpoint=False)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        skimage_result = skimage_radon(phantom, theta=angles_deg, circle=True)
        torch_result = T.radon_transform(
            image_torch, angles_rad, circle=True
        ).numpy()

        # skimage returns (num_bins, num_angles)
        skimage_result = skimage_result.T

        # Find peak positions for each angle
        skimage_peaks = np.argmax(skimage_result, axis=1)
        torch_peaks = np.argmax(torch_result, axis=1)

        # Peaks should follow similar pattern (allowing for different centering)
        # Normalize to center
        skimage_peaks_centered = skimage_peaks - skimage_peaks.mean()
        torch_peaks_centered = torch_peaks - torch_peaks.mean()

        correlation = np.corrcoef(
            skimage_peaks_centered, torch_peaks_centered
        )[0, 1]
        assert correlation > 0.9, (
            f"Peak pattern correlation too low: {correlation}"
        )
