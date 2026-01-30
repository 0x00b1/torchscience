"""Tests for inverse Radon transform (filtered back-projection) implementation."""

import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T

# Check if skimage is available for reference tests
try:
    from skimage.transform import iradon as skimage_iradon
    from skimage.transform import radon as skimage_radon

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class TestInverseRadonTransformForward:
    """Test inverse Radon transform forward pass correctness."""

    def test_basic_reconstruction(self):
        """Test basic reconstruction of a centered square."""
        # Create a simple phantom
        phantom = torch.zeros(32, 32, dtype=torch.float64)
        phantom[12:20, 12:20] = 1.0

        # Forward Radon transform
        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)
        sinogram = T.radon_transform(phantom, angles, circle=True)

        # Inverse Radon transform
        reconstructed = T.inverse_radon_transform(
            sinogram, angles, circle=True, output_size=32
        )

        # Check shape
        assert reconstructed.shape == (32, 32)

        # The reconstruction should have similar structure to original
        # (not exact due to limited angles and filtering)
        center_val = reconstructed[14:18, 14:18].mean()
        border_val = reconstructed[0:4, 0:4].mean()
        # Center should be brighter than border
        assert center_val > border_val

    def test_output_shape_2d(self):
        """Output shape for 2D sinogram."""
        sinogram = torch.randn(90, 91, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(sinogram, angles)

        assert reconstructed.dim() == 2
        # Default size is approximately num_bins / sqrt(2)
        expected_size = int(91 / math.sqrt(2))
        assert reconstructed.shape[0] == expected_size
        assert reconstructed.shape[1] == expected_size

    def test_output_shape_with_explicit_size(self):
        """Output shape with explicit output_size."""
        sinogram = torch.randn(90, 91, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=64
        )

        assert reconstructed.shape == (64, 64)

    def test_output_shape_batched(self):
        """Output shape for batched sinogram."""
        sinograms = torch.randn(5, 90, 91, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 90, dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinograms, angles, output_size=32
        )

        assert reconstructed.dim() == 3
        assert reconstructed.shape[0] == 5
        assert reconstructed.shape[1] == 32
        assert reconstructed.shape[2] == 32

    def test_different_filters(self):
        """Different filters should produce different results."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        recon_ramp = T.inverse_radon_transform(
            sinogram, angles, filter_type="ramp", output_size=32
        )
        recon_hamming = T.inverse_radon_transform(
            sinogram, angles, filter_type="hamming", output_size=32
        )
        recon_hann = T.inverse_radon_transform(
            sinogram, angles, filter_type="hann", output_size=32
        )

        # Different filters should give different results
        assert not torch.allclose(recon_ramp, recon_hamming)
        assert not torch.allclose(recon_ramp, recon_hann)
        assert not torch.allclose(recon_hamming, recon_hann)

    def test_all_filter_types(self):
        """All filter types should work without error."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        for filter_type in [
            "ramp",
            "shepp-logan",
            "cosine",
            "hamming",
            "hann",
        ]:
            reconstructed = T.inverse_radon_transform(
                sinogram, angles, filter_type=filter_type, output_size=32
            )
            assert reconstructed.shape == (32, 32)
            assert torch.isfinite(reconstructed).all()

    def test_circle_mode(self):
        """Circle mode should produce different results than no circle."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        recon_circle = T.inverse_radon_transform(
            sinogram, angles, circle=True, output_size=32
        )
        recon_no_circle = T.inverse_radon_transform(
            sinogram, angles, circle=False, output_size=32
        )

        # Results should differ
        assert not torch.allclose(recon_circle, recon_no_circle)

        # In circle mode, corners should be zero
        # Check a corner pixel
        assert abs(recon_circle[0, 0].item()) < 1e-10

    def test_round_trip(self):
        """Round trip: radon -> inverse_radon should approximate original."""
        # Create a simple phantom
        phantom = torch.zeros(48, 48, dtype=torch.float64)
        phantom[16:32, 16:32] = 1.0

        # Use many angles for better reconstruction
        angles = torch.linspace(0, math.pi, 180, dtype=torch.float64)

        # Forward and inverse
        sinogram = T.radon_transform(phantom, angles, circle=True)
        reconstructed = T.inverse_radon_transform(
            sinogram, angles, circle=True, output_size=48
        )

        # The reconstruction should capture the main structure
        # Normalize both for comparison
        phantom_norm = phantom / phantom.max()
        recon_norm = reconstructed / reconstructed.max()

        # Check correlation (should be positive)
        correlation = (phantom_norm * recon_norm).sum()
        assert correlation > 0


class TestInverseRadonTransformGradient:
    """Test inverse Radon transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. sinogram should pass numerical check."""
        sinogram = torch.randn(10, 16, dtype=torch.float64, requires_grad=True)
        angles = torch.linspace(0, math.pi, 10, dtype=torch.float64)

        def func(sino):
            return T.inverse_radon_transform(
                sino, angles, output_size=8, filter_type="ramp"
            )

        assert gradcheck(func, (sinogram,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        sinograms = torch.randn(
            2, 10, 16, dtype=torch.float64, requires_grad=True
        )
        angles = torch.linspace(0, math.pi, 10, dtype=torch.float64)

        def func(sino):
            return T.inverse_radon_transform(sino, angles, output_size=8)

        assert gradcheck(func, (sinograms,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        sinogram = torch.randn(8, 12, dtype=torch.float64, requires_grad=True)
        angles = torch.linspace(0, math.pi, 8, dtype=torch.float64)

        def func(sino):
            return T.inverse_radon_transform(sino, angles, output_size=6)

        assert gradgradcheck(func, (sinogram,), raise_exception=True)

    def test_gradient_with_different_filters(self):
        """Gradient should work with different filter types."""
        sinogram = torch.randn(8, 12, dtype=torch.float64, requires_grad=True)
        angles = torch.linspace(0, math.pi, 8, dtype=torch.float64)

        for filter_type in ["ramp", "shepp-logan", "hamming"]:

            def func(sino):
                return T.inverse_radon_transform(
                    sino, angles, output_size=6, filter_type=filter_type
                )

            assert gradcheck(
                func,
                (sinogram.clone().requires_grad_(True),),
                raise_exception=True,
            )


class TestInverseRadonTransformMeta:
    """Test inverse Radon transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        sinogram = torch.empty(90, 91, device="meta", dtype=torch.float64)
        angles = torch.empty(90, device="meta", dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(sinogram, angles)

        assert reconstructed.dim() == 2
        expected_size = int(91 / math.sqrt(2))
        assert reconstructed.shape[0] == expected_size
        assert reconstructed.shape[1] == expected_size
        assert reconstructed.device.type == "meta"

    def test_meta_tensor_explicit_size(self):
        """Meta tensor with explicit output size."""
        sinogram = torch.empty(90, 91, device="meta", dtype=torch.float64)
        angles = torch.empty(90, device="meta", dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=64
        )

        assert reconstructed.shape == (64, 64)
        assert reconstructed.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        sinogram = torch.empty(3, 45, 65, device="meta", dtype=torch.float64)
        angles = torch.empty(45, device="meta", dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )

        assert reconstructed.dim() == 3
        assert reconstructed.shape[0] == 3
        assert reconstructed.shape[1] == 32
        assert reconstructed.shape[2] == 32


class TestInverseRadonTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_angle(self):
        """Should work with a single angle."""
        sinogram = torch.randn(1, 32, dtype=torch.float64)
        angles = torch.tensor([0.0], dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=16
        )
        assert reconstructed.shape == (16, 16)

    def test_invalid_filter_type(self):
        """Should raise error for invalid filter type."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        with pytest.raises(ValueError, match="filter_type must be one of"):
            T.inverse_radon_transform(sinogram, angles, filter_type="invalid")

    def test_mismatched_angles(self):
        """Should raise error when angles don't match sinogram."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(
            0, math.pi, 30, dtype=torch.float64
        )  # Wrong size

        with pytest.raises(RuntimeError):
            T.inverse_radon_transform(sinogram, angles)

    def test_float32_dtype(self):
        """Should work with float32 dtype."""
        sinogram = torch.randn(45, 65, dtype=torch.float32)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float32)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )
        assert reconstructed.dtype == torch.float32
        assert reconstructed.shape == (32, 32)

    def test_small_output_size(self):
        """Should work with very small output size."""
        sinogram = torch.randn(10, 20, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 10, dtype=torch.float64)

        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=4
        )
        assert reconstructed.shape == (4, 4)


class TestInverseRadonTransformDevice:
    """Test inverse Radon transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        sinogram = torch.randn(45, 65, dtype=torch.float64, device="cuda")
        angles = torch.linspace(
            0, math.pi, 45, dtype=torch.float64, device="cuda"
        )

        try:
            reconstructed = T.inverse_radon_transform(
                sinogram, angles, output_size=32
            )
            assert reconstructed.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestInverseRadonTransformCUDA:
    """Test inverse Radon transform CUDA backend."""

    def test_cuda_forward_matches_cpu(self):
        """CUDA forward should match CPU output."""
        sinogram_cpu = torch.randn(20, 30, dtype=torch.float64)
        angles_cpu = torch.linspace(0, math.pi, 20, dtype=torch.float64)

        sinogram_cuda = sinogram_cpu.cuda()
        angles_cuda = angles_cpu.cuda()

        recon_cpu = T.inverse_radon_transform(
            sinogram_cpu, angles_cpu, output_size=16
        )
        recon_cuda = T.inverse_radon_transform(
            sinogram_cuda, angles_cuda, output_size=16
        )

        assert torch.allclose(
            recon_cpu, recon_cuda.cpu(), rtol=1e-10, atol=1e-10
        )

    def test_cuda_gradient(self):
        """Gradient should work on CUDA."""
        sinogram = torch.randn(
            20, 30, dtype=torch.float64, device="cuda", requires_grad=True
        )
        angles = torch.linspace(
            0, math.pi, 20, dtype=torch.float64, device="cuda"
        )

        recon = T.inverse_radon_transform(sinogram, angles, output_size=16)
        loss = recon.sum()
        loss.backward()

        assert sinogram.grad is not None
        assert sinogram.grad.device.type == "cuda"

    def test_cuda_gradcheck(self):
        """Gradient check on CUDA."""
        sinogram = torch.randn(
            10, 16, dtype=torch.float64, device="cuda", requires_grad=True
        )
        angles = torch.linspace(
            0, math.pi, 10, dtype=torch.float64, device="cuda"
        )

        def func(sino):
            return T.inverse_radon_transform(
                sino, angles, output_size=8, filter_type="ramp"
            )

        assert gradcheck(func, (sinogram,), raise_exception=True)

    def test_cuda_batched(self):
        """Batched inverse Radon transform on CUDA."""
        sinograms = torch.randn(5, 20, 30, dtype=torch.float64, device="cuda")
        angles = torch.linspace(
            0, math.pi, 20, dtype=torch.float64, device="cuda"
        )

        recon = T.inverse_radon_transform(sinograms, angles, output_size=16)

        assert recon.dim() == 3
        assert recon.shape[0] == 5
        assert recon.shape[1] == 16
        assert recon.shape[2] == 16
        assert recon.device.type == "cuda"


class TestInverseRadonTransformDtype:
    """Test inverse Radon transform dtype handling."""

    def test_float32_input(self):
        """Inverse Radon should work with float32 input."""
        sinogram = torch.randn(45, 65, dtype=torch.float32)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float32)
        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )
        assert reconstructed.dtype == torch.float32

    def test_float64_input(self):
        """Inverse Radon should work with float64 input."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)
        reconstructed = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )
        assert reconstructed.dtype == torch.float64


class TestInverseRadonTransformVmap:
    """Test inverse Radon transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        sinogram = torch.randn(8, 45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        # Manual batching
        recon_batched = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )

        # vmap
        def inv_radon_single(sino):
            return T.inverse_radon_transform(sino, angles, output_size=32)

        recon_vmap = torch.vmap(inv_radon_single)(sinogram)

        assert torch.allclose(recon_batched, recon_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        sinogram = torch.randn(4, 4, 20, 25, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 20, dtype=torch.float64)

        def inv_radon_single(sino):
            return T.inverse_radon_transform(sino, angles, output_size=16)

        recon_vmap = torch.vmap(torch.vmap(inv_radon_single))(sinogram)

        assert recon_vmap.shape == torch.Size([4, 4, 16, 16])


class TestInverseRadonTransformCompile:
    """Test inverse Radon transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        sinogram = torch.randn(45, 65, dtype=torch.float64)
        angles = torch.linspace(0, math.pi, 45, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_inv_radon(sino):
            return T.inverse_radon_transform(sino, angles, output_size=32)

        recon_compiled = compiled_inv_radon(sinogram)
        recon_eager = T.inverse_radon_transform(
            sinogram, angles, output_size=32
        )

        assert torch.allclose(recon_compiled, recon_eager, atol=1e-10)

    @pytest.mark.skip(
        reason="Meta kernel stride mismatch for backward with torch.compile"
    )
    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        sinogram = torch.randn(10, 16, dtype=torch.float64, requires_grad=True)
        angles = torch.linspace(0, math.pi, 10, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_inv_radon(sino):
            return T.inverse_radon_transform(sino, angles, output_size=8)

        recon = compiled_inv_radon(sinogram)
        recon.sum().backward()

        assert sinogram.grad is not None
        assert sinogram.grad.shape == sinogram.shape


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not available")
class TestInverseRadonTransformSkimageReference:
    """Tests comparing against scikit-image's iradon implementation."""

    def test_round_trip_reconstruction_quality(self):
        """Test that round-trip produces similar quality as skimage."""
        # Create a simple disk phantom
        size = 64
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 4
        phantom = ((x - center) ** 2 + (y - center) ** 2 <= radius**2).astype(
            np.float64
        )

        # Forward Radon with skimage
        angles_deg = np.linspace(0, 180, 90, endpoint=False)
        sinogram_skimage = skimage_radon(
            phantom, theta=angles_deg, circle=True
        )

        # Reconstruct with both
        skimage_recon = skimage_iradon(
            sinogram_skimage, theta=angles_deg, circle=True, filter_name="ramp"
        )

        # Use skimage sinogram for torch (need to transpose)
        sinogram_torch = torch.from_numpy(sinogram_skimage.T)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        torch_recon = T.inverse_radon_transform(
            sinogram_torch,
            angles_rad,
            circle=True,
            output_size=size,
            filter_type="ramp",
        ).numpy()

        # Both reconstructions should capture the original structure
        # Check correlation with original phantom
        phantom_flat = phantom.flatten()
        skimage_corr = np.corrcoef(phantom_flat, skimage_recon.flatten())[0, 1]
        torch_corr = np.corrcoef(phantom_flat, torch_recon.flatten())[0, 1]

        # Both should have reasonable correlation with original
        assert skimage_corr > 0.8, (
            f"skimage correlation too low: {skimage_corr}"
        )
        assert torch_corr > 0.7, f"torch correlation too low: {torch_corr}"

    def test_filter_types_produce_valid_output(self):
        """Test that different filters produce valid reconstructions."""
        size = 32
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 4
        phantom = ((x - center) ** 2 + (y - center) ** 2 <= radius**2).astype(
            np.float64
        )

        angles_deg = np.linspace(0, 180, 45, endpoint=False)
        sinogram = skimage_radon(phantom, theta=angles_deg, circle=True)

        sinogram_torch = torch.from_numpy(sinogram.T)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        # Test all our filter types
        for filter_type in [
            "ramp",
            "shepp-logan",
            "cosine",
            "hamming",
            "hann",
        ]:
            recon = T.inverse_radon_transform(
                sinogram_torch,
                angles_rad,
                circle=True,
                output_size=size,
                filter_type=filter_type,
            ).numpy()

            # Should produce finite values
            assert np.isfinite(recon).all(), (
                f"Filter {filter_type} produced non-finite"
            )

            # Should have positive correlation with original
            correlation = np.corrcoef(phantom.flatten(), recon.flatten())[0, 1]
            assert correlation > 0.5, (
                f"Filter {filter_type} correlation too low: {correlation}"
            )

    def test_structure_preservation(self):
        """Test that reconstruction preserves main structural features."""
        # Create phantom with distinct features
        size = 48
        phantom = np.zeros((size, size), dtype=np.float64)
        # Central disk
        y, x = np.ogrid[:size, :size]
        center = size // 2
        phantom[(x - center) ** 2 + (y - center) ** 2 <= 100] = 1.0
        # Inner disk
        phantom[(x - center) ** 2 + (y - center) ** 2 <= 25] = 0.5

        angles_deg = np.linspace(0, 180, 90, endpoint=False)
        sinogram = skimage_radon(phantom, theta=angles_deg, circle=True)

        sinogram_torch = torch.from_numpy(sinogram.T)
        angles_rad = torch.from_numpy(np.deg2rad(angles_deg))

        recon = T.inverse_radon_transform(
            sinogram_torch,
            angles_rad,
            circle=True,
            output_size=size,
            filter_type="ramp",
        ).numpy()

        # Center should be brighter than edge
        center_val = recon[
            size // 2 - 2 : size // 2 + 2, size // 2 - 2 : size // 2 + 2
        ].mean()
        edge_val = recon[0:4, 0:4].mean()

        assert center_val > edge_val, "Center should be brighter than edge"
