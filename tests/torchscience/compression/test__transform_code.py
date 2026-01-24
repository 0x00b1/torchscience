"""Tests for transform coding."""

import pytest
import torch

from torchscience.compression import transform_code


class TestTransformCodeBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns coefficients, reconstruction, and mask."""
        x = torch.randn(10, 32)
        coeffs, recon, mask = transform_code(x)
        assert isinstance(coeffs, torch.Tensor)
        assert isinstance(recon, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_output_shapes(self):
        """Output shapes match input."""
        x = torch.randn(5, 4, 64)
        coeffs, recon, mask = transform_code(x)
        assert coeffs.shape == x.shape
        assert recon.shape == x.shape
        assert mask.shape == x.shape

    def test_mask_dtype(self):
        """Mask is boolean."""
        x = torch.randn(10, 16)
        _, _, mask = transform_code(x)
        assert mask.dtype == torch.bool


class TestTransformCodeDCT:
    """Tests for DCT transform."""

    def test_dct_real_output(self):
        """DCT produces real coefficients."""
        x = torch.randn(10, 32)
        coeffs, _, _ = transform_code(x, transform="dct")
        assert not coeffs.is_complex()

    def test_dct_invertible(self):
        """DCT is invertible (without quantization)."""
        x = torch.randn(10, 32)
        # Very small step to minimize quantization error
        coeffs, recon, _ = transform_code(
            x, transform="dct", quantization_step=1e-6
        )
        assert torch.allclose(x, recon, atol=1e-4)

    def test_dct_energy_compaction(self):
        """DCT concentrates energy in low frequencies for smooth signals."""
        # Smooth signal: low frequencies dominate
        t = torch.linspace(0, 1, 64)
        x = torch.sin(2 * torch.pi * t).unsqueeze(0)  # Single low-freq sine

        coeffs, _, _ = transform_code(
            x, transform="dct", quantization_step=1e-6
        )

        # First few coefficients should have most energy
        energy_low = (coeffs[..., :8] ** 2).sum()
        energy_total = (coeffs**2).sum()
        assert energy_low / energy_total > 0.9


class TestTransformCodeDFT:
    """Tests for DFT transform."""

    def test_dft_complex_output(self):
        """DFT produces complex coefficients."""
        x = torch.randn(10, 32)
        coeffs, _, _ = transform_code(x, transform="dft")
        assert coeffs.is_complex()

    def test_dft_real_reconstruction(self):
        """DFT reconstruction is real."""
        x = torch.randn(10, 32)
        _, recon, _ = transform_code(x, transform="dft")
        assert not recon.is_complex()

    def test_dft_invertible(self):
        """DFT is invertible (without quantization)."""
        x = torch.randn(10, 32)
        coeffs, recon, _ = transform_code(
            x, transform="dft", quantization_step=1e-6
        )
        assert torch.allclose(x, recon, atol=1e-4)


class TestTransformCodeIdentity:
    """Tests for identity transform."""

    def test_identity_no_transform(self):
        """Identity transform doesn't change signal (except quantization)."""
        x = torch.randn(10, 32)
        coeffs, recon, _ = transform_code(
            x, transform="identity", quantization_step=1e-6
        )
        assert torch.allclose(coeffs, x, atol=1e-4)
        assert torch.allclose(recon, x, atol=1e-4)


class TestTransformCodeQuantization:
    """Tests for quantization."""

    def test_quantization_step_affects_output(self):
        """Larger step size gives coarser quantization."""
        x = torch.randn(10, 32)
        coeffs_fine, _, _ = transform_code(x, quantization_step=0.1)
        coeffs_coarse, _, _ = transform_code(x, quantization_step=1.0)

        # Fine should have more unique values
        assert len(coeffs_fine.unique()) > len(coeffs_coarse.unique())

    def test_quantization_to_step_multiples(self):
        """Quantized values are multiples of step size."""
        x = torch.randn(10, 32)
        step = 0.5
        coeffs, _, _ = transform_code(
            x, transform="identity", quantization_step=step
        )

        # All values should be multiples of step (within tolerance)
        remainders = (coeffs / step).round() - (coeffs / step)
        assert torch.allclose(
            remainders, torch.zeros_like(remainders), atol=1e-6
        )


class TestTransformCodeKeepRatio:
    """Tests for coefficient thresholding."""

    def test_keep_ratio_zeros_coefficients(self):
        """Keep ratio zeros out small coefficients."""
        x = torch.randn(100, 64)
        _, _, mask = transform_code(x, keep_ratio=0.5)

        # About half should be kept (allow some slack for threshold ties)
        keep_fraction = mask.float().mean().item()
        assert 0.45 < keep_fraction < 0.60

    def test_keep_ratio_1_keeps_all(self):
        """Keep ratio 1.0 keeps all coefficients."""
        x = torch.randn(10, 32)
        _, _, mask = transform_code(x, keep_ratio=1.0)
        assert mask.all()

    def test_reconstruction_with_partial_coefficients(self):
        """Reconstruction works with partial coefficients."""
        torch.manual_seed(42)
        x = torch.randn(10, 64)
        _, recon_full, _ = transform_code(
            x, keep_ratio=1.0, quantization_step=0.01
        )
        _, recon_half, _ = transform_code(
            x, keep_ratio=0.5, quantization_step=0.01
        )

        # Both should reconstruct something reasonable
        mse_full = ((x - recon_full) ** 2).mean()
        mse_half = ((x - recon_half) ** 2).mean()

        # Full should be better than half
        assert mse_full < mse_half


class TestTransformCodeGradients:
    """Tests for gradient modes."""

    def test_ste_gradient_passes_through(self):
        """STE mode passes gradients through."""
        x = torch.randn(5, 16, requires_grad=True)
        coeffs, _, _ = transform_code(x, gradient_mode="ste")
        loss = coeffs.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x))

    def test_none_gradient_blocks(self):
        """None mode blocks gradients."""
        x = torch.randn(5, 16, requires_grad=True)
        coeffs, _, _ = transform_code(x, gradient_mode="none")
        # Coefficients are detached, so no grad_fn
        assert not coeffs.requires_grad

    def test_soft_mode_works(self):
        """Soft mode produces valid output."""
        x = torch.randn(5, 16)
        coeffs, recon, _ = transform_code(x, gradient_mode="soft")
        assert coeffs.shape == x.shape


class TestTransformCodeDistortion:
    """Tests for compression distortion."""

    def test_compression_increases_mse(self):
        """More aggressive compression increases MSE."""
        torch.manual_seed(42)
        x = torch.randn(100, 64)

        _, recon_low, _ = transform_code(
            x, quantization_step=0.1, keep_ratio=0.9
        )
        _, recon_high, _ = transform_code(
            x, quantization_step=1.0, keep_ratio=0.3
        )

        mse_low = ((x - recon_low) ** 2).mean()
        mse_high = ((x - recon_high) ** 2).mean()

        assert mse_high > mse_low

    def test_smooth_signals_compress_well(self):
        """Smooth signals compress better than noise."""
        # Smooth signal
        t = torch.linspace(0, 1, 64).unsqueeze(0).expand(100, -1)
        smooth = torch.sin(4 * torch.pi * t)

        # Noise
        noise = torch.randn(100, 64)

        _, recon_smooth, _ = transform_code(smooth, keep_ratio=0.2)
        _, recon_noise, _ = transform_code(noise, keep_ratio=0.2)

        mse_smooth = ((smooth - recon_smooth) ** 2).mean()
        mse_noise = ((noise - recon_noise) ** 2).mean()

        # Smooth should compress better
        assert mse_smooth < mse_noise * 0.5


class TestTransformCodeEdgeCases:
    """Edge case tests."""

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            transform_code([1.0, 2.0, 3.0])

    def test_scalar_raises(self):
        """Raises error for scalar input."""
        x = torch.tensor(1.0)
        with pytest.raises(ValueError, match="at least 1D"):
            transform_code(x)

    def test_invalid_transform_raises(self):
        """Raises error for invalid transform."""
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="transform must be"):
            transform_code(x, transform="invalid")

    def test_invalid_gradient_mode_raises(self):
        """Raises error for invalid gradient mode."""
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="gradient_mode"):
            transform_code(x, gradient_mode="invalid")

    def test_invalid_keep_ratio_raises(self):
        """Raises error for invalid keep_ratio."""
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="keep_ratio"):
            transform_code(x, keep_ratio=0)
        with pytest.raises(ValueError, match="keep_ratio"):
            transform_code(x, keep_ratio=1.5)

    def test_invalid_quantization_step_raises(self):
        """Raises error for invalid quantization_step."""
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="quantization_step"):
            transform_code(x, quantization_step=0)
        with pytest.raises(ValueError, match="quantization_step"):
            transform_code(x, quantization_step=-1)

    def test_1d_input(self):
        """Works with 1D input."""
        x = torch.randn(32)
        coeffs, recon, mask = transform_code(x)
        assert coeffs.shape == x.shape


class TestTransformCodeDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(10, 32, device="cpu")
        coeffs, recon, mask = transform_code(x)
        assert coeffs.device.type == "cpu"
        assert recon.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(10, 32, device="cuda")
        coeffs, recon, mask = transform_code(x)
        assert coeffs.device.type == "cuda"
        assert recon.device.type == "cuda"
