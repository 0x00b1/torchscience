import pytest
import torch
import torch.testing
from torchscience.signal_processing.noise import shot_noise


class TestShotNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = shot_noise([100], rate=5.0)
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = shot_noise([4, 100], rate=5.0)
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = shot_noise([2, 3, 100], rate=5.0)
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = shot_noise([10, 0], rate=5.0)
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = shot_noise([0, 100], rate=5.0)
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0


class TestShotNoiseDtype:
    """Tests for dtype handling."""

    def test_default_dtype_float32(self):
        """Test default dtype is float32."""
        result = shot_noise([100], rate=5.0)
        assert result.dtype == torch.float32

    def test_explicit_float64(self):
        """Test float64 dtype."""
        result = shot_noise([100], rate=5.0, dtype=torch.float64)
        assert result.dtype == torch.float64

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_half_precision_dtypes(self, dtype):
        """Test half-precision dtypes."""
        result = shot_noise([100], rate=5.0, dtype=dtype)
        assert result.dtype == dtype


class TestShotNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = shot_noise([100], rate=5.0, device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        result = shot_noise([100], rate=5.0, device="cuda")
        assert result.device.type == "cuda"


class TestShotNoiseStatistics:
    """Tests for statistical properties."""

    def test_mean_equals_rate_high(self):
        """Test mean approximately equals rate for high rates."""
        torch.manual_seed(42)
        rate = 100.0
        result = shot_noise([10000], rate=rate, dtype=torch.float64)
        mean = result.mean().item()
        # Allow 5% deviation for statistical noise
        assert abs(mean - rate) / rate < 0.05, (
            f"Mean {mean} not near rate {rate}"
        )

    def test_variance_approximately_rate_high(self):
        """Test variance approximately equals rate for high rates."""
        torch.manual_seed(42)
        rate = 100.0
        result = shot_noise([10000], rate=rate, dtype=torch.float64)
        var = result.var().item()
        # Allow 15% deviation - Gaussian approximation plus ReLU affects variance
        assert abs(var - rate) / rate < 0.15, (
            f"Variance {var} not near rate {rate}"
        )

    def test_non_negative_output(self):
        """Test all outputs are non-negative."""
        torch.manual_seed(42)
        result = shot_noise([1000], rate=1.0)
        assert (result >= 0).all()

    def test_zero_rate(self):
        """Test rate=0 gives all zeros (after ReLU clamp)."""
        torch.manual_seed(42)
        result = shot_noise([100], rate=0.0)
        # With rate=0, output is N(0,0) = 0, so should be all zeros
        assert (result == 0).all()

    def test_low_rate_mostly_positive(self):
        """Test low rate produces mostly positive values due to ReLU."""
        torch.manual_seed(42)
        rate = 5.0
        result = shot_noise([1000], rate=rate)
        # Most should be positive
        assert (result > 0).float().mean() > 0.8


class TestShotNoiseTensorRate:
    """Tests for tensor rate parameter."""

    def test_tensor_rate_broadcasting(self):
        """Test rate tensor broadcasts with size."""
        rates = torch.tensor([[1.0], [10.0]])  # [2, 1]
        result = shot_noise([2, 100], rate=rates, dtype=torch.float64)
        assert result.shape == torch.Size([2, 100])

    def test_tensor_rate_spatially_varying(self):
        """Test different rates produce different means per position."""
        torch.manual_seed(42)
        rates = torch.tensor([10.0, 50.0, 100.0])
        # Generate many samples per rate
        result = shot_noise(
            [3, 10000], rate=rates.unsqueeze(1), dtype=torch.float64
        )

        means = result.mean(dim=1)

        # Check each mean is close to its rate
        for i, expected in enumerate([10.0, 50.0, 100.0]):
            assert abs(means[i].item() - expected) / expected < 0.1

    def test_tensor_rate_device_inheritance(self):
        """Test result inherits device from rate tensor."""
        rates = torch.tensor([5.0])
        result = shot_noise([100], rate=rates)
        assert result.device == rates.device


class TestShotNoiseGradient:
    """Tests for gradient support."""

    def test_requires_grad_propagates(self):
        """Test requires_grad parameter works."""
        result = shot_noise([100], rate=5.0, requires_grad=True)
        assert result.requires_grad

    def test_gradient_flows(self):
        """Test gradients can be computed."""
        result = shot_noise(
            [100], rate=10.0, dtype=torch.float64, requires_grad=True
        )
        loss = result.sum()
        loss.backward()
        # Should not raise

    def test_gradient_wrt_rate(self):
        """Test gradient w.r.t. rate tensor."""
        rate = torch.tensor([10.0], requires_grad=True)
        # Expand rate to match size for gradient computation
        result = shot_noise([100], rate=rate, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        # Rate gradient should be approximately 100 (number of samples)
        # because d/d_rate(E[X]) = d/d_rate(rate) = 1 per sample
        assert rate.grad is not None
        # The gradient magnitude depends on the implementation
        assert rate.grad.abs().item() > 0


class TestShotNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = shot_noise([100], rate=5.0, generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = shot_noise([100], rate=5.0, generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = shot_noise([100], rate=5.0, generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = shot_noise([100], rate=5.0, generator=g2)

        assert not torch.allclose(result1, result2)


class TestShotNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(shot_noise)
        result = compiled([100], rate=5.0)
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = shot_noise([100], rate=5.0, generator=g1)

        compiled = torch.compile(shot_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled([100], rate=5.0, generator=g2)

        torch.testing.assert_close(eager, compiled_result)


class TestShotNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = shot_noise([100000], rate=5.0)
        assert result.shape == torch.Size([100000])

    def test_very_high_rate(self):
        """Test with very high rate."""
        result = shot_noise([100], rate=1000.0, dtype=torch.float64)
        mean = result.mean().item()
        # Mean should be close to rate
        assert abs(mean - 1000.0) / 1000.0 < 0.1

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = shot_noise([100], rate=5.0)
        assert result.is_contiguous()

    def test_single_element(self):
        """Test single element output."""
        result = shot_noise([1], rate=5.0)
        assert result.shape == torch.Size([1])
        assert result.item() >= 0

    def test_finite_output(self):
        """Test all outputs are finite."""
        result = shot_noise([1000], rate=5.0)
        assert torch.isfinite(result).all()
