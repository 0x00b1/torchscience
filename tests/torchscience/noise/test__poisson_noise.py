import pytest
import torch
import torch.testing

from torchscience.noise import poisson_noise


class TestPoissonNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = poisson_noise([100], rate=5.0)
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = poisson_noise([4, 100], rate=5.0)
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = poisson_noise([2, 3, 100], rate=5.0)
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = poisson_noise([10, 0], rate=5.0)
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = poisson_noise([0, 100], rate=5.0)
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0


class TestPoissonNoiseDtype:
    """Tests for dtype handling."""

    def test_default_dtype_int64(self):
        """Test default dtype is int64."""
        result = poisson_noise([100], rate=5.0)
        assert result.dtype == torch.int64

    def test_explicit_int32(self):
        """Test int32 dtype."""
        result = poisson_noise([100], rate=5.0, dtype=torch.int32)
        assert result.dtype == torch.int32

    def test_explicit_float32(self):
        """Test float32 dtype for convenience."""
        result = poisson_noise([100], rate=5.0, dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_explicit_float64(self):
        """Test float64 dtype."""
        result = poisson_noise([100], rate=5.0, dtype=torch.float64)
        assert result.dtype == torch.float64


class TestPoissonNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = poisson_noise([100], rate=5.0, device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        result = poisson_noise([100], rate=5.0, device="cuda")
        assert result.device.type == "cuda"


class TestPoissonNoiseStatistics:
    """Tests for statistical properties."""

    def test_mean_equals_rate(self):
        """Test mean approximately equals rate."""
        torch.manual_seed(42)
        rate = 10.0
        result = poisson_noise([10000], rate=rate, dtype=torch.float64)
        mean = result.mean().item()
        # Allow 5% deviation for statistical noise
        assert abs(mean - rate) / rate < 0.05, (
            f"Mean {mean} not near rate {rate}"
        )

    def test_variance_equals_rate(self):
        """Test variance approximately equals rate."""
        torch.manual_seed(42)
        rate = 10.0
        result = poisson_noise([10000], rate=rate, dtype=torch.float64)
        var = result.var().item()
        # Allow 10% deviation for statistical noise
        assert abs(var - rate) / rate < 0.1, (
            f"Variance {var} not near rate {rate}"
        )

    def test_integer_values(self):
        """Test outputs are non-negative integers."""
        result = poisson_noise([100], rate=5.0)
        # Check non-negative
        assert (result >= 0).all()
        # Check integer (by default int64)
        assert result.dtype == torch.int64

    def test_zero_rate(self):
        """Test rate=0 gives all zeros."""
        result = poisson_noise([100], rate=0.0)
        assert (result == 0).all()

    def test_low_rate_distribution(self):
        """Test low rate produces expected distribution."""
        torch.manual_seed(42)
        rate = 1.0
        result = poisson_noise([10000], rate=rate, dtype=torch.float64)

        # For Poisson(1), P(0) = e^-1 ≈ 0.368
        p0 = (result == 0).float().mean().item()
        expected_p0 = 0.368
        assert abs(p0 - expected_p0) < 0.03, (
            f"P(0)={p0} not near {expected_p0}"
        )

    def test_high_rate_gaussian_approximation(self):
        """Test high rate approaches Gaussian."""
        torch.manual_seed(42)
        rate = 100.0
        result = poisson_noise([10000], rate=rate, dtype=torch.float64)

        # For large rate, should be approximately N(rate, sqrt(rate))
        mean = result.mean().item()
        std = result.std().item()

        assert abs(mean - rate) / rate < 0.02
        assert abs(std - rate**0.5) / rate**0.5 < 0.1


class TestPoissonNoiseTensorRate:
    """Tests for tensor rate parameter."""

    def test_tensor_rate_broadcasting(self):
        """Test rate tensor broadcasts with size."""
        rates = torch.tensor([[1.0], [10.0]])  # [2, 1]
        result = poisson_noise([2, 100], rate=rates, dtype=torch.float64)
        assert result.shape == torch.Size([2, 100])

    def test_tensor_rate_spatially_varying(self):
        """Test different rates produce different means per position."""
        torch.manual_seed(42)
        rates = torch.tensor([1.0, 10.0, 100.0])
        # Generate many samples per rate
        result = poisson_noise(
            [3, 10000], rate=rates.unsqueeze(1), dtype=torch.float64
        )

        means = result.mean(dim=1)
        expected_means = torch.tensor([1.0, 10.0, 100.0])

        # Check each mean is close to its rate
        for i in range(3):
            assert (
                abs(means[i].item() - expected_means[i].item())
                / expected_means[i].item()
                < 0.1
            )

    def test_tensor_rate_device_inheritance(self):
        """Test result inherits device from rate tensor."""
        rates = torch.tensor([5.0])
        result = poisson_noise([100], rate=rates)
        assert result.device == rates.device


class TestPoissonNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = poisson_noise([100], rate=5.0, generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = poisson_noise([100], rate=5.0, generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = poisson_noise([100], rate=5.0, generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = poisson_noise([100], rate=5.0, generator=g2)

        assert not torch.equal(result1, result2)


class TestPoissonNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(poisson_noise)
        result = compiled([100], rate=5.0)
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = poisson_noise([100], rate=5.0, generator=g1)

        compiled = torch.compile(poisson_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled([100], rate=5.0, generator=g2)

        torch.testing.assert_close(eager, compiled_result)


class TestPoissonNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = poisson_noise([100000], rate=5.0)
        assert result.shape == torch.Size([100000])

    def test_very_low_rate(self):
        """Test with very low rate."""
        torch.manual_seed(42)
        result = poisson_noise([1000], rate=0.01)
        # Most should be zero
        assert (result == 0).float().mean() > 0.95

    def test_very_high_rate(self):
        """Test with very high rate."""
        result = poisson_noise([100], rate=1000.0, dtype=torch.float64)
        mean = result.mean().item()
        # Mean should be close to rate
        assert abs(mean - 1000.0) / 1000.0 < 0.1

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = poisson_noise([100], rate=5.0)
        assert result.is_contiguous()

    def test_single_element(self):
        """Test single element output."""
        result = poisson_noise([1], rate=5.0)
        assert result.shape == torch.Size([1])
        assert result.item() >= 0
