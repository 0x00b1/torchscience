import pytest
import torch
import torch.testing

from torchscience.noise import blue_noise


class TestBlueNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = blue_noise([100])
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = blue_noise([4, 100])
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = blue_noise([2, 3, 100])
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = blue_noise([10, 0])
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = blue_noise([0, 100])
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0

    def test_single_sample(self):
        """Test n=1 returns zeros."""
        result = blue_noise([1])
        assert result.shape == torch.Size([1])
        torch.testing.assert_close(result, torch.zeros(1), rtol=0, atol=0)

    def test_two_samples(self):
        """Test minimal case n=2."""
        result = blue_noise([2])
        assert result.shape == torch.Size([2])


class TestBlueNoiseDtype:
    """Tests for dtype handling."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
        ],
    )
    def test_standard_dtypes(self, dtype):
        """Test standard floating point dtypes."""
        result = blue_noise([100], dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_half_precision_dtypes(self, dtype):
        """Test half-precision dtypes."""
        result = blue_noise([100], dtype=dtype)
        assert result.dtype == dtype

    def test_default_dtype(self):
        """Test default dtype is float32."""
        result = blue_noise([100])
        assert result.dtype == torch.float32


class TestBlueNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = blue_noise([100], device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        result = blue_noise([100], device="cuda")
        assert result.device.type == "cuda"


class TestBlueNoiseStatistics:
    """Tests for statistical properties."""

    def test_approximately_zero_mean(self):
        """Test output has approximately zero mean."""
        torch.manual_seed(42)
        result = blue_noise([10000], dtype=torch.float64)
        mean = result.mean().item()
        assert abs(mean) < 0.1, f"Mean {mean} too far from 0"

    def test_approximately_unit_variance(self):
        """Test output has approximately unit variance."""
        torch.manual_seed(42)
        result = blue_noise([10000], dtype=torch.float64)
        var = result.var().item()
        assert 0.5 < var < 3.0, f"Variance {var} not near 1"

    def test_batched_independence(self):
        """Test different batch elements are independent."""
        torch.manual_seed(42)
        result = blue_noise([100, 1000], dtype=torch.float64)

        # Compute correlation between first two batch elements
        x = result[0] - result[0].mean()
        y = result[1] - result[1].mean()
        corr = (x * y).sum() / (x.norm() * y.norm())

        # Should be near zero (independent)
        assert abs(corr.item()) < 0.2, f"Correlation {corr} too high"


class TestBlueNoiseSpectrum:
    """Tests for spectral properties."""

    def test_power_spectrum_slope(self):
        """Test that power spectrum follows f (slope +1 on log-log)."""
        torch.manual_seed(42)
        n = 4096
        result = blue_noise([n], dtype=torch.float64)

        # Compute power spectrum
        spectrum = torch.fft.rfft(result)
        power = torch.abs(spectrum) ** 2

        # Fit slope on log-log scale (exclude DC and high frequencies)
        freq_start = 10
        freq_end = n // 4
        freqs = torch.arange(freq_start, freq_end, dtype=torch.float64)
        powers = power[freq_start:freq_end]

        # Log-log fit: log(P) = slope * log(f) + intercept
        log_f = torch.log(freqs)
        log_p = torch.log(powers)

        # Linear regression
        n_pts = len(freqs)
        slope = (n_pts * (log_f * log_p).sum() - log_f.sum() * log_p.sum()) / (
            n_pts * (log_f**2).sum() - log_f.sum() ** 2
        )

        # Slope should be approximately +1 for f noise
        assert 0.5 < slope.item() < 1.5, f"Slope {slope} not near +1"

    def test_dc_component_near_zero(self):
        """Test that DC component is near zero (zero mean)."""
        torch.manual_seed(42)
        result = blue_noise([1000], dtype=torch.float64)

        spectrum = torch.fft.rfft(result)
        dc = spectrum[0].abs().item()

        # DC should be small relative to other components
        assert dc < 10, f"DC component {dc} too large"


class TestBlueNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = blue_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = blue_noise([100], generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = blue_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = blue_noise([100], generator=g2)

        assert not torch.allclose(result1, result2)


class TestBlueNoiseGradient:
    """Tests for gradient support."""

    def test_requires_grad_propagates(self):
        """Test requires_grad parameter works."""
        result = blue_noise([100], requires_grad=True)
        assert result.requires_grad

    def test_gradient_flows(self):
        """Test gradients can be computed."""
        result = blue_noise([100], dtype=torch.float64, requires_grad=True)
        loss = result.sum()
        loss.backward()
        # Should not raise


class TestBlueNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(blue_noise)
        result = compiled([100])
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = blue_noise([100], generator=g1)

        compiled = torch.compile(blue_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled([100], generator=g2)

        torch.testing.assert_close(eager, compiled_result)


class TestBlueNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = blue_noise([100000])
        assert result.shape == torch.Size([100000])

    def test_small_tensor(self):
        """Test with small tensor (n=2)."""
        result = blue_noise([2])
        assert result.shape == torch.Size([2])
        assert torch.isfinite(result).all()

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = blue_noise([100])
        assert result.is_contiguous()
