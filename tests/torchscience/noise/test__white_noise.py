import pytest
import torch
import torch.testing

from torchscience.noise import white_noise


class TestWhiteNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = white_noise([100])
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = white_noise([4, 100])
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = white_noise([2, 3, 100])
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = white_noise([10, 0])
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = white_noise([0, 100])
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0

    def test_single_sample(self):
        """Test n=1 works correctly."""
        result = white_noise([1])
        assert result.shape == torch.Size([1])
        # Unlike pink noise, white noise at n=1 is just a random value

    def test_two_samples(self):
        """Test minimal case n=2."""
        result = white_noise([2])
        assert result.shape == torch.Size([2])


class TestWhiteNoiseDtype:
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
        result = white_noise([100], dtype=dtype)
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
        result = white_noise([100], dtype=dtype)
        assert result.dtype == dtype

    def test_default_dtype(self):
        """Test default dtype is float32."""
        result = white_noise([100])
        assert result.dtype == torch.float32


class TestWhiteNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = white_noise([100], device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        result = white_noise([100], device="cuda")
        assert result.device.type == "cuda"


class TestWhiteNoiseStatistics:
    """Tests for statistical properties."""

    def test_approximately_zero_mean(self):
        """Test output has approximately zero mean."""
        torch.manual_seed(42)
        result = white_noise([10000], dtype=torch.float64)
        mean = result.mean().item()
        assert abs(mean) < 0.1, f"Mean {mean} too far from 0"

    def test_approximately_unit_variance(self):
        """Test output has approximately unit variance."""
        torch.manual_seed(42)
        result = white_noise([10000], dtype=torch.float64)
        var = result.var().item()
        assert 0.8 < var < 1.2, f"Variance {var} not near 1"

    def test_batched_independence(self):
        """Test different batch elements are independent."""
        torch.manual_seed(42)
        result = white_noise([100, 1000], dtype=torch.float64)

        # Compute correlation between first two batch elements
        x = result[0] - result[0].mean()
        y = result[1] - result[1].mean()
        corr = (x * y).sum() / (x.norm() * y.norm())

        # Should be near zero (independent)
        assert abs(corr.item()) < 0.1, f"Correlation {corr} too high"

    def test_samples_uncorrelated(self):
        """Test that adjacent samples are uncorrelated (white noise property)."""
        torch.manual_seed(42)
        result = white_noise([10000], dtype=torch.float64)

        # Autocorrelation at lag 1
        x = result[:-1] - result[:-1].mean()
        y = result[1:] - result[1:].mean()
        autocorr = (x * y).sum() / (x.norm() * y.norm())

        # Should be near zero for white noise
        assert abs(autocorr.item()) < 0.05, (
            f"Autocorrelation {autocorr} too high"
        )


class TestWhiteNoiseSpectrum:
    """Tests for spectral properties."""

    def test_flat_power_spectrum(self):
        """Test that power spectrum is approximately flat."""
        torch.manual_seed(42)
        n = 4096
        result = white_noise([n], dtype=torch.float64)

        # Compute power spectrum
        spectrum = torch.fft.rfft(result)
        power = torch.abs(spectrum) ** 2

        # Check that low and high frequency bands have similar power
        low_band = power[10:100].mean()
        mid_band = power[100:500].mean()
        high_band = power[500:1000].mean()

        # All bands should be within a factor of 3 of each other
        # (statistical variation is expected)
        min_power = min(low_band, mid_band, high_band)
        max_power = max(low_band, mid_band, high_band)
        ratio = max_power / min_power
        assert ratio < 3.0, f"Power ratio {ratio} indicates non-flat spectrum"

    def test_power_spectrum_slope_near_zero(self):
        """Test that power spectrum slope is near zero on log-log scale."""
        torch.manual_seed(42)
        n = 4096
        result = white_noise([n], dtype=torch.float64)

        # Compute power spectrum
        spectrum = torch.fft.rfft(result)
        power = torch.abs(spectrum) ** 2

        # Fit slope on log-log scale (exclude DC and very high frequencies)
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

        # Slope should be approximately 0 for white noise (allowing some variance)
        assert -0.5 < slope.item() < 0.5, f"Slope {slope} not near 0"


class TestWhiteNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = white_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = white_noise([100], generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = white_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = white_noise([100], generator=g2)

        assert not torch.allclose(result1, result2)


class TestWhiteNoiseGradient:
    """Tests for gradient support."""

    def test_requires_grad_propagates(self):
        """Test requires_grad parameter works."""
        result = white_noise([100], requires_grad=True)
        assert result.requires_grad

    def test_gradient_flows(self):
        """Test gradients can be computed."""
        result = white_noise([100], dtype=torch.float64, requires_grad=True)
        loss = result.sum()
        loss.backward()
        # Should not raise


class TestWhiteNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(white_noise)
        result = compiled([100])
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = white_noise([100], generator=g1)

        compiled = torch.compile(white_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled([100], generator=g2)

        torch.testing.assert_close(eager, compiled_result)


class TestWhiteNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = white_noise([100000])
        assert result.shape == torch.Size([100000])

    def test_small_tensor(self):
        """Test with small tensor (n=2)."""
        result = white_noise([2])
        assert result.shape == torch.Size([2])
        assert torch.isfinite(result).all()

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = white_noise([100])
        assert result.is_contiguous()

    def test_matches_torch_randn(self):
        """Test that white_noise matches torch.randn with same generator."""
        g1 = torch.Generator().manual_seed(42)
        result = white_noise([100], generator=g1, dtype=torch.float64)

        g2 = torch.Generator().manual_seed(42)
        expected = torch.randn([100], generator=g2, dtype=torch.float64)

        torch.testing.assert_close(result, expected)
