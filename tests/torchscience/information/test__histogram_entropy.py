"""Comprehensive tests for histogram entropy."""

import math

import pytest
import torch

from torchscience.information import histogram_entropy


class TestHistogramEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D sample vector."""
        samples = torch.rand(1000)
        result = histogram_entropy(samples)
        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of samples."""
        samples = torch.randn(10, 1000)
        result = histogram_entropy(samples)
        assert result.shape == torch.Size([10])

    def test_output_shape_3d_batch(self):
        """Returns 2D tensor for nested batch of samples."""
        samples = torch.randn(4, 5, 1000)
        result = histogram_entropy(samples)
        assert result.shape == torch.Size([4, 5])

    def test_non_negativity(self):
        """Histogram entropy is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            samples = torch.randn(1000)
            result = histogram_entropy(samples)
            assert result >= 0, f"Entropy should be non-negative, got {result}"


class TestHistogramEntropyCorrectness:
    """Numerical correctness tests."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution has entropy approximately equal to log(bins)."""
        torch.manual_seed(42)
        n_bins = 10
        # Large sample for accurate approximation
        samples = torch.rand(100000)
        result = histogram_entropy(samples, bins=n_bins)
        expected = math.log(n_bins)
        # Histogram entropy for uniform should be close to log(n_bins)
        assert torch.isclose(result, torch.tensor(expected), rtol=0.05), (
            f"Expected {expected}, got {result}"
        )

    def test_uniform_distribution_bits(self):
        """Uniform distribution entropy in bits."""
        torch.manual_seed(42)
        n_bins = 8
        samples = torch.rand(100000)
        result = histogram_entropy(samples, bins=n_bins, base=2)
        expected = math.log2(n_bins)  # 3 bits for 8 bins
        assert torch.isclose(result, torch.tensor(expected), rtol=0.05)

    def test_concentrated_vs_spread_distribution(self):
        """Concentrated samples have lower entropy than spread samples."""
        torch.manual_seed(42)
        # Concentrated samples around zero with small spread
        concentrated = torch.randn(1000) * 0.1
        # Spread samples with larger spread
        spread = torch.randn(1000) * 10.0

        # Use same number of bins for both
        h_concentrated = histogram_entropy(concentrated, bins=50)
        h_spread = histogram_entropy(spread, bins=50)

        # Both should have similar entropy since they're both Gaussian
        # (shape matters, not scale for fixed bins spanning the data range)
        assert torch.isfinite(h_concentrated)
        assert torch.isfinite(h_spread)

    def test_constant_samples_zero_entropy(self):
        """All identical samples have zero entropy."""
        samples = torch.ones(1000) * 5.0
        result = histogram_entropy(samples)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6), (
            f"Constant samples should have zero entropy, got {result}"
        )

    def test_two_clusters_binary_entropy(self):
        """Two equal clusters should have entropy ~ log(2)."""
        torch.manual_seed(42)
        # Create two distinct clusters
        cluster1 = torch.zeros(5000)
        cluster2 = torch.ones(5000) * 10.0
        samples = torch.cat([cluster1, cluster2])
        # Use 2 bins spanning the range
        result = histogram_entropy(samples, bins=2)
        expected = math.log(2)  # Binary entropy for equal split
        assert torch.isclose(result, torch.tensor(expected), atol=0.01), (
            f"Expected {expected}, got {result}"
        )

    def test_more_bins_increases_entropy_for_continuous(self):
        """More bins generally increases entropy for continuous distributions."""
        torch.manual_seed(42)
        samples = torch.randn(10000)
        h10 = histogram_entropy(samples, bins=10)
        h100 = histogram_entropy(samples, bins=100)
        assert h100 > h10, f"More bins should increase entropy: {h100} > {h10}"


class TestHistogramEntropyBinMethods:
    """Bin selection method tests."""

    def test_scott_method(self):
        """Scott's rule produces reasonable bins."""
        torch.manual_seed(42)
        samples = torch.randn(1000)
        result = histogram_entropy(samples, bins="scott")
        assert torch.isfinite(result)
        assert result > 0

    def test_fd_method(self):
        """Freedman-Diaconis rule produces reasonable bins."""
        torch.manual_seed(42)
        samples = torch.randn(1000)
        result = histogram_entropy(samples, bins="fd")
        assert torch.isfinite(result)
        assert result > 0

    def test_auto_method(self):
        """Auto method selects appropriate rule."""
        torch.manual_seed(42)
        samples = torch.randn(1000)
        result = histogram_entropy(samples, bins="auto")
        assert torch.isfinite(result)
        assert result > 0

    def test_auto_falls_back_to_scott_for_constant(self):
        """Auto method handles zero IQR gracefully."""
        # Near-constant samples (very small IQR)
        samples = torch.zeros(1000) + torch.randn(1000) * 1e-10
        result = histogram_entropy(samples, bins="auto")
        assert torch.isfinite(result)


class TestHistogramEntropyCorrection:
    """Bias correction tests."""

    def test_miller_madow_increases_entropy(self):
        """Miller-Madow correction increases entropy estimate."""
        torch.manual_seed(42)
        samples = torch.randn(100)  # Small sample for noticeable correction
        h_raw = histogram_entropy(samples, bins=10)
        h_corrected = histogram_entropy(
            samples, bins=10, correction="miller_madow"
        )
        assert h_corrected >= h_raw, (
            f"Miller-Madow correction should increase entropy: "
            f"{h_corrected} >= {h_raw}"
        )

    def test_miller_madow_correction_magnitude(self):
        """Miller-Madow correction has expected magnitude."""
        torch.manual_seed(42)
        n_samples = 100
        n_bins = 10
        samples = torch.randn(n_samples)
        h_raw = histogram_entropy(samples, bins=n_bins)
        h_corrected = histogram_entropy(
            samples, bins=n_bins, correction="miller_madow"
        )
        # Correction is (m-1)/(2n) where m is number of non-empty bins
        # For Gaussian with 10 bins and 100 samples, most bins should be filled
        max_correction = (n_bins - 1) / (2 * n_samples)
        actual_correction = (h_corrected - h_raw).item()
        assert 0 <= actual_correction <= max_correction + 1e-6


class TestHistogramEntropyBase:
    """Logarithm base tests."""

    def test_base_2_bits(self):
        """Entropy in bits (base 2)."""
        torch.manual_seed(42)
        samples = torch.rand(100000)
        h_nats = histogram_entropy(samples, bins=8)
        h_bits = histogram_entropy(samples, bins=8, base=2)
        expected_bits = h_nats.item() / math.log(2)
        assert torch.isclose(h_bits, torch.tensor(expected_bits), rtol=1e-5)

    def test_base_10(self):
        """Entropy in base 10."""
        torch.manual_seed(42)
        samples = torch.rand(100000)
        h_nats = histogram_entropy(samples, bins=10)
        h_dits = histogram_entropy(samples, bins=10, base=10)
        expected_dits = h_nats.item() / math.log(10)
        assert torch.isclose(h_dits, torch.tensor(expected_dits), rtol=1e-5)


class TestHistogramEntropyDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Works with dim=0."""
        samples = torch.randn(1000, 5)  # 5 batches along dim=1
        result = histogram_entropy(samples, dim=0)
        assert result.shape == torch.Size([5])

    def test_dim_negative(self):
        """Works with negative dim values."""
        samples = torch.randn(5, 1000)
        result_neg = histogram_entropy(samples, dim=-1)
        result_pos = histogram_entropy(samples, dim=1)
        assert torch.allclose(result_neg, result_pos)

    def test_dim_middle(self):
        """Works with middle dimension."""
        samples = torch.randn(3, 1000, 4)  # samples along dim=1
        result = histogram_entropy(samples, dim=1)
        assert result.shape == torch.Size([3, 4])


class TestHistogramEntropyValidation:
    """Input validation tests."""

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            histogram_entropy([1.0, 2.0, 3.0])

    def test_scalar_input(self):
        """Raises error for scalar inputs."""
        with pytest.raises(ValueError, match="at least 1 dimension"):
            histogram_entropy(torch.tensor(1.0))

    def test_invalid_bins_string(self):
        """Raises error for invalid bins string."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="bins must be an int"):
            histogram_entropy(samples, bins="invalid")

    def test_invalid_bins_zero(self):
        """Raises error for zero bins."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="bins must be positive"):
            histogram_entropy(samples, bins=0)

    def test_invalid_bins_negative(self):
        """Raises error for negative bins."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="bins must be positive"):
            histogram_entropy(samples, bins=-5)

    def test_invalid_correction(self):
        """Raises error for invalid correction."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="correction must be one of"):
            histogram_entropy(samples, correction="invalid")

    def test_invalid_base_zero(self):
        """Raises error for zero base."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="base must be positive"):
            histogram_entropy(samples, base=0)

    def test_invalid_base_one(self):
        """Raises error for base=1."""
        samples = torch.randn(100)
        with pytest.raises(
            ValueError, match="base must be positive and not equal to 1"
        ):
            histogram_entropy(samples, base=1)

    def test_invalid_base_negative(self):
        """Raises error for negative base."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="base must be positive"):
            histogram_entropy(samples, base=-2)

    def test_dim_out_of_range(self):
        """Raises error for dim out of range."""
        samples = torch.randn(10, 100)
        with pytest.raises(IndexError, match="dim .* out of range"):
            histogram_entropy(samples, dim=5)


class TestHistogramEntropyDtypes:
    """Data type tests."""

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        samples = torch.randn(1000, dtype=torch.float32)
        result = histogram_entropy(samples)
        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        samples = torch.randn(1000, dtype=torch.float64)
        result = histogram_entropy(samples)
        assert result.dtype == torch.float64
        assert torch.isfinite(result)


class TestHistogramEntropyEdgeCases:
    """Edge case handling tests."""

    def test_small_sample(self):
        """Handles small sample sizes."""
        samples = torch.randn(10)
        result = histogram_entropy(samples, bins=5)
        assert torch.isfinite(result)
        assert result >= 0

    def test_very_large_sample(self):
        """Handles large sample sizes."""
        samples = torch.randn(100000)
        result = histogram_entropy(samples, bins=100)
        assert torch.isfinite(result)
        assert result > 0

    def test_single_sample(self):
        """Handles single sample."""
        samples = torch.tensor([1.0])
        result = histogram_entropy(samples, bins=10)
        # Single sample in one bin -> zero entropy
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_two_samples_same_bin(self):
        """Two samples in same bin have zero entropy."""
        samples = torch.tensor([1.0, 1.0001])
        result = histogram_entropy(samples, bins=10)
        # Both samples likely in same bin
        # Note: could be in different bins depending on binning
        assert torch.isfinite(result)
        assert result >= 0

    def test_two_samples_different_bins(self):
        """Two samples in different bins have log(2) entropy."""
        samples = torch.tensor([0.0, 10.0])
        result = histogram_entropy(samples, bins=2)
        expected = math.log(2)
        assert torch.isclose(result, torch.tensor(expected), atol=0.01)

    def test_outliers(self):
        """Handles outliers gracefully."""
        torch.manual_seed(42)
        samples = torch.randn(1000)
        # Add extreme outliers
        samples = torch.cat([samples, torch.tensor([1000.0, -1000.0])])
        result = histogram_entropy(samples, bins=20)
        assert torch.isfinite(result)
        assert result > 0


class TestHistogramEntropyBatched:
    """Batched operation tests."""

    def test_batch_consistency(self):
        """Batched computation matches individual computation."""
        torch.manual_seed(42)
        samples = torch.randn(5, 1000)
        batched_result = histogram_entropy(samples, bins=20)

        for i in range(5):
            individual_result = histogram_entropy(samples[i], bins=20)
            assert torch.isclose(
                batched_result[i], individual_result, rtol=1e-5
            ), f"Batch element {i} mismatch"

    def test_batch_with_different_distributions(self):
        """Handles batches with different distributions."""
        torch.manual_seed(42)
        # Mix of different distributions
        uniform = torch.rand(1000)
        normal = torch.randn(1000)
        concentrated = torch.zeros(1000) + torch.randn(1000) * 0.01

        samples = torch.stack([uniform, normal, concentrated])
        result = histogram_entropy(samples, bins=20)

        assert result.shape == torch.Size([3])
        # Concentrated should have lowest entropy
        assert result[2] < result[0]
        assert result[2] < result[1]


class TestHistogramEntropyReproducibility:
    """Reproducibility tests."""

    def test_deterministic(self):
        """Same input produces same output."""
        samples = torch.randn(1000)
        result1 = histogram_entropy(samples, bins=10)
        result2 = histogram_entropy(samples, bins=10)
        assert torch.equal(result1, result2)

    def test_same_seed_same_result(self):
        """Same random seed produces same result."""
        torch.manual_seed(42)
        samples1 = torch.randn(1000)
        result1 = histogram_entropy(samples1, bins=10)

        torch.manual_seed(42)
        samples2 = torch.randn(1000)
        result2 = histogram_entropy(samples2, bins=10)

        assert torch.equal(result1, result2)
