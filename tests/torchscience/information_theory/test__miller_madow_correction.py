"""Comprehensive tests for Miller-Madow correction."""

import pytest
import torch

from torchscience.information import histogram_entropy, miller_madow_correction


class TestMillerMadowCorrectionBasic:
    """Basic functionality tests."""

    def test_output_shape_scalar(self):
        """Returns scalar for scalar entropy input."""
        entropy = torch.tensor(1.5)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.shape == torch.Size([])

    def test_output_shape_1d(self):
        """Returns 1D tensor for 1D entropy input."""
        entropy = torch.tensor([1.0, 1.5, 2.0])
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.shape == torch.Size([3])

    def test_output_shape_2d(self):
        """Returns 2D tensor for 2D entropy input."""
        entropy = torch.randn(4, 5)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.shape == torch.Size([4, 5])

    def test_non_negativity_of_correction(self):
        """Correction term is always non-negative for num_bins >= 1."""
        entropy = torch.zeros(10)
        for num_bins in [1, 5, 10, 100]:
            result = miller_madow_correction(
                entropy, num_bins=num_bins, num_samples=100
            )
            # For num_bins >= 1, correction is (m-1)/(2n) >= 0
            assert (result >= 0).all(), (
                f"Correction should be non-negative for num_bins={num_bins}"
            )


class TestMillerMadowCorrectionCorrectness:
    """Numerical correctness tests."""

    def test_correction_formula_basic(self):
        """Verify correction formula: H_corrected = H_naive + (m-1)/(2n)."""
        entropy = torch.tensor(1.5)
        num_bins = 10
        num_samples = 100
        expected_correction = (num_bins - 1) / (2 * num_samples)
        expected = entropy + expected_correction
        result = miller_madow_correction(entropy, num_bins, num_samples)
        assert torch.isclose(result, expected, rtol=1e-6)

    def test_correction_formula_various_values(self):
        """Verify correction formula with various parameter combinations."""
        test_cases = [
            # (entropy, num_bins, num_samples, expected_correction)
            (1.0, 5, 50, (5 - 1) / (2 * 50)),
            (2.5, 20, 1000, (20 - 1) / (2 * 1000)),
            (0.0, 2, 10, (2 - 1) / (2 * 10)),
            (3.14, 100, 500, (100 - 1) / (2 * 500)),
        ]
        for entropy_val, m, n, expected_corr in test_cases:
            entropy = torch.tensor(entropy_val)
            result = miller_madow_correction(entropy, m, n)
            expected = entropy_val + expected_corr
            assert torch.isclose(result, torch.tensor(expected), rtol=1e-6), (
                f"Failed for m={m}, n={n}: expected {expected}, got {result}"
            )

    def test_single_bin_no_correction(self):
        """Single bin (m=1) gives zero correction."""
        entropy = torch.tensor(0.0)
        result = miller_madow_correction(entropy, num_bins=1, num_samples=100)
        # (1-1)/(2*100) = 0
        assert torch.isclose(result, entropy, rtol=1e-6)

    def test_correction_increases_entropy(self):
        """Correction always increases entropy for m >= 1."""
        entropy = torch.tensor(1.5)
        for num_bins in [1, 2, 5, 10, 50]:
            result = miller_madow_correction(
                entropy, num_bins=num_bins, num_samples=100
            )
            assert result >= entropy, (
                f"Corrected entropy should be >= naive entropy for num_bins={num_bins}"
            )

    def test_correction_magnitude_decreases_with_sample_size(self):
        """Larger sample size gives smaller correction."""
        entropy = torch.tensor(1.5)
        num_bins = 10
        result_small = miller_madow_correction(
            entropy, num_bins, num_samples=50
        )
        result_large = miller_madow_correction(
            entropy, num_bins, num_samples=500
        )
        correction_small = result_small - entropy
        correction_large = result_large - entropy
        assert correction_small > correction_large, (
            "Larger sample should give smaller correction"
        )


class TestMillerMadowCorrectionBatched:
    """Batched operation tests."""

    def test_batched_entropy_scalar_params(self):
        """Batched entropy with scalar num_bins and num_samples."""
        entropy = torch.tensor([1.0, 1.5, 2.0, 2.5])
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        expected_correction = (10 - 1) / (2 * 100)
        expected = entropy + expected_correction
        assert torch.allclose(result, expected, rtol=1e-6)

    def test_batched_all_tensor_inputs(self):
        """All inputs as tensors with broadcasting."""
        entropy = torch.tensor([1.0, 1.5])
        num_bins = torch.tensor([5, 8])
        num_samples = torch.tensor([100, 200])
        result = miller_madow_correction(entropy, num_bins, num_samples)
        expected = torch.tensor(
            [
                1.0 + (5 - 1) / (2 * 100),
                1.5 + (8 - 1) / (2 * 200),
            ]
        )
        assert torch.allclose(result, expected, rtol=1e-6)

    def test_broadcasting_entropy_bins(self):
        """Test broadcasting between entropy and num_bins."""
        entropy = torch.tensor([[1.0], [2.0]])  # Shape (2, 1)
        num_bins = torch.tensor([5, 10])  # Shape (2,) -> broadcasts to (2, 2)
        result = miller_madow_correction(entropy, num_bins, num_samples=100)
        assert result.shape == torch.Size([2, 2])

    def test_batch_consistency(self):
        """Batched computation matches individual computation."""
        entropy = torch.tensor([1.0, 1.5, 2.0])
        num_bins = torch.tensor([5, 8, 10])
        num_samples = torch.tensor([50, 100, 200])
        batched = miller_madow_correction(entropy, num_bins, num_samples)

        for i in range(3):
            individual = miller_madow_correction(
                entropy[i], num_bins[i].item(), num_samples[i].item()
            )
            assert torch.isclose(batched[i], individual, rtol=1e-6), (
                f"Batch element {i} mismatch"
            )


class TestMillerMadowCorrectionEdgeCases:
    """Edge case handling tests."""

    def test_num_samples_zero_returns_unchanged(self):
        """Zero samples returns entropy unchanged (no correction)."""
        entropy = torch.tensor(1.5)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=0)
        assert torch.isclose(result, entropy, rtol=1e-6), (
            "Zero samples should return entropy unchanged"
        )

    def test_num_samples_zero_batched(self):
        """Zero samples in batch returns those elements unchanged."""
        entropy = torch.tensor([1.0, 1.5, 2.0])
        num_samples = torch.tensor([100, 0, 50])
        result = miller_madow_correction(
            entropy, num_bins=10, num_samples=num_samples
        )
        # Element 1 should be unchanged, others should be corrected
        assert torch.isclose(result[1], entropy[1], rtol=1e-6)
        assert result[0] > entropy[0]
        assert result[2] > entropy[2]

    def test_num_bins_zero(self):
        """Zero bins produces negative correction (edge case)."""
        # This is an unusual case, but the formula still applies
        # Correction = (0-1)/(2*100) = -0.005
        entropy = torch.tensor(1.0)
        result = miller_madow_correction(entropy, num_bins=0, num_samples=100)
        expected = 1.0 + (-1) / (2 * 100)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-6)

    def test_large_num_bins_small_samples(self):
        """Large correction for many bins with few samples."""
        entropy = torch.tensor(1.0)
        # Correction = (100-1)/(2*10) = 4.95
        result = miller_madow_correction(entropy, num_bins=100, num_samples=10)
        expected = 1.0 + (100 - 1) / (2 * 10)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-6)

    def test_very_large_samples(self):
        """Very large samples give negligible correction."""
        entropy = torch.tensor(2.0)
        result = miller_madow_correction(
            entropy, num_bins=10, num_samples=1_000_000
        )
        # Correction = 9 / 2000000 = 4.5e-6
        assert torch.isclose(result, entropy, atol=1e-5)


class TestMillerMadowCorrectionDtypes:
    """Data type tests."""

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        entropy = torch.tensor(1.5, dtype=torch.float32)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        entropy = torch.tensor(1.5, dtype=torch.float64)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.dtype == torch.float64
        assert torch.isfinite(result)

    def test_tensor_params_converted_to_entropy_dtype(self):
        """Tensor parameters are converted to entropy dtype."""
        entropy = torch.tensor(1.5, dtype=torch.float64)
        num_bins = torch.tensor(10, dtype=torch.int32)
        num_samples = torch.tensor(100, dtype=torch.int64)
        result = miller_madow_correction(entropy, num_bins, num_samples)
        assert result.dtype == torch.float64


class TestMillerMadowCorrectionValidation:
    """Input validation tests."""

    def test_non_tensor_entropy(self):
        """Raises error for non-tensor entropy."""
        with pytest.raises(TypeError, match="entropy must be a Tensor"):
            miller_madow_correction(1.5, num_bins=10, num_samples=100)

    def test_non_tensor_non_int_num_bins(self):
        """Raises error for invalid num_bins type."""
        entropy = torch.tensor(1.5)
        with pytest.raises(
            TypeError, match="num_bins must be a Tensor or int"
        ):
            miller_madow_correction(entropy, num_bins=10.5, num_samples=100)

    def test_non_tensor_non_int_num_samples(self):
        """Raises error for invalid num_samples type."""
        entropy = torch.tensor(1.5)
        with pytest.raises(
            TypeError, match="num_samples must be a Tensor or int"
        ):
            miller_madow_correction(entropy, num_bins=10, num_samples=100.5)


class TestMillerMadowCorrectionDevice:
    """Device handling tests."""

    def test_result_on_same_device(self):
        """Result is on the same device as input."""
        entropy = torch.tensor(1.5)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.device == entropy.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Works on CUDA device."""
        entropy = torch.tensor(1.5, device="cuda")
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        assert result.device.type == "cuda"


class TestMillerMadowCorrectionIntegration:
    """Integration tests with histogram_entropy."""

    def test_matches_histogram_entropy_correction(self):
        """Manual correction matches histogram_entropy with correction param."""
        torch.manual_seed(42)
        samples = torch.randn(100)
        n_bins = 10

        # Get uncorrected entropy from histogram_entropy
        h_uncorrected = histogram_entropy(samples, bins=n_bins)

        # Get corrected entropy from histogram_entropy
        h_corrected = histogram_entropy(
            samples, bins=n_bins, correction="miller_madow"
        )

        # The difference should be (m-1)/(2n) where m is non-empty bins
        # We can't know exactly how many non-empty bins, but correction
        # should be non-negative and bounded by (n_bins-1)/(2*n)
        correction_applied = h_corrected - h_uncorrected
        max_correction = (n_bins - 1) / (2 * 100)

        assert correction_applied >= 0
        assert correction_applied <= max_correction + 1e-6

    def test_manual_correction_pipeline(self):
        """Can manually apply correction to histogram_entropy result."""
        torch.manual_seed(42)
        samples = torch.randn(200)
        n_bins = 15

        # Get uncorrected entropy
        h_naive = histogram_entropy(samples, bins=n_bins)

        # For simplicity, assume all bins are filled (worst case correction)
        # In practice, you'd count non-empty bins
        h_manual = miller_madow_correction(
            h_naive, num_bins=n_bins, num_samples=200
        )

        # The manual correction should increase entropy
        assert h_manual >= h_naive


class TestMillerMadowCorrectionReproducibility:
    """Reproducibility tests."""

    def test_deterministic(self):
        """Same input produces same output."""
        entropy = torch.tensor(1.5)
        result1 = miller_madow_correction(
            entropy, num_bins=10, num_samples=100
        )
        result2 = miller_madow_correction(
            entropy, num_bins=10, num_samples=100
        )
        assert torch.equal(result1, result2)

    def test_associativity(self):
        """Multiple small corrections equal one large correction."""
        entropy = torch.tensor(1.0)
        # Two corrections with same bins and samples
        result1 = miller_madow_correction(
            entropy, num_bins=10, num_samples=100
        )
        result2 = miller_madow_correction(
            result1, num_bins=10, num_samples=100
        )

        # Should equal entropy + 2 * correction
        single_correction = (10 - 1) / (2 * 100)
        expected = entropy + 2 * single_correction
        assert torch.isclose(result2, expected, rtol=1e-6)


class TestMillerMadowCorrectionGradients:
    """Gradient computation tests."""

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        entropy = torch.tensor(1.5, requires_grad=True)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        result.backward()
        assert entropy.grad is not None
        assert torch.isfinite(entropy.grad)

    def test_gradient_is_one(self):
        """Gradient with respect to entropy is 1 (linear operation)."""
        entropy = torch.tensor(1.5, requires_grad=True)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        result.backward()
        assert torch.isclose(entropy.grad, torch.tensor(1.0), rtol=1e-6)

    def test_gradient_batched(self):
        """Gradients work for batched inputs."""
        entropy = torch.tensor([1.0, 1.5, 2.0], requires_grad=True)
        result = miller_madow_correction(entropy, num_bins=10, num_samples=100)
        loss = result.sum()
        loss.backward()
        assert entropy.grad is not None
        assert torch.allclose(entropy.grad, torch.ones(3), rtol=1e-6)
