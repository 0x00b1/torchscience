"""Tests for anderson_darling function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import anderson_darling


class TestAndersonDarling:
    """Tests for anderson_darling function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.anderson."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(50, dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )
        scipy_result = scipy_stats.anderson(sample.numpy(), dist="norm")

        # Allow some tolerance for approximation differences
        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=0.02,
        )

    def test_normal_sample(self):
        """Test that a normal sample has low statistic."""
        torch.manual_seed(123)
        sample = torch.randn(100, dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        # Should not reject at 5% level (statistic < critical value)
        # significance_levels is [0.15, 0.10, 0.05, 0.025, 0.01]
        # critical_values[2] corresponds to 5% level
        assert statistic < critical_values[2]

    def test_uniform_sample(self):
        """Test that a uniform sample has high statistic."""
        torch.manual_seed(42)
        sample = torch.rand(100, dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        # Should reject at 5% level
        assert statistic > critical_values[2]

    def test_critical_values(self):
        """Test that critical values are returned correctly."""
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        # Should have 5 significance levels: 15%, 10%, 5%, 2.5%, 1%
        assert len(significance_levels) == 5
        assert torch.allclose(
            significance_levels,
            torch.tensor([0.15, 0.10, 0.05, 0.025, 0.01], dtype=torch.float64),
        )
        # Critical values should be increasing
        for i in range(len(critical_values) - 1):
            assert critical_values[i] < critical_values[i + 1]

    def test_batched(self):
        """Test batched computation."""
        torch.manual_seed(42)
        samples = torch.randn(5, 50, dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            samples
        )

        assert statistic.shape == (5,)
        # Critical values: (batch, 5) - 5 critical values per batch element
        assert critical_values.shape == (5, 5)

    def test_insufficient_samples(self):
        """Test that n < 8 returns NaN."""
        sample = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        # Need at least 8 samples for reliable test
        assert torch.isnan(statistic)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        sample = torch.randn(50, dtype=torch.float64, device="meta")

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        assert statistic.device.type == "meta"
        assert statistic.shape == ()

    def test_meta_tensor_batched(self):
        """Test meta tensor shape inference with batched input."""
        sample = torch.randn(5, 50, dtype=torch.float64, device="meta")

        statistic, critical_values, significance_levels = anderson_darling(
            sample
        )

        assert statistic.shape == (5,)
        assert critical_values.shape == (5, 5)

    def test_no_gradients(self):
        """Test that gradients are not supported."""
        sample = torch.randn(50, dtype=torch.float64, requires_grad=True)

        with pytest.raises(RuntimeError, match="does not support gradients"):
            anderson_darling(sample)


class TestAndersonDarlingTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=dtype)

        with torch.no_grad():
            statistic, critical_values, significance_levels = anderson_darling(
                sample
            )

        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
