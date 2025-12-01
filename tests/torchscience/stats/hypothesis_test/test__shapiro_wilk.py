"""Tests for shapiro_wilk function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import shapiro_wilk


class TestShapiroWilk:
    """Tests for shapiro_wilk function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.shapiro."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(50, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)
        scipy_result = scipy_stats.shapiro(sample.numpy())

        # Our implementation uses Blom's approximation for expected order
        # statistics rather than scipy's exact tabulated coefficients,
        # so we allow a larger tolerance.
        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=0.02,  # 2% tolerance for approximation
        )
        # P-values can differ more due to differences in W statistic
        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=0.5,  # P-values are sensitive to small W changes
        )

    def test_normal_sample_high_pvalue(self):
        """Test that a normal sample has high p-value."""
        torch.manual_seed(123)
        sample = torch.randn(100, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)

        # Cannot reject normality for normal data
        assert pvalue > 0.05
        # W should be close to 1 for normal data
        assert statistic > 0.9

    def test_uniform_sample_low_pvalue(self):
        """Test that a uniform sample has low p-value."""
        torch.manual_seed(42)
        sample = torch.rand(100, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)

        # Should reject normality for uniform data
        assert pvalue < 0.05

    def test_exponential_sample_low_pvalue(self):
        """Test that an exponential sample has low p-value."""
        torch.manual_seed(42)
        sample = (
            torch.distributions.Exponential(1.0)
            .sample((100,))
            .to(torch.float64)
        )

        statistic, pvalue = shapiro_wilk(sample)

        # Should reject normality for exponential data
        assert pvalue < 0.05

    def test_small_sample(self):
        """Test with small sample (n=10)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(10, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)
        scipy_result = scipy_stats.shapiro(sample.numpy())

        # For small samples, Blom's approximation differs more from
        # scipy's exact coefficients, so we use a larger tolerance.
        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=0.05,  # 5% tolerance for small sample approximation
        )

    def test_batched(self):
        """Test batched computation."""
        torch.manual_seed(42)
        samples = torch.randn(5, 50, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(samples)

        assert statistic.shape == (5,)
        assert pvalue.shape == (5,)
        # All normal samples should have high p-values
        assert (pvalue > 0.01).all()

    def test_insufficient_samples(self):
        """Test that n < 3 returns NaN."""
        sample = torch.tensor([1.0, 2.0], dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)

        # Need at least 3 samples
        assert torch.isnan(statistic)
        assert torch.isnan(pvalue)

    def test_sample_size_limit(self):
        """Test behavior near sample size limit."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        # Test at 5000 (scipy's limit)
        sample = torch.randn(5000, dtype=torch.float64)

        statistic, pvalue = shapiro_wilk(sample)

        # Should still work
        assert torch.isfinite(statistic)
        assert torch.isfinite(pvalue)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        sample = torch.randn(50, dtype=torch.float64, device="meta")

        statistic, pvalue = shapiro_wilk(sample)

        assert statistic.device.type == "meta"
        assert pvalue.device.type == "meta"
        assert statistic.shape == ()

    def test_meta_tensor_batched(self):
        """Test meta tensor shape inference with batched input."""
        sample = torch.randn(5, 50, dtype=torch.float64, device="meta")

        statistic, pvalue = shapiro_wilk(sample)

        assert statistic.shape == (5,)
        assert pvalue.shape == (5,)

    def test_no_gradients(self):
        """Test that gradients are not supported."""
        sample = torch.randn(50, dtype=torch.float64, requires_grad=True)

        with pytest.raises(RuntimeError, match="does not support gradients"):
            shapiro_wilk(sample)


class TestShapiroWilkTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=dtype)

        with torch.no_grad():
            statistic, pvalue = shapiro_wilk(sample)

        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
        assert 0 <= pvalue <= 1
