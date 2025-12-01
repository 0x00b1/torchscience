"""Tests for t-test functions in torchscience.statistics.hypothesis_test."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import (
    one_sample_t_test,
    paired_t_test,
    two_sample_t_test,
)


class TestOneSampleTTest:
    """Tests for one_sample_t_test function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.ttest_1samp."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(100, dtype=torch.float64)

        t_stat, p_value, df = one_sample_t_test(sample)
        scipy_result = scipy_stats.ttest_1samp(sample.numpy(), 0.0)

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            df, torch.tensor(99.0, dtype=torch.float64), rtol=1e-5
        )

    def test_nonzero_popmean(self):
        """Test with popmean != 0."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(100, dtype=torch.float64) + 5.0

        t_stat, p_value, df = one_sample_t_test(sample, popmean=5.0)
        scipy_result = scipy_stats.ttest_1samp(sample.numpy(), 5.0)

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_alternative_less(self):
        """Test alternative='less' hypothesis."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = (
            torch.randn(100, dtype=torch.float64) - 0.5
        )  # Mean shifted below 0

        t_stat, p_value, df = one_sample_t_test(sample, alternative="less")
        scipy_result = scipy_stats.ttest_1samp(
            sample.numpy(), 0.0, alternative="less"
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_alternative_greater(self):
        """Test alternative='greater' hypothesis."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = (
            torch.randn(100, dtype=torch.float64) + 0.5
        )  # Mean shifted above 0

        t_stat, p_value, df = one_sample_t_test(sample, alternative="greater")
        scipy_result = scipy_stats.ttest_1samp(
            sample.numpy(), 0.0, alternative="greater"
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_batched(self):
        """Test batched computation with shape (5, 100) -> output (5,)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        samples = torch.randn(5, 100, dtype=torch.float64)

        t_stat, p_value, df = one_sample_t_test(samples)

        assert t_stat.shape == (5,)
        assert p_value.shape == (5,)
        assert df.shape == (5,)

        # Verify each batch matches scipy
        for i in range(5):
            scipy_result = scipy_stats.ttest_1samp(samples[i].numpy(), 0.0)
            assert torch.allclose(
                t_stat[i],
                torch.tensor(scipy_result.statistic, dtype=torch.float64),
                rtol=1e-5,
            )
            assert torch.allclose(
                p_value[i],
                torch.tensor(scipy_result.pvalue, dtype=torch.float64),
                rtol=1e-5,
            )

    def test_multidim_batch(self):
        """Test multidimensional batch with shape (2, 3, 50) -> output (2, 3)."""
        torch.manual_seed(42)
        samples = torch.randn(2, 3, 50, dtype=torch.float64)

        t_stat, p_value, df = one_sample_t_test(samples)

        assert t_stat.shape == (2, 3)
        assert p_value.shape == (2, 3)
        assert df.shape == (2, 3)
        # All df should be n-1 = 49
        assert torch.allclose(
            df, torch.full((2, 3), 49.0, dtype=torch.float64)
        )

    def test_insufficient_samples(self):
        """Test that n=1 returns NaN for statistic, p-value, and df."""
        sample = torch.tensor([1.0], dtype=torch.float64)

        t_stat, p_value, df = one_sample_t_test(sample)

        assert torch.isnan(t_stat)
        assert torch.isnan(p_value)
        assert torch.isnan(df)

    def test_zero_variance(self):
        """Test that all same values returns NaN."""
        sample = torch.full((10,), 5.0, dtype=torch.float64)

        t_stat, p_value, df = one_sample_t_test(sample)

        assert torch.isnan(t_stat) or torch.isinf(t_stat)
        assert torch.isnan(p_value)

    def test_invalid_alternative(self):
        """Test that invalid alternative raises ValueError."""
        sample = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="alternative must be one of"):
            one_sample_t_test(sample, alternative="invalid")


class TestTwoSampleTTest:
    """Tests for two_sample_t_test function."""

    def test_welch_correctness(self):
        """Test Welch's t-test against scipy.stats.ttest_ind(equal_var=False)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(50, dtype=torch.float64)
        sample2 = torch.randn(50, dtype=torch.float64) + 0.5

        t_stat, p_value, df = two_sample_t_test(
            sample1, sample2, equal_var=False
        )
        scipy_result = scipy_stats.ttest_ind(
            sample1.numpy(), sample2.numpy(), equal_var=False
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_student_correctness(self):
        """Test Student's t-test against scipy.stats.ttest_ind(equal_var=True)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(50, dtype=torch.float64)
        sample2 = torch.randn(50, dtype=torch.float64) + 0.5

        t_stat, p_value, df = two_sample_t_test(
            sample1, sample2, equal_var=True
        )
        scipy_result = scipy_stats.ttest_ind(
            sample1.numpy(), sample2.numpy(), equal_var=True
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )
        # Student's df should be n1 + n2 - 2 = 98
        assert torch.allclose(
            df, torch.tensor(98.0, dtype=torch.float64), rtol=1e-5
        )

    def test_different_sample_sizes(self):
        """Test with different sample sizes n1=50, n2=100."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(50, dtype=torch.float64)
        sample2 = torch.randn(100, dtype=torch.float64)

        t_stat, p_value, df = two_sample_t_test(sample1, sample2)
        scipy_result = scipy_stats.ttest_ind(
            sample1.numpy(), sample2.numpy(), equal_var=False
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_alternative_less(self):
        """Test alternative='less' hypothesis."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(50, dtype=torch.float64)
        sample2 = torch.randn(50, dtype=torch.float64) + 0.5

        t_stat, p_value, df = two_sample_t_test(
            sample1, sample2, alternative="less"
        )
        scipy_result = scipy_stats.ttest_ind(
            sample1.numpy(),
            sample2.numpy(),
            equal_var=False,
            alternative="less",
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_batched(self):
        """Test batched computation with shapes (5, 100) and (5, 80)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        samples1 = torch.randn(5, 100, dtype=torch.float64)
        samples2 = torch.randn(5, 80, dtype=torch.float64)

        t_stat, p_value, df = two_sample_t_test(samples1, samples2)

        assert t_stat.shape == (5,)
        assert p_value.shape == (5,)
        assert df.shape == (5,)

        # Verify each batch matches scipy
        for i in range(5):
            scipy_result = scipy_stats.ttest_ind(
                samples1[i].numpy(), samples2[i].numpy(), equal_var=False
            )
            assert torch.allclose(
                t_stat[i],
                torch.tensor(scipy_result.statistic, dtype=torch.float64),
                rtol=1e-5,
            )
            assert torch.allclose(
                p_value[i],
                torch.tensor(scipy_result.pvalue, dtype=torch.float64),
                rtol=1e-5,
            )

    def test_equal_var_differs(self):
        """Verify equal_var=True/False give different degrees of freedom."""
        torch.manual_seed(42)
        # Use samples with different variances
        sample1 = torch.randn(50, dtype=torch.float64)
        sample2 = (
            torch.randn(50, dtype=torch.float64) * 2
        )  # Different variance

        _, _, df_welch = two_sample_t_test(sample1, sample2, equal_var=False)
        _, _, df_student = two_sample_t_test(sample1, sample2, equal_var=True)

        # Student's df is always n1 + n2 - 2 = 98
        assert torch.allclose(
            df_student, torch.tensor(98.0, dtype=torch.float64)
        )
        # Welch's df should be different (and typically less than Student's)
        assert not torch.allclose(df_welch, df_student, rtol=1e-3)

    def test_insufficient_samples_welch(self):
        """Test that n=1 returns NaN for Welch's t-test."""
        sample1 = torch.tensor([1.0], dtype=torch.float64)
        sample2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        t_stat, p_value, df = two_sample_t_test(sample1, sample2)

        assert torch.isnan(t_stat) or torch.isinf(t_stat)
        assert torch.isnan(p_value)


class TestPairedTTest:
    """Tests for paired_t_test function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.ttest_rel."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(30, dtype=torch.float64)
        sample2 = sample1 + torch.randn(30, dtype=torch.float64) * 0.5

        t_stat, p_value, df = paired_t_test(sample1, sample2)
        scipy_result = scipy_stats.ttest_rel(sample1.numpy(), sample2.numpy())

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            df, torch.tensor(29.0, dtype=torch.float64), rtol=1e-5
        )

    def test_alternative_less(self):
        """Test alternative='less' hypothesis."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample1 = torch.randn(30, dtype=torch.float64)
        sample2 = sample1 + 0.5  # sample2 is larger on average

        t_stat, p_value, df = paired_t_test(
            sample1, sample2, alternative="less"
        )
        scipy_result = scipy_stats.ttest_rel(
            sample1.numpy(), sample2.numpy(), alternative="less"
        )

        assert torch.allclose(
            t_stat,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            p_value,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_batched(self):
        """Test batched computation with shape (5, 30)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        samples1 = torch.randn(5, 30, dtype=torch.float64)
        samples2 = samples1 + torch.randn(5, 30, dtype=torch.float64) * 0.5

        t_stat, p_value, df = paired_t_test(samples1, samples2)

        assert t_stat.shape == (5,)
        assert p_value.shape == (5,)
        assert df.shape == (5,)

        # Verify each batch matches scipy
        for i in range(5):
            scipy_result = scipy_stats.ttest_rel(
                samples1[i].numpy(), samples2[i].numpy()
            )
            assert torch.allclose(
                t_stat[i],
                torch.tensor(scipy_result.statistic, dtype=torch.float64),
                rtol=1e-5,
            )
            assert torch.allclose(
                p_value[i],
                torch.tensor(scipy_result.pvalue, dtype=torch.float64),
                rtol=1e-5,
            )

    def test_insufficient_samples(self):
        """Test that n=1 returns NaN."""
        sample1 = torch.tensor([1.0], dtype=torch.float64)
        sample2 = torch.tensor([2.0], dtype=torch.float64)

        t_stat, p_value, df = paired_t_test(sample1, sample2)

        assert torch.isnan(t_stat)
        assert torch.isnan(p_value)
        assert torch.isnan(df)

    def test_shape_mismatch_error(self):
        """Test that different shapes raise RuntimeError."""
        sample1 = torch.randn(10, dtype=torch.float64)
        sample2 = torch.randn(15, dtype=torch.float64)

        with pytest.raises(RuntimeError):
            paired_t_test(sample1, sample2)


class TestDtypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_one_sample_dtypes(self, dtype):
        """Test one_sample_t_test with different float dtypes."""
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=dtype)

        t_stat, p_value, df = one_sample_t_test(sample)

        assert t_stat.dtype == dtype
        assert p_value.dtype == dtype
        assert df.dtype == dtype
        # Check values are finite
        assert torch.isfinite(t_stat)
        assert torch.isfinite(p_value)
        assert torch.isfinite(df)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_two_sample_dtypes(self, dtype):
        """Test two_sample_t_test with different float dtypes."""
        torch.manual_seed(42)
        sample1 = torch.randn(50, dtype=dtype)
        sample2 = torch.randn(50, dtype=dtype)

        t_stat, p_value, df = two_sample_t_test(sample1, sample2)

        assert t_stat.dtype == dtype
        assert p_value.dtype == dtype
        assert df.dtype == dtype
        # Check values are finite
        assert torch.isfinite(t_stat)
        assert torch.isfinite(p_value)
        assert torch.isfinite(df)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_paired_dtypes(self, dtype):
        """Test paired_t_test with different float dtypes."""
        torch.manual_seed(42)
        sample1 = torch.randn(30, dtype=dtype)
        sample2 = torch.randn(30, dtype=dtype)

        t_stat, p_value, df = paired_t_test(sample1, sample2)

        assert t_stat.dtype == dtype
        assert p_value.dtype == dtype
        assert df.dtype == dtype
        # Check values are finite
        assert torch.isfinite(t_stat)
        assert torch.isfinite(p_value)
        assert torch.isfinite(df)
