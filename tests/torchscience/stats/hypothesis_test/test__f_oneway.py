"""Tests for f_oneway function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import f_oneway


class TestFOneway:
    """Tests for f_oneway function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.f_oneway."""
        scipy_stats = pytest.importorskip("scipy.stats")

        group1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        group2 = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float64)
        group3 = torch.tensor([1.5, 3.0, 4.5, 6.0, 7.5], dtype=torch.float64)

        statistic, pvalue = f_oneway(group1, group2, group3)
        scipy_result = scipy_stats.f_oneway(
            group1.numpy(), group2.numpy(), group3.numpy()
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_two_groups(self):
        """Test with two groups (should match two-sample t-test)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        group1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        group2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

        statistic, pvalue = f_oneway(group1, group2)
        scipy_result = scipy_stats.f_oneway(group1.numpy(), group2.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_equal_means_high_pvalue(self):
        """Test that equal group means produce high p-value."""
        torch.manual_seed(42)
        # Same population, random samples
        group1 = torch.randn(50, dtype=torch.float64)
        group2 = torch.randn(50, dtype=torch.float64)
        group3 = torch.randn(50, dtype=torch.float64)

        statistic, pvalue = f_oneway(group1, group2, group3)

        # Cannot reject null hypothesis of equal means
        assert pvalue > 0.05

    def test_different_means_low_pvalue(self):
        """Test that different group means produce low p-value."""
        torch.manual_seed(42)
        group1 = torch.randn(50, dtype=torch.float64)
        group2 = torch.randn(50, dtype=torch.float64) + 5.0  # Different mean
        group3 = torch.randn(50, dtype=torch.float64) + 10.0  # Different mean

        statistic, pvalue = f_oneway(group1, group2, group3)

        # Reject null hypothesis
        assert pvalue < 0.05

    def test_unequal_group_sizes(self):
        """Test with unequal group sizes."""
        scipy_stats = pytest.importorskip("scipy.stats")

        group1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        group2 = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        group3 = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        statistic, pvalue = f_oneway(group1, group2, group3)
        scipy_result = scipy_stats.f_oneway(
            group1.numpy(), group2.numpy(), group3.numpy()
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_many_groups(self):
        """Test with many groups."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        groups = [torch.randn(20, dtype=torch.float64) + i for i in range(5)]

        statistic, pvalue = f_oneway(*groups)
        scipy_result = scipy_stats.f_oneway(*[g.numpy() for g in groups])

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_insufficient_groups(self):
        """Test that single group raises error."""
        group1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="at least 2 groups"):
            f_oneway(group1)

    def test_insufficient_samples(self):
        """Test that group with single sample still computes a result."""
        scipy_stats = pytest.importorskip("scipy.stats")

        group1 = torch.tensor([1.0], dtype=torch.float64)
        group2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        # scipy computes a valid result even with a single-sample group
        statistic, pvalue = f_oneway(group1, group2)
        scipy_result = scipy_stats.f_oneway(group1.numpy(), group2.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        group1 = torch.randn(10, dtype=torch.float64, device="meta")
        group2 = torch.randn(10, dtype=torch.float64, device="meta")

        statistic, pvalue = f_oneway(group1, group2)

        assert statistic.device.type == "meta"
        assert pvalue.device.type == "meta"
        assert statistic.shape == ()
        assert pvalue.shape == ()


class TestFOnewayGradients:
    """Tests for f_oneway gradient support."""

    def test_gradcheck(self):
        """Test gradient computation."""
        group1 = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )
        group2 = torch.tensor(
            [5.0, 6.0, 7.0, 8.0], dtype=torch.float64, requires_grad=True
        )

        def func(g1, g2):
            stat, _ = f_oneway(g1, g2)
            return stat

        assert torch.autograd.gradcheck(
            func, (group1, group2), raise_exception=True
        )


class TestFOnewayTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        """Test f_oneway with different float dtypes."""
        group1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        group2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=dtype)

        statistic, pvalue = f_oneway(group1, group2)

        assert statistic.dtype == dtype
        assert pvalue.dtype == dtype
        assert torch.isfinite(statistic)
        assert torch.isfinite(pvalue)
