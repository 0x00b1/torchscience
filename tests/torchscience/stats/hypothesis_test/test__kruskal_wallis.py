"""Tests for kruskal_wallis function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import kruskal_wallis


class TestKruskalWallis:
    """Tests for kruskal_wallis function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.kruskal."""
        scipy_stats = pytest.importorskip("scipy.stats")

        # Three groups with different distributions
        g1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        g2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        g3 = torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64)

        # Concatenate data and provide group sizes
        data = torch.cat([g1, g2, g3])
        group_sizes = torch.tensor([4, 4, 4], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)
        scipy_result = scipy_stats.kruskal(g1.numpy(), g2.numpy(), g3.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-2,
        )

    def test_identical_groups_high_pvalue(self):
        """Test that identical groups have high p-value."""
        torch.manual_seed(42)
        g1 = torch.randn(20, dtype=torch.float64)
        g2 = torch.randn(20, dtype=torch.float64)
        g3 = torch.randn(20, dtype=torch.float64)

        data = torch.cat([g1, g2, g3])
        group_sizes = torch.tensor([20, 20, 20], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)

        # Cannot reject null hypothesis
        assert pvalue > 0.05

    def test_different_groups_low_pvalue(self):
        """Test that different groups have low p-value."""
        torch.manual_seed(42)
        g1 = torch.randn(20, dtype=torch.float64)
        g2 = torch.randn(20, dtype=torch.float64) + 5.0
        g3 = torch.randn(20, dtype=torch.float64) + 10.0

        data = torch.cat([g1, g2, g3])
        group_sizes = torch.tensor([20, 20, 20], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)

        # Reject null hypothesis
        assert pvalue < 0.05

    def test_two_groups(self):
        """Test with only two groups (equivalent to Mann-Whitney)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        g1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        g2 = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)

        data = torch.cat([g1, g2])
        group_sizes = torch.tensor([5, 5], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)
        scipy_result = scipy_stats.kruskal(g1.numpy(), g2.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_with_ties(self):
        """Test with tied values."""
        scipy_stats = pytest.importorskip("scipy.stats")

        g1 = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
        g2 = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        g3 = torch.tensor([3.0, 4.0, 4.0, 5.0], dtype=torch.float64)

        data = torch.cat([g1, g2, g3])
        group_sizes = torch.tensor([4, 4, 4], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)
        scipy_result = scipy_stats.kruskal(g1.numpy(), g2.numpy(), g3.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_unequal_group_sizes(self):
        """Test with unequal group sizes."""
        scipy_stats = pytest.importorskip("scipy.stats")

        g1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        g2 = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        g3 = torch.tensor([9.0, 10.0], dtype=torch.float64)

        data = torch.cat([g1, g2, g3])
        group_sizes = torch.tensor([3, 5, 2], dtype=torch.int64)

        statistic, pvalue = kruskal_wallis(data, group_sizes)
        scipy_result = scipy_stats.kruskal(g1.numpy(), g2.numpy(), g3.numpy())

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        data = torch.randn(30, dtype=torch.float64, device="meta")
        group_sizes = torch.tensor(
            [10, 10, 10], dtype=torch.int64, device="meta"
        )

        statistic, pvalue = kruskal_wallis(data, group_sizes)

        assert statistic.device.type == "meta"
        assert pvalue.device.type == "meta"
        assert statistic.shape == ()

    def test_no_gradients(self):
        """Test that gradients are not supported."""
        data = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        group_sizes = torch.tensor([3, 3], dtype=torch.int64)

        with pytest.raises(RuntimeError, match="does not support gradients"):
            kruskal_wallis(data, group_sizes)


class TestKruskalWallisTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
        group_sizes = torch.tensor([3, 3], dtype=torch.int64)

        with torch.no_grad():
            statistic, pvalue = kruskal_wallis(data, group_sizes)

        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
