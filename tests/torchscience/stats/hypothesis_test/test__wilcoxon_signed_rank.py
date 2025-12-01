"""Tests for wilcoxon_signed_rank function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import wilcoxon_signed_rank


class TestWilcoxonSignedRank:
    """Tests for wilcoxon_signed_rank function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.wilcoxon."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)

        statistic, pvalue = wilcoxon_signed_rank(x, y)
        # Use method="approx" to match our asymptotic implementation
        scipy_result = scipy_stats.wilcoxon(
            x.numpy(), y.numpy(), alternative="two-sided", method="approx"
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_one_sample(self):
        """Test one-sample version (test median = 0)."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        statistic, pvalue = wilcoxon_signed_rank(x)
        scipy_result = scipy_stats.wilcoxon(
            x.numpy(), alternative="two-sided", method="approx"
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_zero_differences_excluded(self):
        """Test that zero differences are excluded with wilcox method."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )  # First pair has zero diff

        statistic, pvalue = wilcoxon_signed_rank(x, y, zero_method="wilcox")
        scipy_result = scipy_stats.wilcoxon(
            x.numpy(), y.numpy(), zero_method="wilcox", method="approx"
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_alternative_less(self):
        """Test one-sided alternative 'less'."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)

        statistic, pvalue = wilcoxon_signed_rank(x, y, alternative="less")
        scipy_result = scipy_stats.wilcoxon(
            x.numpy(), y.numpy(), alternative="less", method="approx"
        )

        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-2,
        )

    def test_alternative_greater(self):
        """Test one-sided alternative 'greater'."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        statistic, pvalue = wilcoxon_signed_rank(x, y, alternative="greater")
        scipy_result = scipy_stats.wilcoxon(
            x.numpy(), y.numpy(), alternative="greater", method="approx"
        )

        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-2,
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.randn(10, dtype=torch.float64, device="meta")
        y = torch.randn(10, dtype=torch.float64, device="meta")

        statistic, pvalue = wilcoxon_signed_rank(x, y)

        assert statistic.device.type == "meta"
        assert statistic.shape == ()

    def test_no_gradients(self):
        """Test that gradients are not supported."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        y = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        with pytest.raises(RuntimeError, match="does not support gradients"):
            wilcoxon_signed_rank(x, y)


class TestWilcoxonSignedRankTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype)

        with torch.no_grad():
            statistic, pvalue = wilcoxon_signed_rank(x, y)

        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
