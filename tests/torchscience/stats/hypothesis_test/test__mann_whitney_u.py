"""Tests for mann_whitney_u function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import mann_whitney_u


class TestMannWhitneyU:
    """Tests for mann_whitney_u function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.mannwhitneyu."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        y = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)

        statistic, pvalue = mann_whitney_u(x, y)
        # Use method="asymptotic" to match our implementation
        scipy_result = scipy_stats.mannwhitneyu(
            x.numpy(), y.numpy(), alternative="two-sided", method="asymptotic"
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )
        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-2,  # Asymptotic approximation may differ slightly
        )

    def test_identical_samples_high_pvalue(self):
        """Test that identical distributions have high p-value."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64)
        y = torch.randn(50, dtype=torch.float64)

        statistic, pvalue = mann_whitney_u(x, y)

        # Cannot reject null hypothesis
        assert pvalue > 0.05

    def test_different_samples_low_pvalue(self):
        """Test that different distributions have low p-value."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64)
        y = torch.randn(50, dtype=torch.float64) + 5.0  # Shifted

        statistic, pvalue = mann_whitney_u(x, y)

        # Reject null hypothesis
        assert pvalue < 0.05

    def test_with_ties(self):
        """Test with tied values."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        statistic, pvalue = mann_whitney_u(x, y)
        scipy_result = scipy_stats.mannwhitneyu(
            x.numpy(), y.numpy(), alternative="two-sided"
        )

        assert torch.allclose(
            statistic,
            torch.tensor(scipy_result.statistic, dtype=torch.float64),
            rtol=1e-5,
        )

    def test_alternative_less(self):
        """Test one-sided alternative 'less'."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)

        statistic, pvalue = mann_whitney_u(x, y, alternative="less")
        # Use method="asymptotic" to match our implementation
        scipy_result = scipy_stats.mannwhitneyu(
            x.numpy(), y.numpy(), alternative="less", method="asymptotic"
        )

        assert torch.allclose(
            pvalue,
            torch.tensor(scipy_result.pvalue, dtype=torch.float64),
            rtol=1e-2,
        )

    def test_alternative_greater(self):
        """Test one-sided alternative 'greater'."""
        scipy_stats = pytest.importorskip("scipy.stats")

        x = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        statistic, pvalue = mann_whitney_u(x, y, alternative="greater")
        # Use method="asymptotic" to match our implementation
        scipy_result = scipy_stats.mannwhitneyu(
            x.numpy(), y.numpy(), alternative="greater", method="asymptotic"
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

        statistic, pvalue = mann_whitney_u(x, y)

        assert statistic.device.type == "meta"
        assert pvalue.device.type == "meta"
        assert statistic.shape == ()

    def test_no_gradients(self):
        """Test that gradients are not supported."""
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        y = torch.tensor(
            [4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True
        )

        with pytest.raises(RuntimeError, match="does not support gradients"):
            mann_whitney_u(x, y)


class TestMannWhitneyUTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        y = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=dtype)

        with torch.no_grad():
            statistic, pvalue = mann_whitney_u(x, y)

        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
