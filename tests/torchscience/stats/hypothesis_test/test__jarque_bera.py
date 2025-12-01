"""Tests for jarque_bera function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import jarque_bera


class TestJarqueBera:
    """Tests for jarque_bera function."""

    def test_basic_correctness(self):
        """Test basic correctness against scipy.stats.jarque_bera."""
        scipy_stats = pytest.importorskip("scipy.stats")

        torch.manual_seed(42)
        sample = torch.randn(100, dtype=torch.float64)

        statistic, pvalue = jarque_bera(sample)
        scipy_result = scipy_stats.jarque_bera(sample.numpy())

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

    def test_normal_sample_high_pvalue(self):
        """Test that a normal sample has high p-value."""
        torch.manual_seed(123)
        sample = torch.randn(1000, dtype=torch.float64)
        statistic, pvalue = jarque_bera(sample)
        assert pvalue > 0.05

    def test_uniform_sample_low_pvalue(self):
        """Test that a uniform sample has low p-value."""
        torch.manual_seed(42)
        sample = torch.rand(1000, dtype=torch.float64)
        statistic, pvalue = jarque_bera(sample)
        assert pvalue < 0.05

    def test_batched(self):
        """Test batched computation."""
        scipy_stats = pytest.importorskip("scipy.stats")
        torch.manual_seed(42)
        samples = torch.randn(5, 100, dtype=torch.float64)

        statistic, pvalue = jarque_bera(samples)
        assert statistic.shape == (5,)
        assert pvalue.shape == (5,)

        for i in range(5):
            scipy_result = scipy_stats.jarque_bera(samples[i].numpy())
            assert torch.allclose(
                statistic[i],
                torch.tensor(scipy_result.statistic, dtype=torch.float64),
                rtol=1e-5,
            )

    def test_insufficient_samples(self):
        """Test that n < 3 returns NaN."""
        sample = torch.tensor([1.0, 2.0], dtype=torch.float64)
        statistic, pvalue = jarque_bera(sample)
        assert torch.isnan(statistic)
        assert torch.isnan(pvalue)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        sample = torch.randn(100, dtype=torch.float64, device="meta")
        statistic, pvalue = jarque_bera(sample)
        assert statistic.device.type == "meta"
        assert statistic.shape == ()


class TestJarqueBeraGradients:
    """Tests for gradient support."""

    def test_gradcheck(self):
        """Test gradient computation."""
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=torch.float64, requires_grad=True)

        def func(x):
            stat, _ = jarque_bera(x)
            return stat

        assert torch.autograd.gradcheck(func, (sample,), raise_exception=True)


class TestJarqueBeraTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        torch.manual_seed(42)
        sample = torch.randn(50, dtype=dtype)
        statistic, pvalue = jarque_bera(sample)
        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
