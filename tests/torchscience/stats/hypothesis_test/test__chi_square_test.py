"""Tests for chi_square_test function."""

import pytest
import torch

from torchscience.statistics.hypothesis_test import chi_square_test


class TestChiSquareTest:
    """Tests for chi_square_test function."""

    def test_basic_correctness_uniform(self):
        """Test against scipy.stats.chisquare with uniform expected."""
        scipy_stats = pytest.importorskip("scipy.stats")

        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0], dtype=torch.float64
        )
        statistic, pvalue = chi_square_test(observed)
        scipy_result = scipy_stats.chisquare(observed.numpy())

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

    def test_with_expected_frequencies(self):
        """Test with explicit expected frequencies."""
        scipy_stats = pytest.importorskip("scipy.stats")

        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0], dtype=torch.float64
        )
        expected = torch.tensor(
            [16.0, 16.0, 16.0, 16.0, 12.0, 12.0], dtype=torch.float64
        )

        statistic, pvalue = chi_square_test(observed, expected)
        scipy_result = scipy_stats.chisquare(
            observed.numpy(), expected.numpy()
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

    def test_ddof(self):
        """Test with delta degrees of freedom."""
        scipy_stats = pytest.importorskip("scipy.stats")

        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0], dtype=torch.float64
        )
        statistic, pvalue = chi_square_test(observed, ddof=1)
        scipy_result = scipy_stats.chisquare(observed.numpy(), ddof=1)

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

    def test_batched(self):
        """Test batched computation."""
        scipy_stats = pytest.importorskip("scipy.stats")
        torch.manual_seed(42)
        observed = torch.rand(3, 6, dtype=torch.float64) * 20 + 10

        statistic, pvalue = chi_square_test(observed)
        assert statistic.shape == (3,)
        assert pvalue.shape == (3,)

        for i in range(3):
            scipy_result = scipy_stats.chisquare(observed[i].numpy())
            assert torch.allclose(
                statistic[i],
                torch.tensor(scipy_result.statistic, dtype=torch.float64),
                rtol=1e-5,
            )

    def test_perfect_fit(self):
        """Test that perfectly matching frequencies give high p-value."""
        observed = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float64)
        statistic, pvalue = chi_square_test(observed)
        assert statistic == 0.0
        assert pvalue == 1.0

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        observed = torch.randn(6, dtype=torch.float64, device="meta")
        statistic, pvalue = chi_square_test(observed)
        assert statistic.device.type == "meta"
        assert statistic.shape == ()


class TestChiSquareTestGradients:
    """Tests for gradient support."""

    def test_gradcheck(self):
        """Test gradient computation."""
        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(x):
            stat, _ = chi_square_test(x)
            return stat

        assert torch.autograd.gradcheck(
            func, (observed,), raise_exception=True
        )

    def test_gradcheck_with_expected(self):
        """Test gradient computation with expected frequencies."""
        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected = torch.tensor(
            [16.0, 16.0, 16.0, 16.0, 12.0, 12.0],
            dtype=torch.float64,
        )

        def func(x):
            stat, _ = chi_square_test(x, expected)
            return stat

        assert torch.autograd.gradcheck(
            func, (observed,), raise_exception=True
        )


class TestChiSquareTestTypes:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        observed = torch.tensor(
            [16.0, 18.0, 16.0, 14.0, 12.0, 12.0], dtype=dtype
        )
        statistic, pvalue = chi_square_test(observed)
        assert statistic.dtype == dtype
        assert torch.isfinite(statistic)
