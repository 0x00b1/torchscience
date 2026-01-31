import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401 - loads the C++ extension


class TestChi2CdfForward:
    """Test chi2_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.cdf."""
        x = torch.linspace(0.1, 20, 100)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_cumulative_distribution(x, df)
        expected = torch.tensor(
            scipy.stats.chi2.cdf(x.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_at_zero(self):
        """CDF(0) = 0."""
        x = torch.tensor([0.0])
        df = torch.tensor(5.0)
        result = torch.ops.torchscience.chi2_cumulative_distribution(x, df)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_chi2_2_is_exponential(self):
        """Chi2(2) = Exponential(1/2), so CDF = 1 - exp(-x/2)."""
        x = torch.linspace(0.1, 10, 50)
        df = torch.tensor(2.0)

        result = torch.ops.torchscience.chi2_cumulative_distribution(x, df)
        expected = 1 - torch.exp(-x / 2)

        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("df", [1.0, 2.0, 5.0, 10.0, 30.0])
    def test_various_df(self, df):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 30, 50)
        df_t = torch.tensor(df)

        result = torch.ops.torchscience.chi2_cumulative_distribution(x, df_t)
        expected = torch.tensor(
            scipy.stats.chi2.cdf(x.numpy(), df=df),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestChi2CdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.chi2_cumulative_distribution(x_, df)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_df(self):
        """Gradient check for df parameter."""
        x = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(df_):
            return torch.ops.torchscience.chi2_cumulative_distribution(x, df_)

        assert torch.autograd.gradcheck(fn, (df,), eps=1e-6, atol=1e-4)

    def test_gradcheck_both(self):
        """Gradient check for both parameters."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, df_):
            return torch.ops.torchscience.chi2_cumulative_distribution(x_, df_)

        assert torch.autograd.gradcheck(fn, (x, df), eps=1e-6, atol=1e-4)


class TestChi2Pdf:
    """Test chi2_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.pdf."""
        x = torch.linspace(0.1, 20, 100)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_probability_density(x, df)
        expected = torch.tensor(
            scipy.stats.chi2.pdf(x.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_at_zero(self):
        """PDF at x=0."""
        x = torch.tensor([0.0])
        df = torch.tensor(5.0)
        result = torch.ops.torchscience.chi2_probability_density(x, df)
        # For df > 2, pdf(0) = 0
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    @pytest.mark.parametrize("df", [2.0, 5.0, 10.0])
    def test_various_df(self, df):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 20, 50)
        df_t = torch.tensor(df)

        result = torch.ops.torchscience.chi2_probability_density(x, df_t)
        expected = torch.tensor(
            scipy.stats.chi2.pdf(x.numpy(), df=df),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestChi2Ppf:
    """Test chi2_quantile forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.ppf."""
        p = torch.linspace(0.01, 0.99, 50)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_quantile(p, df)
        expected = torch.tensor(
            scipy.stats.chi2.ppf(p.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_inverse_of_cumulative_distribution(self):
        """PPF should be inverse of CDF."""
        p = torch.linspace(0.1, 0.9, 20)
        df = torch.tensor(5.0)

        x = torch.ops.torchscience.chi2_quantile(p, df)
        p_recovered = torch.ops.torchscience.chi2_cumulative_distribution(
            x, df
        )

        assert torch.allclose(p, p_recovered, atol=1e-5)

    @pytest.mark.parametrize("df", [2.0, 5.0, 10.0])
    def test_various_df(self, df):
        """Test various degrees of freedom."""
        p = torch.linspace(0.1, 0.9, 20)
        df_t = torch.tensor(df)

        result = torch.ops.torchscience.chi2_quantile(p, df_t)
        expected = torch.tensor(
            scipy.stats.chi2.ppf(p.numpy(), df=df),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestChi2Sf:
    """Test chi2_survival forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.sf."""
        x = torch.linspace(0.1, 20, 100)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_survival(x, df)
        expected = torch.tensor(
            scipy.stats.chi2.sf(x.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cumulative_distribution_plus_survival_equals_one(self):
        """CDF + SF = 1."""
        x = torch.linspace(0.1, 20, 50)
        df = torch.tensor(5.0)

        cdf = torch.ops.torchscience.chi2_cumulative_distribution(x, df)
        sf = torch.ops.torchscience.chi2_survival(x, df)

        assert torch.allclose(cdf + sf, torch.ones_like(cdf), atol=1e-5)

    def test_at_zero(self):
        """SF(0) = 1."""
        x = torch.tensor([0.0])
        df = torch.tensor(5.0)
        result = torch.ops.torchscience.chi2_survival(x, df)
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-6)


class TestChi2LogPdf:
    """Test chi2_log_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.logpdf."""
        x = torch.linspace(0.1, 20, 100)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_log_probability_density(x, df)
        expected = torch.tensor(
            scipy.stats.chi2.logpdf(x.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_consistency_with_pdf(self):
        """log(pdf) should equal logpdf for positive values."""
        x = torch.linspace(0.1, 20, 50)
        df = torch.tensor(5.0)

        pdf = torch.ops.torchscience.chi2_probability_density(x, df)
        log_pdf = torch.ops.torchscience.chi2_log_probability_density(x, df)

        assert torch.allclose(torch.log(pdf), log_pdf, atol=1e-5)

    def test_at_zero(self):
        """log PDF at x=0 should be -inf."""
        x = torch.tensor([0.0])
        df = torch.tensor(5.0)
        result = torch.ops.torchscience.chi2_log_probability_density(x, df)
        assert torch.isinf(result).all()
        assert (result < 0).all()

    @pytest.mark.parametrize("df", [2.0, 5.0, 10.0])
    def test_various_df(self, df):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 20, 50)
        df_t = torch.tensor(df)

        result = torch.ops.torchscience.chi2_log_probability_density(x, df_t)
        expected = torch.tensor(
            scipy.stats.chi2.logpdf(x.numpy(), df=df),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestChi2LogPdfGradients:
    """Test chi2_log_probability_density gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.chi2_log_probability_density(x_, df)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_df(self):
        """Gradient check for df parameter."""
        x = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(df_):
            return torch.ops.torchscience.chi2_log_probability_density(x, df_)

        assert torch.autograd.gradcheck(fn, (df,), eps=1e-6, atol=1e-4)

    def test_gradcheck_both(self):
        """Gradient check for both parameters."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, df_):
            return torch.ops.torchscience.chi2_log_probability_density(x_, df_)

        assert torch.autograd.gradcheck(fn, (x, df), eps=1e-6, atol=1e-4)
