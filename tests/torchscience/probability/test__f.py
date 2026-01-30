import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401
from torchscience import _csrc  # noqa: F401 - loads the C++ extension


class TestFCdfForward:
    """Test f_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.f.cdf."""
        x = torch.linspace(0.1, 10, 100)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        result = torch.ops.torchscience.f_cumulative_distribution(x, dfn, dfd)
        expected = torch.tensor(
            scipy.stats.f.cdf(x.numpy(), dfn=5, dfd=10),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_symmetric_case(self):
        """F(1, d, d) has CDF(1) = 0.5."""
        x = torch.tensor([1.0])
        d = torch.tensor(10.0)

        result = torch.ops.torchscience.f_cumulative_distribution(x, d, d)
        assert torch.allclose(result, torch.tensor([0.5]), atol=1e-4)

    def test_at_zero(self):
        """CDF(0) = 0."""
        x = torch.tensor([0.0])
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)
        result = torch.ops.torchscience.f_cumulative_distribution(x, dfn, dfd)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    @pytest.mark.parametrize(
        "dfn,dfd", [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    )
    def test_various_df(self, dfn, dfd):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 10, 50)
        dfn_t = torch.tensor(dfn)
        dfd_t = torch.tensor(dfd)

        result = torch.ops.torchscience.f_cumulative_distribution(
            x, dfn_t, dfd_t
        )
        expected = torch.tensor(
            scipy.stats.f.cdf(x.numpy(), dfn=dfn, dfd=dfd),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestFCdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        dfn = torch.tensor(5.0, dtype=torch.float64)
        dfd = torch.tensor(10.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.f_cumulative_distribution(
                x_, dfn, dfd
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_dfn(self):
        """Gradient check for dfn parameter."""
        x = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        dfn = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        dfd = torch.tensor(10.0, dtype=torch.float64)

        def fn(dfn_):
            return torch.ops.torchscience.f_cumulative_distribution(
                x, dfn_, dfd
            )

        assert torch.autograd.gradcheck(fn, (dfn,), eps=1e-6, atol=1e-4)

    def test_gradcheck_dfd(self):
        """Gradient check for dfd parameter."""
        x = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        dfn = torch.tensor(5.0, dtype=torch.float64)
        dfd = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)

        def fn(dfd_):
            return torch.ops.torchscience.f_cumulative_distribution(
                x, dfn, dfd_
            )

        assert torch.autograd.gradcheck(fn, (dfd,), eps=1e-6, atol=1e-4)


class TestFPdf:
    """Test f_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.f.pdf."""
        x = torch.linspace(0.1, 10, 100)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        result = torch.ops.torchscience.f_probability_density(x, dfn, dfd)
        expected = torch.tensor(
            scipy.stats.f.pdf(x.numpy(), dfn=5, dfd=10),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_at_zero(self):
        """PDF at x=0."""
        x = torch.tensor([0.0])
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)
        result = torch.ops.torchscience.f_probability_density(x, dfn, dfd)
        # For dfn > 2, pdf(0) = 0
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    @pytest.mark.parametrize(
        "dfn,dfd", [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    )
    def test_various_df(self, dfn, dfd):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 10, 50)
        dfn_t = torch.tensor(dfn)
        dfd_t = torch.tensor(dfd)

        result = torch.ops.torchscience.f_probability_density(x, dfn_t, dfd_t)
        expected = torch.tensor(
            scipy.stats.f.pdf(x.numpy(), dfn=dfn, dfd=dfd),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestFPpf:
    """Test f_quantile forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.f.ppf."""
        p = torch.linspace(0.01, 0.99, 50)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        result = torch.ops.torchscience.f_quantile(p, dfn, dfd)
        expected = torch.tensor(
            scipy.stats.f.ppf(p.numpy(), dfn=5, dfd=10),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_inverse_of_cumulative_distribution(self):
        """PPF should be inverse of CDF."""
        p = torch.linspace(0.1, 0.9, 20)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        x = torch.ops.torchscience.f_quantile(p, dfn, dfd)
        p_recovered = torch.ops.torchscience.f_cumulative_distribution(
            x, dfn, dfd
        )

        assert torch.allclose(p, p_recovered, atol=1e-5)

    @pytest.mark.parametrize(
        "dfn,dfd", [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    )
    def test_various_df(self, dfn, dfd):
        """Test various degrees of freedom."""
        p = torch.linspace(0.1, 0.9, 20)
        dfn_t = torch.tensor(dfn)
        dfd_t = torch.tensor(dfd)

        result = torch.ops.torchscience.f_quantile(p, dfn_t, dfd_t)
        expected = torch.tensor(
            scipy.stats.f.ppf(p.numpy(), dfn=dfn, dfd=dfd),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestFSf:
    """Test f_survival forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.f.sf."""
        x = torch.linspace(0.1, 10, 100)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        result = torch.ops.torchscience.f_survival(x, dfn, dfd)
        expected = torch.tensor(
            scipy.stats.f.sf(x.numpy(), dfn=5, dfd=10),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cumulative_distribution_plus_survival_equals_one(self):
        """CDF + SF = 1."""
        x = torch.linspace(0.1, 10, 50)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        cdf = torch.ops.torchscience.f_cumulative_distribution(x, dfn, dfd)
        sf = torch.ops.torchscience.f_survival(x, dfn, dfd)

        assert torch.allclose(cdf + sf, torch.ones_like(cdf), atol=1e-5)

    def test_at_zero(self):
        """SF(0) = 1."""
        x = torch.tensor([0.0])
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)
        result = torch.ops.torchscience.f_survival(x, dfn, dfd)
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-6)


class TestFLogPdfForward:
    """Test f_log_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.f.logpdf."""
        x = torch.linspace(0.1, 10, 100)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        result = torch.ops.torchscience.f_log_probability_density(x, dfn, dfd)
        expected = torch.tensor(
            scipy.stats.f.logpdf(x.numpy(), dfn=5, dfd=10),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_exp_equals_pdf(self):
        """exp(logpdf) should equal pdf."""
        x = torch.linspace(0.1, 10, 50)
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)

        log_pdf = torch.ops.torchscience.f_log_probability_density(x, dfn, dfd)
        pdf = torch.ops.torchscience.f_probability_density(x, dfn, dfd)

        assert torch.allclose(torch.exp(log_pdf), pdf, atol=1e-5)

    def test_at_zero(self):
        """Log PDF at x=0 should be -inf for dfn > 2."""
        x = torch.tensor([0.0])
        dfn = torch.tensor(5.0)
        dfd = torch.tensor(10.0)
        result = torch.ops.torchscience.f_log_probability_density(x, dfn, dfd)
        assert torch.isinf(result).all()
        assert (result < 0).all()

    @pytest.mark.parametrize(
        "dfn,dfd", [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    )
    def test_various_df(self, dfn, dfd):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 10, 50)
        dfn_t = torch.tensor(dfn)
        dfd_t = torch.tensor(dfd)

        result = torch.ops.torchscience.f_log_probability_density(
            x, dfn_t, dfd_t
        )
        expected = torch.tensor(
            scipy.stats.f.logpdf(x.numpy(), dfn=dfn, dfd=dfd),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestFLogPdfGradients:
    """Test gradient computation for f_log_probability_density."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        dfn = torch.tensor(5.0, dtype=torch.float64)
        dfd = torch.tensor(10.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.f_log_probability_density(
                x_, dfn, dfd
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_dfn(self):
        """Gradient check for dfn parameter."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        dfn = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        dfd = torch.tensor(10.0, dtype=torch.float64)

        def fn(dfn_):
            return torch.ops.torchscience.f_log_probability_density(
                x, dfn_, dfd
            )

        assert torch.autograd.gradcheck(fn, (dfn,), eps=1e-6, atol=1e-4)

    def test_gradcheck_dfd(self):
        """Gradient check for dfd parameter."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        dfn = torch.tensor(5.0, dtype=torch.float64)
        dfd = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)

        def fn(dfd_):
            return torch.ops.torchscience.f_log_probability_density(
                x, dfn, dfd_
            )

        assert torch.autograd.gradcheck(fn, (dfd,), eps=1e-6, atol=1e-4)

    def test_gradcheck_all_params(self):
        """Gradient check for all parameters."""
        x = torch.tensor(
            [0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        dfn = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        dfd = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, dfn_, dfd_):
            return torch.ops.torchscience.f_log_probability_density(
                x_, dfn_, dfd_
            )

        assert torch.autograd.gradcheck(fn, (x, dfn, dfd), eps=1e-6, atol=1e-4)
