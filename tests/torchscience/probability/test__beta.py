import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401 - Load C++ extensions


class TestBetaCdfForward:
    """Test beta_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.beta.cdf."""
        x = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        result = torch.ops.torchscience.beta_cumulative_distribution(x, a, b)
        expected = torch.tensor(
            scipy.stats.beta.cdf(x.numpy(), a=2, b=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_boundaries(self):
        """CDF(0) = 0, CDF(1) = 1."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)

        assert torch.allclose(
            torch.ops.torchscience.beta_cumulative_distribution(
                torch.tensor([0.0]), a, b
            ),
            torch.tensor([0.0]),
            atol=1e-6,
        )
        assert torch.allclose(
            torch.ops.torchscience.beta_cumulative_distribution(
                torch.tensor([1.0]), a, b
            ),
            torch.tensor([1.0]),
            atol=1e-6,
        )

    def test_uniform_case(self):
        """Beta(1, 1) is Uniform(0, 1), so CDF(x) = x."""
        x = torch.linspace(0.01, 0.99, 50)
        a = torch.tensor(1.0)
        b = torch.tensor(1.0)

        result = torch.ops.torchscience.beta_cumulative_distribution(x, a, b)
        assert torch.allclose(result, x, atol=1e-5)

    @pytest.mark.parametrize(
        "a,b", [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (10, 10)]
    )
    def test_various_parameters(self, a, b):
        """Test various parameter combinations."""
        x = torch.linspace(0.01, 0.99, 50)
        a_t = torch.tensor(float(a))
        b_t = torch.tensor(float(b))

        result = torch.ops.torchscience.beta_cumulative_distribution(
            x, a_t, b_t
        )
        expected = torch.tensor(
            scipy.stats.beta.cdf(x.numpy(), a=a, b=b),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestBetaCdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.beta_cumulative_distribution(
                x_, a, b
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_a(self):
        """Gradient check for a parameter."""
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(a_):
            return torch.ops.torchscience.beta_cumulative_distribution(
                x, a_, b
            )

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    def test_gradcheck_b(self):
        """Gradient check for b parameter."""
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def fn(b_):
            return torch.ops.torchscience.beta_cumulative_distribution(
                x, a, b_
            )

        assert torch.autograd.gradcheck(fn, (b,), eps=1e-6, atol=1e-4)

    def test_grad_x_is_probability_density(self):
        """dCDF/dx = PDF."""
        x = torch.linspace(
            0.1, 0.9, 50, dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64)

        cdf = torch.ops.torchscience.beta_cumulative_distribution(x, a, b)
        grad_x = torch.autograd.grad(cdf.sum(), x)[0]

        # Compare to scipy PDF
        pdf_expected = torch.tensor(
            scipy.stats.beta.pdf(x.detach().numpy(), a=2, b=3),
            dtype=torch.float64,
        )
        assert torch.allclose(grad_x, pdf_expected, atol=1e-5)


class TestBetaPdfForward:
    """Test beta_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.beta.pdf."""
        x = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        result = torch.ops.torchscience.beta_probability_density(x, a, b)
        expected = torch.tensor(
            scipy.stats.beta.pdf(x.numpy(), a=2, b=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestBetaPpfForward:
    """Test beta_quantile forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.beta.ppf."""
        p = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        result = torch.ops.torchscience.beta_quantile(p, a, b)
        expected = torch.tensor(
            scipy.stats.beta.ppf(p.numpy(), a=2, b=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-4)

    def test_cumulative_distribution_quantile_roundtrip(self):
        """ppf(cdf(x)) = x."""
        x = torch.linspace(0.1, 0.9, 50)
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)

        p = torch.ops.torchscience.beta_cumulative_distribution(x, a, b)
        x_recovered = torch.ops.torchscience.beta_quantile(p, a, b)

        assert torch.allclose(x, x_recovered, atol=1e-5)


class TestBetaLogPdfForward:
    """Test beta_log_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.beta.logpdf."""
        x = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        result = torch.ops.torchscience.beta_log_probability_density(x, a, b)
        expected = torch.tensor(
            scipy.stats.beta.logpdf(x.numpy(), a=2, b=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_consistency_with_pdf(self):
        """log(pdf(x)) should equal logpdf(x)."""
        x = torch.linspace(0.1, 0.9, 50)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        log_pdf = torch.ops.torchscience.beta_log_probability_density(x, a, b)
        pdf = torch.ops.torchscience.beta_probability_density(x, a, b)

        assert torch.allclose(log_pdf, torch.log(pdf), atol=1e-5)

    def test_boundary_values(self):
        """Log PDF returns -inf at boundaries x=0 and x=1."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)

        # At x=0 and x=1, log PDF should be -inf
        result_0 = torch.ops.torchscience.beta_log_probability_density(
            torch.tensor([0.0]), a, b
        )
        result_1 = torch.ops.torchscience.beta_log_probability_density(
            torch.tensor([1.0]), a, b
        )

        assert torch.isinf(result_0).all()
        assert torch.isinf(result_1).all()
        assert (result_0 < 0).all()
        assert (result_1 < 0).all()

    @pytest.mark.parametrize(
        "a,b", [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (10, 10)]
    )
    def test_various_parameters(self, a, b):
        """Test various parameter combinations."""
        x = torch.linspace(0.01, 0.99, 50)
        a_t = torch.tensor(float(a))
        b_t = torch.tensor(float(b))

        result = torch.ops.torchscience.beta_log_probability_density(
            x, a_t, b_t
        )
        expected = torch.tensor(
            scipy.stats.beta.logpdf(x.numpy(), a=a, b=b),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestBetaLogPdfGradients:
    """Test beta_log_probability_density gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.beta_log_probability_density(
                x_, a, b
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_a(self):
        """Gradient check for a parameter."""
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(a_):
            return torch.ops.torchscience.beta_log_probability_density(
                x, a_, b
            )

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-3)

    def test_gradcheck_b(self):
        """Gradient check for b parameter."""
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def fn(b_):
            return torch.ops.torchscience.beta_log_probability_density(
                x, a, b_
            )

        assert torch.autograd.gradcheck(fn, (b,), eps=1e-6, atol=1e-3)


class TestBetaSurvivalForward:
    """Test beta_survival forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.beta.sf (survival function)."""
        x = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        result = torch.ops.torchscience.beta_survival(x, a, b)
        expected = torch.tensor(
            scipy.stats.beta.sf(x.numpy(), a=2, b=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_boundaries(self):
        """SF(0) = 1, SF(1) = 0."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)

        assert torch.allclose(
            torch.ops.torchscience.beta_survival(torch.tensor([0.0]), a, b),
            torch.tensor([1.0]),
            atol=1e-6,
        )
        assert torch.allclose(
            torch.ops.torchscience.beta_survival(torch.tensor([1.0]), a, b),
            torch.tensor([0.0]),
            atol=1e-6,
        )

    def test_uniform_case(self):
        """Beta(1, 1) is Uniform(0, 1), so SF(x) = 1 - x."""
        x = torch.linspace(0.01, 0.99, 50)
        a = torch.tensor(1.0)
        b = torch.tensor(1.0)

        result = torch.ops.torchscience.beta_survival(x, a, b)
        assert torch.allclose(result, 1 - x, atol=1e-5)

    @pytest.mark.parametrize(
        "a,b", [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (10, 10)]
    )
    def test_various_parameters(self, a, b):
        """Test various parameter combinations."""
        x = torch.linspace(0.01, 0.99, 50)
        a_t = torch.tensor(float(a))
        b_t = torch.tensor(float(b))

        result = torch.ops.torchscience.beta_survival(x, a_t, b_t)
        expected = torch.tensor(
            scipy.stats.beta.sf(x.numpy(), a=a, b=b),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cdf_sf_sum_to_one(self):
        """CDF(x) + SF(x) = 1."""
        x = torch.linspace(0.01, 0.99, 100)
        a = torch.tensor(2.0)
        b = torch.tensor(5.0)

        cdf = torch.ops.torchscience.beta_cumulative_distribution(x, a, b)
        sf = torch.ops.torchscience.beta_survival(x, a, b)

        assert torch.allclose(cdf + sf, torch.ones_like(cdf), atol=1e-6)


class TestBetaSurvivalGradients:
    """Test beta_survival gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.beta_survival(x_, a, b)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_a(self):
        """Gradient check for a parameter."""
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float64)

        def fn(a_):
            return torch.ops.torchscience.beta_survival(x, a_, b)

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    def test_gradcheck_b(self):
        """Gradient check for b parameter."""
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def fn(b_):
            return torch.ops.torchscience.beta_survival(x, a, b_)

        assert torch.autograd.gradcheck(fn, (b,), eps=1e-6, atol=1e-4)

    def test_grad_x_is_negative_pdf(self):
        """dSF/dx = -PDF."""
        x = torch.linspace(
            0.1, 0.9, 50, dtype=torch.float64, requires_grad=True
        )
        a = torch.tensor(2.0, dtype=torch.float64)
        b = torch.tensor(3.0, dtype=torch.float64)

        sf = torch.ops.torchscience.beta_survival(x, a, b)
        grad_x = torch.autograd.grad(sf.sum(), x)[0]

        # dSF/dx should be -PDF
        pdf_expected = torch.tensor(
            scipy.stats.beta.pdf(x.detach().numpy(), a=2, b=3),
            dtype=torch.float64,
        )
        assert torch.allclose(grad_x, -pdf_expected, atol=1e-5)
