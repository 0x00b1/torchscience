import math

import scipy.stats
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.probability import (
    normal_cumulative_distribution,
    normal_log_probability_density,
    normal_probability_density,
    normal_quantile,
    normal_survival,
)


class TestNormalCdfForward:
    """Test normal_cumulative_distribution forward correctness."""

    def test_standard_normal(self):
        """Test standard normal CDF at key quantiles."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = torch.ops.torchscience.normal_cumulative_distribution(
            x, torch.tensor(0.0), torch.tensor(1.0)
        )
        expected = torch.tensor([0.0228, 0.1587, 0.5000, 0.8413, 0.9772])
        assert torch.allclose(result, expected, atol=1e-4)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.cdf."""
        x = torch.linspace(-4, 4, 100)
        loc = torch.tensor(1.5)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )
        expected = torch.tensor(
            scipy.stats.norm.cdf(x.numpy(), loc=1.5, scale=2.0),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)

    def test_symmetry(self):
        """CDF(mu - x) + CDF(mu + x) = 1."""
        x = torch.linspace(0.1, 3, 50)
        mu = torch.tensor(2.0)
        sigma = torch.tensor(1.5)

        cdf_minus = torch.ops.torchscience.normal_cumulative_distribution(
            mu - x, mu, sigma
        )
        cdf_plus = torch.ops.torchscience.normal_cumulative_distribution(
            mu + x, mu, sigma
        )

        assert torch.allclose(
            cdf_minus + cdf_plus, torch.ones_like(x), atol=1e-6
        )


class TestNormalCdfMeta:
    """Test meta tensor support."""

    def test_meta_tensor_shape(self):
        """Meta tensors should produce correct output shape."""
        x = torch.randn(3, 4, device="meta")
        loc = torch.randn(3, 4, device="meta")
        scale = torch.randn(3, 4, device="meta")
        result = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcast(self):
        """Meta tensors should broadcast correctly."""
        x = torch.randn(3, 4, device="meta")
        loc = torch.randn(1, device="meta")
        scale = torch.randn(1, device="meta")
        result = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )
        assert result.shape == (3, 4)


class TestNormalCdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.normal_cumulative_distribution(
                x_, loc, scale
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_cumulative_distribution(
                x, loc_, scale
            )

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_cumulative_distribution(
                x, loc, scale_
            )

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)

    def test_grad_x_is_probability_density(self):
        """dCDF/dx should equal PDF."""
        x = torch.linspace(-3, 3, 100, dtype=torch.float64, requires_grad=True)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        cdf = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )
        grad_x = torch.autograd.grad(cdf.sum(), x)[0]

        # PDF = exp(-z^2/2) / (sigma * sqrt(2*pi))
        z = (x - loc) / scale
        pdf = torch.exp(-0.5 * z * z) / (scale * math.sqrt(2 * math.pi))

        assert torch.allclose(grad_x, pdf.detach(), atol=1e-6)

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
        loc = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, loc_, scale_):
            return torch.ops.torchscience.normal_cumulative_distribution(
                x_, loc_, scale_
            )

        assert torch.autograd.gradgradcheck(
            fn, (x, loc, scale), eps=1e-6, atol=1e-3
        )


class TestNormalPdfForward:
    """Test normal_probability_density forward correctness."""

    def test_standard_normal_peak(self):
        """PDF at mean should equal 1 / sqrt(2*pi)."""
        x = torch.tensor([0.0])
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        result = torch.ops.torchscience.normal_probability_density(
            x, loc, scale
        )
        expected = torch.tensor([1 / math.sqrt(2 * math.pi)])

        assert torch.allclose(result, expected, atol=1e-6)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.pdf."""
        x = torch.linspace(-4, 4, 100)
        loc = torch.tensor(1.0)
        scale = torch.tensor(0.5)

        result = torch.ops.torchscience.normal_probability_density(
            x, loc, scale
        )
        expected = torch.tensor(
            scipy.stats.norm.pdf(x.numpy(), loc=1.0, scale=0.5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)

    def test_symmetry(self):
        """PDF(mu - x) = PDF(mu + x)."""
        x = torch.linspace(0.1, 3, 50)
        mu = torch.tensor(2.0)
        sigma = torch.tensor(1.5)

        pdf_minus = torch.ops.torchscience.normal_probability_density(
            mu - x, mu, sigma
        )
        pdf_plus = torch.ops.torchscience.normal_probability_density(
            mu + x, mu, sigma
        )

        assert torch.allclose(pdf_minus, pdf_plus, atol=1e-6)


class TestNormalPdfGradients:
    """Test normal_probability_density gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.normal_probability_density(
                x_, loc, scale
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_probability_density(
                x, loc_, scale
            )

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_probability_density(
                x, loc, scale_
            )

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)

    def test_grad_at_mean(self):
        """dPDF/dx at x=mean should be 0."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        pdf = torch.ops.torchscience.normal_probability_density(x, loc, scale)
        grad_x = torch.autograd.grad(pdf.sum(), x)[0]

        assert torch.allclose(grad_x, torch.zeros_like(grad_x), atol=1e-6)


class TestNormalPpfForward:
    """Test normal_quantile (quantile function) forward correctness."""

    def test_standard_quantiles(self):
        """Test standard normal quantiles."""
        p = torch.tensor([0.5, 0.025, 0.975])
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        result = torch.ops.torchscience.normal_quantile(p, loc, scale)
        expected = torch.tensor([0.0, -1.96, 1.96])

        assert torch.allclose(result, expected, atol=0.01)

    def test_cumulative_distribution_quantile_roundtrip(self):
        """ppf(cdf(x)) should equal x."""
        x = torch.linspace(-3, 3, 100)
        loc = torch.tensor(1.0)
        scale = torch.tensor(2.0)

        p = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )
        x_recovered = torch.ops.torchscience.normal_quantile(p, loc, scale)

        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.ppf."""
        p = torch.linspace(0.01, 0.99, 100)
        loc = torch.tensor(-1.0)
        scale = torch.tensor(1.5)

        result = torch.ops.torchscience.normal_quantile(p, loc, scale)
        expected = torch.tensor(
            scipy.stats.norm.ppf(p.numpy(), loc=-1.0, scale=1.5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestNormalPpfGradients:
    """Test normal_quantile gradient computation."""

    def test_gradcheck_p(self):
        """Gradient check for p parameter."""
        p = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(p_):
            return torch.ops.torchscience.normal_quantile(p_, loc, scale)

        assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        p = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_quantile(p, loc_, scale)

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        p = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_quantile(p, loc, scale_)

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)


class TestNormalSfForward:
    """Test normal_survival (survival function) forward correctness."""

    def test_survival_plus_cumulative_distribution_equals_one(self):
        """SF(x) + CDF(x) = 1."""
        x = torch.linspace(-3, 3, 100)
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        sf = torch.ops.torchscience.normal_survival(x, loc, scale)
        cdf = torch.ops.torchscience.normal_cumulative_distribution(
            x, loc, scale
        )

        assert torch.allclose(sf + cdf, torch.ones_like(sf), atol=1e-6)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.sf."""
        x = torch.linspace(-4, 4, 100)
        loc = torch.tensor(1.0)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.normal_survival(x, loc, scale)
        expected = torch.tensor(
            scipy.stats.norm.sf(x.numpy(), loc=1.0, scale=2.0),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_x_stability(self):
        """SF should be more accurate than 1-CDF for large x."""
        x = torch.tensor([5.0, 6.0, 7.0])
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        sf = torch.ops.torchscience.normal_survival(x, loc, scale)
        expected = torch.tensor(
            scipy.stats.norm.sf(x.numpy(), loc=0.0, scale=1.0),
            dtype=torch.float32,
        )
        assert torch.allclose(sf, expected, rtol=1e-5)


class TestNormalSfGradients:
    """Test normal_survival gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.normal_survival(x_, loc, scale)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_survival(x, loc_, scale)

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_survival(x, loc, scale_)

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)


class TestNormalLogProbabilityDensityForward:
    """Test normal_log_probability_density forward correctness."""

    def test_log_of_probability_density(self):
        """logpdf(x) = log(pdf(x))."""
        x = torch.linspace(-3, 3, 100)
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        logpdf = torch.ops.torchscience.normal_log_probability_density(
            x, loc, scale
        )
        pdf = torch.ops.torchscience.normal_probability_density(x, loc, scale)

        assert torch.allclose(logpdf, torch.log(pdf), atol=1e-6)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.logpdf."""
        x = torch.linspace(-4, 4, 100)
        loc = torch.tensor(1.0)
        scale = torch.tensor(0.5)

        result = torch.ops.torchscience.normal_log_probability_density(
            x, loc, scale
        )
        expected = torch.tensor(
            scipy.stats.norm.logpdf(x.numpy(), loc=1.0, scale=0.5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_peak_value(self):
        """logpdf at mean should be -log(scale * sqrt(2*pi))."""
        loc = torch.tensor(2.0)
        scale = torch.tensor(1.5)
        x = loc.clone()

        result = torch.ops.torchscience.normal_log_probability_density(
            x, loc, scale
        )
        expected = -torch.log(scale * math.sqrt(2 * math.pi))

        assert torch.allclose(result, expected, atol=1e-6)


class TestNormalLogProbabilityDensityGradients:
    """Test normal_log_probability_density gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.normal_log_probability_density(
                x_, loc, scale
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_log_probability_density(
                x, loc_, scale
            )

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_log_probability_density(
                x, loc, scale_
            )

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)

    def test_grad_at_mean(self):
        """dlogpdf/dx at x=mean should be 0."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        logpdf = torch.ops.torchscience.normal_log_probability_density(
            x, loc, scale
        )
        grad_x = torch.autograd.grad(logpdf.sum(), x)[0]

        assert torch.allclose(grad_x, torch.zeros_like(grad_x), atol=1e-6)


class TestNormalPythonAPI:
    """Test Python API."""

    def test_imports(self):
        """All functions should be importable."""
        assert all(
            callable(f)
            for f in [
                normal_cumulative_distribution,
                normal_probability_density,
                normal_quantile,
                normal_survival,
                normal_log_probability_density,
            ]
        )

    def test_default_parameters(self):
        """Should work with default loc=0, scale=1."""
        x = torch.tensor([0.0, 1.0])
        result = normal_cumulative_distribution(x)
        expected = normal_cumulative_distribution(x, loc=0.0, scale=1.0)
        assert torch.allclose(result, expected)

    def test_float_parameters(self):
        """Should accept float parameters for loc and scale."""
        x = torch.tensor([0.0, 1.0, 2.0])

        # All functions should accept float parameters
        cdf = normal_cumulative_distribution(x, loc=1.0, scale=2.0)
        pdf = normal_probability_density(x, loc=1.0, scale=2.0)
        sf = normal_survival(x, loc=1.0, scale=2.0)
        logpdf = normal_log_probability_density(x, loc=1.0, scale=2.0)

        assert cdf.shape == x.shape
        assert pdf.shape == x.shape
        assert sf.shape == x.shape
        assert logpdf.shape == x.shape

    def test_quantile_float_parameters(self):
        """PPF should accept float parameters."""
        p = torch.tensor([0.1, 0.5, 0.9])
        result = normal_quantile(p, loc=1.0, scale=2.0)
        assert result.shape == p.shape

    def test_consistency_with_ops(self):
        """Python API should produce same results as torch.ops."""
        x = torch.tensor([0.0, 1.0, 2.0])
        loc = torch.tensor(1.0)
        scale = torch.tensor(2.0)

        # Compare Python API with torch.ops
        assert torch.allclose(
            normal_cumulative_distribution(x, loc, scale),
            torch.ops.torchscience.normal_cumulative_distribution(
                x, loc, scale
            ),
        )
        assert torch.allclose(
            normal_probability_density(x, loc, scale),
            torch.ops.torchscience.normal_probability_density(x, loc, scale),
        )
        assert torch.allclose(
            normal_survival(x, loc, scale),
            torch.ops.torchscience.normal_survival(x, loc, scale),
        )
        assert torch.allclose(
            normal_log_probability_density(x, loc, scale),
            torch.ops.torchscience.normal_log_probability_density(
                x, loc, scale
            ),
        )

        p = torch.tensor([0.1, 0.5, 0.9])
        assert torch.allclose(
            normal_quantile(p, loc, scale),
            torch.ops.torchscience.normal_quantile(p, loc, scale),
        )
