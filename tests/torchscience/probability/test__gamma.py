import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401 - Load C++ extensions


class TestGammaCdfForward:
    """Test gamma_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.gamma.cdf with scale."""
        x = torch.linspace(0.1, 20, 100)
        shape = torch.tensor(2.0)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape, scale
        )
        expected = torch.tensor(
            scipy.stats.gamma.cdf(x.numpy(), a=2, scale=2),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_exponential_case(self):
        """Gamma(1, scale) is Exponential(1/scale), CDF = 1 - exp(-x/scale)."""
        x = torch.linspace(0.1, 10, 50)
        shape = torch.tensor(1.0)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape, scale
        )
        expected = 1 - torch.exp(-x / scale)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_at_zero(self):
        """CDF(0) = 0."""
        x = torch.tensor([0.0])
        shape = torch.tensor(2.0)
        scale = torch.tensor(1.0)
        result = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape, scale
        )
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    @pytest.mark.parametrize(
        "shape,scale", [(1, 1), (2, 1), (5, 2), (0.5, 1), (10, 0.5)]
    )
    def test_various_parameters(self, shape, scale):
        """Test various parameter combinations."""
        x = torch.linspace(0.1, 20, 50)
        shape_t = torch.tensor(float(shape))
        scale_t = torch.tensor(float(scale))

        result = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape_t, scale_t
        )
        expected = torch.tensor(
            scipy.stats.gamma.cdf(x.numpy(), a=shape, scale=scale),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestGammaPdfForward:
    """Test gamma_probability_density forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.gamma.pdf."""
        x = torch.linspace(0.1, 20, 100)
        shape = torch.tensor(2.0)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.gamma_probability_density(
            x, shape, scale
        )
        expected = torch.tensor(
            scipy.stats.gamma.pdf(x.numpy(), a=2, scale=2),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestGammaPpfForward:
    """Test gamma_quantile forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.gamma.ppf."""
        p = torch.linspace(0.01, 0.99, 100)
        shape = torch.tensor(2.0)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.gamma_quantile(p, shape, scale)
        expected = torch.tensor(
            scipy.stats.gamma.ppf(p.numpy(), a=2, scale=2),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-4)

    def test_cumulative_distribution_quantile_roundtrip(self):
        """ppf(cdf(x)) = x."""
        x = torch.linspace(0.5, 10, 50)
        shape = torch.tensor(2.0)
        scale = torch.tensor(2.0)

        p = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape, scale
        )
        x_recovered = torch.ops.torchscience.gamma_quantile(p, shape, scale)

        assert torch.allclose(x, x_recovered, atol=1e-4)


class TestGammaCdfGradients:
    """Test gamma_cumulative_distribution gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        shape = torch.tensor(2.0, dtype=torch.float64)
        scale = torch.tensor(2.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.gamma_cumulative_distribution(
                x_, shape, scale
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_shape(self):
        """Gradient check for shape parameter."""
        x = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        shape = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(2.0, dtype=torch.float64)

        def fn(shape_):
            return torch.ops.torchscience.gamma_cumulative_distribution(
                x, shape_, scale
            )

        assert torch.autograd.gradcheck(fn, (shape,), eps=1e-6, atol=1e-3)

    def test_grad_x_is_probability_density(self):
        """dCDF/dx = PDF."""
        x = torch.linspace(
            0.5, 10, 50, dtype=torch.float64, requires_grad=True
        )
        shape = torch.tensor(2.0, dtype=torch.float64)
        scale = torch.tensor(2.0, dtype=torch.float64)

        cdf = torch.ops.torchscience.gamma_cumulative_distribution(
            x, shape, scale
        )
        grad_x = torch.autograd.grad(cdf.sum(), x)[0]

        # Compare to scipy PDF
        pdf_expected = torch.tensor(
            scipy.stats.gamma.pdf(x.detach().numpy(), a=2, scale=2),
            dtype=torch.float64,
        )
        assert torch.allclose(grad_x, pdf_expected, atol=1e-5)
