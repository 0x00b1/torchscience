import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401 - Load C++ extension


class TestBinomialCdfForward:
    """Test binomial_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.binom.cdf."""
        k = torch.arange(0, 11, dtype=torch.float32)
        n = torch.tensor(10.0)
        p = torch.tensor(0.3)

        result = torch.ops.torchscience.binomial_cumulative_distribution(
            k, n, p
        )
        expected = torch.tensor(
            scipy.stats.binom.cdf(k.numpy(), n=10, p=0.3),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_boundaries(self):
        """CDF(-1) = 0, CDF(n) = 1."""
        n = torch.tensor(10.0)
        p = torch.tensor(0.5)

        assert torch.allclose(
            torch.ops.torchscience.binomial_cumulative_distribution(
                torch.tensor([-1.0]), n, p
            ),
            torch.tensor([0.0]),
            atol=1e-6,
        )
        assert torch.allclose(
            torch.ops.torchscience.binomial_cumulative_distribution(
                torch.tensor([10.0]), n, p
            ),
            torch.tensor([1.0]),
            atol=1e-6,
        )

    def test_bernoulli_case(self):
        """Binomial(1, p) = Bernoulli(p): CDF(0) = 1-p, CDF(1) = 1."""
        n = torch.tensor(1.0)
        p = torch.tensor(0.3)

        cdf_0 = torch.ops.torchscience.binomial_cumulative_distribution(
            torch.tensor([0.0]), n, p
        )
        cdf_1 = torch.ops.torchscience.binomial_cumulative_distribution(
            torch.tensor([1.0]), n, p
        )

        assert torch.allclose(cdf_0, torch.tensor([0.7]), atol=1e-5)
        assert torch.allclose(cdf_1, torch.tensor([1.0]), atol=1e-5)

    @pytest.mark.parametrize(
        "n,p", [(5, 0.2), (10, 0.5), (20, 0.7), (50, 0.3)]
    )
    def test_various_parameters(self, n, p):
        """Test various parameter combinations."""
        k = torch.arange(0, n + 1, dtype=torch.float32)
        n_t = torch.tensor(float(n))
        p_t = torch.tensor(p)

        result = torch.ops.torchscience.binomial_cumulative_distribution(
            k, n_t, p_t
        )
        expected = torch.tensor(
            scipy.stats.binom.cdf(k.numpy(), n=n, p=p),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestBinomialCdfGradients:
    """Test gradient computation for p."""

    def test_gradcheck_p(self):
        """Gradient check for p parameter."""
        k = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64)
        n = torch.tensor(10.0, dtype=torch.float64)
        p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        def fn(p_):
            return torch.ops.torchscience.binomial_cumulative_distribution(
                k, n, p_
            )

        assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-4)

    def test_gradient_sign(self):
        """dCDF/dp should be negative (higher p -> lower CDF for fixed k)."""
        k = torch.tensor([3.0], dtype=torch.float64)
        n = torch.tensor(10.0, dtype=torch.float64)
        p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        cdf = torch.ops.torchscience.binomial_cumulative_distribution(k, n, p)
        grad_p = torch.autograd.grad(cdf.sum(), p)[0]

        # For k < n*p (expectation), increasing p decreases CDF
        assert grad_p < 0


class TestBinomialProbabilityMassForward:
    """Test binomial_probability_mass forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.binom.pmf."""
        k = torch.arange(0, 11, dtype=torch.float32)
        n = torch.tensor(10.0)
        p = torch.tensor(0.3)

        result = torch.ops.torchscience.binomial_probability_mass(k, n, p)
        expected = torch.tensor(
            scipy.stats.binom.pmf(k.numpy(), n=10, p=0.3),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sum_to_one(self):
        """PMF should sum to 1."""
        k = torch.arange(0, 11, dtype=torch.float32)
        n = torch.tensor(10.0)
        p = torch.tensor(0.5)

        pmf = torch.ops.torchscience.binomial_probability_mass(k, n, p)
        assert torch.allclose(pmf.sum(), torch.tensor(1.0), atol=1e-5)

    def test_mode(self):
        """Mode should be near n*p."""
        k = torch.arange(0, 21, dtype=torch.float32)
        n = torch.tensor(20.0)
        p = torch.tensor(0.4)

        pmf = torch.ops.torchscience.binomial_probability_mass(k, n, p)
        mode = k[pmf.argmax()]

        # Mode should be floor(n*p+p) or floor(n*p+p)-1
        expected_mode = torch.floor(n * p + p)
        assert torch.abs(mode - expected_mode) <= 1

    def test_negative_k(self):
        """PMF should be 0 for negative k."""
        k = torch.tensor([-1.0, -2.0, -5.0])
        n = torch.tensor(10.0)
        p = torch.tensor(0.5)

        pmf = torch.ops.torchscience.binomial_probability_mass(k, n, p)
        assert torch.allclose(pmf, torch.zeros_like(pmf))


class TestBinomialProbabilityMassGradients:
    """Test gradient computation for p."""

    def test_gradcheck_p(self):
        """Gradient check for p parameter."""
        k = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64)
        n = torch.tensor(10.0, dtype=torch.float64)
        p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        def fn(p_):
            return torch.ops.torchscience.binomial_probability_mass(k, n, p_)

        assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-4)
