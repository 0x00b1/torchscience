import pytest
import scipy.stats
import torch

import torchscience._csrc  # noqa: F401 - Load C++ extension


class TestPoissonCdfForward:
    """Test poisson_cumulative_distribution forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.poisson.cdf."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(5.0)

        result = torch.ops.torchscience.poisson_cumulative_distribution(
            k, rate
        )
        expected = torch.tensor(
            scipy.stats.poisson.cdf(k.numpy(), mu=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_boundary(self):
        """CDF(-1) = 0."""
        rate = torch.tensor(5.0)
        result = torch.ops.torchscience.poisson_cumulative_distribution(
            torch.tensor([-1.0]), rate
        )
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_at_zero(self):
        """CDF(0) = exp(-lambda)."""
        k = torch.tensor([0.0])
        rate = torch.tensor(3.0)

        result = torch.ops.torchscience.poisson_cumulative_distribution(
            k, rate
        )
        expected = torch.exp(-rate)

        assert torch.allclose(result, expected.unsqueeze(0), atol=1e-5)

    @pytest.mark.parametrize("rate", [0.5, 1.0, 5.0, 10.0, 20.0])
    def test_various_rates(self, rate):
        """Test various rate values."""
        k = torch.arange(0, 30, dtype=torch.float32)
        rate_t = torch.tensor(rate)

        result = torch.ops.torchscience.poisson_cumulative_distribution(
            k, rate_t
        )
        expected = torch.tensor(
            scipy.stats.poisson.cdf(k.numpy(), mu=rate),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestPoissonCdfGradients:
    """Test gradient computation for rate."""

    def test_gradcheck_rate(self):
        """Gradient check for rate parameter."""
        k = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(rate_):
            return torch.ops.torchscience.poisson_cumulative_distribution(
                k, rate_
            )

        assert torch.autograd.gradcheck(fn, (rate,), eps=1e-6, atol=1e-4)

    def test_gradient_sign(self):
        """dCDF/drate should be negative (higher rate -> more probability mass above k)."""
        k = torch.tensor([3.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        cdf = torch.ops.torchscience.poisson_cumulative_distribution(k, rate)
        grad_rate = torch.autograd.grad(cdf.sum(), rate)[0]

        # For k < rate (expectation), increasing rate decreases CDF
        assert grad_rate < 0


class TestPoissonProbabilityMassForward:
    """Test poisson_probability_mass forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.poisson.pmf."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(5.0)

        result = torch.ops.torchscience.poisson_probability_mass(k, rate)
        expected = torch.tensor(
            scipy.stats.poisson.pmf(k.numpy(), mu=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sum_to_one(self):
        """PMF should sum to approximately 1 (truncated)."""
        k = torch.arange(0, 50, dtype=torch.float32)
        rate = torch.tensor(10.0)

        pmf = torch.ops.torchscience.poisson_probability_mass(k, rate)
        # For rate=10, 50 terms should capture almost all probability
        assert torch.allclose(pmf.sum(), torch.tensor(1.0), atol=1e-5)

    def test_mode(self):
        """Mode should be floor(lambda)."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(7.3)

        pmf = torch.ops.torchscience.poisson_probability_mass(k, rate)
        mode = k[pmf.argmax()]

        assert mode == 7.0  # floor(7.3)

    def test_negative_k(self):
        """PMF should be 0 for negative k."""
        k = torch.tensor([-1.0, -2.0, -5.0])
        rate = torch.tensor(5.0)

        pmf = torch.ops.torchscience.poisson_probability_mass(k, rate)
        assert torch.allclose(pmf, torch.zeros_like(pmf))


class TestPoissonProbabilityMassGradients:
    """Test gradient computation for rate."""

    def test_gradcheck_rate(self):
        """Gradient check for rate parameter."""
        k = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(rate_):
            return torch.ops.torchscience.poisson_probability_mass(k, rate_)

        assert torch.autograd.gradcheck(fn, (rate,), eps=1e-6, atol=1e-4)


class TestPoissonLogProbabilityMassForward:
    """Test poisson_log_probability_mass forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.poisson.logpmf."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(5.0)

        result = torch.ops.torchscience.poisson_log_probability_mass(k, rate)
        expected = torch.tensor(
            scipy.stats.poisson.logpmf(k.numpy(), mu=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_consistency_with_pmf(self):
        """log(PMF) should equal log_probability_mass."""
        k = torch.arange(0, 15, dtype=torch.float32)
        rate = torch.tensor(7.0)

        pmf = torch.ops.torchscience.poisson_probability_mass(k, rate)
        log_pmf = torch.ops.torchscience.poisson_log_probability_mass(k, rate)

        assert torch.allclose(log_pmf, torch.log(pmf), atol=1e-5)

    def test_negative_k(self):
        """log_pmf should be -inf for negative k."""
        k = torch.tensor([-1.0, -2.0, -5.0])
        rate = torch.tensor(5.0)

        log_pmf = torch.ops.torchscience.poisson_log_probability_mass(k, rate)
        assert torch.all(torch.isinf(log_pmf) & (log_pmf < 0))

    def test_zero_rate_k_zero(self):
        """log_pmf(0, 0) = 0 (since P(X=0) = 1)."""
        k = torch.tensor([0.0])
        rate = torch.tensor(0.0)

        log_pmf = torch.ops.torchscience.poisson_log_probability_mass(k, rate)
        assert torch.allclose(log_pmf, torch.tensor([0.0]), atol=1e-6)

    def test_zero_rate_k_positive(self):
        """log_pmf(k>0, 0) = -inf."""
        k = torch.tensor([1.0, 2.0, 5.0])
        rate = torch.tensor(0.0)

        log_pmf = torch.ops.torchscience.poisson_log_probability_mass(k, rate)
        assert torch.all(torch.isinf(log_pmf) & (log_pmf < 0))

    @pytest.mark.parametrize("rate", [0.5, 1.0, 5.0, 10.0, 20.0])
    def test_various_rates(self, rate):
        """Test various rate values."""
        k = torch.arange(0, 30, dtype=torch.float32)
        rate_t = torch.tensor(rate)

        result = torch.ops.torchscience.poisson_log_probability_mass(k, rate_t)
        expected = torch.tensor(
            scipy.stats.poisson.logpmf(k.numpy(), mu=rate),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestPoissonLogProbabilityMassGradients:
    """Test gradient computation for rate."""

    def test_gradcheck_rate(self):
        """Gradient check for rate parameter."""
        k = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(rate_):
            return torch.ops.torchscience.poisson_log_probability_mass(
                k, rate_
            )

        assert torch.autograd.gradcheck(fn, (rate,), eps=1e-6, atol=1e-4)

    def test_gradient_manual(self):
        """Test gradient d(log_pmf)/drate = k/rate - 1."""
        k = torch.tensor([3.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        log_pmf = torch.ops.torchscience.poisson_log_probability_mass(k, rate)
        grad_rate = torch.autograd.grad(log_pmf.sum(), rate)[0]

        # d(log_pmf)/drate = k/rate - 1 = 3/5 - 1 = -0.4
        expected_grad = k / rate - 1
        assert torch.allclose(grad_rate, expected_grad.sum(), atol=1e-6)


class TestPoissonSurvivalForward:
    """Test poisson_survival forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.poisson.sf."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(5.0)

        result = torch.ops.torchscience.poisson_survival(k, rate)
        expected = torch.tensor(
            scipy.stats.poisson.sf(k.numpy(), mu=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cdf_sf_complement(self):
        """CDF + SF should equal 1."""
        k = torch.arange(0, 20, dtype=torch.float32)
        rate = torch.tensor(7.0)

        cdf = torch.ops.torchscience.poisson_cumulative_distribution(k, rate)
        sf = torch.ops.torchscience.poisson_survival(k, rate)

        assert torch.allclose(cdf + sf, torch.ones_like(cdf), atol=1e-5)

    def test_boundary_negative_k(self):
        """SF(-1) = 1 (all probability mass exceeds k)."""
        rate = torch.tensor(5.0)
        result = torch.ops.torchscience.poisson_survival(
            torch.tensor([-1.0]), rate
        )
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-6)

    def test_at_zero(self):
        """SF(0) = 1 - exp(-lambda)."""
        k = torch.tensor([0.0])
        rate = torch.tensor(3.0)

        result = torch.ops.torchscience.poisson_survival(k, rate)
        expected = 1.0 - torch.exp(-rate)

        assert torch.allclose(result, expected.unsqueeze(0), atol=1e-5)

    @pytest.mark.parametrize("rate", [0.5, 1.0, 5.0, 10.0, 20.0])
    def test_various_rates(self, rate):
        """Test various rate values."""
        k = torch.arange(0, 30, dtype=torch.float32)
        rate_t = torch.tensor(rate)

        result = torch.ops.torchscience.poisson_survival(k, rate_t)
        expected = torch.tensor(
            scipy.stats.poisson.sf(k.numpy(), mu=rate),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestPoissonSurvivalGradients:
    """Test gradient computation for rate."""

    def test_gradcheck_rate(self):
        """Gradient check for rate parameter."""
        k = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(rate_):
            return torch.ops.torchscience.poisson_survival(k, rate_)

        assert torch.autograd.gradcheck(fn, (rate,), eps=1e-6, atol=1e-4)

    def test_gradient_sign(self):
        """dSF/drate should be positive (higher rate -> more probability mass above k)."""
        k = torch.tensor([3.0], dtype=torch.float64)
        rate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        sf = torch.ops.torchscience.poisson_survival(k, rate)
        grad_rate = torch.autograd.grad(sf.sum(), rate)[0]

        # For k < rate (expectation), increasing rate increases SF
        assert grad_rate > 0
