import numpy as np
import pytest
import scipy.integrate
import torch


class TestQuad:
    def test_basic_integration(self):
        """Integrate sin(x) from 0 to pi"""
        from torchscience.integration.quadrature import quad

        result = quad(torch.sin, 0, torch.pi)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-6
        )

    def test_matches_scipy(self):
        """Compare with scipy.integrate.quad"""
        from torchscience.integration.quadrature import quad

        result = quad(lambda x: torch.exp(-(x**2)), -2, 2)
        expected, _ = scipy.integrate.quad(lambda x: np.exp(-(x**2)), -2, 2)

        assert torch.allclose(
            result, torch.tensor(expected, dtype=result.dtype), rtol=1e-6
        )

    def test_oscillatory(self):
        """Handle oscillatory integrand"""
        from torchscience.integration.quadrature import quad

        result = quad(lambda x: torch.sin(20 * x), 0, torch.pi, limit=100)
        expected = (1 - torch.cos(torch.tensor(20 * torch.pi))) / 20

        assert torch.allclose(result, expected.to(result.dtype), rtol=1e-4)

    def test_convergence_failure_raises(self):
        """Should raise IntegrationError when convergence fails"""
        from torchscience.integration.quadrature import IntegrationError, quad

        # Very difficult integrand with very tight tolerance and low limit
        with pytest.raises(IntegrationError, match="failed to converge"):
            quad(
                lambda x: torch.sin(1000 * x),
                0,
                torch.pi,
                epsabs=1e-15,
                limit=5,
            )


class TestQuadInfo:
    def test_returns_error_and_info(self):
        """quad_info returns error estimate and info dict"""
        from torchscience.integration.quadrature import quad_info

        result, error, info = quad_info(torch.sin, 0, torch.pi)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-6
        )
        assert error < 1e-6
        assert "neval" in info
        assert "nsubintervals" in info
        assert "converged" in info
        assert info["converged"]

    def test_info_shows_convergence_status(self):
        """Info dict correctly reports non-convergence"""
        from torchscience.integration.quadrature import quad_info

        # Very difficult integrand
        result, error, info = quad_info(
            lambda x: torch.sin(1000 * x),
            0,
            torch.pi,
            epsabs=1e-15,
            limit=5,
        )

        assert not info["converged"]


class TestQuadGradients:
    def test_gradient_closure(self):
        """Gradient through closure parameters (within tolerance)"""
        from torchscience.integration.quadrature import quad

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        result = quad(lambda x: theta * x**2, 0, 1)
        result.backward()

        # Note: Adaptive quad supports gradients through closure params
        # because function evaluations still track the graph
        assert theta.grad is not None
        assert torch.allclose(
            theta.grad, torch.tensor(1 / 3, dtype=torch.float64), rtol=1e-3
        )


class TestQuadTolerance:
    def test_epsabs(self):
        """Test absolute tolerance"""
        from torchscience.integration.quadrature import quad_info

        _, error, _ = quad_info(torch.sin, 0, torch.pi, epsabs=1e-10)

        assert error < 1e-9

    def test_epsrel(self):
        """Test relative tolerance"""
        from torchscience.integration.quadrature import quad_info

        result, error, _ = quad_info(torch.sin, 0, torch.pi, epsrel=1e-8)

        assert error < 1e-7 * abs(result.item())
