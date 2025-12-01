import pytest
import torch


class TestRegularizedGammaQForward:
    """Tests for regularized_gamma_q forward pass."""

    def test_basic(self):
        """Test basic Q(a, x) = 1 - P(a, x) relationship."""
        a = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([0.5, 1.0, 2.0])
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        p_result = torch.ops.torchscience.regularized_gamma_p(a, x)
        expected = 1.0 - p_result
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_q_at_zero(self):
        """Q(a, 0) = 1 for all a > 0."""
        a = torch.tensor([1.0, 2.0, 5.0, 10.0])
        x = torch.zeros_like(a)
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        expected = torch.ones_like(a)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_exponential_survival(self):
        """Q(1, x) = e^(-x) (exponential survival function)."""
        x = torch.tensor([0.5, 1.0, 2.0, 5.0])
        a = torch.ones_like(x)
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        expected = torch.exp(-x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_range_0_1(self):
        """Q(a, x) should be in [0, 1]."""
        a = torch.tensor([0.5, 1.0, 2.0, 5.0])
        x = torch.tensor([0.1, 1.0, 2.0, 10.0])
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_large_x(self):
        """Q(a, x) approaches 0 as x -> infinity."""
        a = torch.tensor([1.0, 2.0])
        x = torch.tensor([20.0, 30.0])
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        assert torch.all(result < 0.01)

    def test_scipy_comparison(self):
        """Compare against scipy values."""
        scipy_special = pytest.importorskip("scipy.special")
        a = torch.tensor([1.0, 2.0, 3.0, 5.0])
        x = torch.tensor([0.5, 1.0, 2.0, 3.0])
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        expected = torch.tensor(
            [
                scipy_special.gammaincc(1.0, 0.5),
                scipy_special.gammaincc(2.0, 1.0),
                scipy_special.gammaincc(3.0, 2.0),
                scipy_special.gammaincc(5.0, 3.0),
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


class TestRegularizedGammaQGradients:
    """Tests for regularized_gamma_q gradients."""

    def test_gradcheck_x(self):
        """First-order gradient w.r.t. x."""
        a = torch.tensor([1.5, 2.5], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x_: torch.ops.torchscience.regularized_gamma_q(a, x_),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_a(self):
        """First-order gradient w.r.t. a."""
        a = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.autograd.gradcheck(
            lambda a_: torch.ops.torchscience.regularized_gamma_q(a_, x),
            (a,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradient_sign(self):
        """Gradient of Q w.r.t. x should be negative (Q decreases with x)."""
        a = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() < 0  # Q decreases with x

    @pytest.mark.xfail(
        reason="Second-order gradients use numerical differentiation"
    )
    def test_gradgradcheck(self):
        """Second-order gradients."""
        a = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torch.ops.torchscience.regularized_gamma_q,
            (a, x),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-2,
        )


class TestRegularizedGammaQMeta:
    """Tests for regularized_gamma_q with meta tensors."""

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        a = torch.empty(5, 3, device="meta")
        x = torch.empty(5, 3, device="meta")
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        assert result.device.type == "meta"
        assert result.shape == (5, 3)

    def test_meta_broadcast(self):
        """Test meta tensor broadcasting."""
        a = torch.empty(1, 3, device="meta")
        x = torch.empty(5, 1, device="meta")
        result = torch.ops.torchscience.regularized_gamma_q(a, x)
        assert result.shape == (5, 3)


class TestRegularizedGammaQPythonAPI:
    """Tests for the Python wrapper."""

    def test_wrapper_exists(self):
        """Test that the wrapper is exposed."""
        from torchscience.special_functions import regularized_gamma_q

        assert callable(regularized_gamma_q)

    def test_wrapper_call(self):
        """Test calling the wrapper."""
        from torchscience.special_functions import regularized_gamma_q

        a = torch.tensor([1.0, 2.0])
        x = torch.tensor([0.5, 1.0])
        result = regularized_gamma_q(a, x)
        assert result.shape == (2,)
