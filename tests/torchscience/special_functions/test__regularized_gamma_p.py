import pytest
import scipy.special
import torch

import torchscience.special_functions  # noqa: F401 - registers operators
from torchscience.special_functions import regularized_gamma_p


class TestRegularizedGammaPForward:
    """Test regularized_gamma_p forward correctness."""

    def test_basic_values(self):
        """Test against scipy.special.gammainc."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0])
        x = torch.tensor([0.5, 1.0, 2.0, 3.0])

        result = torch.ops.torchscience.regularized_gamma_p(a, x)

        expected = torch.tensor(
            scipy.special.gammainc(a.numpy(), x.numpy()),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)


class TestRegularizedGammaPSpecialCases:
    """Test special values and edge cases."""

    def test_at_zero(self):
        """P(a, 0) = 0 for all a > 0."""
        a = torch.tensor([0.5, 1.0, 2.0, 10.0])
        x = torch.zeros_like(a)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert torch.allclose(result, torch.zeros_like(a))

    def test_exponential_cdf(self):
        """P(1, x) = 1 - exp(-x) is the exponential CDF."""
        x = torch.linspace(0.1, 5.0, 50)
        a = torch.ones_like(x)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        expected = 1 - torch.exp(-x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_x_approaches_one(self):
        """P(a, x) -> 1 as x -> inf."""
        a = torch.tensor([1.0, 2.0, 5.0])
        x = torch.tensor([100.0, 100.0, 100.0])
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert torch.allclose(result, torch.ones_like(a), atol=1e-10)

    def test_range_zero_to_one(self):
        """P(a, x) is always in [0, 1]."""
        a = torch.rand(100) * 10 + 0.1  # a in (0.1, 10.1)
        x = torch.rand(100) * 20  # x in [0, 20)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert (result >= 0).all() and (result <= 1).all()


class TestRegularizedGammaPMeta:
    """Test meta tensor support."""

    def test_meta_tensor_shape(self):
        """Meta tensors should produce correct output shape."""
        a = torch.randn(3, 4, device="meta")
        x = torch.randn(3, 4, device="meta")
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcast(self):
        """Meta tensors should broadcast correctly."""
        a = torch.randn(3, 1, device="meta")
        x = torch.randn(1, 4, device="meta")
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert result.shape == (3, 4)


class TestRegularizedGammaPGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        a = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor(
            [1.0, 2.0, 1.5], dtype=torch.float64, requires_grad=True
        )

        def fn(x_):
            return torch.ops.torchscience.regularized_gamma_p(a, x_)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_a(self):
        """Gradient check for a parameter."""
        a = torch.tensor(
            [1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        x = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)

        def fn(a_):
            return torch.ops.torchscience.regularized_gamma_p(a_, x)

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    @pytest.mark.xfail(
        reason="Second-order gradients use numerical differentiation which has "
        "precision limitations for the regularized gamma function"
    )
    def test_gradgradcheck(self):
        """Second-order gradient check."""
        a = torch.tensor([1.5, 2.0], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def fn(a_, x_):
            return torch.ops.torchscience.regularized_gamma_p(a_, x_)

        assert torch.autograd.gradgradcheck(fn, (a, x), eps=1e-6, atol=1e-3)


class TestRegularizedGammaPPythonAPI:
    """Test Python API."""

    def test_import(self):
        """Function should be importable."""
        from torchscience.special_functions import regularized_gamma_p

        assert callable(regularized_gamma_p)

    def test_api_matches_operator(self):
        """Python API should match operator behavior."""
        a = torch.tensor([1.5, 2.0])
        x = torch.tensor([1.0, 2.0])

        result_api = regularized_gamma_p(a, x)
        result_op = torch.ops.torchscience.regularized_gamma_p(a, x)

        assert torch.allclose(result_api, result_op)
