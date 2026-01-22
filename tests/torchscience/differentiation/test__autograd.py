"""Tests for autograd support of differentiation operators."""

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import gradient


class TestGradientAutograd:
    """Tests for gradient autograd support."""

    def test_gradient_gradcheck(self):
        """Gradient passes torch.autograd.gradcheck."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def grad_fn(f):
            return gradient(f, dx=0.1, accuracy=2, boundary="replicate")

        assert gradcheck(grad_fn, (field,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_gradgradcheck(self):
        """Gradient passes torch.autograd.gradgradcheck."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def grad_fn(f):
            g = gradient(f, dx=0.1, accuracy=2, boundary="replicate")
            return g.sum()

        assert gradgradcheck(grad_fn, (field,), eps=1e-6, atol=1e-4, rtol=1e-3)
