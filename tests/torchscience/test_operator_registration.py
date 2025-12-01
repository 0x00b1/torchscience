"""Test that operator registration via X-macros works correctly."""

import torch

import torchscience.special_functions as sf


def test_all_special_functions_registered():
    """Verify all special functions from the X-macro are accessible."""
    x = torch.randn(10)

    # Unary
    sf.gamma(x.abs() + 0.1)

    # Binary
    sf.chebyshev_polynomial_t(torch.tensor([2.0]), x)

    # Ternary
    sf.incomplete_beta(x.sigmoid(), torch.tensor([2.0]), torch.tensor([3.0]))

    # Quaternary
    sf.hypergeometric_2_f_1(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        x.tanh() * 0.9,
    )


def test_gradients_work():
    """Verify autograd integration via X-macro registration."""
    x = torch.randn(10, requires_grad=True)
    y = sf.gamma(x.abs() + 1.0)
    y.sum().backward()
    assert x.grad is not None


def test_second_order_gradients_work():
    """Verify second-order gradients work via X-macro registration."""
    x = torch.randn(5, requires_grad=True)

    def func(x):
        return sf.gamma(x.abs() + 1.0).sum()

    # First order gradient
    grad = torch.autograd.grad(func(x), x, create_graph=True)[0]

    # Second order gradient
    grad2 = torch.autograd.grad(grad.sum(), x)[0]
    assert grad2 is not None


def test_meta_tensors_work():
    """Verify meta tensor shape inference via X-macro registration."""
    x = torch.randn(10, 20, device="meta")

    # Unary
    result = sf.gamma(x)
    assert result.shape == (10, 20)
    assert result.device.type == "meta"

    # Binary
    a = torch.randn(10, 20, device="meta")
    b = torch.randn(10, 20, device="meta")
    result = sf.chebyshev_polynomial_t(a, b)
    assert result.shape == (10, 20)


def test_broadcasting_works():
    """Verify broadcasting works across all arities."""
    # Binary broadcasting
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([0.5])
    result = sf.chebyshev_polynomial_t(a, b)
    assert result.shape == (3,)

    # Ternary broadcasting
    z = torch.tensor([[0.3], [0.5]])  # (2, 1)
    a = torch.tensor([2.0, 3.0])  # (2,)
    b = torch.tensor([1.0])  # (1,)
    result = sf.incomplete_beta(z, a, b)
    assert result.shape == (2, 2)
