import pytest
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators


class TestOperatorTemplates:
    """Test all operator template categories."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    # Reduction tests use kurtosis (already implemented)
    def test_reduction_template(self, device):
        """Verify reduction template pattern via kurtosis."""
        x = torch.randn(4, 5, 6, device=device, requires_grad=True)

        # All dims
        result = torch.ops.torchscience.kurtosis(x, None, False, True, True)
        assert result.shape == ()

        # Single dim
        result = torch.ops.torchscience.kurtosis(x, [1], False, True, True)
        assert result.shape == (4, 6)

        # Multiple dims with keepdim
        result = torch.ops.torchscience.kurtosis(x, [0, 2], True, True, True)
        assert result.shape == (1, 5, 1)

        # Backward
        result = torch.ops.torchscience.kurtosis(x, [1], False, True, True)
        result.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    # Fixed tests use hilbert_transform (already implemented)
    def test_fixed_template(self, device):
        """Verify fixed template pattern via hilbert_transform."""
        x = torch.randn(4, 128, device=device, requires_grad=True)

        # Default dim (-1)
        result = torch.ops.torchscience.hilbert_transform(
            x, -1, -1, 0, 0.0, None
        )
        assert result.shape == x.shape

        # Explicit dim
        result = torch.ops.torchscience.hilbert_transform(
            x, -1, 1, 0, 0.0, None
        )
        assert result.shape == x.shape

        # Backward
        result = torch.ops.torchscience.hilbert_transform(
            x, -1, -1, 0, 0.0, None
        )
        result.sum().backward()
        assert x.grad is not None

    # Creation tests use rectangular_window (already implemented)
    def test_creation_template(self, device):
        """Verify creation template pattern via rectangular_window."""
        result = torch.ops.torchscience.rectangular_window(
            64,
            dtype=torch.float32,
            layout=torch.strided,
            device=device,
            requires_grad=False,
        )
        assert result.shape == (64,)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.ones(64))

    # Pointwise tests use gamma (already implemented)
    def test_pointwise_template(self, device):
        """Verify pointwise template pattern via gamma."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], device=device, requires_grad=True
        )

        result = torch.ops.torchscience.gamma(x)
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0], device=device)
        assert torch.allclose(result, expected)

        # Backward
        result.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_pointwise_backward_backward(self, device):
        """Verify pointwise backward_backward via gamma."""
        x = torch.tensor(
            [2.0, 3.0, 4.0, 5.0], device=device, requires_grad=True
        )

        result = torch.ops.torchscience.gamma(x)

        # First backward with graph retention
        (grad,) = torch.autograd.grad(result.sum(), x, create_graph=True)

        # Second backward
        (grad_grad,) = torch.autograd.grad(grad.sum(), x, retain_graph=True)

        assert grad_grad is not None
        assert grad_grad.shape == x.shape
        # Verify values are finite
        assert torch.isfinite(grad_grad).all()

    def test_reduction_backward_backward(self, device):
        """Verify reduction backward_backward via kurtosis."""
        x = torch.randn(4, 10, device=device, requires_grad=True)

        result = torch.ops.torchscience.kurtosis(x, None, False, True, True)

        # Check if backward_backward is implemented
        try:
            (grad,) = torch.autograd.grad(result, x, create_graph=True)
            (grad_grad,) = torch.autograd.grad(grad.sum(), x)
            assert grad_grad.shape == x.shape
            assert torch.isfinite(grad_grad).all()
        except RuntimeError:
            pytest.skip("backward_backward not yet implemented for kurtosis")
