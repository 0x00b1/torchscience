# tests/torchscience/test__reduction_xmacro.py
"""Integration tests for reduction X-macro infrastructure."""

import torch

import torchscience


class TestReductionXMacro:
    """Test that reduction operators registered via X-macro work correctly."""

    def test_kurtosis_forward(self):
        """Test kurtosis forward pass."""
        x = torch.randn(100, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.shape == ()
        assert result.dtype == torch.float64

    def test_kurtosis_backward(self):
        """Test kurtosis backward pass."""
        x = torch.randn(100, dtype=torch.float64, requires_grad=True)
        result = torchscience.statistics.descriptive.kurtosis(x)
        result.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_kurtosis_dim_reduction(self):
        """Test kurtosis with dimension specification."""
        x = torch.randn(10, 20, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1)
        assert result.shape == (10,)

    def test_kurtosis_keepdim(self):
        """Test kurtosis with keepdim=True."""
        x = torch.randn(10, 20, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(
            x, dim=1, keepdim=True
        )
        assert result.shape == (10, 1)

    def test_kurtosis_extra_args(self):
        """Test kurtosis with fisher and bias arguments."""
        x = torch.randn(100, dtype=torch.float64)

        # Fisher kurtosis (excess kurtosis)
        fisher_result = torchscience.statistics.descriptive.kurtosis(
            x, fisher=True
        )

        # Pearson kurtosis
        pearson_result = torchscience.statistics.descriptive.kurtosis(
            x, fisher=False
        )

        # Fisher should be Pearson - 3
        assert torch.allclose(fisher_result, pearson_result - 3, atol=1e-6)
