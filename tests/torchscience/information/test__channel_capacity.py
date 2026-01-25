"""Tests for channel capacity."""

import math

import pytest
import torch

from torchscience.information import channel_capacity


class TestChannelCapacityBasic:
    """Basic functionality tests."""

    def test_output_shape(self):
        """Returns scalar for 2D input."""
        P = torch.eye(2)
        result = channel_capacity(P)
        assert result.shape == torch.Size([])

    def test_return_distribution(self):
        """Returns tuple when return_distribution=True."""
        P = torch.eye(2)
        result = channel_capacity(P, return_distribution=True)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestChannelCapacityCorrectness:
    """Numerical correctness tests."""

    def test_noiseless_channel(self):
        """Noiseless channel has capacity log(n)."""
        n = 4
        P = torch.eye(n)
        result = channel_capacity(P, base=2.0)
        expected = math.log2(n)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_binary_symmetric_channel(self):
        """BSC with crossover p has capacity 1 - H(p)."""
        p = 0.1
        P = torch.tensor([[1 - p, p], [p, 1 - p]])
        result = channel_capacity(P, base=2.0)
        h_p = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        expected = 1.0 - h_p
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_binary_erasure_channel(self):
        """BEC with erasure probability e has capacity 1 - e."""
        e = 0.3
        P = torch.tensor([[1 - e, 0.0, e], [0.0, 1 - e, e]])
        result = channel_capacity(P, base=2.0)
        expected = 1.0 - e
        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_useless_channel(self):
        """Useless channel (all rows equal) has capacity 0."""
        P = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = channel_capacity(P, base=2.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_optimal_distribution_bsc(self):
        """BSC optimal input is uniform."""
        p = 0.2
        P = torch.tensor([[1 - p, p], [p, 1 - p]])
        _, px = channel_capacity(P, return_distribution=True)
        expected = torch.tensor([0.5, 0.5])
        assert torch.allclose(px, expected, atol=1e-4)


class TestChannelCapacityBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual."""
        P1 = torch.eye(2)
        P2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        P_batch = torch.stack([P1, P2])

        r1 = channel_capacity(P1)
        r2 = channel_capacity(P2)
        r_batch = channel_capacity(P_batch)

        assert torch.isclose(r_batch[0], r1, atol=1e-5)
        assert torch.isclose(r_batch[1], r2, atol=1e-5)


class TestChannelCapacityEdgeCases:
    """Tests for edge cases."""

    def test_single_input(self):
        """Single input has capacity 0."""
        P = torch.tensor([[0.5, 0.5]])
        result = channel_capacity(P, base=2.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_single_output(self):
        """Single output has capacity 0."""
        P = torch.tensor([[1.0], [1.0]])
        result = channel_capacity(P, base=2.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)


class TestChannelCapacityDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        P = torch.eye(2, device="cpu")
        result = channel_capacity(P)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        P = torch.eye(2, device="cuda")
        result = channel_capacity(P)
        assert result.device.type == "cuda"
