"""Tests for vector quantization."""

import pytest
import torch

from torchscience.compression import vector_quantize


class TestVectorQuantizeBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns quantized tensor, indices, and loss."""
        x = torch.randn(4, 8)
        codebook = torch.randn(16, 8)
        q, idx, loss = vector_quantize(x, codebook)
        assert isinstance(q, torch.Tensor)
        assert isinstance(idx, torch.Tensor)
        assert isinstance(loss, torch.Tensor)
        assert idx.dtype == torch.long
        assert loss.dim() == 0  # Scalar

    def test_output_shapes(self):
        """Output shapes are correct."""
        x = torch.randn(3, 4, 8)  # Batch of 3x4 vectors
        codebook = torch.randn(16, 8)
        q, idx, loss = vector_quantize(x, codebook)
        assert q.shape == x.shape
        assert idx.shape == x.shape[:-1]  # (3, 4)

    def test_indices_in_range(self):
        """Indices are within codebook size."""
        x = torch.randn(100, 8)
        K = 32
        codebook = torch.randn(K, 8)
        _, idx, _ = vector_quantize(x, codebook)
        assert idx.min() >= 0
        assert idx.max() < K


class TestVectorQuantizeCorrectness:
    """Correctness tests."""

    def test_quantized_is_codebook_entry(self):
        """Quantized vectors are codebook entries."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16, 8)
        q, idx, _ = vector_quantize(x, codebook)
        # Each quantized vector should equal its codebook entry
        for i in range(len(x)):
            assert torch.allclose(q[i], codebook[idx[i]])

    def test_nearest_neighbor(self):
        """Finds nearest codebook entry."""
        # Simple case: codebook is identity-like
        D = 4
        codebook = torch.eye(D)
        x = torch.tensor([[0.9, 0.1, 0.0, 0.0]])  # Closest to [1,0,0,0]
        q, idx, _ = vector_quantize(x, codebook)
        assert idx.item() == 0
        assert torch.allclose(q[0], codebook[0])

    def test_commitment_loss_positive(self):
        """Commitment loss is non-negative."""
        x = torch.randn(20, 8)
        codebook = torch.randn(16, 8)
        _, _, loss = vector_quantize(x, codebook, beta=0.25)
        assert loss >= 0

    def test_commitment_loss_scales_with_beta(self):
        """Higher beta gives higher commitment loss."""
        x = torch.randn(20, 8)
        codebook = torch.randn(16, 8)
        _, _, loss_low = vector_quantize(x, codebook, beta=0.1)
        _, _, loss_high = vector_quantize(x, codebook, beta=1.0)
        # With same data, higher beta means higher loss
        assert loss_high > loss_low


class TestVectorQuantizeGradients:
    """Tests for gradient computation."""

    def test_ste_gradient_to_input(self):
        """STE mode provides gradients to input."""
        x = torch.randn(10, 8, requires_grad=True)
        codebook = torch.randn(16, 8)
        q, _, loss = vector_quantize(x, codebook, gradient_mode="ste")
        total = q.sum() + loss
        total.backward()
        assert x.grad is not None

    def test_ste_gradient_to_codebook(self):
        """STE mode provides gradients to codebook."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16, 8, requires_grad=True)
        q, _, loss = vector_quantize(x, codebook, gradient_mode="ste")
        q.sum().backward()
        assert codebook.grad is not None

    def test_gumbel_gradient_to_input(self):
        """Gumbel mode provides gradients to input."""
        x = torch.randn(10, 8, requires_grad=True)
        codebook = torch.randn(16, 8)
        q, _, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", temperature=1.0
        )
        q.sum().backward()
        assert x.grad is not None

    def test_gumbel_gradient_to_codebook(self):
        """Gumbel mode provides gradients to codebook."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16, 8, requires_grad=True)
        q, _, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", temperature=1.0
        )
        q.sum().backward()
        assert codebook.grad is not None


class TestVectorQuantizeGumbel:
    """Tests for Gumbel-Softmax mode."""

    def test_gumbel_hard_mode(self):
        """Hard Gumbel mode produces discrete indices."""
        x = torch.randn(20, 8)
        codebook = torch.randn(16, 8)
        q, idx, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", hard=True, temperature=0.1
        )
        # With hard=True, quantized should be exactly codebook entries
        for i in range(len(x)):
            assert torch.allclose(q[i], codebook[idx[i]], atol=1e-5)

    def test_gumbel_soft_mode(self):
        """Soft Gumbel mode produces weighted combinations."""
        x = torch.randn(20, 8)
        codebook = torch.randn(16, 8)
        q, idx, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", hard=False, temperature=10.0
        )
        # With high temperature and soft mode, output is a blend
        # Check that output is not exactly a codebook entry (usually)
        # This is a probabilistic test
        exact_matches = sum(
            any(
                torch.allclose(q[i], codebook[j]) for j in range(len(codebook))
            )
            for i in range(len(x))
        )
        # Not all should be exact matches with high temperature
        assert exact_matches < len(x)

    def test_temperature_affects_softness(self):
        """Lower temperature gives harder assignments."""
        torch.manual_seed(42)
        x = torch.randn(20, 8)
        codebook = torch.randn(16, 8)

        # Low temperature - should produce outputs closer to hard quantization
        q_low, _, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", hard=False, temperature=0.001
        )

        # High temperature - should produce smoother/blended outputs
        q_high, _, _ = vector_quantize(
            x, codebook, gradient_mode="gumbel", hard=False, temperature=100.0
        )

        # Compare to STE (hard quantization)
        q_ste, _, _ = vector_quantize(x, codebook, gradient_mode="ste")

        # Low temperature should be closer to STE than high temperature
        diff_low = (q_low - q_ste).norm()
        diff_high = (q_high - q_ste).norm()
        assert diff_low < diff_high


class TestVectorQuantizeEdgeCases:
    """Edge case tests."""

    def test_single_vector(self):
        """Works with single vector."""
        x = torch.randn(8)
        codebook = torch.randn(16, 8)
        q, idx, loss = vector_quantize(x.unsqueeze(0), codebook)
        assert q.shape == (1, 8)
        assert idx.shape == (1,)

    def test_single_codebook_entry(self):
        """Works with single codebook entry."""
        x = torch.randn(10, 8)
        codebook = torch.randn(1, 8)
        q, idx, _ = vector_quantize(x, codebook)
        # All should map to entry 0
        assert (idx == 0).all()
        assert torch.allclose(q, codebook[0].expand_as(q))

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        codebook = torch.randn(16, 8)
        with pytest.raises(TypeError, match="must be a Tensor"):
            vector_quantize([[0.1] * 8], codebook)

    def test_dimension_mismatch_raises(self):
        """Raises error for dimension mismatch."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16, 4)  # Wrong dimension
        with pytest.raises(ValueError, match="dimension"):
            vector_quantize(x, codebook)

    def test_codebook_not_2d_raises(self):
        """Raises error for non-2D codebook."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16)  # 1D
        with pytest.raises(ValueError, match="2D"):
            vector_quantize(x, codebook)

    def test_invalid_gradient_mode_raises(self):
        """Raises error for invalid gradient mode."""
        x = torch.randn(10, 8)
        codebook = torch.randn(16, 8)
        with pytest.raises(ValueError, match="gradient_mode"):
            vector_quantize(x, codebook, gradient_mode="invalid")


class TestVectorQuantizeDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(10, 8, device="cpu")
        codebook = torch.randn(16, 8, device="cpu")
        q, idx, loss = vector_quantize(x, codebook)
        assert q.device.type == "cpu"
        assert idx.device.type == "cpu"
        assert loss.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(10, 8, device="cuda")
        codebook = torch.randn(16, 8, device="cuda")
        q, idx, loss = vector_quantize(x, codebook)
        assert q.device.type == "cuda"
        assert idx.device.type == "cuda"
        assert loss.device.type == "cuda"
