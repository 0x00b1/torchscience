"""Tests for scalar quantization."""

import pytest
import torch

from torchscience.information.compression import scalar_quantize


class TestScalarQuantizeBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns quantized tensor and indices."""
        x = torch.tensor([0.1, 0.5, 0.9])
        q, idx = scalar_quantize(x, levels=4)
        assert isinstance(q, torch.Tensor)
        assert isinstance(idx, torch.Tensor)
        assert idx.dtype == torch.long

    def test_output_shapes(self):
        """Output shapes match input."""
        x = torch.randn(3, 4, 5)
        q, idx = scalar_quantize(x, levels=8)
        assert q.shape == x.shape
        assert idx.shape == x.shape

    def test_indices_in_range(self):
        """Indices are within valid range."""
        x = torch.randn(100)
        levels = 16
        q, idx = scalar_quantize(x, levels=levels)
        assert idx.min() >= 0
        assert idx.max() < levels


class TestScalarQuantizeUniform:
    """Tests for uniform quantization mode."""

    def test_uniform_quantizes_to_levels(self):
        """Uniform mode produces correct number of unique values."""
        x = torch.linspace(0, 1, 100)
        levels = 4
        q, idx = scalar_quantize(x, levels=levels)
        unique_q = q.unique()
        assert len(unique_q) == levels

    def test_uniform_boundaries(self):
        """Values at boundaries quantize correctly."""
        x = torch.tensor([0.0, 0.5, 1.0])
        q, idx = scalar_quantize(x, levels=3)
        # Min should stay at 0, max at 1, middle at 0.5
        assert torch.isclose(q[0], torch.tensor(0.0))
        assert torch.isclose(q[2], torch.tensor(1.0))

    def test_uniform_constant_input(self):
        """Handles constant input gracefully."""
        x = torch.ones(10) * 0.5
        q, idx = scalar_quantize(x, levels=4)
        assert torch.allclose(q, x)
        assert (idx == 0).all()


class TestScalarQuantizeNonuniform:
    """Tests for non-uniform quantization mode."""

    def test_nonuniform_uses_codebook(self):
        """Non-uniform mode uses provided codebook."""
        codebook = torch.tensor([0.0, 0.3, 0.7, 1.0])
        x = torch.tensor([0.1, 0.4, 0.8])
        q, idx = scalar_quantize(x, levels=codebook, mode="nonuniform")
        # 0.1 closest to 0.0, 0.4 closest to 0.3, 0.8 closest to 0.7 or 1.0
        assert q[0] == 0.0
        assert q[1] == 0.3

    def test_nonuniform_indices_match_codebook(self):
        """Indices correctly index the codebook."""
        codebook = torch.tensor([0.0, 0.5, 1.0])
        x = torch.tensor([0.1, 0.4, 0.6, 0.9])
        q, idx = scalar_quantize(x, levels=codebook, mode="nonuniform")
        for i in range(len(x)):
            assert torch.isclose(q[i], codebook[idx[i]])


class TestScalarQuantizeGradients:
    """Tests for gradient computation."""

    def test_ste_gradient_passes_through(self):
        """STE mode passes gradients through."""
        x = torch.randn(10, requires_grad=True)
        q, idx = scalar_quantize(x, levels=4, gradient_mode="ste")
        loss = q.sum()
        loss.backward()
        assert x.grad is not None
        # STE: gradient should be approximately ones (passed through)
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_soft_gradient_exists(self):
        """Soft mode computes gradients."""
        x = torch.randn(10, requires_grad=True)
        q, idx = scalar_quantize(
            x, levels=4, gradient_mode="soft", temperature=0.1
        )
        loss = q.sum()
        loss.backward()
        assert x.grad is not None

    def test_none_gradient_no_grad(self):
        """None mode produces no gradients."""
        x = torch.randn(10, requires_grad=True)
        q, idx = scalar_quantize(x, levels=4, gradient_mode="none")
        # Output should be detached
        assert not q.requires_grad

    def test_soft_temperature_affects_output(self):
        """Lower temperature gives harder assignments."""
        x = torch.tensor([0.25, 0.75])
        codebook = torch.tensor([0.0, 0.5, 1.0])

        # High temperature: softer
        q_high, _ = scalar_quantize(
            x,
            levels=codebook,
            mode="nonuniform",
            gradient_mode="soft",
            temperature=10.0,
        )

        # Low temperature: harder
        q_low, _ = scalar_quantize(
            x,
            levels=codebook,
            mode="nonuniform",
            gradient_mode="soft",
            temperature=0.01,
        )

        # Low temp should be closer to hard quantization
        q_hard, _ = scalar_quantize(
            x, levels=codebook, mode="nonuniform", gradient_mode="ste"
        )

        # q_low should be closer to q_hard than q_high is
        diff_low = (q_low - q_hard).abs().sum()
        diff_high = (q_high - q_hard).abs().sum()
        assert diff_low <= diff_high


class TestScalarQuantizeEdgeCases:
    """Edge case tests."""

    def test_single_element(self):
        """Works with single element tensor."""
        x = torch.tensor([0.5])
        q, idx = scalar_quantize(x, levels=4)
        assert q.shape == x.shape
        assert idx.shape == x.shape

    def test_large_levels(self):
        """Works with many quantization levels."""
        x = torch.randn(100)
        q, idx = scalar_quantize(x, levels=1024)
        assert idx.max() < 1024

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            scalar_quantize([0.1, 0.5, 0.9], levels=4)

    def test_invalid_gradient_mode_raises(self):
        """Raises error for invalid gradient mode."""
        x = torch.tensor([0.5])
        with pytest.raises(ValueError, match="gradient_mode"):
            scalar_quantize(x, levels=4, gradient_mode="invalid")

    def test_nonuniform_without_codebook_raises(self):
        """Raises error for nonuniform mode without codebook."""
        x = torch.tensor([0.5])
        with pytest.raises(ValueError, match="nonuniform"):
            scalar_quantize(x, levels=4, mode="nonuniform")


class TestScalarQuantizeReversibility:
    """Tests for quantization properties."""

    def test_idempotent(self):
        """Quantizing twice gives same result."""
        x = torch.randn(50)
        q1, idx1 = scalar_quantize(x, levels=8)
        q2, idx2 = scalar_quantize(q1, levels=8)
        assert torch.allclose(q1, q2)
        assert torch.equal(idx1, idx2)

    def test_preserves_range(self):
        """Quantized values are within input range."""
        x = torch.randn(100)
        q, _ = scalar_quantize(x, levels=16)
        assert q.min() >= x.min() - 1e-5
        assert q.max() <= x.max() + 1e-5


class TestScalarQuantizeDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(10, device="cpu")
        q, idx = scalar_quantize(x, levels=4)
        assert q.device.type == "cpu"
        assert idx.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(10, device="cuda")
        q, idx = scalar_quantize(x, levels=4)
        assert q.device.type == "cuda"
        assert idx.device.type == "cuda"
