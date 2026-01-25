"""Tests for lattice quantization."""

import pytest
import torch

from torchscience.information.compression import lattice_quantize


class TestLatticeQuantizeBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns quantized tensor and indices."""
        x = torch.randn(10, 3)
        q, idx = lattice_quantize(x)
        assert isinstance(q, torch.Tensor)
        assert isinstance(idx, torch.Tensor)

    def test_output_shapes(self):
        """Output shapes match input."""
        x = torch.randn(5, 4, 3)
        q, idx = lattice_quantize(x)
        assert q.shape == x.shape
        assert idx.shape == x.shape

    def test_default_is_zn(self):
        """Default lattice is Zn."""
        x = torch.tensor([[1.2, 2.7, -0.3]])
        q, _ = lattice_quantize(x)
        # Zn rounds to integers
        expected = torch.tensor([[1.0, 3.0, 0.0]])
        assert torch.allclose(q, expected)


class TestLatticeZn:
    """Tests for integer lattice Zn."""

    def test_rounds_to_integers(self):
        """Zn rounds each coordinate to nearest integer."""
        x = torch.tensor([[0.4, 0.6, -0.4, -0.6]])
        q, idx = lattice_quantize(x, lattice="Zn")
        expected = torch.tensor([[0.0, 1.0, 0.0, -1.0]])
        assert torch.allclose(q, expected)
        assert torch.equal(idx, expected.long())

    def test_1d_works(self):
        """Zn works with 1D vectors."""
        x = torch.tensor([[1.5]])
        q, _ = lattice_quantize(x, lattice="Zn")
        assert q.shape == (1, 1)

    def test_scale_factor(self):
        """Scale factor affects quantization."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        q_scale1, _ = lattice_quantize(x, lattice="Zn", scale=1.0)
        q_scale2, _ = lattice_quantize(x, lattice="Zn", scale=2.0)

        # scale=2 means input is divided by 2 before rounding
        # x/2 = [0.5, 1.0, 1.5] -> round -> [0, 1, 2] -> *2 -> [0, 2, 4]
        assert torch.allclose(q_scale1, x)  # Already integers
        assert torch.allclose(q_scale2, torch.tensor([[0.0, 2.0, 4.0]]))


class TestLatticeDn:
    """Tests for checkerboard lattice Dn."""

    def test_even_coordinate_sum(self):
        """Dn quantizes to points with even coordinate sum."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        q, _ = lattice_quantize(x, lattice="Dn")

        # All coordinate sums should be even
        coord_sums = q.sum(dim=-1)
        assert torch.all(coord_sums % 2 == 0)

    def test_close_to_zn(self):
        """Dn is similar to Zn when rounding would give even sum."""
        x = torch.tensor(
            [[0.1, 0.1, 0.1, 0.1]]
        )  # Rounds to [0,0,0,0], sum=0 (even)
        q_dn, _ = lattice_quantize(x, lattice="Dn")
        q_zn, _ = lattice_quantize(x, lattice="Zn")
        assert torch.allclose(q_dn, q_zn)

    def test_differs_from_zn_when_odd_sum(self):
        """Dn differs from Zn when naive rounding gives odd sum."""
        x = torch.tensor(
            [[0.1, 0.1, 0.1, 0.9]]
        )  # Rounds to [0,0,0,1], sum=1 (odd)
        q_dn, _ = lattice_quantize(x, lattice="Dn")
        q_zn, _ = lattice_quantize(x, lattice="Zn")

        # Dn should adjust one coordinate
        assert not torch.allclose(q_dn, q_zn)
        assert q_dn.sum() % 2 == 0

    def test_requires_dimension_2(self):
        """Dn requires at least 2 dimensions."""
        x = torch.tensor([[1.5]])
        with pytest.raises(ValueError, match="dimension >= 2"):
            lattice_quantize(x, lattice="Dn")


class TestLatticeAn:
    """Tests for simplex lattice An."""

    def test_output_shape_preserved(self):
        """An preserves input shape."""
        x = torch.randn(10, 5)
        q, _ = lattice_quantize(x, lattice="An")
        assert q.shape == x.shape

    def test_integer_coordinates(self):
        """An produces integer coordinates."""
        x = torch.randn(10, 3)
        q, idx = lattice_quantize(x, lattice="An")
        # Check that quantized values are integers
        assert torch.allclose(q, q.round())

    def test_requires_dimension_2(self):
        """An requires at least 2 dimensions."""
        x = torch.tensor([[1.5]])
        with pytest.raises(ValueError, match="dimension >= 2"):
            lattice_quantize(x, lattice="An")


class TestLatticeE8:
    """Tests for E8 lattice."""

    def test_requires_dimension_8(self):
        """E8 requires exactly 8 dimensions."""
        x = torch.randn(10, 4)
        with pytest.raises(ValueError, match="dimension 8"):
            lattice_quantize(x, lattice="E8")

    def test_8d_works(self):
        """E8 works with 8D input."""
        x = torch.randn(10, 8)
        q, _ = lattice_quantize(x, lattice="E8")
        assert q.shape == (10, 8)

    def test_e8_union_of_d8_and_shifted(self):
        """E8 points are either in D8 or D8 + 1/2."""
        torch.manual_seed(42)
        x = torch.randn(100, 8)
        q, _ = lattice_quantize(x, lattice="E8")

        for i in range(q.shape[0]):
            point = q[i]
            # Check if in D8 (integer coords, even sum)
            is_integer = torch.allclose(point, point.round())
            in_d8 = is_integer and (point.sum() % 2 == 0)

            # Check if in D8 + 1/2 (half-integer coords, even sum of doubled)
            doubled = point * 2
            is_half_int = torch.allclose(doubled, doubled.round())
            in_d8_half = is_half_int and (doubled.sum() % 2 == 0)

            assert in_d8 or in_d8_half


class TestLatticeQuantizeGradients:
    """Tests for gradient modes."""

    def test_ste_gradient_passes_through(self):
        """STE mode passes gradients through."""
        x = torch.randn(5, 3, requires_grad=True)
        q, _ = lattice_quantize(x, gradient_mode="ste")
        loss = q.sum()
        loss.backward()
        # Gradients should be 1 (straight-through)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_none_gradient_blocks(self):
        """None mode blocks gradients."""
        x = torch.randn(5, 3, requires_grad=True)
        q, _ = lattice_quantize(x, gradient_mode="none")
        # In none mode, output is detached and has no grad_fn
        assert not q.requires_grad

    def test_soft_mode_returns_quantized(self):
        """Soft mode returns quantized values (no STE applied)."""
        x = torch.tensor([[0.4, 0.6, -0.4]])
        q_soft, _ = lattice_quantize(x, gradient_mode="soft")
        q_ste, _ = lattice_quantize(x, gradient_mode="ste")
        # Both should produce same quantized values
        assert torch.allclose(q_soft, q_ste)


class TestLatticeQuantizeDistortion:
    """Tests for quantization distortion properties."""

    def test_zn_mse_formula(self):
        """Zn MSE matches theoretical formula."""
        torch.manual_seed(42)
        # For Zn with uniform distribution on [0, 1], MSE = 1/12
        x = torch.rand(100000, 1)
        q, _ = lattice_quantize(x, lattice="Zn")
        mse = ((x - q) ** 2).mean().item()
        # Theoretical: 1/12 ≈ 0.0833
        assert abs(mse - 1 / 12) < 0.005

    def test_dn_distortion_bounded(self):
        """Dn distortion is bounded and reasonable."""
        torch.manual_seed(42)
        x = torch.randn(10000, 4)

        q_dn, _ = lattice_quantize(x, lattice="Dn")
        dist_dn = ((x - q_dn) ** 2).sum(dim=-1).mean()

        # Dn distortion should be reasonable (not much worse than Zn)
        # The distortion per dimension for uniform rounding is ~1/12
        # For 4D, expect total MSE around 4/12 = 0.33
        # Dn may be slightly higher due to adjustments for even sum
        assert dist_dn < 0.6  # Allow reasonable overhead


class TestLatticeQuantizeEdgeCases:
    """Edge case tests."""

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            lattice_quantize([1.0, 2.0, 3.0])

    def test_scalar_raises(self):
        """Raises error for scalar input."""
        x = torch.tensor(1.0)
        with pytest.raises(ValueError, match="at least 1D"):
            lattice_quantize(x)

    def test_invalid_lattice_raises(self):
        """Raises error for invalid lattice."""
        x = torch.randn(10, 3)
        with pytest.raises(ValueError, match="lattice must be"):
            lattice_quantize(x, lattice="invalid")

    def test_invalid_gradient_mode_raises(self):
        """Raises error for invalid gradient mode."""
        x = torch.randn(10, 3)
        with pytest.raises(ValueError, match="gradient_mode"):
            lattice_quantize(x, gradient_mode="invalid")

    def test_batch_dimensions(self):
        """Works with batch dimensions."""
        x = torch.randn(2, 3, 4, 5)
        q, idx = lattice_quantize(x)
        assert q.shape == x.shape


class TestLatticeQuantizeDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(10, 4, device="cpu")
        q, idx = lattice_quantize(x, lattice="Dn")
        assert q.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(10, 4, device="cuda")
        q, idx = lattice_quantize(x, lattice="Dn")
        assert q.device.type == "cuda"
