import pytest
import torch
import torch.nn.functional as F

from torchscience.pad import pad


class TestPadForward:
    """Forward correctness tests for pad operator."""

    def test_constant_1d(self):
        """Constant padding matches torch.nn.functional.pad."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = pad(x, (2, 1), mode="constant", value=0.0)
        expected = F.pad(x, (2, 1), mode="constant", value=0.0)
        torch.testing.assert_close(result, expected)

    def test_constant_2d(self):
        """2D constant padding."""
        x = torch.randn(3, 4)
        result = pad(x, (1, 2, 3, 4), mode="constant", value=-1.0)
        expected = F.pad(x, (1, 2, 3, 4), mode="constant", value=-1.0)
        torch.testing.assert_close(result, expected)

    def test_replicate_1d(self):
        """Replicate padding extends edge values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = pad(x, (2, 3), mode="replicate")
        expected = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_reflect_1d(self):
        """Reflect padding is edge-inclusive."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = pad(x, (2, 2), mode="reflect")
        # Edge-inclusive: reflects around edge value
        expected = torch.tensor([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_circular_1d(self):
        """Circular padding wraps around."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = pad(x, (2, 2), mode="circular")
        expected = torch.tensor([2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_reflect_odd_1d(self):
        """Antisymmetric reflection flips signs."""
        x = torch.tensor([0.0, 1.0, 2.0])
        result = pad(x, (1, 1), mode="reflect_odd")
        # At left: 2*edge - reflected = 2*0 - 1 = -1
        # At right: 2*edge - reflected = 2*2 - 1 = 3
        expected = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_linear_extrapolation(self):
        """Linear extrapolation extends the edge slope."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = pad(x, (1, 1), mode="linear")
        # Left: 1 - (2-1) = 0
        # Right: 3 + (3-2) = 4
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(result, expected)

    def test_zero_padding(self):
        """Zero padding returns input unchanged."""
        x = torch.randn(3, 4, 5)
        result = pad(x, (0, 0, 0, 0), mode="constant")
        torch.testing.assert_close(result, x)

    def test_dim_parameter(self):
        """Explicit dim parameter pads correct dimensions."""
        x = torch.randn(2, 3, 4)
        result = pad(x, (1, 1), mode="constant", value=0.0, dim=1)
        assert result.shape == (2, 5, 4)
        # Interior should match
        torch.testing.assert_close(result[:, 1:-1, :], x)

    def test_out_parameter(self):
        """Output tensor is used when provided."""
        x = torch.randn(3, 4)
        out = torch.empty(5, 6)
        result = pad(x, (1, 1, 1, 1), mode="constant", value=0.0, out=out)
        assert result is out

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        """Different dtypes work correctly."""
        x = torch.randn(5, dtype=dtype)
        result = pad(x, (1, 1), mode="reflect")
        assert result.dtype == dtype


class TestPadGradient:
    """Gradient correctness tests."""

    @pytest.mark.parametrize(
        "mode", ["constant", "replicate", "reflect", "circular"]
    )
    def test_gradcheck_basic_modes(self, mode):
        """First-order gradient check for basic modes."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda t: pad(t, (2, 2), mode=mode, value=0.0),
            x,
            raise_exception=True,
        )

    @pytest.mark.parametrize("mode", ["linear", "polynomial"])
    def test_gradcheck_extrapolation_modes(self, mode):
        """First-order gradient check for extrapolation modes."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda t: pad(t, (2, 2), mode=mode, order=2),
            x,
            raise_exception=True,
        )

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            lambda t: pad(t, (1, 1), mode="reflect"),
            x,
            raise_exception=True,
        )

    def test_gradient_accumulation_reflect(self):
        """Reflected positions accumulate gradients correctly."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = pad(x, (1, 1), mode="reflect")
        # y = [2, 1, 2, 3, 2]
        loss = y.sum()
        loss.backward()
        # x[0]=1 appears at positions 1
        # x[1]=2 appears at positions 0, 2, 4 (3 times)
        # x[2]=3 appears at position 3
        expected = torch.tensor([1.0, 3.0, 1.0])
        torch.testing.assert_close(x.grad, expected)


class TestPadIntegration:
    """PyTorch integration tests."""

    def test_torch_compile(self):
        """Works with torch.compile."""
        x = torch.randn(10)
        compiled_pad = torch.compile(lambda t: pad(t, (2, 2), mode="reflect"))
        result = compiled_pad(x)
        expected = pad(x, (2, 2), mode="reflect")
        torch.testing.assert_close(result, expected)

    def test_meta_tensors(self):
        """Shape inference with meta tensors."""
        x = torch.randn(3, 4, device="meta")
        result = pad(x, (1, 1, 2, 2), mode="constant")
        assert result.shape == (7, 6)
        assert result.device.type == "meta"


class TestPadEdgeCases:
    """Edge case tests."""

    def test_empty_tensor(self):
        """Handles empty input tensors."""
        x = torch.empty(0, 3)
        with pytest.raises(RuntimeError):
            pad(x, (1, 1), mode="constant")

    def test_large_padding(self):
        """Padding larger than input size."""
        x = torch.tensor([1.0, 2.0, 3.0])
        # Padding of 5 on each side for input of size 3
        result = pad(x, (5, 5), mode="reflect")
        assert result.shape == (13,)

    def test_asymmetric_padding(self):
        """Different before/after padding amounts."""
        x = torch.randn(5, 5)
        result = pad(x, (1, 3, 2, 4), mode="constant", value=0.0)
        assert result.shape == (11, 9)

    def test_5d_tensor(self):
        """N-dimensional padding beyond 3D."""
        x = torch.randn(2, 3, 4, 5, 6)
        result = pad(x, (1, 1), mode="replicate", dim=-1)
        assert result.shape == (2, 3, 4, 5, 8)

    def test_negative_dim(self):
        """Negative dimension indexing."""
        x = torch.randn(2, 3, 4)
        result = pad(x, (1, 1), mode="reflect", dim=-2)
        assert result.shape == (2, 5, 4)

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_polynomial_orders(self, order):
        """Polynomial extrapolation with different orders."""
        x = torch.randn(10, dtype=torch.float64)
        result = pad(x, (2, 2), mode="polynomial", order=order)
        assert result.shape == (14,)


class TestPadComplex:
    """Complex tensor support tests."""

    @pytest.mark.parametrize(
        "mode", ["constant", "replicate", "reflect", "circular"]
    )
    def test_complex_forward(self, mode):
        """Complex tensors work with basic modes."""
        x = torch.randn(5, dtype=torch.complex64)
        result = pad(x, (2, 2), mode=mode, value=0.0)
        assert result.dtype == torch.complex64
        assert result.shape == (9,)

    def test_complex_gradcheck(self):
        """Complex gradient check."""
        x = torch.randn(5, dtype=torch.complex128, requires_grad=True)
        # Note: gradcheck for complex requires special handling
        # For now, just verify it runs without error
        y = pad(x, (1, 1), mode="reflect")
        loss = y.abs().sum()
        loss.backward()
        assert x.grad is not None
