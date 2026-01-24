"""Tests for inverse_discrete_wavelet_transform."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.transform import (
    discrete_wavelet_transform,
    inverse_discrete_wavelet_transform,
)


class TestInverseDiscreteWaveletTransformRoundtrip:
    """Tests for DWT/IDWT roundtrip reconstruction."""

    def test_roundtrip_level_1_haar(self):
        """Test roundtrip reconstruction with level 1 Haar wavelet."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=1
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_roundtrip_level_2_haar(self):
        """Test roundtrip reconstruction with level 2 Haar wavelet."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=2
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_roundtrip_level_3_haar(self):
        """Test roundtrip reconstruction with level 3 Haar wavelet."""
        x = torch.randn(256)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    @pytest.mark.xfail(
        reason="DWT uses 'same' mode which loses boundary information for longer filters"
    )
    def test_roundtrip_db2(self):
        """Test roundtrip reconstruction with db2 wavelet.

        Note: This test may fail because the current DWT implementation uses
        'same' mode padding which discards some boundary information for
        filters longer than 2. For perfect reconstruction with db2/db3/db4,
        the DWT would need to use 'full' mode padding.
        """
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db2", level=2)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="db2"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-4, rtol=1e-4)

    @pytest.mark.xfail(
        reason="DWT uses 'same' mode which loses boundary information for longer filters"
    )
    def test_roundtrip_db3(self):
        """Test roundtrip reconstruction with db3 wavelet.

        Note: See test_roundtrip_db2 for explanation.
        """
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db3", level=2)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="db3"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-4, rtol=1e-4)

    @pytest.mark.xfail(
        reason="DWT uses 'same' mode which loses boundary information for longer filters"
    )
    def test_roundtrip_db4(self):
        """Test roundtrip reconstruction with db4 wavelet.

        Note: See test_roundtrip_db2 for explanation.
        """
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db4", level=2)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="db4"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-4, rtol=1e-4)

    def test_roundtrip_odd_length(self):
        """Test roundtrip with odd-length input.

        Note: For odd-length signals, the DWT output length is ceil(N/2).
        Since ceil(127/2) = 64 = ceil(128/2), the IDWT cannot distinguish
        between original lengths 127 and 128. The reconstruction will
        produce a signal of length 128.

        To handle this, users should trim the output to the known original
        length, or use an even-length signal.
        """
        x = torch.randn(127)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=1
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        # IDWT produces length 128 (2 * 64), not 127
        # We need to trim to match original length
        assert x_rec.shape[-1] == 128
        x_rec_trimmed = x_rec[:127]
        torch.testing.assert_close(x, x_rec_trimmed, atol=1e-5, rtol=1e-5)


class TestInverseDiscreteWaveletTransformBatched:
    """Tests for batched IDWT."""

    def test_batched_1d(self):
        """Test IDWT with 2D batched input."""
        x = torch.randn(8, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_batched_2d(self):
        """Test IDWT with 3D batched input."""
        x = torch.randn(2, 8, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_batched_multi_level(self):
        """Test multi-level IDWT with batched input."""
        x = torch.randn(4, 128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=2
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)


class TestInverseDiscreteWaveletTransformDim:
    """Tests for dim parameter handling."""

    def test_dim_last(self):
        """Test IDWT on last dimension (default)."""
        x = torch.randn(4, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=-1)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", dim=-1
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_dim_first(self):
        """Test IDWT on first dimension."""
        x = torch.randn(128, 4)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=0)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", dim=0
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_dim_middle(self):
        """Test IDWT on middle dimension."""
        x = torch.randn(4, 128, 3)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=1)
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", dim=1
        )

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)


class TestInverseDiscreteWaveletTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_haar(self):
        """Test gradient correctness for Haar wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Detach for gradcheck input
        approx_in = approx.detach().clone().requires_grad_(True)
        details_in = [d.detach().clone().requires_grad_(True) for d in details]

        def idwt_wrapper(approx, detail):
            return inverse_discrete_wavelet_transform(
                (approx, [detail]), wavelet="haar"
            ).sum()

        assert gradcheck(
            idwt_wrapper,
            (approx_in, details_in[0]),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_db2(self):
        """Test gradient correctness for db2 wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        approx, details = discrete_wavelet_transform(x, wavelet="db2")

        approx_in = approx.detach().clone().requires_grad_(True)
        details_in = [d.detach().clone().requires_grad_(True) for d in details]

        def idwt_wrapper(approx, detail):
            return inverse_discrete_wavelet_transform(
                (approx, [detail]), wavelet="db2"
            ).sum()

        assert gradcheck(
            idwt_wrapper,
            (approx_in, details_in[0]),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Make inputs require grad
        approx = approx.detach().clone().requires_grad_(True)
        details = [d.detach().clone().requires_grad_(True) for d in details]

        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        loss = x_rec.sum()
        loss.backward()

        assert approx.grad is not None
        assert approx.grad.shape == approx.shape
        assert details[0].grad is not None
        assert details[0].grad.shape == details[0].shape

    def test_backward_multi_level(self):
        """Test backward pass with multi-level IDWT."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3
        )

        # Make inputs require grad
        approx = approx.detach().clone().requires_grad_(True)
        details = [d.detach().clone().requires_grad_(True) for d in details]

        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        loss = x_rec.sum()
        loss.backward()

        assert approx.grad is not None
        for d in details:
            assert d.grad is not None


class TestInverseDiscreteWaveletTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        # First get coefficients from regular tensor
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Create meta versions
        approx_meta = torch.empty_like(approx, device="meta")
        details_meta = [torch.empty_like(d, device="meta") for d in details]

        x_rec = inverse_discrete_wavelet_transform(
            (approx_meta, details_meta), wavelet="haar"
        )

        assert x_rec.device == torch.device("meta")
        assert x_rec.shape == x.shape

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        x = torch.randn(4, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        approx_meta = torch.empty_like(approx, device="meta")
        details_meta = [torch.empty_like(d, device="meta") for d in details]

        x_rec = inverse_discrete_wavelet_transform(
            (approx_meta, details_meta), wavelet="haar"
        )

        assert x_rec.device == torch.device("meta")
        assert x_rec.shape == x.shape


class TestInverseDiscreteWaveletTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(128, device="cuda")
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.device.type == "cuda"
        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)


class TestInverseDiscreteWaveletTransformDtype:
    """Tests for dtype handling."""

    def test_float32(self):
        """Test float32 input."""
        x = torch.randn(128, dtype=torch.float32)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.dtype == torch.float32

    def test_float64(self):
        """Test float64 input."""
        x = torch.randn(128, dtype=torch.float64)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        assert x_rec.dtype == torch.float64


class TestInverseDiscreteWaveletTransformSpecialCases:
    """Tests for special cases and edge cases."""

    def test_constant_signal_reconstruction(self):
        """Test reconstruction of constant signal."""
        x = torch.ones(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_alternating_signal_reconstruction(self):
        """Test reconstruction of alternating signal."""
        x = torch.tensor([1.0, -1.0] * 64)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_minimum_length(self):
        """Test IDWT with minimum valid length."""
        x = torch.randn(4)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_invalid_wavelet_raises(self):
        """Test that invalid wavelet name raises error."""
        approx = torch.randn(64)
        details = [torch.randn(64)]

        with pytest.raises(ValueError, match="wavelet"):
            inverse_discrete_wavelet_transform(
                (approx, details), wavelet="invalid_wavelet"
            )


class TestInverseDiscreteWaveletTransformPadding:
    """Tests for padding modes."""

    def test_symmetric_padding_roundtrip(self):
        """Test roundtrip with symmetric padding."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="symmetric"
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", padding_mode="symmetric"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_reflect_padding_roundtrip(self):
        """Test roundtrip with reflect padding."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="reflect"
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", padding_mode="reflect"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_periodic_padding_roundtrip(self):
        """Test roundtrip with periodic padding."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="periodic"
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", padding_mode="periodic"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)

    def test_zero_padding_roundtrip(self):
        """Test roundtrip with zero padding."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="zero"
        )
        x_rec = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar", padding_mode="zero"
        )

        torch.testing.assert_close(x, x_rec, atol=1e-5, rtol=1e-5)


class TestInverseDiscreteWaveletTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that IDWT works with vmap."""
        x = torch.randn(8, 128)

        # Direct batched call
        approx, details = discrete_wavelet_transform(x, wavelet="haar")
        x_rec_batched = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        # vmap version
        def idwt_single(approx_i, detail_i):
            return inverse_discrete_wavelet_transform(
                (approx_i, [detail_i]), wavelet="haar"
            )

        x_rec_vmap = torch.vmap(idwt_single)(approx, details[0])

        torch.testing.assert_close(x_rec_batched, x_rec_vmap)


class TestInverseDiscreteWaveletTransformCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that IDWT works with torch.compile."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Compiled version
        @torch.compile(fullgraph=True)
        def compiled_idwt(approx, detail):
            return inverse_discrete_wavelet_transform(
                (approx, [detail]), wavelet="haar"
            )

        x_rec_compiled = compiled_idwt(approx, details[0])

        # Direct version
        x_rec_direct = inverse_discrete_wavelet_transform(
            (approx, details), wavelet="haar"
        )

        torch.testing.assert_close(x_rec_compiled, x_rec_direct)
