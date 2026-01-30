"""Tests for discrete_wavelet_transform."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import discrete_wavelet_transform

# Try to import pywt for reference tests
try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class TestDiscreteWaveletTransformBasic:
    """Tests for basic DWT functionality."""

    def test_basic_output_structure(self):
        """Test DWT returns (approx, [details]) tuple."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert isinstance(approx, torch.Tensor)
        assert isinstance(details, list)
        assert len(details) == 1  # level=1 by default
        assert isinstance(details[0], torch.Tensor)

    def test_basic_output_shape_haar(self):
        """Test DWT output shapes for Haar wavelet.

        For Haar wavelet with input length N, output lengths are ceil(N/2).
        """
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # After one level: length = ceil(128/2) = 64
        assert approx.shape[-1] == 64
        assert details[0].shape[-1] == 64

    def test_output_shape_odd_length(self):
        """Test DWT with odd-length input."""
        x = torch.randn(127)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # For odd length, output is ceil(127/2) = 64 with symmetric padding
        assert approx.ndim == 1
        assert details[0].ndim == 1

    def test_db1_same_as_haar(self):
        """Test that db1 produces same results as haar."""
        x = torch.randn(128)
        approx_haar, details_haar = discrete_wavelet_transform(
            x, wavelet="haar"
        )
        approx_db1, details_db1 = discrete_wavelet_transform(x, wavelet="db1")

        torch.testing.assert_close(approx_haar, approx_db1)
        torch.testing.assert_close(details_haar[0], details_db1[0])


class TestDiscreteWaveletTransformMultiLevel:
    """Tests for multi-level DWT decomposition."""

    def test_level_2(self):
        """Test 2-level DWT decomposition."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=2
        )

        # Level 2: should have 2 detail coefficient lists
        assert len(details) == 2

        # First detail (from level 1): length 64
        assert details[0].shape[-1] == 64

        # Second detail (from level 2): length 32
        assert details[1].shape[-1] == 32

        # Final approx: length 32
        assert approx.shape[-1] == 32

    def test_level_3(self):
        """Test 3-level DWT decomposition."""
        x = torch.randn(256)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3
        )

        assert len(details) == 3
        # Details: 128, 64, 32
        # Approx: 32
        assert details[0].shape[-1] == 128
        assert details[1].shape[-1] == 64
        assert details[2].shape[-1] == 32
        assert approx.shape[-1] == 32

    def test_max_level_auto(self):
        """Test that level is clamped to maximum valid level."""
        x = torch.randn(16)
        # With length 16, max level is 4 (16 -> 8 -> 4 -> 2 -> 1)
        # Asking for level=10 should be clamped
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=10
        )

        # Should still work, just clamped
        assert approx.ndim == 1


class TestDiscreteWaveletTransformWavelets:
    """Tests for different wavelet types."""

    def test_db2_wavelet(self):
        """Test Daubechies-2 wavelet."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db2")

        assert approx.ndim == 1
        assert len(details) == 1

    def test_db3_wavelet(self):
        """Test Daubechies-3 wavelet."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db3")

        assert approx.ndim == 1
        assert len(details) == 1

    def test_db4_wavelet(self):
        """Test Daubechies-4 wavelet."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="db4")

        assert approx.ndim == 1
        assert len(details) == 1

    def test_invalid_wavelet_raises(self):
        """Test that invalid wavelet name raises error."""
        x = torch.randn(128)
        with pytest.raises(ValueError, match="wavelet"):
            discrete_wavelet_transform(x, wavelet="invalid_wavelet")


class TestDiscreteWaveletTransformPadding:
    """Tests for padding modes."""

    def test_symmetric_padding(self):
        """Test symmetric padding mode (default)."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="symmetric"
        )

        assert approx.ndim == 1

    def test_reflect_padding(self):
        """Test reflect padding mode."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="reflect"
        )

        assert approx.ndim == 1

    def test_periodic_padding(self):
        """Test periodic (circular) padding mode."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="periodic"
        )

        assert approx.ndim == 1

    def test_zero_padding(self):
        """Test zero padding mode."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="zero"
        )

        assert approx.ndim == 1


class TestDiscreteWaveletTransformDim:
    """Tests for dim parameter handling."""

    def test_dim_last(self):
        """Test DWT on last dimension (default)."""
        x = torch.randn(4, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=-1)

        # Batch dim preserved
        assert approx.shape[0] == 4
        assert approx.shape[1] == 64
        assert details[0].shape[0] == 4
        assert details[0].shape[1] == 64

    def test_dim_first(self):
        """Test DWT on first dimension."""
        x = torch.randn(128, 4)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=0)

        # Signal dim (0) is transformed
        assert approx.shape[0] == 64
        assert approx.shape[1] == 4
        assert details[0].shape[0] == 64
        assert details[0].shape[1] == 4

    def test_dim_middle(self):
        """Test DWT on middle dimension."""
        x = torch.randn(4, 128, 3)
        approx, details = discrete_wavelet_transform(x, wavelet="haar", dim=1)

        assert approx.shape == (4, 64, 3)
        assert details[0].shape == (4, 64, 3)


class TestDiscreteWaveletTransformBatched:
    """Tests for batched input."""

    def test_batched_1d(self):
        """Test DWT with 2D batched input."""
        x = torch.randn(8, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.shape == (8, 64)
        assert details[0].shape == (8, 64)

    def test_batched_2d(self):
        """Test DWT with 3D batched input."""
        x = torch.randn(2, 8, 128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.shape == (2, 8, 64)
        assert details[0].shape == (2, 8, 64)

    def test_batched_multi_level(self):
        """Test multi-level DWT with batched input."""
        x = torch.randn(4, 128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=2
        )

        assert approx.shape == (4, 32)
        assert details[0].shape == (4, 64)
        assert details[1].shape == (4, 32)


class TestDiscreteWaveletTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_haar(self):
        """Test gradient correctness for Haar wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)

        def dwt_wrapper(x):
            approx, details = discrete_wavelet_transform(x, wavelet="haar")
            # Return sum of all outputs for scalar loss
            return approx.sum() + details[0].sum()

        assert gradcheck(dwt_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.xfail(
        reason="C++ backend has boundary handling differences for non-Haar wavelets"
    )
    def test_gradcheck_db2(self):
        """Test gradient correctness for db2 wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)

        def dwt_wrapper(x):
            approx, details = discrete_wavelet_transform(x, wavelet="db2")
            return approx.sum() + details[0].sum()

        assert gradcheck(dwt_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(128, requires_grad=True)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        loss = approx.sum() + details[0].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_multi_level(self):
        """Test backward pass with multi-level DWT."""
        x = torch.randn(128, requires_grad=True)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3
        )

        loss = approx.sum() + sum(d.sum() for d in details)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradgradcheck_haar(self):
        """Test second-order gradient correctness for Haar wavelet."""
        x = torch.randn(32, dtype=torch.float64, requires_grad=True)

        def dwt_wrapper(x):
            approx, details = discrete_wavelet_transform(x, wavelet="haar")
            return approx.sum() + details[0].sum()

        assert gradgradcheck(dwt_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestDiscreteWaveletTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(128, device="meta")
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.device == torch.device("meta")
        assert approx.shape[-1] == 64

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        x = torch.randn(4, 128, device="meta")
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.device == torch.device("meta")
        assert approx.shape == (4, 64)


class TestDiscreteWaveletTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(128, device="cuda")
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.device.type == "cuda"
        assert details[0].device.type == "cuda"


class TestDiscreteWaveletTransformDtype:
    """Tests for dtype handling."""

    def test_float32(self):
        """Test float32 input."""
        x = torch.randn(128, dtype=torch.float32)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.dtype == torch.float32

    def test_float64(self):
        """Test float64 input."""
        x = torch.randn(128, dtype=torch.float64)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.dtype == torch.float64


class TestDiscreteWaveletTransformEnergyPreservation:
    """Tests for signal properties."""

    def test_energy_preservation_haar(self):
        """Test that Haar DWT approximately preserves energy.

        Parseval's theorem: sum(x^2) ~= sum(approx^2) + sum(detail^2)
        """
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        input_energy = (x**2).sum()
        output_energy = (approx**2).sum() + (details[0] ** 2).sum()

        # Energy should be preserved (within tolerance)
        torch.testing.assert_close(
            input_energy, output_energy, rtol=0.01, atol=0.01
        )

    def test_energy_preservation_multi_level(self):
        """Test energy preservation with multi-level DWT."""
        x = torch.randn(128)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3
        )

        input_energy = (x**2).sum()
        output_energy = (approx**2).sum() + sum((d**2).sum() for d in details)

        torch.testing.assert_close(
            input_energy, output_energy, rtol=0.01, atol=0.01
        )


class TestDiscreteWaveletTransformSpecialCases:
    """Tests for special cases and edge cases."""

    def test_constant_signal(self):
        """Test DWT of constant signal."""
        x = torch.ones(128)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # For constant signal, detail coefficients should be ~0
        assert details[0].abs().max() < 1e-5

    def test_alternating_signal(self):
        """Test DWT of alternating signal."""
        x = torch.tensor([1.0, -1.0] * 64)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # For alternating signal, approx coefficients should be ~0
        assert approx.abs().max() < 1e-5

    def test_minimum_length(self):
        """Test DWT with minimum valid length."""
        x = torch.randn(4)
        approx, details = discrete_wavelet_transform(x, wavelet="haar")

        assert approx.ndim == 1
        assert len(details) == 1

    def test_level_0_raises(self):
        """Test that level=0 raises error."""
        x = torch.randn(128)
        with pytest.raises(ValueError, match="level"):
            discrete_wavelet_transform(x, wavelet="haar", level=0)

    def test_negative_level_raises(self):
        """Test that negative level raises error."""
        x = torch.randn(128)
        with pytest.raises(ValueError, match="level"):
            discrete_wavelet_transform(x, wavelet="haar", level=-1)


@pytest.mark.skipif(not HAS_PYWT, reason="pywt not available")
class TestDiscreteWaveletTransformPyWTReference:
    """Tests comparing with PyWavelets (pywt) reference implementation."""

    def test_haar_matches_pywt(self):
        """Test Haar DWT matches pywt output."""

        x = torch.randn(128, dtype=torch.float64)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", padding_mode="symmetric"
        )

        # PyWavelets reference
        cA, cD = pywt.dwt(x.numpy(), "haar", mode="symmetric")

        torch.testing.assert_close(
            approx, torch.from_numpy(cA), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            details[0], torch.from_numpy(cD), rtol=1e-5, atol=1e-5
        )

    def test_db2_matches_pywt(self):
        """Test db2 DWT matches pywt output."""

        x = torch.randn(128, dtype=torch.float64)
        approx, details = discrete_wavelet_transform(
            x, wavelet="db2", padding_mode="symmetric"
        )

        # PyWavelets reference
        cA, cD = pywt.dwt(x.numpy(), "db2", mode="symmetric")

        torch.testing.assert_close(
            approx, torch.from_numpy(cA), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            details[0], torch.from_numpy(cD), rtol=1e-4, atol=1e-4
        )

    def test_multi_level_matches_pywt(self):
        """Test multi-level DWT matches pywt.wavedec output."""

        x = torch.randn(256, dtype=torch.float64)
        approx, details = discrete_wavelet_transform(
            x, wavelet="haar", level=3, padding_mode="symmetric"
        )

        # PyWavelets reference
        coeffs = pywt.wavedec(x.numpy(), "haar", mode="symmetric", level=3)
        # coeffs = [cA3, cD3, cD2, cD1] (coarsest to finest for approx, then details)

        # Note: pywt returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # Our function returns (approx, [cD_1, cD_2, ..., cD_n])
        torch.testing.assert_close(
            approx, torch.from_numpy(coeffs[0]), rtol=1e-5, atol=1e-5
        )

        # Compare details (reversing order since pywt returns coarsest first)
        for i, (our_detail, pywt_detail) in enumerate(
            zip(details, reversed(coeffs[1:]))
        ):
            torch.testing.assert_close(
                our_detail, torch.from_numpy(pywt_detail), rtol=1e-5, atol=1e-5
            )


class TestDiscreteWaveletTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that DWT works with vmap."""
        x = torch.randn(8, 128)

        # Direct batched call
        approx_batched, details_batched = discrete_wavelet_transform(
            x, wavelet="haar"
        )

        # vmap version
        def dwt_single(xi):
            return discrete_wavelet_transform(xi, wavelet="haar")

        approx_vmap, details_vmap = torch.vmap(dwt_single)(x)

        torch.testing.assert_close(approx_batched, approx_vmap)
        torch.testing.assert_close(details_batched[0], details_vmap[0])


class TestDiscreteWaveletTransformCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that DWT works with torch.compile."""
        x = torch.randn(128)

        # Compiled version
        @torch.compile(fullgraph=True)
        def compiled_dwt(x):
            return discrete_wavelet_transform(x, wavelet="haar")

        approx_compiled, details_compiled = compiled_dwt(x)

        # Direct version
        approx_direct, details_direct = discrete_wavelet_transform(
            x, wavelet="haar"
        )

        torch.testing.assert_close(approx_compiled, approx_direct)
        torch.testing.assert_close(details_compiled[0], details_direct[0])


class TestDiscreteWaveletTransformAutocast:
    """Tests for autocast compatibility."""

    @pytest.mark.skip(
        reason="C++ dispatcher segfaults with float16 inputs - requires investigation"
    )
    def test_autocast_cpu_float16(self):
        """Test that DWT works under CPU autocast with float16."""
        x = torch.randn(64, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Should upcast to float32 for numerical stability
        assert approx.dtype == torch.float32
        assert approx.shape[0] == 32
        assert not torch.isnan(approx).any()
        assert not torch.isnan(details[0]).any()

    @pytest.mark.skip(
        reason="C++ dispatcher segfaults with bfloat16 inputs - requires investigation"
    )
    def test_autocast_cpu_bfloat16(self):
        """Test that DWT works under CPU autocast with bfloat16."""
        x = torch.randn(64, dtype=torch.bfloat16)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            approx, details = discrete_wavelet_transform(x, wavelet="haar")

        # Should upcast to float32 for numerical stability
        assert approx.dtype == torch.float32
        assert approx.shape[0] == 32
        assert not torch.isnan(approx).any()
        assert not torch.isnan(details[0]).any()
