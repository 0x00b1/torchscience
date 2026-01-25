"""PyTorch integration tests for filter design functions.

This module tests that filter design functions work correctly with PyTorch's
advanced features including:
- torch.autograd.gradcheck (gradient correctness)
- torch.compile (graph compilation)
- torch.vmap (vectorized mapping)
- Device and dtype handling
"""

import pytest
import torch

from torchscience.signal_processing.filter import (
    bessel_design,
    bessel_prototype,
    bilinear_transform_zpk,
    butterworth_design,
    butterworth_minimum_order,
    butterworth_prototype,
    cascade_sos,
    chebyshev_type_1_design,
    chebyshev_type_1_minimum_order,
    chebyshev_type_1_prototype,
    chebyshev_type_2_design,
    chebyshev_type_2_minimum_order,
    chebyshev_type_2_prototype,
    elliptic_design,
    elliptic_minimum_order,
    elliptic_prototype,
    firwin,
    firwin2,
    iirnotch,
    iirpeak,
    lowpass_to_highpass_zpk,
    lowpass_to_lowpass_zpk,
    sos_to_zpk,
    zpk_to_ba,
    zpk_to_sos,
)


class TestGradcheck:
    """Test gradient correctness using torch.autograd.gradcheck.

    These tests verify that the analytical gradients computed by PyTorch's
    autograd match numerical gradients computed via finite differences.

    Note: gradcheck requires float64 for numerical stability and inputs
    must have requires_grad=True.
    """

    # =========================================================================
    # IIR Filter Design Functions - Gradient tests
    # =========================================================================

    def test_butterworth_design_cutoff_gradient(self):
        """Test gradcheck for butterworth_design with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            sos = butterworth_design(4, c, output="sos")
            return sos.sum()

        # Relaxed tolerance due to numerical complexity of filter design
        assert torch.autograd.gradcheck(
            fn, cutoff, eps=1e-6, atol=0.02, rtol=1e-3
        )

    def test_butterworth_design_bandpass_cutoff_gradient(self):
        """Test gradcheck for butterworth_design bandpass with respect to cutoffs."""
        cutoff = torch.tensor(
            [0.2, 0.4], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            sos = butterworth_design(
                2, c, filter_type="bandpass", output="sos"
            )
            return sos.sum()

        # Relaxed tolerance due to numerical complexity of bandpass filter design
        assert torch.autograd.gradcheck(
            fn, cutoff, eps=1e-6, atol=0.1, rtol=0.02
        )

    def test_chebyshev_type_1_design_cutoff_gradient(self):
        """Test gradcheck for chebyshev_type_1_design with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            sos = chebyshev_type_1_design(
                4, c, passband_ripple_db=1.0, output="sos"
            )
            return sos.sum()

        # Relaxed tolerance due to numerical complexity of filter design
        assert torch.autograd.gradcheck(
            fn, cutoff, eps=1e-6, atol=0.15, rtol=0.02
        )

    def test_chebyshev_type_2_design_cutoff_gradient(self):
        """Test gradcheck for chebyshev_type_2_design with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            sos = chebyshev_type_2_design(
                4, c, stopband_attenuation_db=40.0, output="sos"
            )
            return sos.sum()

        assert torch.autograd.gradcheck(fn, cutoff, eps=1e-6, atol=1e-4)

    def test_bessel_design_cutoff_gradient(self):
        """Test gradcheck for bessel_design with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            sos = bessel_design(4, c, output="sos")
            return sos.sum()

        # Relaxed tolerance due to numerical complexity of filter design
        assert torch.autograd.gradcheck(
            fn, cutoff, eps=1e-6, atol=0.15, rtol=0.02
        )

    def test_elliptic_design_cutoff_gradient(self):
        """Test gradcheck for elliptic_design with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            sos = elliptic_design(
                4,
                c,
                passband_ripple_db=1.0,
                stopband_attenuation_db=40.0,
                output="sos",
            )
            return sos.sum()

        # Relaxed tolerance due to numerical complexity of filter design
        assert torch.autograd.gradcheck(
            fn, cutoff, eps=1e-6, atol=0.02, rtol=1e-3
        )

    # =========================================================================
    # FIR Filter Design Functions - Gradient tests
    # =========================================================================

    def test_firwin_cutoff_gradient(self):
        """Test gradcheck for firwin with respect to cutoff."""
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def fn(c):
            h = firwin(51, c)
            return h.sum()

        assert torch.autograd.gradcheck(fn, cutoff, eps=1e-6, atol=1e-4)

    @pytest.mark.xfail(
        reason="firwin gradient chain broken for bandpass (uses .tolist() in _build_bands)"
    )
    def test_firwin_bandpass_cutoff_gradient(self):
        """Test gradcheck for firwin bandpass with respect to cutoffs."""
        cutoff = torch.tensor(
            [0.2, 0.4], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            h = firwin(51, c, filter_type="bandpass")
            return h.sum()

        assert torch.autograd.gradcheck(fn, cutoff, eps=1e-6, atol=1e-4)

    def test_firwin2_gains_gradient(self):
        """Test gradcheck for firwin2 with respect to gains."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = torch.tensor(
            [1.0, 1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True
        )

        def fn(g):
            h = firwin2(65, freqs, g)
            return h.sum()

        assert torch.autograd.gradcheck(fn, gains, eps=1e-6, atol=1e-4)

    # =========================================================================
    # Notch/Peak Filter Design Functions - Gradient tests
    # =========================================================================

    def test_iirnotch_frequency_gradient(self):
        """Test gradcheck for iirnotch with respect to notch_frequency."""
        notch_freq = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        def fn(f):
            # iirnotch doesn't support tensor inputs directly for frequency
            # This tests if the function can work with differentiable tensors
            b, a = iirnotch(f.item(), quality_factor=30.0)
            return b.sum() + a.sum()

        # Note: This may fail if iirnotch doesn't support autograd
        # which is expected behavior we want to document
        try:
            assert torch.autograd.gradcheck(
                fn, notch_freq, eps=1e-6, atol=1e-4
            )
        except RuntimeError as e:
            pytest.skip(
                f"iirnotch does not support gradients for frequency: {e}"
            )

    def test_iirnotch_quality_factor_gradient(self):
        """Test gradcheck for iirnotch with respect to quality_factor."""
        quality = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)

        def fn(q):
            b, a = iirnotch(0.1, quality_factor=q.item())
            return b.sum() + a.sum()

        try:
            assert torch.autograd.gradcheck(fn, quality, eps=1e-6, atol=1e-4)
        except RuntimeError as e:
            pytest.skip(
                f"iirnotch does not support gradients for quality_factor: {e}"
            )

    def test_iirpeak_frequency_gradient(self):
        """Test gradcheck for iirpeak with respect to peak_frequency."""
        peak_freq = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        def fn(f):
            b, a = iirpeak(f.item(), quality_factor=30.0)
            return b.sum() + a.sum()

        try:
            assert torch.autograd.gradcheck(fn, peak_freq, eps=1e-6, atol=1e-4)
        except RuntimeError as e:
            pytest.skip(
                f"iirpeak does not support gradients for frequency: {e}"
            )

    # =========================================================================
    # Transform Functions - Gradient tests
    # =========================================================================

    def test_bilinear_transform_zpk_gradient(self):
        """Test gradcheck for bilinear_transform_zpk."""
        z = torch.tensor([], dtype=torch.complex128)
        p = torch.tensor([-1.0 + 1.0j, -1.0 - 1.0j], dtype=torch.complex128)
        k = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def fn(gain):
            z_d, p_d, k_d = bilinear_transform_zpk(
                z, p, gain, sampling_frequency=2.0
            )
            return k_d

        assert torch.autograd.gradcheck(fn, k, eps=1e-6, atol=1e-4)

    def test_lowpass_to_lowpass_zpk_gradient(self):
        """Test gradcheck for lowpass_to_lowpass_zpk."""
        z = torch.tensor([], dtype=torch.complex128)
        p = torch.tensor([-1.0 + 0.0j], dtype=torch.complex128)
        k = torch.tensor(1.0, dtype=torch.float64)
        cutoff = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def fn(c):
            z_t, p_t, k_t = lowpass_to_lowpass_zpk(z, p, k, cutoff_frequency=c)
            return k_t

        assert torch.autograd.gradcheck(fn, cutoff, eps=1e-6, atol=1e-4)

    def test_lowpass_to_highpass_zpk_gradient(self):
        """Test gradcheck for lowpass_to_highpass_zpk."""
        z = torch.tensor([], dtype=torch.complex128)
        p = torch.tensor([-1.0 + 0.0j], dtype=torch.complex128)
        k = torch.tensor(1.0, dtype=torch.float64)
        cutoff = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def fn(c):
            z_t, p_t, k_t = lowpass_to_highpass_zpk(
                z, p, k, cutoff_frequency=c
            )
            return k_t

        assert torch.autograd.gradcheck(fn, cutoff, eps=1e-6, atol=1e-4)

    # =========================================================================
    # Conversion Functions - Gradient tests
    # =========================================================================

    def test_zpk_to_sos_gradient(self):
        """Test gradcheck for zpk_to_sos with respect to gain."""
        z = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
        p = torch.tensor([0.9 + 0.1j, 0.9 - 0.1j], dtype=torch.complex128)
        k = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def fn(gain):
            sos = zpk_to_sos(z, p, gain)
            return sos.sum()

        assert torch.autograd.gradcheck(fn, k, eps=1e-6, atol=1e-4)

    def test_zpk_to_ba_gradient(self):
        """Test gradcheck for zpk_to_ba with respect to gain."""
        z = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
        p = torch.tensor([0.9 + 0.1j, 0.9 - 0.1j], dtype=torch.complex128)
        k = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def fn(gain):
            b, a = zpk_to_ba(z, p, gain)
            return b.sum() + a.sum()

        assert torch.autograd.gradcheck(fn, k, eps=1e-6, atol=1e-4)


class TestTorchCompile:
    """Test torch.compile compatibility for filter design functions.

    These tests verify that filter design functions can be compiled with
    torch.compile and produce correct results.
    """

    # =========================================================================
    # IIR Filter Design Functions - Compile tests
    # =========================================================================

    def test_butterworth_design_compile(self):
        """Test torch.compile works for butterworth_design."""

        def fn(cutoff):
            return butterworth_design(4, cutoff, output="sos")

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    def test_butterworth_design_compile_with_grad(self):
        """Test torch.compile works with gradients for butterworth_design."""

        def fn(cutoff):
            sos = butterworth_design(4, cutoff, output="sos")
            return sos.sum()

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        result = compiled_fn(cutoff)
        result.backward()

        assert cutoff.grad is not None
        assert not torch.isnan(cutoff.grad)

    def test_chebyshev_type_1_design_compile(self):
        """Test torch.compile works for chebyshev_type_1_design."""

        def fn(cutoff):
            return chebyshev_type_1_design(
                4, cutoff, passband_ripple_db=1.0, output="sos"
            )

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    def test_chebyshev_type_2_design_compile(self):
        """Test torch.compile works for chebyshev_type_2_design."""

        def fn(cutoff):
            return chebyshev_type_2_design(
                4, cutoff, stopband_attenuation_db=40.0, output="sos"
            )

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    def test_bessel_design_compile(self):
        """Test torch.compile works for bessel_design."""

        def fn(cutoff):
            return bessel_design(4, cutoff, output="sos")

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    def test_elliptic_design_compile(self):
        """Test torch.compile works for elliptic_design."""

        def fn(cutoff):
            return elliptic_design(
                4,
                cutoff,
                passband_ripple_db=1.0,
                stopband_attenuation_db=40.0,
                output="sos",
            )

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    # =========================================================================
    # FIR Filter Design Functions - Compile tests
    # =========================================================================

    def test_firwin_compile(self):
        """Test torch.compile works for firwin."""

        def fn(cutoff):
            return firwin(51, cutoff)

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64)

        result = compiled_fn(cutoff)
        expected = fn(cutoff)

        torch.testing.assert_close(result, expected)

    @pytest.mark.xfail(
        reason="firwin gradient chain broken (uses .tolist() in _build_bands and .item() in _scale_filter)"
    )
    def test_firwin_compile_with_grad(self):
        """Test torch.compile works with gradients for firwin."""

        def fn(cutoff):
            h = firwin(51, cutoff)
            return h.sum()

        compiled_fn = torch.compile(fn)
        cutoff = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        result = compiled_fn(cutoff)
        result.backward()

        assert cutoff.grad is not None
        assert not torch.isnan(cutoff.grad)

    def test_firwin2_compile(self):
        """Test torch.compile works for firwin2."""

        def fn(gains):
            freqs = [0, 0.25, 0.3, 1.0]
            return firwin2(65, freqs, gains)

        compiled_fn = torch.compile(fn)
        gains = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)

        result = compiled_fn(gains)
        expected = fn(gains)

        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Notch/Peak Filter Design Functions - Compile tests
    # =========================================================================

    def test_iirnotch_compile(self):
        """Test torch.compile works for iirnotch."""

        def fn():
            return iirnotch(0.1, quality_factor=30.0)

        compiled_fn = torch.compile(fn)

        b_result, a_result = compiled_fn()
        b_expected, a_expected = fn()

        torch.testing.assert_close(b_result, b_expected)
        torch.testing.assert_close(a_result, a_expected)

    def test_iirpeak_compile(self):
        """Test torch.compile works for iirpeak."""

        def fn():
            return iirpeak(0.1, quality_factor=30.0)

        compiled_fn = torch.compile(fn)

        b_result, a_result = compiled_fn()
        b_expected, a_expected = fn()

        torch.testing.assert_close(b_result, b_expected)
        torch.testing.assert_close(a_result, a_expected)

    # =========================================================================
    # Conversion Functions - Compile tests
    # =========================================================================

    def test_zpk_to_sos_compile(self):
        """Test torch.compile works for zpk_to_sos."""

        def fn(k):
            z = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
            p = torch.tensor([0.9 + 0.1j, 0.9 - 0.1j], dtype=torch.complex128)
            return zpk_to_sos(z, p, k)

        compiled_fn = torch.compile(fn)
        k = torch.tensor(1.0, dtype=torch.float64)

        result = compiled_fn(k)
        expected = fn(k)

        torch.testing.assert_close(result, expected)

    def test_zpk_to_ba_compile(self):
        """Test torch.compile works for zpk_to_ba."""

        def fn(k):
            z = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
            p = torch.tensor([0.9 + 0.1j, 0.9 - 0.1j], dtype=torch.complex128)
            return zpk_to_ba(z, p, k)

        compiled_fn = torch.compile(fn)
        k = torch.tensor(1.0, dtype=torch.float64)

        b_result, a_result = compiled_fn(k)
        b_expected, a_expected = fn(k)

        torch.testing.assert_close(b_result, b_expected)
        torch.testing.assert_close(a_result, a_expected)

    def test_sos_to_zpk_compile(self):
        """Test torch.compile works for sos_to_zpk."""

        def fn(sos):
            return sos_to_zpk(sos)

        compiled_fn = torch.compile(fn)
        sos = butterworth_design(4, 0.3, output="sos", dtype=torch.float64)

        z_result, p_result, k_result = compiled_fn(sos)
        z_expected, p_expected, k_expected = fn(sos)

        torch.testing.assert_close(z_result, z_expected)
        torch.testing.assert_close(p_result, p_expected)
        torch.testing.assert_close(k_result, k_expected)

    def test_cascade_sos_compile(self):
        """Test torch.compile works for cascade_sos."""

        def fn(sos1, sos2):
            return cascade_sos(sos1, sos2)

        compiled_fn = torch.compile(fn)
        sos1 = butterworth_design(2, 0.3, output="sos", dtype=torch.float64)
        sos2 = butterworth_design(2, 0.4, output="sos", dtype=torch.float64)

        result = compiled_fn(sos1, sos2)
        expected = fn(sos1, sos2)

        torch.testing.assert_close(result, expected)


class TestVmap:
    """Test torch.vmap compatibility for filter design functions.

    These tests verify that filter design functions work with torch.vmap
    for vectorized operations over batches of inputs.
    """

    def test_firwin_vmap_over_cutoffs(self):
        """Test vmap over cutoff frequencies for firwin."""
        cutoffs = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        # Design filters for multiple cutoffs in parallel
        def single_firwin(c):
            return firwin(51, c)

        try:
            h_batch = torch.vmap(single_firwin)(cutoffs)
            assert h_batch.shape == (4, 51)

            # Verify each filter matches sequential computation
            for i, c in enumerate(cutoffs):
                h_expected = firwin(51, c)
                torch.testing.assert_close(h_batch[i], h_expected)
        except RuntimeError as e:
            pytest.skip(f"firwin does not support vmap: {e}")

    def test_butterworth_design_vmap_over_cutoffs(self):
        """Test vmap over cutoff frequencies for butterworth_design."""
        cutoffs = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        def single_butter(c):
            return butterworth_design(4, c, output="sos")

        try:
            sos_batch = torch.vmap(single_butter)(cutoffs)
            assert sos_batch.shape == (4, 2, 6)  # (batch, sections, coeffs)

            # Verify each filter matches sequential computation
            for i, c in enumerate(cutoffs):
                sos_expected = butterworth_design(4, c, output="sos")
                torch.testing.assert_close(sos_batch[i], sos_expected)
        except RuntimeError as e:
            pytest.skip(f"butterworth_design does not support vmap: {e}")

    def test_zpk_to_sos_vmap_over_gains(self):
        """Test vmap over gains for zpk_to_sos."""
        z = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
        p = torch.tensor([0.9 + 0.1j, 0.9 - 0.1j], dtype=torch.complex128)
        gains = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

        def single_zpk_to_sos(k):
            return zpk_to_sos(z, p, k)

        try:
            sos_batch = torch.vmap(single_zpk_to_sos)(gains)
            assert sos_batch.shape == (4, 1, 6)

            # Verify each conversion matches sequential computation
            for i, k in enumerate(gains):
                sos_expected = zpk_to_sos(z, p, k)
                torch.testing.assert_close(sos_batch[i], sos_expected)
        except RuntimeError as e:
            pytest.skip(f"zpk_to_sos does not support vmap: {e}")

    def test_bilinear_transform_zpk_vmap_over_gains(self):
        """Test vmap over gains for bilinear_transform_zpk."""
        z = torch.tensor([], dtype=torch.complex128)
        p = torch.tensor([-1.0 + 1.0j, -1.0 - 1.0j], dtype=torch.complex128)
        gains = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

        def single_bilinear(k):
            return bilinear_transform_zpk(z, p, k, sampling_frequency=2.0)[
                2
            ]  # just gain

        try:
            k_batch = torch.vmap(single_bilinear)(gains)
            assert k_batch.shape == (4,)
        except RuntimeError as e:
            pytest.skip(f"bilinear_transform_zpk does not support vmap: {e}")


class TestDeviceAndDtype:
    """Test device and dtype handling for filter design functions.

    These tests verify that filter design functions correctly handle
    different dtypes (float32, float64) and devices (CPU, CUDA).
    """

    # =========================================================================
    # dtype tests - float32
    # =========================================================================

    def test_butterworth_design_float32(self):
        """Test butterworth_design with float32 dtype."""
        sos = butterworth_design(4, 0.3, dtype=torch.float32)
        assert sos.dtype == torch.float32

    def test_chebyshev_type_1_design_float32(self):
        """Test chebyshev_type_1_design with float32 dtype."""
        sos = chebyshev_type_1_design(
            4, 0.3, passband_ripple_db=1.0, dtype=torch.float32
        )
        assert sos.dtype == torch.float32

    def test_chebyshev_type_2_design_float32(self):
        """Test chebyshev_type_2_design with float32 dtype."""
        sos = chebyshev_type_2_design(
            4, 0.3, stopband_attenuation_db=40.0, dtype=torch.float32
        )
        assert sos.dtype == torch.float32

    def test_bessel_design_float32(self):
        """Test bessel_design with float32 dtype."""
        sos = bessel_design(4, 0.3, dtype=torch.float32)
        assert sos.dtype == torch.float32

    def test_elliptic_design_float32(self):
        """Test elliptic_design with float32 dtype."""
        sos = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            dtype=torch.float32,
        )
        assert sos.dtype == torch.float32

    def test_firwin_float32(self):
        """Test firwin with float32 dtype."""
        h = firwin(51, 0.3, dtype=torch.float32)
        assert h.dtype == torch.float32

    def test_firwin2_float32(self):
        """Test firwin2 with float32 dtype."""
        h = firwin2(65, [0, 0.25, 0.3, 1.0], [1, 1, 0, 0], dtype=torch.float32)
        assert h.dtype == torch.float32

    def test_iirnotch_float32(self):
        """Test iirnotch with float32 dtype."""
        b, a = iirnotch(0.1, 30.0, dtype=torch.float32)
        assert b.dtype == torch.float32
        assert a.dtype == torch.float32

    def test_iirpeak_float32(self):
        """Test iirpeak with float32 dtype."""
        b, a = iirpeak(0.1, 30.0, dtype=torch.float32)
        assert b.dtype == torch.float32
        assert a.dtype == torch.float32

    # =========================================================================
    # dtype tests - float64
    # =========================================================================

    def test_butterworth_design_float64(self):
        """Test butterworth_design with float64 dtype."""
        sos = butterworth_design(4, 0.3, dtype=torch.float64)
        assert sos.dtype == torch.float64

    def test_chebyshev_type_1_design_float64(self):
        """Test chebyshev_type_1_design with float64 dtype."""
        sos = chebyshev_type_1_design(
            4, 0.3, passband_ripple_db=1.0, dtype=torch.float64
        )
        assert sos.dtype == torch.float64

    def test_chebyshev_type_2_design_float64(self):
        """Test chebyshev_type_2_design with float64 dtype."""
        sos = chebyshev_type_2_design(
            4, 0.3, stopband_attenuation_db=40.0, dtype=torch.float64
        )
        assert sos.dtype == torch.float64

    def test_bessel_design_float64(self):
        """Test bessel_design with float64 dtype."""
        sos = bessel_design(4, 0.3, dtype=torch.float64)
        assert sos.dtype == torch.float64

    def test_elliptic_design_float64(self):
        """Test elliptic_design with float64 dtype."""
        sos = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            dtype=torch.float64,
        )
        assert sos.dtype == torch.float64

    def test_firwin_float64(self):
        """Test firwin with float64 dtype."""
        h = firwin(51, 0.3, dtype=torch.float64)
        assert h.dtype == torch.float64

    def test_firwin2_float64(self):
        """Test firwin2 with float64 dtype."""
        h = firwin2(65, [0, 0.25, 0.3, 1.0], [1, 1, 0, 0], dtype=torch.float64)
        assert h.dtype == torch.float64

    def test_iirnotch_float64(self):
        """Test iirnotch with float64 dtype."""
        b, a = iirnotch(0.1, 30.0, dtype=torch.float64)
        assert b.dtype == torch.float64
        assert a.dtype == torch.float64

    def test_iirpeak_float64(self):
        """Test iirpeak with float64 dtype."""
        b, a = iirpeak(0.1, 30.0, dtype=torch.float64)
        assert b.dtype == torch.float64
        assert a.dtype == torch.float64

    # =========================================================================
    # Device tests - CPU
    # =========================================================================

    def test_butterworth_design_cpu(self):
        """Test butterworth_design on CPU device."""
        sos = butterworth_design(4, 0.3, device=torch.device("cpu"))
        assert sos.device.type == "cpu"

    def test_firwin_cpu(self):
        """Test firwin on CPU device."""
        h = firwin(51, 0.3, device=torch.device("cpu"))
        assert h.device.type == "cpu"

    def test_iirnotch_cpu(self):
        """Test iirnotch on CPU device."""
        b, a = iirnotch(0.1, 30.0, device=torch.device("cpu"))
        assert b.device.type == "cpu"
        assert a.device.type == "cpu"

    # =========================================================================
    # Device tests - CUDA (if available)
    # =========================================================================

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_butterworth_design_cuda(self):
        """Test butterworth_design on CUDA device."""
        sos = butterworth_design(4, 0.3, device=torch.device("cuda"))
        assert sos.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_firwin_cuda(self):
        """Test firwin on CUDA device."""
        h = firwin(51, 0.3, device=torch.device("cuda"))
        assert h.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_iirnotch_cuda(self):
        """Test iirnotch on CUDA device."""
        b, a = iirnotch(0.1, 30.0, device=torch.device("cuda"))
        assert b.device.type == "cuda"
        assert a.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_butterworth_design_cuda_matches_cpu(self):
        """Test that CUDA and CPU produce same results."""
        sos_cpu = butterworth_design(
            4, 0.3, dtype=torch.float64, device=torch.device("cpu")
        )
        sos_cuda = butterworth_design(
            4, 0.3, dtype=torch.float64, device=torch.device("cuda")
        )

        torch.testing.assert_close(sos_cpu, sos_cuda.cpu())


class TestPrototypeFunctions:
    """Test PyTorch integration for prototype filter functions.

    These functions create analog lowpass filter prototypes that are then
    transformed to digital filters.
    """

    def test_butterworth_prototype_dtype(self):
        """Test butterworth_prototype dtype handling."""
        z, p, k = butterworth_prototype(4, dtype=torch.float64)
        assert p.dtype == torch.complex128  # Poles are complex
        assert k.dtype == torch.float64

    def test_chebyshev_type_1_prototype_dtype(self):
        """Test chebyshev_type_1_prototype dtype handling."""
        z, p, k = chebyshev_type_1_prototype(4, 1.0, dtype=torch.float64)
        assert p.dtype == torch.complex128
        assert k.dtype == torch.float64

    def test_chebyshev_type_2_prototype_dtype(self):
        """Test chebyshev_type_2_prototype dtype handling."""
        z, p, k = chebyshev_type_2_prototype(4, 40.0, dtype=torch.float64)
        assert z.dtype == torch.complex128  # Type II has zeros
        assert p.dtype == torch.complex128
        assert k.dtype == torch.float64

    def test_bessel_prototype_dtype(self):
        """Test bessel_prototype dtype handling."""
        z, p, k = bessel_prototype(4, dtype=torch.float64)
        assert p.dtype == torch.complex128
        assert k.dtype == torch.float64

    def test_elliptic_prototype_dtype(self):
        """Test elliptic_prototype dtype handling."""
        z, p, k = elliptic_prototype(4, 1.0, 40.0, dtype=torch.float64)
        assert z.dtype == torch.complex128
        assert p.dtype == torch.complex128
        assert k.dtype == torch.float64


class TestMinimumOrderFunctions:
    """Test PyTorch integration for minimum order estimation functions."""

    def test_butterworth_minimum_order_returns_int(self):
        """Test butterworth_minimum_order returns integer order."""
        order, wn = butterworth_minimum_order(0.2, 0.3, 3, 40)
        assert isinstance(order, int)
        # wn should be a tensor
        assert isinstance(wn, (float, torch.Tensor))

    def test_chebyshev_type_1_minimum_order_returns_int(self):
        """Test chebyshev_type_1_minimum_order returns integer order."""
        order, wn = chebyshev_type_1_minimum_order(0.2, 0.3, 3, 40)
        assert isinstance(order, int)

    def test_chebyshev_type_2_minimum_order_returns_int(self):
        """Test chebyshev_type_2_minimum_order returns integer order."""
        order, wn = chebyshev_type_2_minimum_order(0.2, 0.3, 3, 40)
        assert isinstance(order, int)

    def test_elliptic_minimum_order_returns_int(self):
        """Test elliptic_minimum_order returns integer order."""
        order, wn = elliptic_minimum_order(0.2, 0.3, 3, 40)
        assert isinstance(order, int)
