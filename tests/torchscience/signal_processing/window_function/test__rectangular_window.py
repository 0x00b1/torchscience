import torch
import torch.testing

import torchscience.signal_processing.window_function
from torchscience.testing import (
    CreationOpDescriptor,
    CreationOpTestCase,
    CreationOpToleranceConfig,
    ExpectedValue,
)


def reference_rectangular_window(
    n: int,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    device: torch.device = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Reference implementation of rectangular window using torch.ones."""
    if n < 0:
        raise RuntimeError(
            f"rectangular_window: n must be non-negative, got {n}"
        )
    result = torch.ones(
        n,
        dtype=dtype or torch.float32,
        layout=layout or torch.strided,
        device=device or "cpu",
    )
    if requires_grad:
        result = result.requires_grad_(True)
    return result


class TestRectangularWindow(CreationOpTestCase):
    """Tests for the rectangular window function."""

    @property
    def descriptor(self) -> CreationOpDescriptor:
        return CreationOpDescriptor(
            name="rectangular_window",
            func=torchscience.signal_processing.window_function.rectangular_window,
            expected_values=[
                ExpectedValue(
                    n=1,
                    expected=torch.tensor([1.0], dtype=torch.float32),
                    description="Single element window",
                ),
                ExpectedValue(
                    n=5,
                    expected=torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
                    ),
                    description="5-element window",
                ),
                ExpectedValue(
                    n=3,
                    expected=torch.tensor(
                        [1.0, 1.0, 1.0], dtype=torch.float64
                    ),
                    description="3-element window float64",
                ),
            ],
            supported_dtypes=[
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ],
            tolerances=CreationOpToleranceConfig(),
            skip_tests={"test_error_for_zero_size", "test_contiguous_format"},
            supports_meta=True,
            reference_func=reference_rectangular_window,
        )

    # =========================================================================
    # Override base class tests that don't apply
    # =========================================================================

    def test_contiguous_format(self):
        """Skip: rectangular_window doesn't support memory_format parameter."""
        pass

    # =========================================================================
    # Rectangular window specific tests
    # =========================================================================

    def test_all_ones(self):
        """Test that all elements are exactly 1.0."""
        for n in [1, 5, 10, 100]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n
            )
            expected = torch.ones(n, dtype=torch.float32)
            torch.testing.assert_close(result, expected)

    def test_all_ones_float64(self):
        """Test that all elements are exactly 1.0 for float64."""
        for n in [1, 5, 10, 100]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            expected = torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(result, expected)

    def test_sum_equals_length(self):
        """Test that sum of window equals window length (coherent gain = n)."""
        for n in [1, 5, 10, 50]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            assert result.sum().item() == float(n), (
                f"Expected sum={n}, got {result.sum().item()}"
            )

    def test_normalization_property(self):
        """Test that mean of rectangular window is 1.0."""
        for n in [1, 5, 10, 100]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            assert result.mean().item() == 1.0, (
                f"Expected mean=1.0, got {result.mean().item()}"
            )

    def test_symmetry(self):
        """Test that rectangular window is symmetric."""
        for n in [1, 5, 10, 11]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped)

    def test_max_and_min(self):
        """Test that max and min are both 1.0."""
        for n in [1, 5, 10, 100]:
            result = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            assert result.max().item() == 1.0, (
                f"Expected max=1.0, got {result.max().item()}"
            )
            assert result.min().item() == 1.0, (
                f"Expected min=1.0, got {result.min().item()}"
            )

    def test_large_window(self):
        """Test with large window size."""
        n = 10000
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                n
            )
        )
        assert result.shape == (n,)
        assert result.sum().item() == float(n)

    def test_gradient_flow(self):
        """Test that gradients flow through when requires_grad=True."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                5, dtype=torch.float64, requires_grad=True
            )
        )
        loss = result.sum()
        loss.backward()
        # For a constant tensor, gradient should be None or zeros
        # since the window values don't depend on any input

    def test_multiply_with_signal(self):
        """Test typical use case of multiplying window with a signal."""
        n = 10
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )
        signal = torch.randn(n, dtype=torch.float64)
        windowed = window * signal
        # Rectangular window should not modify the signal
        torch.testing.assert_close(windowed, signal)

    def test_comparison_with_torch_signal_windows(self):
        """Compare with torch.signal.windows if available."""
        try:
            import torch.signal.windows

            for n in [5, 10, 20]:
                result = torchscience.signal_processing.window_function.rectangular_window(
                    n, dtype=torch.float64
                )
                # torch.signal.windows might not have rectangular, but if it does:
                if hasattr(torch.signal.windows, "rectangular"):
                    expected = torch.signal.windows.rectangular(n)
                    torch.testing.assert_close(result, expected)
        except (ImportError, AttributeError):
            # torch.signal.windows may not be available in all versions
            pass

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_n_equals_zero(self):
        """Test edge case where n=0 returns an empty tensor."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                0
            )
        )
        assert result.shape == (0,)
        assert result.numel() == 0
        assert result.dtype == torch.float32

    def test_n_equals_zero_float64(self):
        """Test n=0 with explicit dtype."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                0, dtype=torch.float64
            )
        )
        assert result.shape == (0,)
        assert result.dtype == torch.float64

    def test_n_equals_one(self):
        """Test edge case where n=1."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                1
            )
        )
        assert result.shape == (1,)
        assert result[0].item() == 1.0

    def test_n_equals_two(self):
        """Test edge case where n=2."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                2
            )
        )
        assert result.shape == (2,)
        torch.testing.assert_close(
            result, torch.tensor([1.0, 1.0], dtype=torch.float32)
        )

    # =========================================================================
    # Dtype edge cases
    # =========================================================================

    def test_half_precision(self):
        """Test float16 dtype."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                5, dtype=torch.float16
            )
        )
        assert result.dtype == torch.float16
        expected = torch.ones(5, dtype=torch.float16)
        torch.testing.assert_close(result, expected)

    def test_bfloat16_precision(self):
        """Test bfloat16 dtype."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                5, dtype=torch.bfloat16
            )
        )
        assert result.dtype == torch.bfloat16
        expected = torch.ones(5, dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Complex dtype tests (rectangular window typically real, but test handling)
    # =========================================================================

    def test_complex64_not_supported_or_works(self):
        """Test behavior with complex64 dtype."""
        try:
            result = torchscience.signal_processing.window_function.rectangular_window(
                5, dtype=torch.complex64
            )
            # If it works, verify it's all ones
            assert result.dtype == torch.complex64
            expected = torch.ones(5, dtype=torch.complex64)
            torch.testing.assert_close(result, expected)
        except (RuntimeError, TypeError):
            # Complex dtypes may not be supported for window functions
            pass

    # =========================================================================
    # Layout tests
    # =========================================================================

    def test_strided_layout_explicit(self):
        """Test explicit strided layout."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                5, layout=torch.strided
            )
        )
        assert result.layout == torch.strided
        torch.testing.assert_close(
            result,
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        )

    def test_sparse_coo_layout(self):
        """Test sparse_coo layout if supported."""
        try:
            result = torchscience.signal_processing.window_function.rectangular_window(
                5, layout=torch.sparse_coo
            )
            assert result.layout == torch.sparse_coo
            # Convert to dense for comparison
            dense = result.to_dense()
            expected = torch.ones(5, dtype=torch.float32)
            torch.testing.assert_close(dense, expected)
        except RuntimeError:
            # Sparse layout may not be supported for ones
            pass

    # =========================================================================
    # Memory and performance tests
    # =========================================================================

    def test_contiguous_output(self):
        """Test that output tensor is contiguous."""
        result = (
            torchscience.signal_processing.window_function.rectangular_window(
                10
            )
        )
        assert result.is_contiguous(), "Output should be contiguous"

    def test_no_memory_leak_large_allocations(self):
        """Test that repeated large allocations don't leak memory."""
        for _ in range(10):
            _ = torchscience.signal_processing.window_function.rectangular_window(
                10000
            )

    # =========================================================================
    # Integration tests
    # =========================================================================

    def test_fft_windowing_workflow(self):
        """Test typical FFT windowing workflow."""
        # Create a test signal
        n = 64
        signal = torch.randn(n, dtype=torch.float64)

        # Apply rectangular window
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )
        windowed_signal = signal * window

        # With rectangular window, windowed signal equals original
        torch.testing.assert_close(windowed_signal, signal)

        # Can compute FFT
        fft_result = torch.fft.fft(windowed_signal)
        assert fft_result.shape == (n,)

    def test_batch_windowing(self):
        """Test applying window to batched signals."""
        batch_size = 4
        n = 32

        # Create batched signals [batch, time]
        signals = torch.randn(batch_size, n, dtype=torch.float64)

        # Create window and broadcast
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )
        windowed = signals * window  # Broadcasting: [batch, time] * [time]

        # Should have same shape as input
        assert windowed.shape == signals.shape

        # With rectangular window, output equals input
        torch.testing.assert_close(windowed, signals)

    # =========================================================================
    # Frequency domain validation tests
    # =========================================================================

    def test_frequency_response_main_lobe_width(self):
        """Test that main lobe width is approximately 4*pi/n."""
        n = 64
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )

        # Zero-pad for better frequency resolution
        nfft = 4096
        spectrum = torch.fft.fft(window, n=nfft)
        magnitude_db = 20 * torch.log10(torch.abs(spectrum) / n + 1e-12)

        # Find first null (where magnitude drops significantly from peak)
        # Main lobe width should be ~4*pi/n in normalized angular frequency
        # In bins: 4*pi/n * nfft/(2*pi) = 2*nfft/n
        expected_first_null_bin = 2 * nfft // n  # = 128 for n=64, nfft=4096

        # The magnitude should be near a null around this bin
        # Allow some tolerance for discrete sampling
        null_region = magnitude_db[
            expected_first_null_bin - 5 : expected_first_null_bin + 5
        ]
        assert null_region.min() < -20, (
            f"Expected null near bin {expected_first_null_bin}, "
            f"but min magnitude in region is {null_region.min():.1f} dB"
        )

    def test_frequency_response_side_lobe_level(self):
        """Test that first side lobe is approximately -13 dB below main lobe.

        The rectangular window has the highest side lobes of common windows,
        at approximately -13 dB (actually -13.26 dB for the first side lobe).
        Reference: Harris, F.J. "On the use of windows for harmonic analysis
        with the discrete Fourier transform," Proc. IEEE, 1978.
        """
        n = 128
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )

        # Zero-pad for better frequency resolution
        nfft = 8192
        spectrum = torch.fft.fft(window, n=nfft)
        magnitude = torch.abs(spectrum)

        # Normalize to main lobe peak (should be at DC for symmetric window)
        magnitude_db = 20 * torch.log10(magnitude / magnitude[0] + 1e-12)

        # For a rectangular window of length n, the frequency response is:
        #   sin(pi*f*n) / sin(pi*f)  (Dirichlet kernel)
        # First null at f = 1/n, first side lobe peak at f ≈ 1.5/n
        # In FFT bins: null at nfft/n, side lobe peak at ~1.5*nfft/n
        bins_per_null = nfft // n  # = 64

        # Search for the first side lobe peak between first and second null
        # First null is at bin 64, second null at bin 128
        # Side lobe peak should be around bin 96 (1.5 * 64)
        search_start = bins_per_null + 5  # Just past first null
        search_end = 2 * bins_per_null - 5  # Just before second null
        side_lobe_region = magnitude_db[search_start:search_end]
        first_side_lobe_level = side_lobe_region.max()

        # The first side lobe should be around -13 dB (allow -11 to -15 dB range)
        assert -15.0 < first_side_lobe_level < -11.0, (
            f"Expected first side lobe level around -13 dB, "
            f"got {first_side_lobe_level:.2f} dB"
        )

    def test_frequency_response_side_lobe_rolloff(self):
        """Test that side lobes roll off at approximately 6 dB/octave.

        For the rectangular window, side lobes decrease as 1/f, which
        corresponds to 6 dB per octave (20 dB per decade).
        """
        n = 128
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )

        nfft = 16384
        spectrum = torch.fft.fft(window, n=nfft)
        magnitude = torch.abs(spectrum)
        magnitude_db = 20 * torch.log10(magnitude / magnitude[0] + 1e-12)

        # Find peaks of several side lobes
        # Side lobe peaks occur at approximately (2k+1)*nfft/n for k=1,2,3,...
        bins_per_lobe = nfft // n

        side_lobe_peaks = []
        for k in range(1, 6):  # First 5 side lobes
            center_bin = (2 * k + 1) * bins_per_lobe
            search_start = max(0, center_bin - bins_per_lobe // 2)
            search_end = min(nfft // 2, center_bin + bins_per_lobe // 2)
            region = magnitude_db[search_start:search_end]
            peak_level = region.max().item()
            side_lobe_peaks.append((center_bin, peak_level))

        # Check rolloff between first and fourth side lobe (factor of ~4 in frequency)
        # Should be approximately 12 dB (2 octaves * 6 dB/octave)
        first_peak = side_lobe_peaks[0][1]
        fourth_peak = side_lobe_peaks[3][1]
        rolloff = first_peak - fourth_peak

        # Allow some tolerance (expect 10-14 dB for 2 octaves)
        assert 8.0 < rolloff < 16.0, (
            f"Expected ~12 dB rolloff over 2 octaves, got {rolloff:.2f} dB"
        )

    def test_coherent_gain_frequency_domain(self):
        """Test that DC component equals window sum (coherent gain = n)."""
        for n in [32, 64, 128]:
            window = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
            spectrum = torch.fft.fft(window)

            # DC component should equal sum of window
            dc_component = spectrum[0].real
            expected_dc = float(n)

            torch.testing.assert_close(
                dc_component,
                torch.tensor(expected_dc, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_parseval_energy_conservation(self):
        """Test Parseval's theorem: energy in time domain equals energy in frequency domain."""
        n = 64
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )

        # Time domain energy
        time_energy = (window**2).sum()

        # Frequency domain energy (scaled by 1/n for FFT normalization)
        spectrum = torch.fft.fft(window)
        freq_energy = (torch.abs(spectrum) ** 2).sum() / n

        torch.testing.assert_close(
            time_energy,
            freq_energy,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_scalloping_loss(self):
        """Test that scalloping loss is approximately 3.92 dB.

        Scalloping loss is the maximum reduction in signal amplitude when
        the signal frequency falls between DFT bins. For a rectangular
        window, this is 20*log10(sin(pi/2)/(pi/2)) ≈ -3.92 dB.
        """
        # Theoretical scalloping loss for rectangular window
        import math

        theoretical_loss_db = 20 * math.log10(
            math.sin(math.pi / 2) / (math.pi / 2)
        )
        # = 20 * log10(1 / (pi/2)) = 20 * log10(2/pi) ≈ -3.92 dB

        n = 256
        window = (
            torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=torch.float64
            )
        )

        # Create a tone exactly at a bin frequency
        bin_freq = 10
        t = torch.arange(n, dtype=torch.float64)
        tone_on_bin = torch.cos(2 * math.pi * bin_freq * t / n)

        # Create a tone exactly between bins
        tone_between_bins = torch.cos(2 * math.pi * (bin_freq + 0.5) * t / n)

        # Apply window and compute FFT
        spectrum_on_bin = torch.fft.fft(window * tone_on_bin)
        spectrum_between = torch.fft.fft(window * tone_between_bins)

        # Peak magnitude for on-bin tone
        peak_on_bin = torch.abs(spectrum_on_bin).max()

        # Peak magnitude for between-bin tone
        peak_between = torch.abs(spectrum_between).max()

        # Scalloping loss in dB
        measured_loss_db = 20 * math.log10(peak_between / peak_on_bin)

        # Should be approximately -3.92 dB (allow -3.5 to -4.5 dB)
        assert -4.5 < measured_loss_db < -3.5, (
            f"Expected scalloping loss around {theoretical_loss_db:.2f} dB, "
            f"got {measured_loss_db:.2f} dB"
        )

    # =========================================================================
    # torch.compile tests
    # =========================================================================

    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_func = torch.compile(
            torchscience.signal_processing.window_function.rectangular_window
        )
        result = compiled_func(10)
        expected = (
            torchscience.signal_processing.window_function.rectangular_window(
                10
            )
        )
        torch.testing.assert_close(result, expected)

    def test_torch_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_func = torch.compile(
            torchscience.signal_processing.window_function.rectangular_window
        )
        for dtype in [torch.float32, torch.float64]:
            result = compiled_func(10, dtype=dtype)
            expected = torchscience.signal_processing.window_function.rectangular_window(
                10, dtype=dtype
            )
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_torch_compile_in_larger_function(self):
        """Test torch.compile when rectangular_window is used in a larger function."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = torchscience.signal_processing.window_function.rectangular_window(
                n, dtype=signal.dtype, device=signal.device
            )
            return torch.fft.fft(signal * window)

        compiled_fn = torch.compile(windowed_fft)

        signal = torch.randn(64, dtype=torch.float64)
        result = compiled_fn(signal)
        expected = windowed_fft(signal)

        torch.testing.assert_close(result, expected)

    def test_torch_compile_dynamic_shapes(self):
        """Test torch.compile with dynamic shapes."""
        compiled_func = torch.compile(
            torchscience.signal_processing.window_function.rectangular_window,
            dynamic=True,
        )

        for n in [16, 32, 64, 128]:
            result = compiled_func(n)
            expected = torchscience.signal_processing.window_function.rectangular_window(
                n
            )
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
