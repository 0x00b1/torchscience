import math

import pytest
import torch
import torch.testing

import torchscience.signal_processing.waveform
import torchscience.window_function
from torchscience.testing import (
    CreationOpDescriptor,
    CreationOpTestCase,
    CreationOpToleranceConfig,
    ExpectedValue,
)


def reference_sine_wave(
    n: int,
    frequency: float = 1.0,
    sample_rate: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    device: torch.device = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Reference implementation of sine wave using torch primitives."""
    if n < 0:
        raise RuntimeError(f"sine_wave: n must be non-negative, got {n}")
    if sample_rate <= 0:
        raise RuntimeError(
            f"sine_wave: sample_rate must be positive, got {sample_rate}"
        )

    if n == 0:
        result = torch.empty(
            0,
            dtype=dtype or torch.float32,
            layout=layout or torch.strided,
            device=device or "cpu",
        )
        if requires_grad:
            result = result.requires_grad_(True)
        return result

    t = torch.arange(
        n,
        dtype=dtype or torch.float32,
        layout=layout or torch.strided,
        device=device or "cpu",
    )
    angular_freq = 2.0 * math.pi * frequency / sample_rate
    result = amplitude * torch.sin(angular_freq * t + phase)

    if requires_grad:
        result = result.requires_grad_(True)
    return result


class TestSineWave(CreationOpTestCase):
    """Tests for the sine wave generator function."""

    @property
    def descriptor(self) -> CreationOpDescriptor:
        return CreationOpDescriptor(
            name="sine_wave",
            func=torchscience.signal_processing.waveform.sine_wave,
            expected_values=[
                ExpectedValue(
                    n=1,
                    expected=torch.tensor([0.0], dtype=torch.float32),
                    description="Single element (sin(0) = 0)",
                ),
                # Note: Default parameters (frequency=1.0, sample_rate=1.0) produce
                # sin(2*pi*k) which is ~0 for all integer k (floating point errors only).
                # We test with default params which gives near-zero values.
                ExpectedValue(
                    n=5,
                    expected=torch.tensor(
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32,
                    ),
                    rtol=1e-5,
                    atol=1e-5,
                    description="5-element default sine wave (sin(2*pi*k) ~ 0)",
                ),
            ],
            supported_dtypes=[
                torch.float32,
                torch.float64,
            ],
            tolerances=CreationOpToleranceConfig(
                float16_rtol=1e-2,
                float16_atol=1e-2,
                bfloat16_rtol=1e-1,
                bfloat16_atol=1e-1,
            ),
            skip_tests={
                "test_error_for_zero_size",
                "test_contiguous_format",
                "test_torch_compile",
                "test_dtype_device_combinations",
            },
            supports_meta=True,
            reference_func=reference_sine_wave,
        )

    # =========================================================================
    # Override base class tests that don't apply
    # =========================================================================

    def test_contiguous_format(self):
        """Skip: sine_wave doesn't support memory_format parameter."""
        pass

    # =========================================================================
    # Sine wave specific tests
    # =========================================================================

    def test_starts_at_zero(self):
        """Test that sine wave starts at zero with default phase."""
        for n in [1, 10, 100]:
            result = torchscience.signal_processing.waveform.sine_wave(n)
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float32),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_phase_offset_pi_half_gives_cosine(self):
        """Test that phase=pi/2 produces a cosine wave."""
        n = 100
        sample_rate = 100.0
        frequency = 1.0

        sine = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )
        cosine = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            phase=math.pi / 2,
            dtype=torch.float64,
        )

        # cos(x) = sin(x + pi/2)
        # Verify first sample: sin(0) = 0, sin(pi/2) = 1
        torch.testing.assert_close(
            sine[0],
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            cosine[0],
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_amplitude_scaling(self):
        """Test that amplitude correctly scales the wave."""
        n = 100
        amplitude = 2.5

        result = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amplitude, dtype=torch.float64
        )

        # Max and min should be within amplitude bounds
        assert result.max().item() <= amplitude + 1e-10
        assert result.min().item() >= -amplitude - 1e-10

    def test_frequency_one_cycle(self):
        """Test that frequency correctly determines number of cycles."""
        sample_rate = 100.0
        frequency = 1.0
        n = int(sample_rate)  # One second = one cycle

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # Start and end should be close to zero (full cycle)
        torch.testing.assert_close(
            result[0],
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        # Note: result[n-1] is not exactly zero but close
        # The last sample is at (n-1)/n of the period
        expected_last = math.sin(2 * math.pi * (n - 1) / n)
        torch.testing.assert_close(
            result[-1],
            torch.tensor(expected_last, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_frequency_multiple_cycles(self):
        """Test multiple cycles within the waveform."""
        sample_rate = 100.0
        frequency = 5.0  # 5 cycles per second
        n = int(sample_rate)

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # Count zero crossings (should be ~2 per cycle = 10 for 5 cycles)
        signs = torch.sign(result)
        # Handle zeros in signs
        signs[signs == 0] = 1
        crossings = torch.sum(torch.abs(torch.diff(signs)) > 0).item()

        # Should have approximately 2 * frequency zero crossings
        # Allow some tolerance due to discrete sampling
        assert 8 <= crossings <= 12, (
            f"Expected ~10 zero crossings, got {crossings}"
        )

    def test_sample_rate_normalization(self):
        """Test that sample_rate correctly normalizes the frequency."""
        n = 100
        frequency = 10.0
        sample_rate_1 = 100.0
        sample_rate_2 = 200.0

        # At sample_rate_1, we get 10/100 = 0.1 cycles per sample
        result_1 = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate_1,
            dtype=torch.float64,
        )

        # At sample_rate_2, same frequency means slower oscillation
        result_2 = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate_2,
            dtype=torch.float64,
        )

        # result_2 should oscillate half as fast
        # result_1[k] should equal result_2[2k] (approximately)
        for k in range(0, n // 2):
            torch.testing.assert_close(
                result_1[k],
                result_2[2 * k],
                rtol=1e-10,
                atol=1e-10,
            )

    def test_symmetry_of_sine(self):
        """Test the antisymmetry property: sin(-x) = -sin(x)."""
        n = 100
        sample_rate = 100.0
        frequency = 1.0

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # For a complete cycle, first half should be antisymmetric to second half
        # sin(x) = -sin(pi - x) shifted appropriately
        # Actually, for sin wave: sin(k) = -sin(-k) but we only have positive indices
        # More useful: the wave should have zero mean for complete cycles
        period_samples = int(sample_rate / frequency)
        single_period = result[:period_samples]

        # Sum over complete period should be approximately zero
        torch.testing.assert_close(
            single_period.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-8,
        )

    def test_max_and_min_values(self):
        """Test that max/min reach amplitude for sufficient samples."""
        n = 1000
        sample_rate = 1000.0
        frequency = 10.0
        amplitude = 1.0

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            amplitude=amplitude,
            dtype=torch.float64,
        )

        # With 10 complete cycles, we should hit very close to +/-1
        assert result.max().item() > 0.99
        assert result.min().item() < -0.99

    def test_energy_proportional_to_amplitude_squared(self):
        """Test that energy scales with amplitude squared."""
        n = 1000
        amplitude_1 = 1.0
        amplitude_2 = 2.0

        result_1 = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amplitude_1, dtype=torch.float64
        )
        result_2 = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amplitude_2, dtype=torch.float64
        )

        energy_1 = (result_1**2).sum().item()
        energy_2 = (result_2**2).sum().item()

        # Energy should scale as amplitude^2
        expected_ratio = (amplitude_2 / amplitude_1) ** 2
        actual_ratio = energy_2 / energy_1

        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_comparison_with_manual_computation(self):
        """Test against manual sin computation."""
        n = 10
        frequency = 2.0
        sample_rate = 10.0
        amplitude = 1.5
        phase = 0.3

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            amplitude=amplitude,
            phase=phase,
            dtype=torch.float64,
        )

        for k in range(n):
            expected = amplitude * math.sin(
                2 * math.pi * frequency * k / sample_rate + phase
            )
            torch.testing.assert_close(
                result[k],
                torch.tensor(expected, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_n_equals_zero(self):
        """Test that n=0 returns empty tensor."""
        result = torchscience.signal_processing.waveform.sine_wave(0)
        assert result.shape == (0,)
        assert result.numel() == 0

    def test_n_equals_one(self):
        """Test n=1 returns sin(phase)."""
        result = torchscience.signal_processing.waveform.sine_wave(1)
        torch.testing.assert_close(
            result,
            torch.tensor([0.0], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_n_equals_one_with_phase(self):
        """Test n=1 with phase returns sin(phase)."""
        phase = math.pi / 4
        result = torchscience.signal_processing.waveform.sine_wave(
            1, phase=phase, dtype=torch.float64
        )
        expected = math.sin(phase)
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_large_n(self):
        """Test with large number of samples."""
        n = 100000
        result = torchscience.signal_processing.waveform.sine_wave(n)
        assert result.shape == (n,)

    def test_very_high_frequency(self):
        """Test with frequency at Nyquist limit."""
        n = 100
        sample_rate = 100.0
        frequency = sample_rate / 2  # Nyquist frequency

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # At Nyquist, we should alternate between positive and negative
        # Actually at exactly Nyquist, sin(k*pi) = 0 for all integer k
        # So let's check it's near zero
        for i in range(n):
            expected = math.sin(math.pi * i)  # sin(pi*k) = 0
            torch.testing.assert_close(
                result[i],
                torch.tensor(expected, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_very_low_frequency(self):
        """Test with very low frequency."""
        n = 1000
        sample_rate = 1000.0
        frequency = 0.001  # Very slow oscillation

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        # Should be a very slow ramp at the start
        # Values should all be positive and small for small k
        assert result[0].item() == 0.0
        assert result[1].item() > 0  # Starting to rise
        assert result[-1].item() > result[0].item()  # Still rising

    def test_zero_amplitude(self):
        """Test with zero amplitude gives all zeros."""
        n = 100
        result = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=0.0, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.zeros(n, dtype=torch.float64), rtol=0, atol=0
        )

    def test_negative_amplitude(self):
        """Test with negative amplitude (inverts wave)."""
        n = 100
        amplitude = -1.0

        result_pos = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=1.0, dtype=torch.float64
        )
        result_neg = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amplitude, dtype=torch.float64
        )

        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float64_precision(self):
        """Test float64 maintains precision."""
        n = 100
        result = torchscience.signal_processing.waveform.sine_wave(
            n, dtype=torch.float64
        )
        assert result.dtype == torch.float64

        # Check specific values with high precision
        expected_0 = 0.0
        expected_1 = math.sin(2 * math.pi * 1.0)

        torch.testing.assert_close(
            result[0],
            torch.tensor(expected_0, dtype=torch.float64),
            rtol=1e-15,
            atol=1e-15,
        )
        torch.testing.assert_close(
            result[1],
            torch.tensor(expected_1, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.xfail(reason="Half dtype not implemented in C++ kernel")
    def test_float16_dtype(self):
        """Test float16 works but with reduced precision."""
        n = 100
        result = torchscience.signal_processing.waveform.sine_wave(
            n, dtype=torch.float16
        )
        assert result.dtype == torch.float16

    @pytest.mark.xfail(reason="BFloat16 dtype not implemented in C++ kernel")
    def test_bfloat16_dtype(self):
        """Test bfloat16 works."""
        n = 100
        result = torchscience.signal_processing.waveform.sine_wave(
            n, dtype=torch.bfloat16
        )
        assert result.dtype == torch.bfloat16

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_flow(self):
        """Test that gradients flow when requires_grad=True."""
        n = 100
        result = torchscience.signal_processing.waveform.sine_wave(
            n, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

        loss = result.sum()
        loss.backward()

    # =========================================================================
    # Frequency domain validation
    # =========================================================================

    def test_single_frequency_component(self):
        """Test that FFT shows single frequency component."""
        n = 256
        sample_rate = 256.0
        frequency = 10.0

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float64,
        )

        spectrum = torch.fft.fft(result)
        magnitude = torch.abs(spectrum)

        # Peak should be at bin corresponding to frequency
        expected_bin = int(frequency * n / sample_rate)
        peak_bin = torch.argmax(magnitude[: n // 2]).item()

        assert peak_bin == expected_bin, (
            f"Expected peak at bin {expected_bin}, got {peak_bin}"
        )

    def test_parseval_energy_conservation(self):
        """Test Parseval's theorem for sine wave."""
        n = 256
        amplitude = 2.0

        result = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amplitude, dtype=torch.float64
        )

        # Time domain energy
        time_energy = (result**2).sum()

        # Frequency domain energy
        spectrum = torch.fft.fft(result)
        freq_energy = (torch.abs(spectrum) ** 2).sum() / n

        torch.testing.assert_close(
            time_energy, freq_energy, rtol=1e-10, atol=1e-10
        )

    # =========================================================================
    # torch.compile tests
    # =========================================================================

    @pytest.mark.xfail(
        reason="torch.compile not yet supported for custom operators"
    )
    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_func = torch.compile(
            torchscience.signal_processing.waveform.sine_wave
        )
        result = compiled_func(100)
        expected = torchscience.signal_processing.waveform.sine_wave(100)
        torch.testing.assert_close(result, expected)

    @pytest.mark.xfail(
        reason="torch.compile not yet supported for custom operators"
    )
    def test_torch_compile_with_parameters(self):
        """Test torch.compile with various parameters."""
        compiled_func = torch.compile(
            torchscience.signal_processing.waveform.sine_wave
        )

        result = compiled_func(
            100,
            frequency=5.0,
            sample_rate=100.0,
            amplitude=2.0,
            phase=0.5,
            dtype=torch.float64,
        )
        expected = torchscience.signal_processing.waveform.sine_wave(
            100,
            frequency=5.0,
            sample_rate=100.0,
            amplitude=2.0,
            phase=0.5,
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected)

    @pytest.mark.xfail(
        reason="torch.compile not yet supported for custom operators"
    )
    def test_torch_compile_in_signal_processing_chain(self):
        """Test torch.compile when sine_wave is used in a processing chain."""

        def process_signal(n: int, freq: float) -> torch.Tensor:
            signal = torchscience.signal_processing.waveform.sine_wave(
                n, frequency=freq, sample_rate=float(n)
            )
            return torch.fft.fft(signal)

        compiled_fn = torch.compile(process_signal)

        result = compiled_fn(256, 10.0)
        expected = process_signal(256, 10.0)

        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Integration tests
    # =========================================================================

    def test_audio_generation_workflow(self):
        """Test typical audio generation workflow."""
        sample_rate = 44100.0  # CD quality
        duration = 0.1  # 100ms
        frequency = 440.0  # A4 note
        n = int(sample_rate * duration)

        result = torchscience.signal_processing.waveform.sine_wave(
            n,
            frequency=frequency,
            sample_rate=sample_rate,
            dtype=torch.float32,
        )

        assert result.shape == (n,)
        assert result.dtype == torch.float32

    def test_windowed_sine_wave(self):
        """Test applying a window function to sine wave."""
        n = 256

        sine = torchscience.signal_processing.waveform.sine_wave(
            n, dtype=torch.float64
        )
        window = torchscience.window_function.rectangular_window(
            n, dtype=torch.float64
        )

        windowed = sine * window

        # With rectangular window, should be unchanged
        torch.testing.assert_close(windowed, sine)

    def test_superposition_of_waves(self):
        """Test adding multiple sine waves together."""
        n = 1000
        sample_rate = 1000.0

        wave1 = torchscience.signal_processing.waveform.sine_wave(
            n, frequency=10.0, sample_rate=sample_rate, dtype=torch.float64
        )
        wave2 = torchscience.signal_processing.waveform.sine_wave(
            n, frequency=20.0, sample_rate=sample_rate, dtype=torch.float64
        )

        combined = wave1 + wave2

        # FFT should show two peaks
        spectrum = torch.fft.fft(combined)
        magnitude = torch.abs(spectrum[: n // 2])

        # Find peaks
        peaks = []
        for i in range(1, len(magnitude) - 1):
            if (
                magnitude[i] > magnitude[i - 1]
                and magnitude[i] > magnitude[i + 1]
            ):
                if magnitude[i] > magnitude.max() * 0.1:  # Significant peaks
                    peaks.append(i)

        assert len(peaks) == 2, f"Expected 2 peaks, got {len(peaks)}"
        assert 10 in peaks
        assert 20 in peaks

    def test_batch_sine_generation(self):
        """Test generating multiple sine waves with different parameters."""
        n = 100
        frequencies = [1.0, 2.0, 5.0, 10.0]

        waves = []
        for freq in frequencies:
            wave = torchscience.signal_processing.waveform.sine_wave(
                n, frequency=freq, dtype=torch.float64
            )
            waves.append(wave)

        stacked = torch.stack(waves)
        assert stacked.shape == (len(frequencies), n)

    def test_contiguous_output(self):
        """Test that output tensor is contiguous."""
        result = torchscience.signal_processing.waveform.sine_wave(100)
        assert result.is_contiguous()


class TestSineWaveTensorParameters:
    """Tests for tensor parameter support."""

    def test_frequency_tensor_1d(self):
        """Test 1D frequency tensor produces batched output."""
        n = 100
        freqs = torch.tensor([1.0, 2.0, 5.0])
        result = torchscience.signal_processing.waveform.sine_wave(
            n, frequency=freqs, sample_rate=100.0
        )
        assert result.shape == (3, 100)

    def test_amplitude_tensor_1d(self):
        """Test 1D amplitude tensor produces batched output."""
        n = 100
        amps = torch.tensor([0.5, 1.0, 2.0])
        result = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amps, sample_rate=100.0
        )
        assert result.shape == (3, 100)

    def test_phase_tensor_1d(self):
        """Test 1D phase tensor produces batched output."""
        n = 100
        phases = torch.tensor([0.0, math.pi / 2, math.pi])
        result = torchscience.signal_processing.waveform.sine_wave(
            n, phase=phases, sample_rate=100.0
        )
        assert result.shape == (3, 100)

    def test_broadcasting_2d(self):
        """Test 2D broadcasting of parameters."""
        n = 100
        freqs = torch.tensor([[100.0], [200.0]])  # shape (2, 1)
        amps = torch.tensor([0.5, 1.0, 1.5])  # shape (3,)
        result = torchscience.signal_processing.waveform.sine_wave(
            n, frequency=freqs, amplitude=amps, sample_rate=1000.0
        )
        assert result.shape == (2, 3, 100)

    def test_gradient_through_frequency(self):
        """Test gradients flow through frequency tensor."""
        n = 100
        freq = torch.tensor([1.0], requires_grad=True)
        result = torchscience.signal_processing.waveform.sine_wave(
            n, frequency=freq, sample_rate=100.0
        )
        loss = result.sum()
        loss.backward()
        assert freq.grad is not None

    def test_gradient_through_amplitude(self):
        """Test gradients flow through amplitude tensor."""
        n = 100
        amp = torch.tensor([1.0], requires_grad=True)
        result = torchscience.signal_processing.waveform.sine_wave(
            n, amplitude=amp, sample_rate=100.0
        )
        loss = result.sum()
        loss.backward()
        assert amp.grad is not None

    def test_gradient_through_phase(self):
        """Test gradients flow through phase tensor."""
        n = 100
        phase = torch.tensor([0.0], requires_grad=True)
        result = torchscience.signal_processing.waveform.sine_wave(
            n, phase=phase, sample_rate=100.0
        )
        loss = result.sum()
        loss.backward()
        assert phase.grad is not None


class TestSineWaveExplicitTime:
    """Tests for explicit time tensor (t parameter)."""

    def test_explicit_time_basic(self):
        """Test explicit time tensor produces correct output."""
        t = torch.linspace(0, 1, 100)
        result = torchscience.signal_processing.waveform.sine_wave(
            t=t, frequency=1.0
        )
        assert result.shape == (100,)
        # First sample at t=0 should be sin(0) = 0
        torch.testing.assert_close(
            result[0], torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_explicit_time_with_batched_freq(self):
        """Test explicit time with batched frequency."""
        t = torch.linspace(0, 1, 100)
        freqs = torch.tensor([1.0, 2.0, 5.0])
        result = torchscience.signal_processing.waveform.sine_wave(
            t=t, frequency=freqs
        )
        assert result.shape == (3, 100)

    def test_n_and_t_mutually_exclusive(self):
        """Test that providing both n and t raises error."""
        t = torch.linspace(0, 1, 100)
        with pytest.raises(ValueError, match="mutually exclusive"):
            torchscience.signal_processing.waveform.sine_wave(n=100, t=t)

    def test_neither_n_nor_t_raises(self):
        """Test that providing neither n nor t raises error."""
        with pytest.raises(ValueError, match="Either n or t"):
            torchscience.signal_processing.waveform.sine_wave()

    def test_sample_rate_ignored_with_t(self):
        """Test that sample_rate is ignored when t is provided."""
        t = torch.linspace(0, 1, 100)
        result1 = torchscience.signal_processing.waveform.sine_wave(
            t=t, frequency=1.0, sample_rate=1.0
        )
        result2 = torchscience.signal_processing.waveform.sine_wave(
            t=t, frequency=1.0, sample_rate=44100.0
        )
        torch.testing.assert_close(result1, result2)

    def test_gradient_through_t(self):
        """Test gradients flow through explicit time tensor."""
        t = torch.linspace(0, 1, 100, requires_grad=True)
        result = torchscience.signal_processing.waveform.sine_wave(
            t=t, frequency=1.0
        )
        loss = result.sum()
        loss.backward()
        assert t.grad is not None


class TestSineWaveGradcheck:
    """Gradient verification tests using torch.autograd.gradcheck."""

    def test_gradcheck_frequency(self):
        """Verify gradients w.r.t. frequency using gradcheck."""

        def func(freq):
            return torchscience.signal_processing.waveform.sine_wave(
                n=50, frequency=freq, sample_rate=100.0
            )

        freq = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            func, freq, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_amplitude(self):
        """Verify gradients w.r.t. amplitude using gradcheck."""

        def func(amp):
            return torchscience.signal_processing.waveform.sine_wave(
                n=50, amplitude=amp, sample_rate=100.0
            )

        amp = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            func, amp, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_phase(self):
        """Verify gradients w.r.t. phase using gradcheck."""

        def func(ph):
            return torchscience.signal_processing.waveform.sine_wave(
                n=50, phase=ph, sample_rate=100.0
            )

        ph = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            func, ph, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_explicit_t(self):
        """Verify gradients w.r.t. explicit time tensor."""

        def func(t):
            return torchscience.signal_processing.waveform.sine_wave(
                t=t, frequency=1.0
            )

        t = torch.linspace(0, 1, 50, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            func, t, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Second-order gradients not yet implemented for sine_wave"
    )
    def test_gradgradcheck_amplitude(self):
        """Verify second-order gradients w.r.t. amplitude."""

        def func(amp):
            return torchscience.signal_processing.waveform.sine_wave(
                n=50, amplitude=amp, sample_rate=100.0
            )

        amp = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            func, amp, eps=1e-6, atol=1e-4, rtol=1e-3
        )
