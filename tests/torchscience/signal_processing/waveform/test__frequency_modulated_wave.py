import math

import pytest
import torch

import torchscience.signal_processing.waveform


class TestFrequencyModulatedWave:
    def test_basic_shape(self):
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=2.0,
                sample_rate=1000.0,
            )
        )
        assert result.shape == (1000,)

    def test_sinusoidal_modulation(self):
        # FM with sinusoidal modulator
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=10.0,
                modulation_index=1.0,
                sample_rate=1000.0,
            )
        )
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_arbitrary_modulating_signal(self):
        # Use a custom modulating waveform
        t = torch.linspace(0, 1, 1000)
        modulating_signal = torch.sin(2 * math.pi * 5 * t)  # 5 Hz modulator
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                t=t,
                carrier_frequency=100.0,
                modulating_signal=modulating_signal,
                modulation_index=2.0,
            )
        )
        assert result.shape == (1000,)

    def test_batched_carrier_frequencies(self):
        carriers = torch.tensor([100.0, 200.0, 300.0])
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=carriers,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
            )
        )
        assert result.shape == (3, 1000)

    def test_explicit_time_tensor(self):
        t = torch.linspace(0, 1, 500)
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                t=t,
                carrier_frequency=50.0,
                modulator_frequency=2.0,
                modulation_index=1.5,
            )
        )
        assert result.shape == (500,)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_amplitude_scaling(self):
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                amplitude=2.5,
            )
        )
        assert result.max() <= 2.5
        assert result.min() >= -2.5

    def test_batched_amplitudes(self):
        amplitudes = torch.tensor([1.0, 2.0, 3.0])
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                amplitude=amplitudes,
            )
        )
        assert result.shape == (3, 1000)

    def test_phase_offset(self):
        result_no_phase = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                phase=0.0,
            )
        )
        result_with_phase = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                phase=math.pi / 2,
            )
        )
        # Results should differ due to phase offset
        assert not torch.allclose(result_no_phase, result_with_phase)

    def test_mutual_exclusivity_n_t(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                t=torch.linspace(0, 1, 1000),
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
            )

    def test_neither_n_nor_t(self):
        with pytest.raises(ValueError, match="Either n or t"):
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
            )

    def test_mutual_exclusivity_modulation_modes(self):
        t = torch.linspace(0, 1, 1000)
        modulating_signal = torch.sin(2 * math.pi * 5 * t)
        with pytest.raises(ValueError, match="mutually exclusive"):
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulating_signal=modulating_signal,
                modulation_index=1.0,
                sample_rate=1000.0,
            )

    def test_neither_modulation_mode(self):
        with pytest.raises(ValueError, match="Either modulator_frequency"):
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulation_index=1.0,
                sample_rate=1000.0,
            )

    def test_zero_modulation_index(self):
        # With zero modulation index, should be a pure carrier
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=0.0,
                sample_rate=1000.0,
            )
        )
        # Compare with pure cosine at carrier frequency
        t = torch.arange(1000, dtype=torch.float32) / 1000.0
        expected = torch.cos(2 * math.pi * 100.0 * t)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_arbitrary_modulation_with_constant_signal(self):
        # With constant zero modulating signal, should be a pure carrier
        t = torch.linspace(0, 1, 1000, dtype=torch.float32)
        modulating_signal = torch.zeros_like(t)
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                t=t,
                carrier_frequency=100.0,
                modulating_signal=modulating_signal,
                modulation_index=2.0,
            )
        )
        # Should be a pure cosine at carrier frequency
        expected = torch.cos(2 * math.pi * 100.0 * t)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batched_modulating_signal(self):
        t = torch.linspace(0, 1, 1000)
        # Create batched modulating signals (3, 1000)
        modulating_signal = torch.stack(
            [
                torch.sin(2 * math.pi * 2 * t),  # 2 Hz
                torch.sin(2 * math.pi * 5 * t),  # 5 Hz
                torch.sin(2 * math.pi * 10 * t),  # 10 Hz
            ]
        )
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                t=t,
                carrier_frequency=100.0,
                modulating_signal=modulating_signal,
                modulation_index=1.0,
            )
        )
        assert result.shape == (3, 1000)

    def test_dtype_float64(self):
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                dtype=torch.float64,
            )
        )
        assert result.dtype == torch.float64

    def test_empty_result_n_zero(self):
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=0,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
            )
        )
        assert result.shape == (0,)

    def test_requires_grad(self):
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=5.0,
                modulation_index=1.0,
                sample_rate=1000.0,
                requires_grad=True,
            )
        )
        assert result.requires_grad

    def test_batched_modulator_frequency(self):
        modulators = torch.tensor([2.0, 5.0, 10.0])
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=100.0,
                modulator_frequency=modulators,
                modulation_index=1.0,
                sample_rate=1000.0,
            )
        )
        assert result.shape == (3, 1000)

    def test_broadcast_carrier_and_modulator(self):
        # (2,) carriers broadcast with (3,) modulators should fail or broadcast appropriately
        carriers = torch.tensor([100.0, 200.0]).unsqueeze(-1)  # (2, 1)
        modulators = torch.tensor([2.0, 5.0, 10.0])  # (3,)
        result = (
            torchscience.signal_processing.waveform.frequency_modulated_wave(
                n=1000,
                carrier_frequency=carriers,
                modulator_frequency=modulators,
                modulation_index=1.0,
                sample_rate=1000.0,
            )
        )
        assert result.shape == (2, 3, 1000)
