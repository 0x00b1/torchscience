import torch
import torchscience.signal_processing.waveform


class TestPulseWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.pulse_wave(
            n=100, sample_rate=100.0
        )
        assert result.shape == (100,)

    def test_duty_cycle_25_percent(self):
        n = 1000
        result = torchscience.signal_processing.waveform.pulse_wave(
            n=n,
            frequency=10.0,
            sample_rate=1000.0,
            duty_cycle=0.25,
            dtype=torch.float64,
        )
        positive_frac = (result > 0).float().mean()
        assert 0.20 < positive_frac < 0.30

    def test_duty_cycle_75_percent(self):
        n = 1000
        result = torchscience.signal_processing.waveform.pulse_wave(
            n=n,
            frequency=10.0,
            sample_rate=1000.0,
            duty_cycle=0.75,
            dtype=torch.float64,
        )
        positive_frac = (result > 0).float().mean()
        assert 0.70 < positive_frac < 0.80

    def test_batched_duty_cycles(self):
        duty_cycles = torch.tensor([0.25, 0.5, 0.75])
        result = torchscience.signal_processing.waveform.pulse_wave(
            n=1000,
            frequency=10.0,
            sample_rate=1000.0,
            duty_cycle=duty_cycles,
        )
        assert result.shape == (3, 1000)
