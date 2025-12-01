# tests/torchscience/signal_processing/waveform/test__step_wave.py
import torch

import torchscience.signal_processing.waveform


class TestStepWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.step_wave(n=100)
        assert result.shape == (100,)

    def test_step_at_origin(self):
        result = torchscience.signal_processing.waveform.step_wave(
            n=100, position=0, amplitude=1.0
        )
        assert (result == 1.0).all()

    def test_step_at_position(self):
        result = torchscience.signal_processing.waveform.step_wave(
            n=100, position=50, amplitude=2.0
        )
        assert (result[:50] == 0.0).all()
        assert (result[50:] == 2.0).all()

    def test_batched_positions(self):
        positions = torch.tensor([10, 20, 30])
        result = torchscience.signal_processing.waveform.step_wave(
            n=100, position=positions
        )
        assert result.shape == (3, 100)
