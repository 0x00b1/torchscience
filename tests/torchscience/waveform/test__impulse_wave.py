# tests/torchscience/signal_processing/waveform/test__impulse_wave.py
import torch

import torchscience.waveform


class TestImpulseWave:
    def test_basic_shape(self):
        result = torchscience.waveform.impulse_wave(n=100)
        assert result.shape == (100,)

    def test_impulse_at_origin(self):
        result = torchscience.waveform.impulse_wave(
            n=100, position=0, amplitude=1.0
        )
        assert result[0] == 1.0
        assert result[1:].sum() == 0.0

    def test_impulse_at_position(self):
        result = torchscience.waveform.impulse_wave(
            n=100, position=50, amplitude=2.0
        )
        assert result[50] == 2.0
        assert result[:50].sum() == 0.0
        assert result[51:].sum() == 0.0

    def test_batched_positions(self):
        positions = torch.tensor([10, 20, 30])
        result = torchscience.waveform.impulse_wave(n=100, position=positions)
        assert result.shape == (3, 100)
        assert result[0, 10] == 1.0
        assert result[1, 20] == 1.0
        assert result[2, 30] == 1.0
