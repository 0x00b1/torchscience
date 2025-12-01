# tests/torchscience/signal_processing/waveform/test__ramp_wave.py
import torch

import torchscience.signal_processing.waveform


class TestRampWave:
    def test_basic_shape(self):
        result = torchscience.signal_processing.waveform.ramp_wave(n=100)
        assert result.shape == (100,)

    def test_ramp_at_origin(self):
        result = torchscience.signal_processing.waveform.ramp_wave(
            n=100, position=0, slope=1.0
        )
        expected = torch.arange(100, dtype=result.dtype)
        torch.testing.assert_close(result, expected)

    def test_ramp_at_position(self):
        result = torchscience.signal_processing.waveform.ramp_wave(
            n=100, position=50, slope=2.0
        )
        assert (result[:50] == 0.0).all()
        expected_ramp = torch.arange(50, dtype=result.dtype) * 2.0
        torch.testing.assert_close(result[50:], expected_ramp)

    def test_batched_positions(self):
        positions = torch.tensor([10, 20, 30])
        result = torchscience.signal_processing.waveform.ramp_wave(
            n=100, position=positions
        )
        assert result.shape == (3, 100)
