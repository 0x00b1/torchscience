import pytest
import torch

from torchscience.encryption import chacha20


class TestChaCha20:
    def test_rfc8439_test_vector(self):
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.tensor(
            [
                0x00,
                0x00,
                0x00,
                0x09,
                0x00,
                0x00,
                0x00,
                0x4A,
                0x00,
                0x00,
                0x00,
                0x00,
            ],
            dtype=torch.uint8,
        )
        output = chacha20(key, nonce, num_bytes=64, counter=1)
        expected_start = torch.tensor(
            [
                0x10,
                0xF1,
                0xE7,
                0xE4,
                0xD1,
                0x3B,
                0x59,
                0x15,
                0x50,
                0x0F,
                0xDD,
                0x1F,
                0xA3,
                0x20,
                0x71,
                0xC4,
            ],
            dtype=torch.uint8,
        )
        assert output.shape == (64,)
        torch.testing.assert_close(output[:16], expected_start)

    def test_key_shape_validation(self):
        with pytest.raises(RuntimeError, match="key"):
            chacha20(
                torch.zeros(16, dtype=torch.uint8),
                torch.zeros(12, dtype=torch.uint8),
                num_bytes=64,
            )

    def test_nonce_shape_validation(self):
        with pytest.raises(RuntimeError, match="nonce"):
            chacha20(
                torch.zeros(32, dtype=torch.uint8),
                torch.zeros(8, dtype=torch.uint8),
                num_bytes=64,
            )

    def test_determinism(self):
        key, nonce = (
            torch.arange(32, dtype=torch.uint8),
            torch.zeros(12, dtype=torch.uint8),
        )
        torch.testing.assert_close(
            chacha20(key, nonce, num_bytes=128),
            chacha20(key, nonce, num_bytes=128),
        )

    def test_different_counters_different_output(self):
        key, nonce = (
            torch.arange(32, dtype=torch.uint8),
            torch.zeros(12, dtype=torch.uint8),
        )
        assert not torch.equal(
            chacha20(key, nonce, num_bytes=64, counter=0),
            chacha20(key, nonce, num_bytes=64, counter=1),
        )

    def test_output_length(self):
        key, nonce = (
            torch.arange(32, dtype=torch.uint8),
            torch.zeros(12, dtype=torch.uint8),
        )
        for length in [1, 63, 64, 65, 128, 1000]:
            assert chacha20(key, nonce, num_bytes=length).shape == (length,)

    def test_meta_tensor(self):
        out = chacha20(
            torch.zeros(32, dtype=torch.uint8, device="meta"),
            torch.zeros(12, dtype=torch.uint8, device="meta"),
            num_bytes=256,
        )
        assert out.device.type == "meta" and out.shape == (256,)
