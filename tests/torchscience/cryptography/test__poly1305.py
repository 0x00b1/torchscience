import torch

from torchscience.cryptography import poly1305


class TestPoly1305:
    def test_rfc7539_test_vector(self):
        # RFC 7539 Section 2.5.2 test vector
        key = torch.tensor(
            [
                0x85,
                0xD6,
                0xBE,
                0x78,
                0x57,
                0x55,
                0x6D,
                0x33,
                0x7F,
                0x44,
                0x52,
                0xFE,
                0x42,
                0xD5,
                0x06,
                0xA8,
                0x01,
                0x03,
                0x80,
                0x8A,
                0xFB,
                0x0D,
                0xB2,
                0xFD,
                0x4A,
                0xBF,
                0xF6,
                0xAF,
                0x41,
                0x49,
                0xF5,
                0x1B,
            ],
            dtype=torch.uint8,
        )
        msg = torch.tensor(
            list(b"Cryptographic Forum Research Group"), dtype=torch.uint8
        )
        expected = torch.tensor(
            [
                0xA8,
                0x06,
                0x1D,
                0xC1,
                0x30,
                0x51,
                0x36,
                0xC6,
                0xC2,
                0x2B,
                0x8B,
                0xAF,
                0x0C,
                0x01,
                0x27,
                0xA9,
            ],
            dtype=torch.uint8,
        )

        result = poly1305(msg, key)
        torch.testing.assert_close(result, expected)

    def test_empty_message(self):
        key = torch.arange(32, dtype=torch.uint8)
        msg = torch.tensor([], dtype=torch.uint8)

        result = poly1305(msg, key)
        assert result.shape == (16,)

    def test_meta_tensor(self):
        key = torch.zeros(32, dtype=torch.uint8, device="meta")
        msg = torch.zeros(100, dtype=torch.uint8, device="meta")

        result = poly1305(msg, key)
        assert result.device.type == "meta"
        assert result.shape == (16,)
