import hashlib

import torch
from torchscience.encryption import sha256


class TestSHA256:
    def test_empty_input(self):
        data = torch.tensor([], dtype=torch.uint8)
        result = sha256(data)
        expected = torch.tensor(
            list(hashlib.sha256(b"").digest()), dtype=torch.uint8
        )
        assert result.shape == (32,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = sha256(data)
        expected = torch.tensor(
            list(hashlib.sha256(b"abc").digest()), dtype=torch.uint8
        )
        torch.testing.assert_close(result, expected)

    def test_various_lengths(self):
        for length in [1, 55, 56, 63, 64, 65, 100, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)
            result = sha256(data)
            expected = torch.tensor(
                list(hashlib.sha256(data_bytes).digest()), dtype=torch.uint8
            )
            torch.testing.assert_close(result, expected)

    def test_determinism(self):
        data = torch.arange(100, dtype=torch.uint8)
        torch.testing.assert_close(sha256(data), sha256(data))

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        result = sha256(data)
        assert result.device.type == "meta"
        assert result.shape == (32,)
