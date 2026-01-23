import hashlib

import torch

from torchscience.cryptography import keccak256, sha3_256, sha3_512


class TestSHA3_256:
    def test_empty_input(self):
        data = torch.tensor([], dtype=torch.uint8)
        result = sha3_256(data)
        expected = torch.tensor(
            list(hashlib.sha3_256(b"").digest()), dtype=torch.uint8
        )
        assert result.shape == (32,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = sha3_256(data)
        expected = torch.tensor(
            list(hashlib.sha3_256(b"abc").digest()), dtype=torch.uint8
        )
        torch.testing.assert_close(result, expected)

    def test_various_lengths(self):
        for length in [1, 55, 56, 63, 64, 65, 100, 135, 136, 137, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)
            result = sha3_256(data)
            expected = torch.tensor(
                list(hashlib.sha3_256(data_bytes).digest()), dtype=torch.uint8
            )
            torch.testing.assert_close(result, expected)

    def test_determinism(self):
        data = torch.arange(100, dtype=torch.uint8)
        torch.testing.assert_close(sha3_256(data), sha3_256(data))

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        result = sha3_256(data)
        assert result.device.type == "meta"
        assert result.shape == (32,)


class TestSHA3_512:
    def test_empty_input(self):
        data = torch.tensor([], dtype=torch.uint8)
        result = sha3_512(data)
        expected = torch.tensor(
            list(hashlib.sha3_512(b"").digest()), dtype=torch.uint8
        )
        assert result.shape == (64,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = sha3_512(data)
        expected = torch.tensor(
            list(hashlib.sha3_512(b"abc").digest()), dtype=torch.uint8
        )
        torch.testing.assert_close(result, expected)

    def test_various_lengths(self):
        for length in [1, 71, 72, 73, 100, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)
            result = sha3_512(data)
            expected = torch.tensor(
                list(hashlib.sha3_512(data_bytes).digest()), dtype=torch.uint8
            )
            torch.testing.assert_close(result, expected)

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        result = sha3_512(data)
        assert result.device.type == "meta"
        assert result.shape == (64,)


class TestKeccak256:
    def test_empty_input(self):
        # Keccak-256 empty string hash (different from SHA3-256)
        data = torch.tensor([], dtype=torch.uint8)
        result = keccak256(data)
        # Known Keccak-256 hash of empty string
        expected_hex = (
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        )
        expected = torch.tensor(
            [int(expected_hex[i : i + 2], 16) for i in range(0, 64, 2)],
            dtype=torch.uint8,
        )
        assert result.shape == (32,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = keccak256(data)
        # Known Keccak-256 hash of "abc"
        expected_hex = (
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        )
        expected = torch.tensor(
            [int(expected_hex[i : i + 2], 16) for i in range(0, 64, 2)],
            dtype=torch.uint8,
        )
        torch.testing.assert_close(result, expected)

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        result = keccak256(data)
        assert result.device.type == "meta"
        assert result.shape == (32,)
