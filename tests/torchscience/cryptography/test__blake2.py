import hashlib

import torch

from torchscience.cryptography import blake2b, blake2s


class TestBLAKE2b:
    def test_empty_input(self):
        data = torch.tensor([], dtype=torch.uint8)
        result = blake2b(data)
        expected = torch.tensor(
            list(hashlib.blake2b(b"").digest()), dtype=torch.uint8
        )
        assert result.shape == (64,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = blake2b(data)
        expected = torch.tensor(
            list(hashlib.blake2b(b"abc").digest()), dtype=torch.uint8
        )
        torch.testing.assert_close(result, expected)

    def test_various_lengths(self):
        for length in [1, 55, 56, 63, 64, 65, 100, 127, 128, 129, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)
            result = blake2b(data)
            expected = torch.tensor(
                list(hashlib.blake2b(data_bytes).digest()), dtype=torch.uint8
            )
            torch.testing.assert_close(result, expected)

    def test_custom_digest_size(self):
        for digest_size in [1, 16, 32, 48, 64]:
            data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
            result = blake2b(data, digest_size=digest_size)
            expected = torch.tensor(
                list(
                    hashlib.blake2b(b"abc", digest_size=digest_size).digest()
                ),
                dtype=torch.uint8,
            )
            assert result.shape == (digest_size,)
            torch.testing.assert_close(result, expected)

    def test_keyed_hash(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        key = torch.tensor(list(b"secret"), dtype=torch.uint8)
        result = blake2b(data, key=key)
        expected = torch.tensor(
            list(hashlib.blake2b(b"abc", key=b"secret").digest()),
            dtype=torch.uint8,
        )
        torch.testing.assert_close(result, expected)

    def test_keyed_hash_custom_digest_size(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        key = torch.tensor(list(b"my_key"), dtype=torch.uint8)
        result = blake2b(data, key=key, digest_size=32)
        expected = torch.tensor(
            list(
                hashlib.blake2b(b"abc", key=b"my_key", digest_size=32).digest()
            ),
            dtype=torch.uint8,
        )
        assert result.shape == (32,)
        torch.testing.assert_close(result, expected)

    def test_keyed_hash_empty_data(self):
        data = torch.tensor([], dtype=torch.uint8)
        key = torch.tensor(list(b"secret"), dtype=torch.uint8)
        result = blake2b(data, key=key)
        expected = torch.tensor(
            list(hashlib.blake2b(b"", key=b"secret").digest()),
            dtype=torch.uint8,
        )
        torch.testing.assert_close(result, expected)

    def test_determinism(self):
        data = torch.arange(100, dtype=torch.uint8)
        torch.testing.assert_close(blake2b(data), blake2b(data))

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        key = torch.zeros(0, dtype=torch.uint8, device="meta")
        result = blake2b(data, key=key)
        assert result.device.type == "meta"
        assert result.shape == (64,)

    def test_meta_tensor_custom_digest_size(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        key = torch.zeros(0, dtype=torch.uint8, device="meta")
        result = blake2b(data, key=key, digest_size=32)
        assert result.device.type == "meta"
        assert result.shape == (32,)


class TestBLAKE2s:
    def test_empty_input(self):
        data = torch.tensor([], dtype=torch.uint8)
        result = blake2s(data)
        expected = torch.tensor(
            list(hashlib.blake2s(b"").digest()), dtype=torch.uint8
        )
        assert result.shape == (32,)
        torch.testing.assert_close(result, expected)

    def test_abc_input(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        result = blake2s(data)
        expected = torch.tensor(
            list(hashlib.blake2s(b"abc").digest()), dtype=torch.uint8
        )
        torch.testing.assert_close(result, expected)

    def test_various_lengths(self):
        for length in [1, 31, 32, 33, 63, 64, 65, 100, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)
            result = blake2s(data)
            expected = torch.tensor(
                list(hashlib.blake2s(data_bytes).digest()), dtype=torch.uint8
            )
            torch.testing.assert_close(result, expected)

    def test_custom_digest_size(self):
        for digest_size in [1, 8, 16, 24, 32]:
            data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
            result = blake2s(data, digest_size=digest_size)
            expected = torch.tensor(
                list(
                    hashlib.blake2s(b"abc", digest_size=digest_size).digest()
                ),
                dtype=torch.uint8,
            )
            assert result.shape == (digest_size,)
            torch.testing.assert_close(result, expected)

    def test_keyed_hash(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        key = torch.tensor(list(b"secret"), dtype=torch.uint8)
        result = blake2s(data, key=key)
        expected = torch.tensor(
            list(hashlib.blake2s(b"abc", key=b"secret").digest()),
            dtype=torch.uint8,
        )
        torch.testing.assert_close(result, expected)

    def test_keyed_hash_custom_digest_size(self):
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
        key = torch.tensor(list(b"my_key"), dtype=torch.uint8)
        result = blake2s(data, key=key, digest_size=16)
        expected = torch.tensor(
            list(
                hashlib.blake2s(b"abc", key=b"my_key", digest_size=16).digest()
            ),
            dtype=torch.uint8,
        )
        assert result.shape == (16,)
        torch.testing.assert_close(result, expected)

    def test_keyed_hash_empty_data(self):
        data = torch.tensor([], dtype=torch.uint8)
        key = torch.tensor(list(b"secret"), dtype=torch.uint8)
        result = blake2s(data, key=key)
        expected = torch.tensor(
            list(hashlib.blake2s(b"", key=b"secret").digest()),
            dtype=torch.uint8,
        )
        torch.testing.assert_close(result, expected)

    def test_determinism(self):
        data = torch.arange(100, dtype=torch.uint8)
        torch.testing.assert_close(blake2s(data), blake2s(data))

    def test_meta_tensor(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        key = torch.zeros(0, dtype=torch.uint8, device="meta")
        result = blake2s(data, key=key)
        assert result.device.type == "meta"
        assert result.shape == (32,)

    def test_meta_tensor_custom_digest_size(self):
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        key = torch.zeros(0, dtype=torch.uint8, device="meta")
        result = blake2s(data, key=key, digest_size=16)
        assert result.device.type == "meta"
        assert result.shape == (16,)
