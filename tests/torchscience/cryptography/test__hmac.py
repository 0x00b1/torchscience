import hashlib
import hmac as py_hmac

import torch
from torchscience.encryption import hmac_sha256


class TestHMAC:
    def test_rfc4231_test_case_1(self):
        """RFC 4231 test case 1."""
        key = torch.tensor([0x0B] * 20, dtype=torch.uint8)
        data = torch.tensor(list(b"Hi There"), dtype=torch.uint8)
        result = hmac_sha256(key, data)
        expected = py_hmac.new(
            bytes([0x0B] * 20), b"Hi There", hashlib.sha256
        ).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)
        torch.testing.assert_close(result, expected_tensor)

    def test_rfc4231_test_case_2(self):
        """RFC 4231 test case 2: key = 'Jefe'."""
        key = torch.tensor(list(b"Jefe"), dtype=torch.uint8)
        data = torch.tensor(
            list(b"what do ya want for nothing?"), dtype=torch.uint8
        )
        result = hmac_sha256(key, data)
        expected = py_hmac.new(
            b"Jefe", b"what do ya want for nothing?", hashlib.sha256
        ).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)
        torch.testing.assert_close(result, expected_tensor)

    def test_long_key(self):
        """Key longer than block size is hashed."""
        key = torch.arange(100, dtype=torch.uint8)
        data = torch.tensor(list(b"test message"), dtype=torch.uint8)
        result = hmac_sha256(key, data)
        expected = py_hmac.new(
            bytes(range(100)), b"test message", hashlib.sha256
        ).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)
        torch.testing.assert_close(result, expected_tensor)

    def test_empty_message(self):
        """HMAC of empty message."""
        key = torch.tensor(list(b"key"), dtype=torch.uint8)
        data = torch.tensor([], dtype=torch.uint8)
        result = hmac_sha256(key, data)
        expected = py_hmac.new(b"key", b"", hashlib.sha256).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)
        torch.testing.assert_close(result, expected_tensor)

    def test_output_shape(self):
        """Output is always 32 bytes."""
        key = torch.zeros(16, dtype=torch.uint8)
        data = torch.zeros(100, dtype=torch.uint8)
        result = hmac_sha256(key, data)
        assert result.shape == (32,)
        assert result.dtype == torch.uint8
