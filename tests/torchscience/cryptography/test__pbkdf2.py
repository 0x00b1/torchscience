import torch

from torchscience.cryptography import pbkdf2_sha256


class TestPBKDF2:
    def test_pbkdf2_sha256_test_vector_1(self):
        """PBKDF2-HMAC-SHA256: 'password', 'salt', 1 iteration, 20 bytes."""
        password = torch.tensor(list(b"password"), dtype=torch.uint8)
        salt = torch.tensor(list(b"salt"), dtype=torch.uint8)
        iterations = 1
        output_len = 20

        expected = torch.tensor(
            [
                0x12,
                0x0F,
                0xB6,
                0xCF,
                0xFC,
                0xF8,
                0xB3,
                0x2C,
                0x43,
                0xE7,
                0x22,
                0x52,
                0x56,
                0xC4,
                0xF8,
                0x37,
                0xA8,
                0x65,
                0x48,
                0xC9,
            ],
            dtype=torch.uint8,
        )

        result = pbkdf2_sha256(password, salt, iterations, output_len)
        torch.testing.assert_close(result, expected)

    def test_pbkdf2_sha256_test_vector_2(self):
        """PBKDF2-HMAC-SHA256: 'password', 'salt', 2 iterations, 20 bytes."""
        password = torch.tensor(list(b"password"), dtype=torch.uint8)
        salt = torch.tensor(list(b"salt"), dtype=torch.uint8)
        iterations = 2
        output_len = 20

        expected = torch.tensor(
            [
                0xAE,
                0x4D,
                0x0C,
                0x95,
                0xAF,
                0x6B,
                0x46,
                0xD3,
                0x2D,
                0x0A,
                0xDF,
                0xF9,
                0x28,
                0xF0,
                0x6D,
                0xD0,
                0x2A,
                0x30,
                0x3F,
                0x8E,
            ],
            dtype=torch.uint8,
        )

        result = pbkdf2_sha256(password, salt, iterations, output_len)
        torch.testing.assert_close(result, expected)

    def test_pbkdf2_sha256_test_vector_3(self):
        """PBKDF2-HMAC-SHA256: 'password', 'salt', 4096 iterations, 20 bytes."""
        password = torch.tensor(list(b"password"), dtype=torch.uint8)
        salt = torch.tensor(list(b"salt"), dtype=torch.uint8)
        iterations = 4096
        output_len = 20

        expected = torch.tensor(
            [
                0xC5,
                0xE4,
                0x78,
                0xD5,
                0x92,
                0x88,
                0xC8,
                0x41,
                0xAA,
                0x53,
                0x0D,
                0xB6,
                0x84,
                0x5C,
                0x4C,
                0x8D,
                0x96,
                0x28,
                0x93,
                0xA0,
            ],
            dtype=torch.uint8,
        )

        result = pbkdf2_sha256(password, salt, iterations, output_len)
        torch.testing.assert_close(result, expected)

    def test_pbkdf2_sha256_test_vector_4(self):
        """PBKDF2-HMAC-SHA256: longer password and salt, 4096 iterations."""
        password = torch.tensor(
            list(b"passwordPASSWORDpassword"), dtype=torch.uint8
        )
        salt = torch.tensor(
            list(b"saltSALTsaltSALTsaltSALTsaltSALTsalt"), dtype=torch.uint8
        )
        iterations = 4096
        output_len = 25

        expected = torch.tensor(
            [
                0x34,
                0x8C,
                0x89,
                0xDB,
                0xCB,
                0xD3,
                0x2B,
                0x2F,
                0x32,
                0xD8,
                0x14,
                0xB8,
                0x11,
                0x6E,
                0x84,
                0xCF,
                0x2B,
                0x17,
                0x34,
                0x7E,
                0xBC,
                0x18,
                0x00,
                0x18,
                0x1C,
            ],
            dtype=torch.uint8,
        )

        result = pbkdf2_sha256(password, salt, iterations, output_len)
        torch.testing.assert_close(result, expected)

    def test_determinism(self):
        """Test that PBKDF2 produces consistent results."""
        password = torch.tensor(list(b"test"), dtype=torch.uint8)
        salt = torch.tensor(list(b"salt"), dtype=torch.uint8)

        result1 = pbkdf2_sha256(password, salt, 100, 32)
        result2 = pbkdf2_sha256(password, salt, 100, 32)

        torch.testing.assert_close(result1, result2)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        password = torch.zeros(8, dtype=torch.uint8, device="meta")
        salt = torch.zeros(16, dtype=torch.uint8, device="meta")

        result = pbkdf2_sha256(password, salt, 1000, 64)

        assert result.device.type == "meta"
        assert result.shape == (64,)
