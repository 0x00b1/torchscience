import torch

from torchscience.cryptography import (
    hkdf_expand_sha256,
    hkdf_extract_sha256,
    hkdf_sha256,
)


class TestHKDFExtract:
    def test_rfc5869_test_case_1_extract(self):
        """RFC 5869 Test Case 1 - Extract phase."""
        ikm = torch.tensor([0x0B] * 22, dtype=torch.uint8)
        salt = torch.tensor(
            [
                0x00,
                0x01,
                0x02,
                0x03,
                0x04,
                0x05,
                0x06,
                0x07,
                0x08,
                0x09,
                0x0A,
                0x0B,
                0x0C,
            ],
            dtype=torch.uint8,
        )

        expected_prk = torch.tensor(
            [
                0x07,
                0x77,
                0x09,
                0x36,
                0x2C,
                0x2E,
                0x32,
                0xDF,
                0x0D,
                0xDC,
                0x3F,
                0x0D,
                0xC4,
                0x7B,
                0xBA,
                0x63,
                0x90,
                0xB6,
                0xC7,
                0x3B,
                0xB5,
                0x0F,
                0x9C,
                0x31,
                0x22,
                0xEC,
                0x84,
                0x4A,
                0xD7,
                0xC2,
                0xB3,
                0xE5,
            ],
            dtype=torch.uint8,
        )

        prk = hkdf_extract_sha256(salt, ikm)
        torch.testing.assert_close(prk, expected_prk)

    def test_empty_salt(self):
        """Test extract with empty salt (uses zero-filled default)."""
        ikm = torch.tensor([0x0B] * 22, dtype=torch.uint8)
        salt = torch.tensor([], dtype=torch.uint8)

        prk = hkdf_extract_sha256(salt, ikm)
        assert prk.shape == (32,)


class TestHKDFExpand:
    def test_rfc5869_test_case_1_expand(self):
        """RFC 5869 Test Case 1 - Expand phase."""
        prk = torch.tensor(
            [
                0x07,
                0x77,
                0x09,
                0x36,
                0x2C,
                0x2E,
                0x32,
                0xDF,
                0x0D,
                0xDC,
                0x3F,
                0x0D,
                0xC4,
                0x7B,
                0xBA,
                0x63,
                0x90,
                0xB6,
                0xC7,
                0x3B,
                0xB5,
                0x0F,
                0x9C,
                0x31,
                0x22,
                0xEC,
                0x84,
                0x4A,
                0xD7,
                0xC2,
                0xB3,
                0xE5,
            ],
            dtype=torch.uint8,
        )
        info = torch.tensor(
            [0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9],
            dtype=torch.uint8,
        )
        output_len = 42

        expected_okm = torch.tensor(
            [
                0x3C,
                0xB2,
                0x5F,
                0x25,
                0xFA,
                0xAC,
                0xD5,
                0x7A,
                0x90,
                0x43,
                0x4F,
                0x64,
                0xD0,
                0x36,
                0x2F,
                0x2A,
                0x2D,
                0x2D,
                0x0A,
                0x90,
                0xCF,
                0x1A,
                0x5A,
                0x4C,
                0x5D,
                0xB0,
                0x2D,
                0x56,
                0xEC,
                0xC4,
                0xC5,
                0xBF,
                0x34,
                0x00,
                0x72,
                0x08,
                0xD5,
                0xB8,
                0x87,
                0x18,
                0x58,
                0x65,
            ],
            dtype=torch.uint8,
        )

        okm = hkdf_expand_sha256(prk, info, output_len)
        torch.testing.assert_close(okm, expected_okm)


class TestHKDFCombined:
    def test_rfc5869_test_case_1(self):
        """RFC 5869 Test Case 1 - Combined extract and expand."""
        ikm = torch.tensor([0x0B] * 22, dtype=torch.uint8)
        salt = torch.tensor(
            [
                0x00,
                0x01,
                0x02,
                0x03,
                0x04,
                0x05,
                0x06,
                0x07,
                0x08,
                0x09,
                0x0A,
                0x0B,
                0x0C,
            ],
            dtype=torch.uint8,
        )
        info = torch.tensor(
            [0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9],
            dtype=torch.uint8,
        )
        output_len = 42

        expected_okm = torch.tensor(
            [
                0x3C,
                0xB2,
                0x5F,
                0x25,
                0xFA,
                0xAC,
                0xD5,
                0x7A,
                0x90,
                0x43,
                0x4F,
                0x64,
                0xD0,
                0x36,
                0x2F,
                0x2A,
                0x2D,
                0x2D,
                0x0A,
                0x90,
                0xCF,
                0x1A,
                0x5A,
                0x4C,
                0x5D,
                0xB0,
                0x2D,
                0x56,
                0xEC,
                0xC4,
                0xC5,
                0xBF,
                0x34,
                0x00,
                0x72,
                0x08,
                0xD5,
                0xB8,
                0x87,
                0x18,
                0x58,
                0x65,
            ],
            dtype=torch.uint8,
        )

        okm = hkdf_sha256(ikm, salt, info, output_len)
        torch.testing.assert_close(okm, expected_okm)

    def test_rfc5869_test_case_2(self):
        """RFC 5869 Test Case 2 - Longer inputs/outputs."""
        ikm = torch.tensor(list(range(0x00, 0x50)), dtype=torch.uint8)
        salt = torch.tensor(list(range(0x60, 0xB0)), dtype=torch.uint8)
        info = torch.tensor(list(range(0xB0, 0x100)), dtype=torch.uint8)
        output_len = 82

        expected_okm = torch.tensor(
            [
                0xB1,
                0x1E,
                0x39,
                0x8D,
                0xC8,
                0x03,
                0x27,
                0xA1,
                0xC8,
                0xE7,
                0xF7,
                0x8C,
                0x59,
                0x6A,
                0x49,
                0x34,
                0x4F,
                0x01,
                0x2E,
                0xDA,
                0x2D,
                0x4E,
                0xFA,
                0xD8,
                0xA0,
                0x50,
                0xCC,
                0x4C,
                0x19,
                0xAF,
                0xA9,
                0x7C,
                0x59,
                0x04,
                0x5A,
                0x99,
                0xCA,
                0xC7,
                0x82,
                0x72,
                0x71,
                0xCB,
                0x41,
                0xC6,
                0x5E,
                0x59,
                0x0E,
                0x09,
                0xDA,
                0x32,
                0x75,
                0x60,
                0x0C,
                0x2F,
                0x09,
                0xB8,
                0x36,
                0x77,
                0x93,
                0xA9,
                0xAC,
                0xA3,
                0xDB,
                0x71,
                0xCC,
                0x30,
                0xC5,
                0x81,
                0x79,
                0xEC,
                0x3E,
                0x87,
                0xC1,
                0x4C,
                0x01,
                0xD5,
                0xC1,
                0xF3,
                0x43,
                0x4F,
                0x1D,
                0x87,
            ],
            dtype=torch.uint8,
        )

        okm = hkdf_sha256(ikm, salt, info, output_len)
        torch.testing.assert_close(okm, expected_okm)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        ikm = torch.zeros(32, dtype=torch.uint8, device="meta")
        salt = torch.zeros(16, dtype=torch.uint8, device="meta")
        info = torch.zeros(10, dtype=torch.uint8, device="meta")

        okm = hkdf_sha256(ikm, salt, info, 64)

        assert okm.device.type == "meta"
        assert okm.shape == (64,)
