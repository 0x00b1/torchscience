import torch

from torchscience.cryptography import (
    ed25519_keypair,
    ed25519_sign,
    ed25519_verify,
)


class TestEd25519:
    def test_rfc8032_test_vector_1(self):
        """RFC 8032 Section 7.1 - Test Vector 1 (empty message)"""
        seed = torch.tensor(
            [
                0x9D,
                0x61,
                0xB1,
                0x9D,
                0xEF,
                0xFD,
                0x5A,
                0x60,
                0xBA,
                0x84,
                0x4A,
                0xF4,
                0x92,
                0xEC,
                0x2C,
                0xC4,
                0x44,
                0x49,
                0xC5,
                0x69,
                0x7B,
                0x32,
                0x69,
                0x19,
                0x70,
                0x3B,
                0xAC,
                0x03,
                0x1C,
                0xAE,
                0x7F,
                0x60,
            ],
            dtype=torch.uint8,
        )

        expected_public = torch.tensor(
            [
                0xD7,
                0x5A,
                0x98,
                0x01,
                0x82,
                0xB1,
                0x0A,
                0xB7,
                0xD5,
                0x4B,
                0xFE,
                0xD3,
                0xC9,
                0x64,
                0x07,
                0x3A,
                0x0E,
                0xE1,
                0x72,
                0xF3,
                0xDA,
                0xA6,
                0x23,
                0x25,
                0xAF,
                0x02,
                0x1A,
                0x68,
                0xF7,
                0x07,
                0x51,
                0x1A,
            ],
            dtype=torch.uint8,
        )

        expected_signature = torch.tensor(
            [
                0xE5,
                0x56,
                0x43,
                0x00,
                0xC3,
                0x60,
                0xAC,
                0x72,
                0x90,
                0x86,
                0xE2,
                0xCC,
                0x80,
                0x6E,
                0x82,
                0x8A,
                0x84,
                0x87,
                0x7F,
                0x1E,
                0xB8,
                0xE5,
                0xD9,
                0x74,
                0xD8,
                0x73,
                0xE0,
                0x65,
                0x22,
                0x49,
                0x01,
                0x55,
                0x5F,
                0xB8,
                0x82,
                0x15,
                0x90,
                0xA3,
                0x3B,
                0xAC,
                0xC6,
                0x1E,
                0x39,
                0x70,
                0x1C,
                0xF9,
                0xB4,
                0x6B,
                0xD2,
                0x5B,
                0xF5,
                0xF0,
                0x59,
                0x5B,
                0xBE,
                0x24,
                0x65,
                0x51,
                0x41,
                0x43,
                0x8E,
                0x7A,
                0x10,
                0x0B,
            ],
            dtype=torch.uint8,
        )

        private_key, public_key = ed25519_keypair(seed)
        torch.testing.assert_close(public_key, expected_public)

        # Sign empty message
        message = torch.tensor([], dtype=torch.uint8)
        signature = ed25519_sign(private_key, message)
        torch.testing.assert_close(signature, expected_signature)

        # Verify signature
        valid = ed25519_verify(public_key, message, signature)
        assert valid.item() == 1

    def test_rfc8032_test_vector_2(self):
        """RFC 8032 Section 7.1 - Test Vector 2 (1-byte message)"""
        seed = torch.tensor(
            [
                0x4C,
                0xCD,
                0x08,
                0x9B,
                0x28,
                0xFF,
                0x96,
                0xDA,
                0x9D,
                0xB6,
                0xC3,
                0x46,
                0xEC,
                0x11,
                0x4E,
                0x0F,
                0x5B,
                0x8A,
                0x31,
                0x9F,
                0x35,
                0xAB,
                0xA6,
                0x24,
                0xDA,
                0x8C,
                0xF6,
                0xED,
                0x4F,
                0xB8,
                0xA6,
                0xFB,
            ],
            dtype=torch.uint8,
        )

        message = torch.tensor([0x72], dtype=torch.uint8)  # "r"

        expected_signature = torch.tensor(
            [
                0x92,
                0xA0,
                0x09,
                0xA9,
                0xF0,
                0xD4,
                0xCA,
                0xB8,
                0x72,
                0x0E,
                0x82,
                0x0B,
                0x5F,
                0x64,
                0x25,
                0x40,
                0xA2,
                0xB2,
                0x7B,
                0x54,
                0x16,
                0x50,
                0x3F,
                0x8F,
                0xB3,
                0x76,
                0x22,
                0x23,
                0xEB,
                0xDB,
                0x69,
                0xDA,
                0x08,
                0x5A,
                0xC1,
                0xE4,
                0x3E,
                0x15,
                0x99,
                0x6E,
                0x45,
                0x8F,
                0x36,
                0x13,
                0xD0,
                0xF1,
                0x1D,
                0x8C,
                0x38,
                0x7B,
                0x2E,
                0xAE,
                0xB4,
                0x30,
                0x2A,
                0xEE,
                0xB0,
                0x0D,
                0x29,
                0x16,
                0x12,
                0xBB,
                0x0C,
                0x00,
            ],
            dtype=torch.uint8,
        )

        private_key, public_key = ed25519_keypair(seed)
        signature = ed25519_sign(private_key, message)
        torch.testing.assert_close(signature, expected_signature)

        valid = ed25519_verify(public_key, message, signature)
        assert valid.item() == 1

    def test_sign_verify_roundtrip(self):
        """Test basic sign/verify roundtrip"""
        seed = torch.arange(32, dtype=torch.uint8)
        private_key, public_key = ed25519_keypair(seed)

        message = torch.tensor(list(b"Hello, World!"), dtype=torch.uint8)
        signature = ed25519_sign(private_key, message)

        assert signature.shape == (64,)

        valid = ed25519_verify(public_key, message, signature)
        assert valid.item() == 1

    def test_invalid_signature(self):
        """Test that invalid signatures are rejected"""
        seed = torch.arange(32, dtype=torch.uint8)
        private_key, public_key = ed25519_keypair(seed)

        message = torch.tensor(list(b"Test"), dtype=torch.uint8)
        signature = ed25519_sign(private_key, message)

        # Corrupt signature
        bad_signature = signature.clone()
        bad_signature[0] ^= 0xFF

        valid = ed25519_verify(public_key, message, bad_signature)
        assert valid.item() == 0

    def test_wrong_message(self):
        """Test that wrong message is rejected"""
        seed = torch.arange(32, dtype=torch.uint8)
        private_key, public_key = ed25519_keypair(seed)

        message = torch.tensor(list(b"Original"), dtype=torch.uint8)
        signature = ed25519_sign(private_key, message)

        wrong_message = torch.tensor(list(b"Modified"), dtype=torch.uint8)
        valid = ed25519_verify(public_key, wrong_message, signature)
        assert valid.item() == 0

    def test_meta_tensor(self):
        """Test shape inference with meta tensors"""
        seed = torch.zeros(32, dtype=torch.uint8, device="meta")
        private_key, public_key = ed25519_keypair(seed)

        assert private_key.device.type == "meta"
        assert public_key.device.type == "meta"
        assert private_key.shape == (64,)
        assert public_key.shape == (32,)
