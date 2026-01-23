import torch

from torchscience.cryptography import x25519, x25519_base, x25519_keypair


class TestX25519:
    def test_rfc7748_test_vector_1(self):
        """RFC 7748 Section 5.2 - Test Vector 1"""
        # Alice's private key
        alice_private = torch.tensor(
            [
                0x77,
                0x07,
                0x6D,
                0x0A,
                0x73,
                0x18,
                0xA5,
                0x7D,
                0x3C,
                0x16,
                0xC1,
                0x72,
                0x51,
                0xB2,
                0x66,
                0x45,
                0xDF,
                0x4C,
                0x2F,
                0x87,
                0xEB,
                0xC0,
                0x99,
                0x2A,
                0xB1,
                0x77,
                0xFB,
                0xA5,
                0x1D,
                0xB9,
                0x2C,
                0x2A,
            ],
            dtype=torch.uint8,
        )

        # Expected public key
        expected_public = torch.tensor(
            [
                0x85,
                0x20,
                0xF0,
                0x09,
                0x89,
                0x30,
                0xA7,
                0x54,
                0x74,
                0x8B,
                0x7D,
                0xDC,
                0xB4,
                0x3E,
                0xF7,
                0x5A,
                0x0D,
                0xBF,
                0x3A,
                0x0D,
                0x26,
                0x38,
                0x1A,
                0xF4,
                0xEB,
                0xA4,
                0xA9,
                0x8E,
                0xAA,
                0x9B,
                0x4E,
                0x6A,
            ],
            dtype=torch.uint8,
        )

        public = x25519_base(alice_private)
        torch.testing.assert_close(public, expected_public)

    def test_rfc7748_test_vector_2(self):
        """RFC 7748 Section 5.2 - Test Vector 2 (shared secret)"""
        # Alice's private key
        alice_private = torch.tensor(
            [
                0x77,
                0x07,
                0x6D,
                0x0A,
                0x73,
                0x18,
                0xA5,
                0x7D,
                0x3C,
                0x16,
                0xC1,
                0x72,
                0x51,
                0xB2,
                0x66,
                0x45,
                0xDF,
                0x4C,
                0x2F,
                0x87,
                0xEB,
                0xC0,
                0x99,
                0x2A,
                0xB1,
                0x77,
                0xFB,
                0xA5,
                0x1D,
                0xB9,
                0x2C,
                0x2A,
            ],
            dtype=torch.uint8,
        )

        # Bob's public key
        bob_public = torch.tensor(
            [
                0xDE,
                0x9E,
                0xDB,
                0x7D,
                0x7B,
                0x7D,
                0xC1,
                0xB4,
                0xD3,
                0x5B,
                0x61,
                0xC2,
                0xEC,
                0xE4,
                0x35,
                0x37,
                0x3F,
                0x83,
                0x43,
                0xC8,
                0x5B,
                0x78,
                0x67,
                0x4D,
                0xAD,
                0xFC,
                0x7E,
                0x14,
                0x6F,
                0x88,
                0x2B,
                0x4F,
            ],
            dtype=torch.uint8,
        )

        # Expected shared secret
        expected_shared = torch.tensor(
            [
                0x4A,
                0x5D,
                0x9D,
                0x5B,
                0xA4,
                0xCE,
                0x2D,
                0xE1,
                0x72,
                0x8E,
                0x3B,
                0xF4,
                0x80,
                0x35,
                0x0F,
                0x25,
                0xE0,
                0x7E,
                0x21,
                0xC9,
                0x47,
                0xD1,
                0x9E,
                0x33,
                0x76,
                0xF0,
                0x9B,
                0x3C,
                0x1E,
                0x16,
                0x17,
                0x42,
            ],
            dtype=torch.uint8,
        )

        shared = x25519(alice_private, bob_public)
        torch.testing.assert_close(shared, expected_shared)

    def test_keypair_roundtrip(self):
        """Test that keypair generation works"""
        seed = torch.arange(32, dtype=torch.uint8)
        private_key, public_key = x25519_keypair(seed)

        assert private_key.shape == (32,)
        assert public_key.shape == (32,)

        # Public key should match base multiplication
        expected_public = x25519_base(private_key)
        torch.testing.assert_close(public_key, expected_public)

    def test_shared_secret_symmetric(self):
        """Test Alice and Bob get same shared secret"""
        alice_seed = torch.arange(32, dtype=torch.uint8)
        bob_seed = torch.arange(32, 64, dtype=torch.uint8)

        alice_private, alice_public = x25519_keypair(alice_seed)
        bob_private, bob_public = x25519_keypair(bob_seed)

        alice_shared = x25519(alice_private, bob_public)
        bob_shared = x25519(bob_private, alice_public)

        torch.testing.assert_close(alice_shared, bob_shared)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors"""
        scalar = torch.zeros(32, dtype=torch.uint8, device="meta")
        point = torch.zeros(32, dtype=torch.uint8, device="meta")

        result = x25519(scalar, point)
        assert result.device.type == "meta"
        assert result.shape == (32,)
