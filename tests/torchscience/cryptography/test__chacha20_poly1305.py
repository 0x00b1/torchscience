import torch

from torchscience.cryptography import (
    chacha20_poly1305_decrypt,
    chacha20_poly1305_encrypt,
)


class TestChaCha20Poly1305:
    def test_rfc8439_test_vector(self):
        # RFC 8439 Section 2.8.2 test vector
        key = torch.tensor(
            [
                0x80,
                0x81,
                0x82,
                0x83,
                0x84,
                0x85,
                0x86,
                0x87,
                0x88,
                0x89,
                0x8A,
                0x8B,
                0x8C,
                0x8D,
                0x8E,
                0x8F,
                0x90,
                0x91,
                0x92,
                0x93,
                0x94,
                0x95,
                0x96,
                0x97,
                0x98,
                0x99,
                0x9A,
                0x9B,
                0x9C,
                0x9D,
                0x9E,
                0x9F,
            ],
            dtype=torch.uint8,
        )
        nonce = torch.tensor(
            [
                0x07,
                0x00,
                0x00,
                0x00,
                0x40,
                0x41,
                0x42,
                0x43,
                0x44,
                0x45,
                0x46,
                0x47,
            ],
            dtype=torch.uint8,
        )
        aad = torch.tensor(
            [
                0x50,
                0x51,
                0x52,
                0x53,
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC4,
                0xC5,
                0xC6,
                0xC7,
            ],
            dtype=torch.uint8,
        )
        plaintext = torch.tensor(
            list(
                b"Ladies and Gentlemen of the class of '99: "
                b"If I could offer you only one tip for the future, sunscreen would be it."
            ),
            dtype=torch.uint8,
        )

        ciphertext, tag = chacha20_poly1305_encrypt(plaintext, key, nonce, aad)

        # Verify we can decrypt
        decrypted = chacha20_poly1305_decrypt(ciphertext, key, nonce, tag, aad)
        torch.testing.assert_close(decrypted, plaintext)

    def test_encrypt_decrypt_roundtrip(self):
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.arange(12, dtype=torch.uint8)
        plaintext = torch.arange(100, dtype=torch.uint8)

        ciphertext, tag = chacha20_poly1305_encrypt(plaintext, key, nonce)
        decrypted = chacha20_poly1305_decrypt(ciphertext, key, nonce, tag)

        torch.testing.assert_close(decrypted, plaintext)

    def test_with_aad(self):
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.arange(12, dtype=torch.uint8)
        plaintext = torch.arange(50, dtype=torch.uint8)
        aad = torch.arange(20, dtype=torch.uint8)

        ciphertext, tag = chacha20_poly1305_encrypt(plaintext, key, nonce, aad)
        decrypted = chacha20_poly1305_decrypt(ciphertext, key, nonce, tag, aad)

        torch.testing.assert_close(decrypted, plaintext)

    def test_meta_tensor(self):
        key = torch.zeros(32, dtype=torch.uint8, device="meta")
        nonce = torch.zeros(12, dtype=torch.uint8, device="meta")
        plaintext = torch.zeros(100, dtype=torch.uint8, device="meta")

        ciphertext, tag = chacha20_poly1305_encrypt(plaintext, key, nonce)
        assert ciphertext.device.type == "meta"
        assert ciphertext.shape == (100,)
        assert tag.shape == (16,)
