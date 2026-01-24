import pytest
import torch

from torchscience.cryptography import decrypt, encrypt


class TestAEAD:
    def test_encrypt_decrypt_roundtrip(self):
        """Test basic encryption and decryption."""
        key = torch.randint(0, 256, (32,), dtype=torch.uint8)
        plaintext = torch.tensor(list(b"Hello, World!"), dtype=torch.uint8)

        ciphertext, nonce, tag = encrypt(plaintext, key)
        recovered = decrypt(ciphertext, key, nonce, tag)

        torch.testing.assert_close(recovered, plaintext)

    def test_with_aad(self):
        """Test encryption with additional authenticated data."""
        key = torch.randint(0, 256, (32,), dtype=torch.uint8)
        plaintext = torch.tensor(list(b"Secret data"), dtype=torch.uint8)
        aad = torch.tensor(list(b"metadata"), dtype=torch.uint8)

        ciphertext, nonce, tag = encrypt(plaintext, key, aad=aad)
        recovered = decrypt(ciphertext, key, nonce, tag, aad=aad)

        torch.testing.assert_close(recovered, plaintext)

    def test_different_key_fails(self):
        """Test that wrong key fails authentication."""
        key1 = torch.randint(0, 256, (32,), dtype=torch.uint8)
        key2 = torch.randint(0, 256, (32,), dtype=torch.uint8)
        plaintext = torch.tensor(list(b"test"), dtype=torch.uint8)

        ciphertext, nonce, tag = encrypt(plaintext, key1)

        with pytest.raises(RuntimeError):
            decrypt(ciphertext, key2, nonce, tag)

    def test_tampered_ciphertext_fails(self):
        """Test that tampered ciphertext fails authentication."""
        key = torch.randint(0, 256, (32,), dtype=torch.uint8)
        plaintext = torch.tensor(list(b"test data"), dtype=torch.uint8)

        ciphertext, nonce, tag = encrypt(plaintext, key)
        ciphertext[0] ^= 0xFF  # Tamper with first byte

        with pytest.raises(RuntimeError):
            decrypt(ciphertext, key, nonce, tag)

    def test_unsupported_algorithm(self):
        """Test that unsupported algorithm raises error."""
        key = torch.randint(0, 256, (32,), dtype=torch.uint8)
        plaintext = torch.tensor(list(b"test"), dtype=torch.uint8)

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            encrypt(plaintext, key, algorithm="aes-gcm")
