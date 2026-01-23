import torch

from torchscience.cryptography import (
    aes_ctr,
    aes_decrypt_block,
    aes_encrypt_block,
)


class TestAESBlock:
    def test_nist_aes128_encrypt(self):
        # NIST FIPS 197 Appendix B test vector
        key = torch.tensor(
            [
                0x2B,
                0x7E,
                0x15,
                0x16,
                0x28,
                0xAE,
                0xD2,
                0xA6,
                0xAB,
                0xF7,
                0x15,
                0x88,
                0x09,
                0xCF,
                0x4F,
                0x3C,
            ],
            dtype=torch.uint8,
        )
        plaintext = torch.tensor(
            [
                0x32,
                0x43,
                0xF6,
                0xA8,
                0x88,
                0x5A,
                0x30,
                0x8D,
                0x31,
                0x31,
                0x98,
                0xA2,
                0xE0,
                0x37,
                0x07,
                0x34,
            ],
            dtype=torch.uint8,
        )
        expected = torch.tensor(
            [
                0x39,
                0x25,
                0x84,
                0x1D,
                0x02,
                0xDC,
                0x09,
                0xFB,
                0xDC,
                0x11,
                0x85,
                0x97,
                0x19,
                0x6A,
                0x0B,
                0x32,
            ],
            dtype=torch.uint8,
        )

        result = aes_encrypt_block(plaintext, key)
        torch.testing.assert_close(result, expected)

    def test_encrypt_decrypt_roundtrip(self):
        key = torch.arange(16, dtype=torch.uint8)
        plaintext = torch.arange(16, dtype=torch.uint8)

        ciphertext = aes_encrypt_block(plaintext, key)
        decrypted = aes_decrypt_block(ciphertext, key)

        torch.testing.assert_close(decrypted, plaintext)

    def test_aes256_roundtrip(self):
        key = torch.arange(32, dtype=torch.uint8)
        plaintext = torch.arange(16, dtype=torch.uint8)

        ciphertext = aes_encrypt_block(plaintext, key)
        decrypted = aes_decrypt_block(ciphertext, key)

        torch.testing.assert_close(decrypted, plaintext)


class TestAESCTR:
    def test_encrypt_decrypt_roundtrip(self):
        key = torch.arange(16, dtype=torch.uint8)
        nonce = torch.arange(12, dtype=torch.uint8)
        plaintext = torch.arange(100, dtype=torch.uint8)

        ciphertext = aes_ctr(plaintext, key, nonce)
        decrypted = aes_ctr(ciphertext, key, nonce)

        torch.testing.assert_close(decrypted, plaintext)

    def test_different_nonce_different_output(self):
        key = torch.arange(16, dtype=torch.uint8)
        nonce1 = torch.zeros(12, dtype=torch.uint8)
        nonce2 = torch.ones(12, dtype=torch.uint8)
        plaintext = torch.arange(32, dtype=torch.uint8)

        ct1 = aes_ctr(plaintext, key, nonce1)
        ct2 = aes_ctr(plaintext, key, nonce2)

        assert not torch.equal(ct1, ct2)

    def test_meta_tensor(self):
        key = torch.zeros(16, dtype=torch.uint8, device="meta")
        nonce = torch.zeros(12, dtype=torch.uint8, device="meta")
        data = torch.zeros(100, dtype=torch.uint8, device="meta")

        result = aes_ctr(data, key, nonce)
        assert result.device.type == "meta"
        assert result.shape == (100,)
