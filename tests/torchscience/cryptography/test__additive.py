import torch

from torchscience.cryptography import additive_reconstruct, additive_split


class TestAdditiveSecretSharing:
    def test_split_reconstruct_roundtrip(self):
        """Test that splitting and reconstructing recovers the secret."""
        secret = torch.tensor(list(b"Secret!"), dtype=torch.uint8)
        n = 4

        shares = additive_split(secret, n)
        assert shares.shape == (n, len(secret))

        recovered = additive_reconstruct(shares)
        torch.testing.assert_close(recovered, secret)

    def test_xor_property(self):
        """Test that XOR of all shares equals secret."""
        secret = torch.tensor([0xDE, 0xAD, 0xBE, 0xEF], dtype=torch.uint8)
        shares = additive_split(secret, 5)

        # Manual XOR of all shares
        result = torch.zeros_like(secret)
        for i in range(5):
            result = result ^ shares[i]

        torch.testing.assert_close(result, secret)

    def test_determinism_with_generator(self):
        """Test that same generator seed produces same shares."""
        secret = torch.tensor([0x11, 0x22, 0x33], dtype=torch.uint8)

        gen1 = torch.Generator().manual_seed(456)
        shares1 = additive_split(secret, 3, generator=gen1)

        gen2 = torch.Generator().manual_seed(456)
        shares2 = additive_split(secret, 3, generator=gen2)

        torch.testing.assert_close(shares1, shares2)

    def test_two_shares(self):
        """Test minimum case of n=2 shares."""
        secret = torch.tensor([0xFF], dtype=torch.uint8)
        shares = additive_split(secret, 2)

        recovered = additive_reconstruct(shares)
        torch.testing.assert_close(recovered, secret)

    def test_large_secret(self):
        """Test with larger secret."""
        secret = torch.randint(0, 256, (1024,), dtype=torch.uint8)
        shares = additive_split(secret, 10)

        recovered = additive_reconstruct(shares)
        torch.testing.assert_close(recovered, secret)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        secret = torch.zeros(64, dtype=torch.uint8, device="meta")
        randomness = torch.zeros(192, dtype=torch.uint8, device="meta")

        shares = torch.ops.torchscience.additive_split(secret, randomness, 4)
        assert shares.device.type == "meta"
        assert shares.shape == (4, 64)
