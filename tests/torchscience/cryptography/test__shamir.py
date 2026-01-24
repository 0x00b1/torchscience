import torch

from torchscience.cryptography import shamir_reconstruct, shamir_split


class TestShamirSecretSharing:
    def test_split_reconstruct_roundtrip(self):
        """Test that splitting and reconstructing recovers the secret."""
        secret = torch.tensor(list(b"Hello, World!"), dtype=torch.uint8)
        n, k = 5, 3

        shares = shamir_split(secret, n, k)
        assert shares.shape == (n, len(secret))

        # Reconstruct using first k shares
        indices = torch.tensor([1, 2, 3], dtype=torch.uint8)
        recovered = shamir_reconstruct(shares[:k], indices)
        torch.testing.assert_close(recovered, secret)

    def test_any_k_shares_work(self):
        """Test that any k shares can reconstruct."""
        secret = torch.tensor([0x42] * 16, dtype=torch.uint8)
        n, k = 5, 3

        generator = torch.Generator().manual_seed(42)
        shares = shamir_split(secret, n, k, generator=generator)

        # Try different combinations of k shares
        for combo in [[0, 1, 2], [0, 2, 4], [1, 3, 4], [2, 3, 4]]:
            selected = shares[combo]
            indices = torch.tensor([c + 1 for c in combo], dtype=torch.uint8)
            recovered = shamir_reconstruct(selected, indices)
            torch.testing.assert_close(recovered, secret)

    def test_determinism_with_generator(self):
        """Test that same generator seed produces same shares."""
        secret = torch.tensor([0xAB, 0xCD, 0xEF], dtype=torch.uint8)

        gen1 = torch.Generator().manual_seed(123)
        shares1 = shamir_split(secret, 5, 3, generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        shares2 = shamir_split(secret, 5, 3, generator=gen2)

        torch.testing.assert_close(shares1, shares2)

    def test_minimum_threshold(self):
        """Test with minimum threshold k=2."""
        secret = torch.tensor([0x01, 0x02, 0x03], dtype=torch.uint8)
        shares = shamir_split(secret, 3, 2)

        indices = torch.tensor([1, 2], dtype=torch.uint8)
        recovered = shamir_reconstruct(shares[:2], indices)
        torch.testing.assert_close(recovered, secret)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        secret = torch.zeros(32, dtype=torch.uint8, device="meta")
        randomness = torch.zeros(64, dtype=torch.uint8, device="meta")

        shares = torch.ops.torchscience.shamir_split(secret, randomness, 5, 3)
        assert shares.device.type == "meta"
        assert shares.shape == (5, 32)
