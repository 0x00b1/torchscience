import torch

from torchscience.cryptography import (
    derive_key,
    x25519_keypair,
    x25519_shared_secret,
)


class TestKeyExchange:
    def test_shared_secret_symmetric(self):
        """Test that both parties compute the same shared secret."""
        alice_seed = torch.randint(0, 256, (32,), dtype=torch.uint8)
        bob_seed = torch.randint(0, 256, (32,), dtype=torch.uint8)

        alice_priv, alice_pub = x25519_keypair(alice_seed)
        bob_priv, bob_pub = x25519_keypair(bob_seed)

        alice_shared = x25519_shared_secret(alice_priv, bob_pub)
        bob_shared = x25519_shared_secret(bob_priv, alice_pub)

        torch.testing.assert_close(alice_shared, bob_shared)

    def test_derive_key_from_shared_secret(self):
        """Test deriving a key from shared secret."""
        shared = torch.randint(0, 256, (32,), dtype=torch.uint8)
        key = derive_key(shared, b"encryption", length=32)

        assert key.shape == (32,)
        assert key.dtype == torch.uint8

    def test_derive_key_different_info(self):
        """Test that different info produces different keys."""
        shared = torch.randint(0, 256, (32,), dtype=torch.uint8)

        key1 = derive_key(shared, b"encryption")
        key2 = derive_key(shared, b"authentication")

        assert not torch.all(key1 == key2)

    def test_derive_key_with_tensor_info(self):
        """Test derive_key with Tensor info parameter."""
        shared = torch.randint(0, 256, (32,), dtype=torch.uint8)
        info = torch.tensor(list(b"test info"), dtype=torch.uint8)

        key = derive_key(shared, info, length=64)
        assert key.shape == (64,)

    def test_derive_key_deterministic(self):
        """Test that derive_key is deterministic."""
        shared = torch.tensor([0x42] * 32, dtype=torch.uint8)

        key1 = derive_key(shared, b"test")
        key2 = derive_key(shared, b"test")

        torch.testing.assert_close(key1, key2)
