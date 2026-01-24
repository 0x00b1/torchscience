import torch

from torchscience.cryptography import password_hash, password_verify


class TestPassword:
    def test_hash_verify_roundtrip(self):
        """Test that correct password verifies."""
        password = b"correct horse battery staple"
        # Use low iterations for fast testing
        hash_tensor = password_hash(password, iterations=1000)

        assert password_verify(password, hash_tensor)

    def test_wrong_password_fails(self):
        """Test that wrong password fails verification."""
        hash_tensor = password_hash(b"correct", iterations=1000)

        assert not password_verify(b"wrong", hash_tensor)

    def test_hash_format(self):
        """Test that hash has correct format."""
        hash_tensor = password_hash(b"test", iterations=1000)
        # salt (16) + iterations (4) + hash (32) = 52 bytes
        assert hash_tensor.shape == (52,)
        assert hash_tensor.dtype == torch.uint8

    def test_different_salts(self):
        """Test that same password produces different hashes."""
        password = b"same_password"
        hash1 = password_hash(password, iterations=1000)
        hash2 = password_hash(password, iterations=1000)

        # Hashes should be different (different random salts)
        assert not torch.all(hash1 == hash2)

    def test_tensor_password(self):
        """Test with Tensor password input."""
        password = torch.tensor(list(b"tensor_password"), dtype=torch.uint8)
        hash_tensor = password_hash(password, iterations=1000)

        assert password_verify(password, hash_tensor)

    def test_iterations_encoded(self):
        """Test that iterations are correctly encoded/decoded."""
        password = b"test"
        iterations = 12345

        hash_tensor = password_hash(password, iterations=iterations)

        # Verify by checking the password verifies (requires correct iteration decode)
        assert password_verify(password, hash_tensor)
