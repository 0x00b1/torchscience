import torch

from torchscience.encryption import ChaCha20Generator


class TestChaCha20Generator:
    def test_construction_with_int_seed(self):
        gen = ChaCha20Generator(seed=42)
        assert gen is not None

    def test_construction_with_tensor_seed(self):
        seed = torch.arange(32, dtype=torch.uint8)
        gen = ChaCha20Generator(seed=seed)
        assert gen is not None

    def test_random_bytes_length(self):
        gen = ChaCha20Generator(seed=42)
        for length in [1, 32, 64, 100, 1000]:
            out = gen.random_bytes(length)
            assert out.shape == (length,)
            assert out.dtype == torch.uint8

    def test_random_shape(self):
        gen = ChaCha20Generator(seed=42)
        out = gen.random((100, 100), dtype=torch.float32)
        assert out.shape == (100, 100)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() < 1.0

    def test_randn_shape(self):
        gen = ChaCha20Generator(seed=42)
        out = gen.randn((100, 100), dtype=torch.float32)
        assert out.shape == (100, 100)
        assert out.dtype == torch.float32

    def test_randn_distribution(self):
        gen = ChaCha20Generator(seed=42)
        out = gen.randn((10000,), dtype=torch.float64)
        assert abs(out.mean().item()) < 0.1
        assert abs(out.std().item() - 1.0) < 0.1

    def test_determinism(self):
        gen1 = ChaCha20Generator(seed=123)
        gen2 = ChaCha20Generator(seed=123)
        out1 = gen1.random_bytes(100)
        out2 = gen2.random_bytes(100)
        torch.testing.assert_close(out1, out2)

    def test_different_seeds_different_output(self):
        gen1 = ChaCha20Generator(seed=1)
        gen2 = ChaCha20Generator(seed=2)
        out1 = gen1.random_bytes(100)
        out2 = gen2.random_bytes(100)
        assert not torch.equal(out1, out2)

    def test_manual_seed(self):
        gen = ChaCha20Generator(seed=42)
        out1 = gen.random_bytes(100)
        gen.manual_seed(42)
        out2 = gen.random_bytes(100)
        torch.testing.assert_close(out1, out2)

    def test_get_set_state(self):
        gen = ChaCha20Generator(seed=42)
        gen.random_bytes(50)
        state = gen.get_state()
        out1 = gen.random_bytes(100)
        gen.set_state(state)
        out2 = gen.random_bytes(100)
        torch.testing.assert_close(out1, out2)

    def test_device_parameter(self):
        gen = ChaCha20Generator(seed=42, device="cpu")
        out = gen.random_bytes(64)
        assert out.device.type == "cpu"
