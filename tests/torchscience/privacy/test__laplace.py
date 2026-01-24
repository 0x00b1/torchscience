import math

import torch

from torchscience.cryptography import ChaCha20Generator
from torchscience.privacy import laplace_mechanism


class TestLaplaceMechanism:
    def test_output_shape(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, 20)
        result = laplace_mechanism(
            x, sensitivity=1.0, epsilon=1.0, generator=gen
        )
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_noise_scale(self):
        gen = ChaCha20Generator(seed=42)
        sensitivity, epsilon = 1.0, 1.0
        b = sensitivity / epsilon
        x = torch.zeros(100000)
        noised = laplace_mechanism(x, sensitivity, epsilon, generator=gen)
        # Laplace std = sqrt(2) * b
        expected_std = math.sqrt(2) * b
        assert abs(noised.std().item() - expected_std) < 0.1

    def test_mean_preserved(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.full((100000,), 5.0)
        noised = laplace_mechanism(
            x, sensitivity=1.0, epsilon=1.0, generator=gen
        )
        assert abs(noised.mean().item() - 5.0) < 0.1

    def test_determinism(self):
        x = torch.randn(100)
        gen1 = ChaCha20Generator(seed=123)
        result1 = laplace_mechanism(
            x, sensitivity=1.0, epsilon=1.0, generator=gen1
        )
        gen2 = ChaCha20Generator(seed=123)
        result2 = laplace_mechanism(
            x, sensitivity=1.0, epsilon=1.0, generator=gen2
        )
        torch.testing.assert_close(result1, result2)

    def test_gradcheck(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)

        def func(x):
            gen.manual_seed(42)
            return laplace_mechanism(
                x, sensitivity=1.0, epsilon=1.0, generator=gen
            )

        torch.autograd.gradcheck(func, x)
