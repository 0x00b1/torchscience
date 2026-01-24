import math

import torch

from torchscience.cryptography import ChaCha20Generator
from torchscience.privacy import gaussian_mechanism


class TestGaussianMechanism:
    def test_output_shape(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, 20)
        result = gaussian_mechanism(
            x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen
        )
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_noise_scale(self):
        gen = ChaCha20Generator(seed=42)
        sensitivity, epsilon, delta = 1.0, 1.0, 1e-5
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        x = torch.zeros(100000)
        noised = gaussian_mechanism(
            x, sensitivity, epsilon, delta, generator=gen
        )
        empirical_std = noised.std().item()
        assert abs(empirical_std - sigma) < 0.1

    def test_mean_preserved(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.full((100000,), 5.0)
        noised = gaussian_mechanism(
            x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen
        )
        assert abs(noised.mean().item() - 5.0) < 0.1

    def test_determinism(self):
        x = torch.randn(100)
        gen1 = ChaCha20Generator(seed=123)
        result1 = gaussian_mechanism(
            x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen1
        )
        gen2 = ChaCha20Generator(seed=123)
        result2 = gaussian_mechanism(
            x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen2
        )
        torch.testing.assert_close(result1, result2)

    def test_gradcheck(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)

        def func(x):
            gen.manual_seed(42)
            return gaussian_mechanism(
                x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen
            )

        torch.autograd.gradcheck(func, x)
