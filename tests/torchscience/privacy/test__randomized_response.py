import math

import torch

from torchscience.encryption import ChaCha20Generator
from torchscience.privacy import randomized_response


class TestRandomizedResponse:
    def test_output_shape(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randint(0, 2, (100,))
        result = randomized_response(x, epsilon=1.0, generator=gen)
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_output_range_binary(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randint(0, 2, (1000,))
        result = randomized_response(x, epsilon=1.0, generator=gen)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_output_range_categorical(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randint(0, 5, (1000,))
        result = randomized_response(
            x, epsilon=1.0, generator=gen, num_categories=5
        )
        assert result.min() >= 0
        assert result.max() < 5

    def test_flip_rate(self):
        """Verify flip rate matches theoretical probability."""
        gen = ChaCha20Generator(seed=42)
        epsilon = 1.0
        p_truth = math.exp(epsilon) / (1 + math.exp(epsilon))

        x = torch.zeros(100000, dtype=torch.int64)
        result = randomized_response(x, epsilon=epsilon, generator=gen)

        # Fraction that stayed 0 should be close to p_truth
        fraction_unchanged = (result == 0).float().mean().item()
        assert abs(fraction_unchanged - p_truth) < 0.02

    def test_high_epsilon_preserves_values(self):
        """High epsilon should mostly preserve original values."""
        gen = ChaCha20Generator(seed=42)
        x = torch.randint(0, 2, (10000,))
        result = randomized_response(x, epsilon=10.0, generator=gen)

        # With high epsilon, most values should be unchanged
        unchanged = (result == x).float().mean().item()
        assert unchanged > 0.99

    def test_low_epsilon_more_random(self):
        """Low epsilon should flip more values."""
        gen = ChaCha20Generator(seed=42)
        x = torch.zeros(10000, dtype=torch.int64)
        result = randomized_response(x, epsilon=0.01, generator=gen)

        # With low epsilon, roughly half should be flipped
        unchanged = (result == 0).float().mean().item()
        assert 0.4 < unchanged < 0.6

    def test_determinism(self):
        x = torch.randint(0, 2, (100,))
        gen1 = ChaCha20Generator(seed=123)
        result1 = randomized_response(x, epsilon=1.0, generator=gen1)
        gen2 = ChaCha20Generator(seed=123)
        result2 = randomized_response(x, epsilon=1.0, generator=gen2)
        torch.testing.assert_close(result1, result2)

    def test_bool_input(self):
        gen = ChaCha20Generator(seed=42)
        x = torch.randint(0, 2, (100,)).bool()
        result = randomized_response(x, epsilon=1.0, generator=gen)
        assert result.dtype == torch.bool
