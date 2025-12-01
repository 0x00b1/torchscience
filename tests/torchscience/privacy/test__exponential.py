import torch

from torchscience.encryption import ChaCha20Generator
from torchscience.privacy import exponential_mechanism


class TestExponentialMechanism:
    def test_output_shape(self):
        gen = ChaCha20Generator(seed=42)
        utilities = torch.randn(10, 5)  # 10 batches, 5 options each
        result = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=1.0, generator=gen
        )
        assert result.shape == (10,)
        assert result.dtype == torch.int64

    def test_output_range(self):
        gen = ChaCha20Generator(seed=42)
        utilities = torch.randn(100, 7)
        result = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=1.0, generator=gen
        )
        assert result.min() >= 0
        assert result.max() < 7

    def test_high_utility_preferred(self):
        """Higher epsilon should more strongly prefer high utility."""
        gen = ChaCha20Generator(seed=42)
        # One option clearly dominates
        utilities = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]] * 1000)

        gen.manual_seed(42)
        result = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=10.0, generator=gen
        )

        # With high epsilon and one dominant option, most should select index 2
        selected_2 = (result == 2).float().mean().item()
        assert selected_2 > 0.9

    def test_low_epsilon_more_random(self):
        """Lower epsilon should be more random."""
        gen = ChaCha20Generator(seed=42)
        utilities = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]] * 1000)

        gen.manual_seed(42)
        result = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=0.01, generator=gen
        )

        # With low epsilon, selection should be more spread out
        selected_2 = (result == 2).float().mean().item()
        assert selected_2 < 0.5  # Less concentrated on the max

    def test_determinism(self):
        utilities = torch.randn(100, 5)
        gen1 = ChaCha20Generator(seed=123)
        result1 = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=1.0, generator=gen1
        )
        gen2 = ChaCha20Generator(seed=123)
        result2 = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=1.0, generator=gen2
        )
        torch.testing.assert_close(result1, result2)

    def test_1d_input(self):
        """Single set of utilities (no batch dimension)."""
        gen = ChaCha20Generator(seed=42)
        utilities = torch.randn(5)
        result = exponential_mechanism(
            utilities, sensitivity=1.0, epsilon=1.0, generator=gen
        )
        assert result.shape == ()
        assert 0 <= result.item() < 5
