import torch
from torch import Tensor

from torchscience.encryption import ChaCha20Generator


def exponential_mechanism(
    utilities: Tensor,
    sensitivity: float,
    epsilon: float,
    generator: ChaCha20Generator,
) -> Tensor:
    """Sample index proportional to exp(epsilon * utility / (2 * sensitivity)).

    The exponential mechanism selects an index with probability proportional
    to exp(epsilon * utility[i] / (2 * sensitivity)).

    Parameters
    ----------
    utilities : Tensor
        (..., k) tensor of utility scores for k options.
    sensitivity : float
        Sensitivity of the utility function.
    epsilon : float
        Privacy parameter epsilon.
    generator : ChaCha20Generator
        Cryptographically secure random number generator.

    Returns
    -------
    Tensor
        (...,) int64 tensor of selected indices.
    """
    # Compute log-probabilities
    scale = epsilon / (2 * sensitivity)
    log_probs = utilities * scale

    # Use Gumbel-max trick for sampling
    # Sample Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)
    u = generator.random(tuple(utilities.shape), utilities.dtype).to(
        utilities.device
    )
    # Clamp to avoid log(0)
    u = u.clamp(min=1e-10, max=1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))

    # Argmax of (log_probs + gumbel) gives sample from softmax distribution
    perturbed = log_probs + gumbel
    return perturbed.argmax(dim=-1)
