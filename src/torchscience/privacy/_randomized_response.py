import math

import torch
from torch import Tensor

from torchscience.cryptography import ChaCha20Generator


def randomized_response(
    x: Tensor,
    epsilon: float,
    generator: ChaCha20Generator,
    num_categories: int = 2,
) -> Tensor:
    """Apply randomized response for differential privacy.

    For binary data: With probability p = exp(epsilon) / (1 + exp(epsilon)),
    return the true value. Otherwise, return a random value.

    Parameters
    ----------
    x : Tensor
        Input tensor (bool or integer in range [0, num_categories)).
    epsilon : float
        Privacy parameter epsilon.
    generator : ChaCha20Generator
        Cryptographically secure random number generator.
    num_categories : int, optional
        Number of categories (2 for binary). Default is 2.

    Returns
    -------
    Tensor
        Randomized responses with same shape and dtype as input.
    """
    original_dtype = x.dtype

    # Probability of telling the truth
    p_truth = math.exp(epsilon) / (1 + math.exp(epsilon))

    # Generate random decisions: tell truth or lie
    u = generator.random(tuple(x.shape), torch.float32).to(x.device)
    tell_truth = u < p_truth

    if num_categories == 2:
        # Binary case: flip the bit if lying
        # Use logical_not for boolean tensors, subtraction for integers
        if original_dtype == torch.bool:
            random_response = ~x
        else:
            random_response = 1 - x
    else:
        # Categorical case: sample uniformly from other categories
        random_vals = torch.randint(
            0, num_categories - 1, x.shape, device=x.device, dtype=x.dtype
        )
        # Shift values >= x to avoid returning the same value
        random_response = torch.where(
            random_vals >= x, random_vals + 1, random_vals
        )

    result = torch.where(tell_truth, x, random_response)
    return result.to(original_dtype)
