import math

import torch
from torch import Tensor

from torchscience.encryption import ChaCha20Generator


def gaussian_mechanism(
    x: Tensor,
    sensitivity: float,
    epsilon: float,
    delta: float,
    generator: ChaCha20Generator,
) -> Tensor:
    """Add Gaussian noise calibrated for (epsilon, delta)-differential privacy.

    Noise scale: sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Parameters
    ----------
    x : Tensor
        Input tensor to be privatized.
    sensitivity : float
        L2 sensitivity of the query.
    epsilon : float
        Privacy parameter epsilon.
    delta : float
        Privacy parameter delta.
    generator : ChaCha20Generator
        Cryptographically secure random number generator.

    Returns
    -------
    Tensor
        Input plus calibrated Gaussian noise.
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = generator.randn(tuple(x.shape), x.dtype).to(x.device)
    return torch.ops.torchscience.gaussian_mechanism(x, noise, sigma)
