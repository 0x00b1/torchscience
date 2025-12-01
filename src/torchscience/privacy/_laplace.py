import torch
from torch import Tensor

from torchscience.encryption import ChaCha20Generator


def laplace_mechanism(
    x: Tensor,
    sensitivity: float,
    epsilon: float,
    generator: ChaCha20Generator,
) -> Tensor:
    """Add Laplace noise calibrated for epsilon-differential privacy.

    Noise scale: b = sensitivity / epsilon

    Parameters
    ----------
    x : Tensor
        Input tensor to be privatized.
    sensitivity : float
        L1 sensitivity of the query.
    epsilon : float
        Privacy parameter epsilon.
    generator : ChaCha20Generator
        Cryptographically secure random number generator.

    Returns
    -------
    Tensor
        Input plus calibrated Laplace noise.
    """
    b = sensitivity / epsilon
    # Generate Laplace samples from uniform: b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
    u = generator.random(tuple(x.shape), x.dtype).to(x.device)
    # Transform uniform to Laplace(0, 1)
    laplace_noise = torch.sign(u - 0.5) * torch.log1p(-2 * torch.abs(u - 0.5))
    return torch.ops.torchscience.laplace_mechanism(x, laplace_noise, b)
