"""Rate-distortion Lagrangian objective."""

from __future__ import annotations

import torch
from torch import Tensor


def rate_distortion_lagrangian(
    rate: Tensor,
    distortion: Tensor,
    *,
    lmbda: float = 0.01,
) -> Tensor:
    """Compute rate-distortion Lagrangian objective.

    The rate-distortion Lagrangian is the standard objective for
    learned compression: J = R + λD, where R is the rate (bits)
    and D is the distortion.

    Parameters
    ----------
    rate : Tensor
        Rate term (bits or nats). Can be scalar or batched.
    distortion : Tensor
        Distortion term (e.g., MSE). Must be broadcastable with rate.
    lmbda : float, default=0.01
        Lagrange multiplier controlling rate-distortion tradeoff.
        Higher λ prioritizes rate (compression), lower λ prioritizes
        distortion (quality).

    Returns
    -------
    Tensor
        Lagrangian objective: J = R + λD

    Examples
    --------
    >>> import torch
    >>> rate = torch.tensor(100.0)  # 100 bits
    >>> distortion = torch.tensor(0.01)  # MSE
    >>> J = rate_distortion_lagrangian(rate, distortion, lmbda=0.1)
    >>> J  # 100 + 0.1 * 0.01 = 100.001

    Notes
    -----
    In neural compression, the rate R is typically computed as:
        R = -log₂(p(z)) summed over latent elements

    The distortion D is typically MSE between original and
    reconstructed images.

    The Lagrange multiplier λ controls the operating point on
    the rate-distortion curve:
    - High λ: Aggressive compression, higher distortion
    - Low λ: Conservative compression, lower distortion

    This formulation arises from the constrained optimization problem:
        minimize D  subject to R ≤ R_target
    which is equivalent to:
        minimize R + λD  (with λ as the constraint's dual variable)

    See Also
    --------
    distortion_measure : Compute distortion between signals.
    """
    if not isinstance(rate, Tensor):
        rate = torch.tensor(rate)
    if not isinstance(distortion, Tensor):
        distortion = torch.tensor(distortion)

    return rate + lmbda * distortion


def estimate_bitrate(
    likelihoods: Tensor,
    *,
    reduction: str = "sum",
) -> Tensor:
    """Estimate bitrate from probability likelihoods.

    Computes the expected code length (in bits) for a set of
    symbols given their probabilities.

    Parameters
    ----------
    likelihoods : Tensor
        Probability of each symbol. Values should be in (0, 1].
        Shape: arbitrary.
    reduction : {"sum", "mean", "none"}, default="sum"
        Reduction mode:
        - "sum": Total bits for all symbols
        - "mean": Average bits per symbol
        - "none": Bits per symbol (no reduction)

    Returns
    -------
    Tensor
        Estimated bitrate.

    Examples
    --------
    >>> import torch
    >>> # Uniform distribution over 8 symbols: -log2(1/8) = 3 bits each
    >>> likelihoods = torch.ones(8) / 8
    >>> bits = estimate_bitrate(likelihoods)
    >>> bits  # 8 * 3 = 24 bits total

    Notes
    -----
    For a symbol with probability p, the optimal code length is
    -log₂(p) bits. This function computes:
        bits = Σᵢ -log₂(pᵢ)

    In practice, arithmetic coding achieves this optimal rate
    asymptotically.

    See Also
    --------
    rate_distortion_lagrangian : Combined rate-distortion objective.
    """
    if not isinstance(likelihoods, Tensor):
        raise TypeError(
            f"likelihoods must be a Tensor, got {type(likelihoods).__name__}"
        )

    # Clamp to avoid log(0)
    likelihoods = likelihoods.clamp(min=1e-10)

    # -log2(p) = -log(p) / log(2)
    bits = -torch.log(likelihoods) / torch.log(
        torch.tensor(2.0, device=likelihoods.device)
    )

    if reduction == "sum":
        return bits.sum()
    elif reduction == "mean":
        return bits.mean()
    elif reduction == "none":
        return bits
    else:
        raise ValueError(
            f"reduction must be 'sum', 'mean', or 'none', got '{reduction}'"
        )
