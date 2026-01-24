"""Entropy rate for stationary Markov chains."""

import math

import torch
from torch import Tensor


def entropy_rate(
    transition_matrix: Tensor,
    *,
    stationary_distribution: Tensor | None = None,
    base: float | None = None,
) -> Tensor:
    r"""Compute entropy rate of a stationary Markov chain.

    For a stationary Markov chain with transition matrix P and stationary
    distribution π, the entropy rate is:

    .. math::

        H_\infty = -\sum_i \pi_i \sum_j P_{ij} \log P_{ij}

    This represents the conditional entropy H(Xₙ|Xₙ₋₁) in steady state,
    which equals the limiting entropy rate.

    Parameters
    ----------
    transition_matrix : Tensor
        Row-stochastic transition matrix P where P[i,j] = P(Xₙ₊₁=j|Xₙ=i).
        Shape: ``(..., n_states, n_states)``. Each row must sum to 1.
    stationary_distribution : Tensor or None, default=None
        Stationary distribution π where πP = π. Shape: ``(..., n_states)``.
        If None, computed as the left eigenvector of P with eigenvalue 1.
    base : float or None, default=None
        Logarithm base. If None, uses natural logarithm (nats).
        Use 2 for bits.

    Returns
    -------
    Tensor
        Entropy rate H∞. Shape is the input shape with last two dimensions
        removed.

    Examples
    --------
    >>> import torch
    >>> # Deterministic chain (identity matrix): entropy rate = 0
    >>> P = torch.eye(3)
    >>> entropy_rate(P)
    tensor(0.)

    >>> # Uniform transitions: entropy rate = log(n_states)
    >>> P = torch.full((3, 3), 1/3)
    >>> entropy_rate(P, base=2.0)
    tensor(1.5850)  # log2(3)

    >>> # Binary symmetric channel-like transition
    >>> p = 0.1  # transition probability
    >>> P = torch.tensor([[1-p, p], [p, 1-p]])
    >>> entropy_rate(P, base=2.0)  # H(p) = -p*log(p) - (1-p)*log(1-p)
    tensor(0.4690)

    Notes
    -----
    - The stationary distribution is computed via power iteration if not provided.
    - For ergodic chains, the entropy rate equals the time-average of H(Xₙ|Xₙ₋₁).
    - If the chain is reducible, the result depends on the choice of
      stationary distribution.

    See Also
    --------
    shannon_entropy : Compute entropy of a distribution.
    conditional_entropy : Conditional entropy H(Y|X).

    References
    ----------
    .. [1] Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory
           (2nd ed.). Wiley-Interscience. Chapter 4.
    """
    if not isinstance(transition_matrix, Tensor):
        raise TypeError(
            f"transition_matrix must be a Tensor, got {type(transition_matrix).__name__}"
        )

    if transition_matrix.dim() < 2:
        raise ValueError("transition_matrix must have at least 2 dimensions")

    n_states = transition_matrix.shape[-1]
    if transition_matrix.shape[-2] != n_states:
        raise ValueError(
            f"transition_matrix must be square, got shape {transition_matrix.shape[-2:]}"
        )

    # Compute stationary distribution if not provided
    if stationary_distribution is None:
        stationary_distribution = _compute_stationary_distribution(
            transition_matrix
        )
    else:
        if not isinstance(stationary_distribution, Tensor):
            raise TypeError(
                f"stationary_distribution must be a Tensor, "
                f"got {type(stationary_distribution).__name__}"
            )
        if stationary_distribution.shape[-1] != n_states:
            raise ValueError(
                f"stationary_distribution last dimension ({stationary_distribution.shape[-1]}) "
                f"must match n_states ({n_states})"
            )

    # Compute H∞ = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ
    # = Σᵢ πᵢ H(P[i, :])
    # where H(P[i, :]) is the entropy of the i-th row

    # Use log base conversion if needed
    if base is not None:
        log_base = math.log(base)
    else:
        log_base = 1.0

    # Compute -P * log(P) element-wise, handling zeros
    P = transition_matrix.float()
    pi = stationary_distribution.float()

    # Compute P * log(P), with 0 * log(0) = 0
    P_safe = torch.clamp(P, min=1e-30)
    P_log_P = P * torch.log(P_safe)
    # Where P is zero, set contribution to zero
    P_log_P = torch.where(P > 0, P_log_P, torch.zeros_like(P_log_P))

    # Row entropies: H(P[i, :]) = -Σⱼ P[i,j] log P[i,j]
    row_entropies = -P_log_P.sum(dim=-1)

    # Weighted by stationary distribution
    # H∞ = Σᵢ πᵢ H(P[i, :])
    entropy = (pi * row_entropies).sum(dim=-1)

    # Convert to requested base
    if base is not None:
        entropy = entropy / log_base

    return entropy


def _compute_stationary_distribution(P: Tensor) -> Tensor:
    """Compute stationary distribution via power iteration.

    Parameters
    ----------
    P : Tensor
        Row-stochastic transition matrix. Shape: (..., n, n)

    Returns
    -------
    Tensor
        Stationary distribution. Shape: (..., n)
    """
    n = P.shape[-1]
    batch_shape = P.shape[:-2]

    # Start with uniform distribution
    pi = torch.ones(*batch_shape, n, dtype=P.dtype, device=P.device) / n

    # Power iteration: π' = π @ P
    # For a row-stochastic matrix, stationary distribution satisfies π @ P = π
    for _ in range(100):  # Usually converges in ~20-30 iterations
        pi_new = torch.einsum("...i,...ij->...j", pi, P)
        if torch.allclose(pi, pi_new, atol=1e-10):
            break
        pi = pi_new

    # Normalize to ensure it sums to 1
    pi = pi / pi.sum(dim=-1, keepdim=True)

    return pi
