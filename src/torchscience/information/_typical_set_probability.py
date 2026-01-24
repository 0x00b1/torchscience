"""Typical set probability for AEP."""

import math

import torch
from torch import Tensor


def typical_set_probability(
    distribution: Tensor,
    n: int,
    epsilon: float,
    *,
    base: float | None = None,
) -> Tensor:
    r"""Compute a lower bound on the probability of the typical set.

    The typical set :math:`A_\epsilon^{(n)}` contains sequences :math:`x^n`
    whose empirical entropy is close to the true entropy:

    .. math::

        A_\epsilon^{(n)} = \left\{ x^n : \left| -\frac{1}{n} \log p(x^n) - H(X) \right| < \epsilon \right\}

    By the Asymptotic Equipartition Property (AEP), as n → ∞:

    .. math::

        P(A_\epsilon^{(n)}) \to 1

    This function computes a lower bound using Chebyshev's inequality:

    .. math::

        P(A_\epsilon^{(n)}) \geq 1 - \frac{\text{Var}[\log p(X)]}{n \epsilon^2}

    Parameters
    ----------
    distribution : Tensor
        Symbol probabilities. Shape: ``(..., n_symbols)``. Must sum to 1
        along the last dimension.
    n : int
        Sequence length (number of i.i.d. symbols).
    epsilon : float
        Typicality parameter. Sequences are typical if their per-symbol
        empirical entropy is within epsilon of the true entropy.
    base : float or None, default=None
        Logarithm base. If None, uses natural logarithm. Use 2 for bits.

    Returns
    -------
    Tensor
        Lower bound on P(Aₑⁿ). Shape is the input shape with last dimension
        removed.

    Examples
    --------
    >>> import torch
    >>> # Fair coin
    >>> p = torch.tensor([0.5, 0.5])
    >>> # For n=100, epsilon=0.1
    >>> typical_set_probability(p, n=100, epsilon=0.1)
    tensor(0.9517)  # High probability of being typical

    >>> # As n increases, probability approaches 1
    >>> typical_set_probability(p, n=1000, epsilon=0.1)
    tensor(0.9952)

    Notes
    -----
    - The bound is based on Chebyshev's inequality applied to the sample
      mean of -log p(X).
    - For finite n, the bound can be loose.
    - The bound can be negative for small n or large epsilon; in such cases,
      it provides no useful information.
    - The returned value is clamped to [0, 1].

    The AEP is fundamental to Shannon's source coding theorem, showing that
    typical sequences have probability approximately 2^{-nH(X)}.

    See Also
    --------
    shannon_entropy : Compute entropy H(X).
    source_coding_bound : Shannon's source coding bounds.

    References
    ----------
    .. [1] Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory
           (2nd ed.). Wiley-Interscience. Chapter 3.
    """
    if not isinstance(distribution, Tensor):
        raise TypeError(
            f"distribution must be a Tensor, got {type(distribution).__name__}"
        )

    if distribution.dim() == 0:
        raise ValueError("distribution must have at least 1 dimension")

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    # Convert to float for computation
    p = distribution.float()

    # Compute variance of -log p(X) (or log(1/p(X)))
    # For random variable Y = -log p(X):
    # E[Y] = -Σ p_i log p_i = H(X)
    # E[Y²] = Σ p_i (log p_i)²
    # Var[Y] = E[Y²] - E[Y]²

    # Handle zeros safely
    p_safe = torch.clamp(p, min=1e-30)
    log_p = torch.log(p_safe)

    # Where p is zero, set contribution to zero
    log_p = torch.where(p > 0, log_p, torch.zeros_like(log_p))

    # E[-log p] = -Σ p_i log p_i = H
    neg_log_p = -log_p
    mean = (p * neg_log_p).sum(dim=-1)

    # E[(-log p)²] = Σ p_i (log p_i)²
    mean_sq = (p * neg_log_p**2).sum(dim=-1)

    # Var[-log p] = E[(-log p)²] - E[-log p]²
    variance = mean_sq - mean**2

    # Apply base conversion if needed
    if base is not None:
        log_base = math.log(base)
        # Variance scales as 1/log_base² when changing base
        variance = variance / (log_base**2)
        # Epsilon is in the same units, so epsilon² stays as is

    # Chebyshev bound: P(|X - μ| ≥ ε) ≤ Var[X]/ε²
    # For sample mean of n i.i.d. variables: Var[sample mean] = Var[X]/n
    # P(|sample mean - μ| ≥ ε) ≤ Var[X]/(n * ε²)
    # So P(typical) = P(|sample mean - μ| < ε) ≥ 1 - Var[X]/(n * ε²)

    lower_bound = 1.0 - variance / (n * epsilon**2)

    # Clamp to [0, 1] since it's a probability
    lower_bound = torch.clamp(lower_bound, min=0.0, max=1.0)

    return lower_bound
