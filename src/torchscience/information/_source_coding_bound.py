"""Shannon's source coding bound."""

from torch import Tensor


def source_coding_bound(
    entropy: Tensor,
    n: int = 1,
    *,
    base: float | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute Shannon's source coding bounds on expected code length.

    For n i.i.d. source symbols with entropy H(X), the expected codeword
    length L of any uniquely decodable code satisfies:

    .. math::

        n H(X) \leq \mathbb{E}[L] < n H(X) + 1

    The lower bound is achieved in the limit as n → ∞ using block coding.

    Parameters
    ----------
    entropy : Tensor
        Source entropy H(X). Shape: ``(...)`` for optional batch dimensions.
        Should be non-negative.
    n : int, default=1
        Number of source symbols to encode jointly (block length).
    base : float or None, default=None
        Logarithm base for entropy (e.g., 2 for bits, e for nats).
        If None, assumes entropy is in nats. The bounds are returned
        in the same base as the input entropy.

    Returns
    -------
    lower_bound : Tensor
        Lower bound on expected code length: n * H(X).
    upper_bound : Tensor
        Upper bound on expected code length: n * H(X) + 1.

    Examples
    --------
    >>> import torch
    >>> # Fair coin: H = 1 bit
    >>> entropy = torch.tensor(1.0)  # 1 bit
    >>> lower, upper = source_coding_bound(entropy, n=1, base=2.0)
    >>> lower
    tensor(1.)
    >>> upper
    tensor(2.)

    >>> # Block coding with n=10
    >>> entropy = torch.tensor(1.0)
    >>> lower, upper = source_coding_bound(entropy, n=10, base=2.0)
    >>> lower
    tensor(10.)
    >>> upper
    tensor(11.)
    >>> # Per-symbol bounds: [1.0, 1.1]

    Notes
    -----
    - The lower bound (n * H) represents the fundamental limit set by entropy.
    - The upper bound (n * H + 1) is achievable using Huffman coding.
    - As n → ∞, the per-symbol overhead (1/n) → 0, achieving the entropy limit.
    - For practical codes, the "+1" term is significant for small block sizes.

    The theorem shows that:
    - Lossless compression below the entropy rate is impossible.
    - Compression arbitrarily close to entropy is achievable with block coding.

    See Also
    --------
    huffman_lengths : Compute optimal code lengths.
    shannon_entropy : Compute source entropy.
    kraft_inequality : Verify code length validity.

    References
    ----------
    .. [1] Shannon, C. E. (1948). A Mathematical Theory of Communication.
           Bell System Technical Journal, 27(3), 379-423.
    """
    if not isinstance(entropy, Tensor):
        raise TypeError(
            f"entropy must be a Tensor, got {type(entropy).__name__}"
        )

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    # The bound is: n * H ≤ E[L] < n * H + 1
    # where H is entropy and n is block length

    # Convert to float for computation
    h = entropy.float()

    # Lower bound: n * H
    lower_bound = n * h

    # Upper bound: n * H + 1
    # Note: The +1 is in the same units as the entropy
    # If entropy is in bits (base=2), then +1 is 1 bit
    # If entropy is in nats (base=e), then +1 is 1 nat
    upper_bound = n * h + 1.0

    # If a different base was specified, the user's entropy is already in that base
    # and the bounds should be returned in the same base (which they are, since
    # we just scaled by n and added 1 in the same units)

    return lower_bound, upper_bound
