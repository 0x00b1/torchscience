"""Channel capacity computation."""

from torch import Tensor

from ._blahut_arimoto import blahut_arimoto


def channel_capacity(
    transition_matrix: Tensor,
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
    return_distribution: bool = False,
    base: float | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Compute channel capacity of a discrete memoryless channel.

    The channel capacity is the maximum mutual information achievable
    over all input distributions:

    .. math::

        C = \max_{p(x)} I(X;Y)

    where I(X;Y) is the mutual information between input X and output Y,
    and the channel is characterized by the transition probabilities P(y|x).

    Parameters
    ----------
    transition_matrix : Tensor
        Channel transition matrix P(y|x) where transition_matrix[x,y] is
        the probability of output y given input x.
        Shape: ``(..., n_inputs, n_outputs)``. Rows must sum to 1.
    max_iters : int, default=100
        Maximum number of Blahut-Arimoto iterations.
    tol : float, default=1e-6
        Convergence tolerance for capacity change between iterations.
    return_distribution : bool, default=False
        If True, also return the capacity-achieving input distribution.
    base : float or None, default=None
        Logarithm base for the output. If None, uses natural logarithm (nats).
        Use 2 for bits.

    Returns
    -------
    capacity : Tensor
        Channel capacity C. Shape is the input shape with last two
        dimensions removed.
    distribution : Tensor (if return_distribution=True)
        Optimal input distribution p*(x) that achieves the capacity.

    Examples
    --------
    >>> import torch
    >>> # Noiseless binary channel (identity matrix)
    >>> P = torch.eye(2)
    >>> channel_capacity(P, base=2.0)
    tensor(1.)  # 1 bit

    >>> # Binary symmetric channel with crossover probability 0.1
    >>> p = 0.1
    >>> P = torch.tensor([[1-p, p], [p, 1-p]])
    >>> channel_capacity(P, base=2.0)
    tensor(0.5310)  # C = 1 - H(p) bits

    >>> # Binary erasure channel with erasure probability 0.2
    >>> e = 0.2
    >>> P = torch.tensor([[1-e, 0, e], [0, 1-e, e]])
    >>> channel_capacity(P, base=2.0)
    tensor(0.8000)  # C = 1 - e bits

    Notes
    -----
    This function uses the Blahut-Arimoto algorithm, which is an iterative
    method that converges to the true capacity from any starting distribution.

    Special cases with known capacities:
    - Noiseless channel (identity P): C = log(n_inputs)
    - Binary symmetric channel: C = 1 - H(p) bits
    - Binary erasure channel: C = 1 - ε bits
    - Useless channel (all rows equal): C = 0

    See Also
    --------
    blahut_arimoto : General algorithm for capacity and rate-distortion.
    mutual_information : Compute I(X;Y) for specific distributions.
    rate_distortion_function : Dual problem of channel capacity.

    References
    ----------
    .. [1] Shannon, C. E. (1948). A Mathematical Theory of Communication.
           Bell System Technical Journal, 27(3), 379-423.
    """
    return blahut_arimoto(
        transition_matrix,
        mode="capacity",
        max_iters=max_iters,
        tol=tol,
        return_distribution=return_distribution,
        base=base,
    )
