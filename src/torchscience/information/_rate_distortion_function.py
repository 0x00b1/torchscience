"""Rate-distortion function computation."""

from torch import Tensor

from ._blahut_arimoto import blahut_arimoto


def rate_distortion_function(
    source_distribution: Tensor,
    distortion_matrix: Tensor,
    lagrange_multiplier: float,
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
    return_distribution: bool = False,
    base: float | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Compute rate-distortion function using Blahut-Arimoto algorithm.

    The rate-distortion function R(D) gives the minimum rate (bits per symbol)
    needed to represent a source with expected distortion at most D:

    .. math::

        R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X;\hat{X})

    This function computes the rate for a given Lagrange multiplier β,
    which implicitly determines the distortion level D.

    Parameters
    ----------
    source_distribution : Tensor
        Source distribution p(x). Shape: ``(..., n_source)``.
        Must sum to 1 along the last dimension.
    distortion_matrix : Tensor
        Distortion matrix d(x, y) where distortion_matrix[x, y] is the
        distortion when source symbol x is represented as y.
        Shape: ``(..., n_source, n_repr)``.
    lagrange_multiplier : float
        Lagrange multiplier β controlling the rate-distortion tradeoff.
        Larger β gives lower rate (and higher distortion).
        β = 0 gives maximum rate (≈ H(X)), β → ∞ gives rate = 0.
    max_iters : int, default=100
        Maximum number of Blahut-Arimoto iterations.
    tol : float, default=1e-6
        Convergence tolerance for rate change between iterations.
    return_distribution : bool, default=False
        If True, also return the optimal reproduction distribution p(y|x).
    base : float or None, default=None
        Logarithm base for the output. If None, uses natural logarithm (nats).
        Use 2 for bits.

    Returns
    -------
    rate : Tensor
        Rate R at the given Lagrange multiplier. Shape is the input shape
        with last dimension removed from source_distribution.
    distribution : Tensor (if return_distribution=True)
        Optimal conditional distribution p(y|x).

    Examples
    --------
    >>> import torch
    >>> # Binary source with Hamming distortion
    >>> px = torch.tensor([0.5, 0.5])
    >>> d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # Hamming

    >>> # Low beta: high rate, low distortion
    >>> rate_distortion_function(px, d, lagrange_multiplier=5.0, base=2.0)
    tensor(0.7...)  # Close to H(X) = 1 bit

    >>> # High beta: low rate, high distortion
    >>> rate_distortion_function(px, d, lagrange_multiplier=0.1, base=2.0)
    tensor(0.0...)  # Close to 0

    Notes
    -----
    The relationship between β and D is monotonic:
    - β = 0: Maximum distortion, rate ≈ 0
    - β → ∞: Zero distortion, rate = H(X)

    To find R(D) for a specific distortion D, binary search over β
    until the achieved distortion matches D.

    Known rate-distortion functions:
    - Binary source with Hamming distortion: R(D) = H(p) - H(D) for D ≤ min(p, 1-p)
    - Gaussian source with MSE distortion: R(D) = 0.5 * log(σ²/D) for D ≤ σ²

    See Also
    --------
    blahut_arimoto : General algorithm for capacity and rate-distortion.
    channel_capacity : Dual problem of rate-distortion.
    shannon_entropy : Source entropy H(X).

    References
    ----------
    .. [1] Shannon, C. E. (1959). Coding theorems for a discrete source
           with a fidelity criterion. IRE Nat. Conv. Rec., 4, 142-163.
    """
    return blahut_arimoto(
        distortion_matrix,
        mode="rate_distortion",
        source_distribution=source_distribution,
        lagrange_multiplier=lagrange_multiplier,
        max_iters=max_iters,
        tol=tol,
        return_distribution=return_distribution,
        base=base,
    )
