"""Blahut-Arimoto algorithm for channel capacity and rate-distortion."""

import math
from typing import Literal

import torch
from torch import Tensor


def blahut_arimoto(
    matrix: Tensor,
    *,
    mode: Literal["capacity", "rate_distortion"] = "capacity",
    source_distribution: Tensor | None = None,
    lagrange_multiplier: float | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    return_distribution: bool = False,
    base: float | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Blahut-Arimoto algorithm for channel capacity or rate-distortion.

    This iterative algorithm computes:

    **Capacity mode (default):**
    Given a channel transition matrix P(y|x), computes the channel capacity:

    .. math::

        C = \max_{p(x)} I(X;Y)

    **Rate-distortion mode:**
    Given a distortion matrix d(x,y) and source distribution p(x), computes
    the rate-distortion function value for a given Lagrange multiplier β:

    .. math::

        R(D) = \min_{p(y|x): E[d] \leq D} I(X;Y)

    Parameters
    ----------
    matrix : Tensor
        For capacity mode: channel transition matrix P(y|x) where matrix[x,y]
        is the probability of output y given input x. Shape: ``(..., n_inputs, n_outputs)``.
        Rows must sum to 1.
        For rate-distortion mode: distortion matrix d(x,y) where matrix[x,y]
        is the distortion from source x to reconstruction y.
    mode : {"capacity", "rate_distortion"}, default="capacity"
        Algorithm mode.
    source_distribution : Tensor or None, default=None
        For rate-distortion mode: source distribution p(x). Required for
        rate-distortion, ignored for capacity.
    lagrange_multiplier : float or None, default=None
        For rate-distortion mode: the Lagrange multiplier β controlling
        the rate-distortion tradeoff. Larger β gives lower rate/higher distortion.
        Required for rate-distortion.
    max_iters : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance for capacity/rate change.
    return_distribution : bool, default=False
        If True, also return the optimal input distribution (capacity mode)
        or conditional distribution p(y|x) (rate-distortion mode).
    base : float or None, default=None
        Logarithm base for the output. If None, uses natural logarithm (nats).
        Use 2 for bits.

    Returns
    -------
    result : Tensor
        Channel capacity (capacity mode) or rate (rate-distortion mode).
    distribution : Tensor (if return_distribution=True)
        Optimal input distribution p(x) (capacity) or p(y|x) (rate-distortion).

    Examples
    --------
    >>> import torch
    >>> # Binary symmetric channel with crossover probability 0.1
    >>> p = 0.1
    >>> P = torch.tensor([[1-p, p], [p, 1-p]])
    >>> capacity = blahut_arimoto(P, base=2.0)
    >>> capacity
    tensor(0.5310)  # C = 1 - H(0.1) bits

    >>> # Get optimal input distribution
    >>> C, px = blahut_arimoto(P, return_distribution=True, base=2.0)
    >>> px
    tensor([0.5000, 0.5000])  # Uniform is optimal for BSC

    Notes
    -----
    The algorithm alternates between:
    1. Fixing p(x) and computing the optimal channel output distribution
    2. Fixing the output distribution and updating p(x)

    Convergence is guaranteed for discrete channels with finite alphabets.

    See Also
    --------
    channel_capacity : Convenience wrapper for capacity computation.
    rate_distortion_function : Convenience wrapper for R-D computation.
    mutual_information : Compute I(X;Y) directly.

    References
    ----------
    .. [1] Blahut, R. (1972). Computation of channel capacity and rate-distortion
           functions. IEEE Trans. Inform. Theory, 18(4), 460-473.
    .. [2] Arimoto, S. (1972). An algorithm for computing the capacity of
           arbitrary discrete memoryless channels. IEEE Trans. Inform. Theory,
           18(1), 14-20.
    """
    if not isinstance(matrix, Tensor):
        raise TypeError(
            f"matrix must be a Tensor, got {type(matrix).__name__}"
        )

    if matrix.dim() < 2:
        raise ValueError("matrix must have at least 2 dimensions")

    if mode == "capacity":
        return _blahut_arimoto_capacity(
            matrix,
            max_iters=max_iters,
            tol=tol,
            return_distribution=return_distribution,
            base=base,
        )
    elif mode == "rate_distortion":
        if source_distribution is None:
            raise ValueError(
                "source_distribution is required for rate_distortion mode"
            )
        if lagrange_multiplier is None:
            raise ValueError(
                "lagrange_multiplier is required for rate_distortion mode"
            )
        return _blahut_arimoto_rate_distortion(
            matrix,
            source_distribution,
            lagrange_multiplier,
            max_iters=max_iters,
            tol=tol,
            return_distribution=return_distribution,
            base=base,
        )
    else:
        raise ValueError(
            f"mode must be 'capacity' or 'rate_distortion', got {mode}"
        )


def _blahut_arimoto_capacity(
    P: Tensor,
    *,
    max_iters: int,
    tol: float,
    return_distribution: bool,
    base: float | None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Blahut-Arimoto for channel capacity.

    P: transition matrix P(y|x), shape (..., n_inputs, n_outputs)
    """
    n_inputs = P.shape[-2]
    n_outputs = P.shape[-1]
    batch_shape = P.shape[:-2]

    # Convert to float
    P = P.float()

    # Initialize uniform input distribution
    px = (
        torch.ones((*batch_shape, n_inputs), dtype=P.dtype, device=P.device)
        / n_inputs
    )

    prev_capacity = torch.zeros(batch_shape, dtype=P.dtype, device=P.device)

    for _ in range(max_iters):
        # Step 1: Compute output distribution p(y) = Σ_x p(x) P(y|x)
        # py[..., j] = Σ_i px[..., i] * P[..., i, j]
        py = torch.einsum("...i,...ij->...j", px, P)

        # Step 2: Compute c(x) = Σ_y P(y|x) log(P(y|x) / p(y))
        # This is the mutual information contribution from each input x
        # c[..., i] = Σ_j P[..., i, j] * log(P[..., i, j] / py[..., j])

        # Avoid log(0) by clamping
        P_safe = torch.clamp(P, min=1e-30)
        py_safe = torch.clamp(py, min=1e-30)

        # log(P(y|x) / p(y))
        log_ratio = torch.log(P_safe) - torch.log(py_safe.unsqueeze(-2))

        # c(x) = Σ_y P(y|x) log(P(y|x) / p(y))
        # Only sum where P > 0
        cx = (P * log_ratio).sum(dim=-1)
        cx = torch.where(P.sum(dim=-1) > 0, cx, torch.zeros_like(cx))

        # Step 3: Update p(x) proportional to exp(c(x))
        px_new = px * torch.exp(cx)
        px_new = px_new / px_new.sum(dim=-1, keepdim=True)

        # Compute capacity: C = Σ_x p(x) c(x)
        capacity = (px_new * cx).sum(dim=-1)

        # Check convergence
        if torch.allclose(capacity, prev_capacity, atol=tol):
            break

        px = px_new
        prev_capacity = capacity

    # Convert to requested base
    if base is not None:
        capacity = capacity / math.log(base)

    if return_distribution:
        return capacity, px
    return capacity


def _blahut_arimoto_rate_distortion(
    d: Tensor,
    px: Tensor,
    beta: float,
    *,
    max_iters: int,
    tol: float,
    return_distribution: bool,
    base: float | None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Blahut-Arimoto for rate-distortion.

    d: distortion matrix d(x,y), shape (..., n_source, n_repr)
    px: source distribution p(x), shape (..., n_source)
    beta: Lagrange multiplier (temperature)
    """
    n_source = d.shape[-2]
    n_repr = d.shape[-1]
    batch_shape = d.shape[:-2]

    # Convert to float
    d = d.float()
    px = px.float()

    # Initialize uniform conditional distribution p(y|x)
    pyx = torch.ones(
        (*batch_shape, n_source, n_repr), dtype=d.dtype, device=d.device
    )
    pyx = pyx / n_repr

    prev_rate = torch.full(
        batch_shape if batch_shape else (1,),
        float("inf"),
        dtype=d.dtype,
        device=d.device,
    )
    if not batch_shape:
        prev_rate = prev_rate.squeeze(0)

    for _ in range(max_iters):
        # Step 1: Compute marginal p(y) = Σ_x p(x) p(y|x)
        py = torch.einsum("...i,...ij->...j", px, pyx)

        # Step 2: Compute rate I(X;Y)
        # I(X;Y) = Σ_x,y p(x) p(y|x) log(p(y|x) / p(y))
        pyx_safe = torch.clamp(pyx, min=1e-30)
        py_safe = torch.clamp(py, min=1e-30)

        log_ratio = torch.log(pyx_safe) - torch.log(py_safe.unsqueeze(-2))

        # I = Σ_x p(x) Σ_y p(y|x) log(p(y|x) / p(y))
        rate = (px.unsqueeze(-1) * pyx * log_ratio).sum(dim=(-2, -1))

        # Step 3: Update p(y|x) ∝ p(y) exp(-β d(x,y))
        log_pyx_new = torch.log(py_safe.unsqueeze(-2)) - beta * d
        # Normalize over y
        pyx_new = torch.softmax(log_pyx_new, dim=-1)

        # Check convergence
        if torch.allclose(rate, prev_rate, atol=tol):
            break

        pyx = pyx_new
        prev_rate = rate

    # Convert to requested base
    if base is not None:
        rate = rate / math.log(base)

    if return_distribution:
        return rate, pyx
    return rate
