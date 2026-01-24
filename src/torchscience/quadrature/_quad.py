"""Adaptive quadrature using Gauss-Kronrod rules."""

import heapq
import warnings
from typing import Callable, Tuple, Union

import torch
from torch import Tensor

from torchscience.quadrature._exceptions import (
    IntegrationError,
    QuadratureWarning,
)
from torchscience.quadrature._rules import GaussKronrod


def quad(
    f: Callable[[Tensor], Tensor],
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    *,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
) -> Tensor:
    """
    Compute definite integral using adaptive quadrature.

    Uses adaptive subdivision with Gauss-Kronrod error estimation.

    Parameters
    ----------
    f : callable
        Integrand function.
    a, b : float or Tensor
        Integration bounds (scalars only, not batched).
    epsabs : float
        Absolute error tolerance.
    epsrel : float
        Relative error tolerance.
    limit : int
        Maximum number of subintervals.

    Returns
    -------
    Tensor
        Integral approximation.

    Raises
    ------
    IntegrationError
        If convergence is not achieved within ``limit`` subdivisions.

    Warns
    -----
    QuadratureWarning
        If the error estimate is larger than requested tolerance.

    Notes
    -----
    Differentiable with respect to parameters captured in f's closure.

    **Limitation:** Gradients through integration limits are NOT supported
    for adaptive quadrature. Use ``fixed_quad`` for limit gradients.

    Examples
    --------
    >>> quad(torch.sin, 0, torch.pi)  # approximately 2.0

    >>> # Gradient through closure parameter
    >>> theta = torch.tensor(2.0, requires_grad=True)
    >>> result = quad(lambda x: theta * torch.sin(x), 0, torch.pi)
    >>> result.backward()  # works
    """
    result, error, info = quad_info(
        f, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit
    )

    if not info["converged"]:
        raise IntegrationError(
            f"Integration failed to converge after {info['nsubintervals']} subintervals. "
            f"Error estimate: {error.item():.2e}, "
            f"tolerance: {epsabs + epsrel * abs(result.item()):.2e}"
        )

    return result


def quad_info(
    f: Callable[[Tensor], Tensor],
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    *,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
) -> Tuple[Tensor, Tensor, dict]:
    """
    Like quad, but returns error estimate and info dict.

    Returns
    -------
    result : Tensor
        Integral approximation.
    error : Tensor
        Estimated absolute error.
    info : dict
        Information dict with keys:
        - "neval": Number of function evaluations
        - "nsubintervals": Number of subintervals used
        - "converged": Whether tolerance was achieved
    """
    gk_rule = GaussKronrod(21)  # G10-K21

    # Infer dtype and device
    if isinstance(a, Tensor):
        dtype = a.dtype
        device = a.device
        a_val = a.detach().item()
    elif isinstance(b, Tensor):
        dtype = b.dtype
        device = b.device
        a_val = float(a)
    else:
        dtype = torch.float64
        device = torch.device("cpu")
        a_val = float(a)

    if isinstance(b, Tensor):
        b_val = b.detach().item()
    else:
        b_val = float(b)

    # Create wrapper that tracks gradient through closure
    def f_wrapper(x: Tensor) -> Tensor:
        return f(x)

    # Initial integration on [a, b]
    a_tensor = torch.tensor(a_val, dtype=dtype, device=device)
    b_tensor = torch.tensor(b_val, dtype=dtype, device=device)

    result_ab, error_ab = gk_rule.integrate_with_error(
        f_wrapper, a_tensor, b_tensor
    )

    neval = 21

    # Check if already converged
    tolerance = epsabs + epsrel * torch.abs(result_ab)
    if error_ab <= tolerance:
        return (
            result_ab,
            error_ab,
            {
                "neval": neval,
                "nsubintervals": 1,
                "converged": True,
            },
        )

    # Adaptive subdivision using priority queue (max-heap by error)
    # Each entry: (-error, left, right, result_tensor, error_tensor)
    # We keep tensors to preserve gradient tracking through f

    # Store interval results as tensors for gradient tracking
    interval_results = [result_ab]
    interval_errors = [error_ab]
    interval_bounds = [(a_val, b_val)]

    # Priority queue: (-error, interval_index)
    heap = [(-error_ab.item(), 0)]
    nsubintervals = 1

    while heap and nsubintervals < limit:
        # Pop interval with largest error
        neg_err, idx = heapq.heappop(heap)

        left, right = interval_bounds[idx]

        # Bisect
        mid = (left + right) / 2

        # Integrate left and right halves
        left_a = torch.tensor(left, dtype=dtype, device=device)
        mid_t = torch.tensor(mid, dtype=dtype, device=device)
        right_b = torch.tensor(right, dtype=dtype, device=device)

        result_left, error_left = gk_rule.integrate_with_error(
            f_wrapper, left_a, mid_t
        )
        result_right, error_right = gk_rule.integrate_with_error(
            f_wrapper, mid_t, right_b
        )

        neval += 2 * 21
        nsubintervals += 1

        # Replace old interval with two new ones
        # Update the result for this index to be the sum of children
        interval_results[idx] = result_left + result_right
        interval_errors[idx] = error_left + error_right
        interval_bounds[idx] = None  # Mark as split

        # Add new intervals
        left_idx = len(interval_results)
        interval_results.append(result_left)
        interval_errors.append(error_left)
        interval_bounds.append((left, mid))

        right_idx = len(interval_results)
        interval_results.append(result_right)
        interval_errors.append(error_right)
        interval_bounds.append((mid, right))

        # Push new intervals to heap
        heapq.heappush(heap, (-error_left.item(), left_idx))
        heapq.heappush(heap, (-error_right.item(), right_idx))

        # Compute total result and error from leaf intervals
        total_result = torch.zeros(1, dtype=dtype, device=device)
        total_error = torch.zeros(1, dtype=dtype, device=device)

        for i, bounds in enumerate(interval_bounds):
            if bounds is not None:  # Leaf interval
                total_result = total_result + interval_results[i]
                total_error = total_error + interval_errors[i]

        # Check convergence
        tolerance = epsabs + epsrel * torch.abs(total_result)
        if total_error <= tolerance:
            return (
                total_result.squeeze(),
                total_error.squeeze(),
                {
                    "neval": neval,
                    "nsubintervals": nsubintervals,
                    "converged": True,
                },
            )

    # Did not converge - compute final result from leaves
    total_result = torch.zeros(1, dtype=dtype, device=device)
    total_error = torch.zeros(1, dtype=dtype, device=device)

    for i, bounds in enumerate(interval_bounds):
        if bounds is not None:
            total_result = total_result + interval_results[i]
            total_error = total_error + interval_errors[i]

    warnings.warn(
        f"Quadrature did not converge. Error: {total_error.item():.2e}",
        QuadratureWarning,
    )

    return (
        total_result.squeeze(),
        total_error.squeeze(),
        {
            "neval": neval,
            "nsubintervals": nsubintervals,
            "converged": False,
        },
    )
