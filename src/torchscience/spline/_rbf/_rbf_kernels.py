"""RBF kernel functions."""

from __future__ import annotations

import torch
from torch import Tensor


def thin_plate_kernel(r: Tensor) -> Tensor:
    """Thin plate spline kernel: r² log(r).

    Parameters
    ----------
    r : Tensor
        Distances, shape (*).

    Returns
    -------
    Tensor
        Kernel values, same shape as r.
    """
    # Handle r=0 case: r² log(r) -> 0 as r -> 0
    result = torch.where(
        r > 0,
        r**2 * torch.log(r),
        torch.zeros_like(r),
    )
    return result


def gaussian_kernel(r: Tensor, epsilon: float) -> Tensor:
    """Gaussian kernel: exp(-ε²r²).

    Parameters
    ----------
    r : Tensor
        Distances.
    epsilon : float
        Shape parameter.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return torch.exp(-(epsilon**2) * r**2)


def multiquadric_kernel(r: Tensor, epsilon: float) -> Tensor:
    """Multiquadric kernel: sqrt(1 + ε²r²).

    Parameters
    ----------
    r : Tensor
        Distances.
    epsilon : float
        Shape parameter.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return torch.sqrt(1 + epsilon**2 * r**2)


def inverse_quadratic_kernel(r: Tensor, epsilon: float) -> Tensor:
    """Inverse quadratic kernel: 1/(1 + ε²r²).

    Parameters
    ----------
    r : Tensor
        Distances.
    epsilon : float
        Shape parameter.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return 1 / (1 + epsilon**2 * r**2)


def inverse_multiquadric_kernel(r: Tensor, epsilon: float) -> Tensor:
    """Inverse multiquadric kernel: 1/sqrt(1 + ε²r²).

    Parameters
    ----------
    r : Tensor
        Distances.
    epsilon : float
        Shape parameter.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return 1 / torch.sqrt(1 + epsilon**2 * r**2)


def cubic_kernel(r: Tensor) -> Tensor:
    """Cubic kernel: r³.

    Parameters
    ----------
    r : Tensor
        Distances.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return r**3


def linear_kernel(r: Tensor) -> Tensor:
    """Linear kernel: r.

    Parameters
    ----------
    r : Tensor
        Distances.

    Returns
    -------
    Tensor
        Kernel values.
    """
    return r


KERNEL_FUNCTIONS = {
    "thin_plate": thin_plate_kernel,
    "gaussian": gaussian_kernel,
    "multiquadric": multiquadric_kernel,
    "inverse_quadratic": inverse_quadratic_kernel,
    "inverse_multiquadric": inverse_multiquadric_kernel,
    "cubic": cubic_kernel,
    "linear": linear_kernel,
}

# Kernels that require epsilon parameter
KERNELS_WITH_EPSILON = {
    "gaussian",
    "multiquadric",
    "inverse_quadratic",
    "inverse_multiquadric",
}

# Kernels that are conditionally positive definite (require polynomial terms)
CONDITIONALLY_POSITIVE_DEFINITE = {
    "thin_plate",  # Order 2 polynomial
    "cubic",  # Order 2 polynomial
    "linear",  # Order 1 polynomial
}


def evaluate_kernel(
    r: Tensor,
    kernel: str,
    epsilon: float | None = None,
) -> Tensor:
    """Evaluate RBF kernel at distances.

    Parameters
    ----------
    r : Tensor
        Pairwise distances.
    kernel : str
        Kernel name.
    epsilon : float, optional
        Shape parameter for kernels that need it.

    Returns
    -------
    Tensor
        Kernel values.
    """
    if kernel not in KERNEL_FUNCTIONS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Available: {list(KERNEL_FUNCTIONS.keys())}"
        )

    if kernel in KERNELS_WITH_EPSILON:
        if epsilon is None:
            raise ValueError(f"Kernel '{kernel}' requires epsilon parameter")
        return KERNEL_FUNCTIONS[kernel](r, epsilon)
    else:
        return KERNEL_FUNCTIONS[kernel](r)
