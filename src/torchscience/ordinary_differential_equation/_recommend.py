"""Solver recommendation based on problem analysis.

This module provides utilities for analyzing ODE problems and recommending
appropriate solvers based on problem characteristics like stiffness, structure,
and desired accuracy.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


def _estimate_jacobian_eigenvalues(
    f: Callable,
    y: torch.Tensor,
    t: float,
    n_samples: int = 5,
) -> torch.Tensor:
    """
    Estimate Jacobian eigenvalues via finite differences.

    Uses random probing to estimate eigenvalue magnitudes without forming
    the full Jacobian matrix.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y) -> dy/dt.
    y : Tensor
        Current state.
    t : float
        Current time.
    n_samples : int
        Number of random probing directions.

    Returns
    -------
    Tensor
        Estimated eigenvalues from Rayleigh quotient probing.
    """
    eps = 1e-7 * (1.0 + torch.abs(y).max().item())

    f0 = f(t, y)
    eigenvalue_estimates = []

    for _ in range(n_samples):
        # Random direction
        v = torch.randn_like(y)
        v_norm = torch.norm(v)
        if v_norm < 1e-10:
            continue
        v = v / v_norm

        # Directional derivative: J @ v ≈ (f(y + eps*v) - f(y)) / eps
        f_perturbed = f(t, y + eps * v)
        Jv = (f_perturbed - f0) / eps

        # Rayleigh quotient estimate
        eigenvalue = torch.dot(v.flatten(), Jv.flatten()).item()
        eigenvalue_estimates.append(eigenvalue)

    if not eigenvalue_estimates:
        return torch.tensor([0.0])

    return torch.tensor(eigenvalue_estimates)


def analyze_problem(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    n_samples: int = 10,
) -> Dict[str, Union[bool, float, int, str, None]]:
    """
    Analyze an ODE problem to determine its characteristics.

    Uses Jacobian eigenvalue sampling to estimate stiffness and detect
    special structure like Hamiltonian systems.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y) -> dy/dt.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval.
    n_samples : int
        Number of sample points for eigenvalue estimation.

    Returns
    -------
    dict
        Analysis results with keys:

        - stiff : bool
            Whether problem appears stiff (stiffness ratio > 100 or
            very negative eigenvalues).
        - stiffness_ratio : float
            Ratio of largest to smallest eigenvalue magnitudes.
        - dimension : int
            State dimension (total number of elements).
        - has_structure : str or None
            Detected structure ("hamiltonian") or None.
        - max_eigenvalue : float
            Estimated maximum eigenvalue magnitude.
        - min_eigenvalue : float
            Estimated minimum non-zero eigenvalue magnitude.

    Examples
    --------
    >>> def decay(t, y):
    ...     return -y
    >>> y0 = torch.tensor([1.0])
    >>> analysis = analyze_problem(decay, y0, (0, 1))
    >>> analysis["stiff"]
    False

    >>> def stiff_decay(t, y):
    ...     return -1000 * y
    >>> analysis = analyze_problem(stiff_decay, y0, (0, 1))
    >>> analysis["stiff"]
    True
    """
    t0, t1 = t_span

    # Handle TensorDict
    is_tensordict = isinstance(y0, TensorDict)
    y_flat, unflatten = flatten_state(y0)

    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    # Sample at multiple points along the trajectory
    eigenvalues_all = []
    sample_times = torch.linspace(
        t0, t0 + 0.1 * abs(t1 - t0), min(n_samples, 5)
    )

    y = y_flat.clone()
    for t in sample_times:
        try:
            eigs = _estimate_jacobian_eigenvalues(f_flat, y, t.item())
            eigenvalues_all.extend(eigs.tolist())
        except Exception:
            # Skip samples that fail
            continue

    if not eigenvalues_all:
        # Fallback if all samples failed
        eigenvalues_all = [0.0]

    eigenvalues = torch.tensor(eigenvalues_all)

    # Compute stiffness metrics
    abs_eigenvalues = torch.abs(eigenvalues)
    max_eig = abs_eigenvalues.max().item()

    # Find minimum non-zero eigenvalue
    nonzero_mask = abs_eigenvalues > 1e-10
    if nonzero_mask.any():
        min_eig = abs_eigenvalues[nonzero_mask].min().item()
    else:
        min_eig = 1e-10

    stiffness_ratio = max_eig / min_eig if min_eig > 0 else float("inf")

    # Detect stiffness: ratio > 100 or very negative eigenvalues
    is_stiff = stiffness_ratio > 100 or eigenvalues.min().item() < -100

    # Detect structure
    structure = None
    n = y_flat.numel()

    # Check for Hamiltonian structure (purely imaginary eigenvalues)
    # This is a heuristic based on eigenvalue patterns
    if n % 2 == 0:
        # For Hamiltonian systems, Jacobian is often close to skew-symmetric
        # which has purely imaginary eigenvalues
        if max_eig < 1e-6 and abs(eigenvalues.mean().item()) < 1e-6:
            structure = "hamiltonian"

    return {
        "stiff": is_stiff,
        "stiffness_ratio": stiffness_ratio,
        "dimension": n,
        "has_structure": structure,
        "max_eigenvalue": max_eig,
        "min_eigenvalue": min_eig,
    }


def recommend_solver(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    *,
    hint: Optional[str] = None,
    accuracy: str = "medium",
) -> str:
    """
    Recommend an ODE solver based on problem characteristics.

    Analyzes the problem using Jacobian eigenvalue sampling and returns
    the name of an appropriate solver.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y) -> dy/dt.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval.
    hint : str, optional
        Problem type hint to skip analysis. Options:

        - "hamiltonian" : Recommend symplectic integrators
        - "neural_ode" : Recommend memory-efficient solvers
        - "stiff" : Recommend implicit solvers

    accuracy : str
        Desired accuracy level. Options: "low", "medium", "high".

    Returns
    -------
    str
        Name of recommended solver. One of:

        - Explicit: "euler", "midpoint", "runge_kutta_4", "dormand_prince_5", "dop853"
        - Implicit: "backward_euler", "bdf", "radau"
        - Symplectic: "stormer_verlet", "yoshida4", "implicit_midpoint"
        - Neural ODE: "reversible_heun", "asynchronous_leapfrog"

    Examples
    --------
    Automatic analysis:

    >>> def decay(t, y):
    ...     return -y
    >>> y0 = torch.tensor([1.0])
    >>> recommend_solver(decay, y0, (0, 10))
    'dormand_prince_5'

    With hint:

    >>> recommend_solver(decay, y0, (0, 10), hint="neural_ode")
    'reversible_heun'

    Stiff problem:

    >>> def stiff(t, y):
    ...     return -1000 * y
    >>> recommend_solver(stiff, y0, (0, 1))
    'bdf'

    High accuracy:

    >>> recommend_solver(decay, y0, (0, 1), accuracy="high")
    'dop853'
    """
    # Use hint if provided
    if hint == "hamiltonian":
        if accuracy == "high":
            return "yoshida4"
        return "stormer_verlet"

    if hint == "neural_ode":
        return "reversible_heun"

    if hint == "stiff":
        return "bdf"

    # Analyze problem
    analysis = analyze_problem(f, y0, t_span)

    # Make recommendation based on analysis
    if analysis["stiff"]:
        # Stiff problem
        if analysis["stiffness_ratio"] > 1e6:
            return "radau"  # Very stiff
        return "bdf"  # Moderately stiff

    if analysis["has_structure"] == "hamiltonian":
        if accuracy == "high":
            return "yoshida4"
        return "stormer_verlet"

    # Non-stiff, general problem
    if accuracy == "high":
        return "dop853"
    elif accuracy == "low":
        return "runge_kutta_4"
    else:
        return "dormand_prince_5"
