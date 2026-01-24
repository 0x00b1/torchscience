"""Utilities for torch.compile optimization of ODE solvers."""

from typing import Callable, Optional

import torch

# Set of solvers known to be compile-compatible
_COMPILE_COMPATIBLE_SOLVERS = {
    "euler",
    "midpoint",
    "runge_kutta_4",
    "reversible_heun",
    "asynchronous_leapfrog",
    "stormer_verlet",
    "yoshida4",
}

# Solvers that require dynamic=True due to adaptive stepping
_DYNAMIC_SOLVERS = {
    "dormand_prince_5",
    "bdf",
    "radau",
}


def is_compile_compatible(solver: Callable) -> bool:
    """
    Check if a solver is compatible with torch.compile.

    Parameters
    ----------
    solver : callable
        An ODE solver function.

    Returns
    -------
    bool
        True if the solver can be compiled without issues.

    Notes
    -----
    Fixed-step explicit solvers are generally compile-compatible.
    Adaptive solvers may require dynamic=True or be incompatible.

    Examples
    --------
    >>> from torchscience.integration.initial_value_problem import euler, dormand_prince_5
    >>> from torchscience.integration.initial_value_problem._compile_utils import (
    ...     is_compile_compatible,
    ... )
    >>> is_compile_compatible(euler)
    True
    >>> is_compile_compatible(dormand_prince_5)  # Adaptive, needs special handling
    False
    """
    name = getattr(solver, "__name__", "")
    return name in _COMPILE_COMPATIBLE_SOLVERS


def compile_solver(
    solver: Callable,
    *,
    mode: Optional[str] = None,
    dynamic: Optional[bool] = None,
    fullgraph: bool = False,
    backend: str = "inductor",
) -> Callable:
    """
    Wrap an ODE solver with torch.compile for potential speedup.

    Parameters
    ----------
    solver : callable
        An ODE solver function (euler, runge_kutta_4, etc.).
    mode : str, optional
        Compilation mode. Options: "default", "reduce-overhead", "max-autotune".
    dynamic : bool, optional
        If True, compile with dynamic shape support.
        Auto-detected based on solver type if not specified.
    fullgraph : bool
        If True, require full graph compilation (no graph breaks).
    backend : str
        Compilation backend. Default is "inductor".

    Returns
    -------
    compiled_solver : callable
        Compiled version of the solver with same signature.

    Examples
    --------
    >>> from torchscience.integration.initial_value_problem import euler
    >>> from torchscience.integration.initial_value_problem._compile_utils import (
    ...     compile_solver,
    ... )
    >>> compiled_euler = compile_solver(euler)
    >>> def f(t, y):
    ...     return -y
    >>> y0 = torch.tensor([1.0])
    >>> y_final, interp = compiled_euler(f, y0, t_span=(0, 1), dt=0.01)

    Notes
    -----
    Performance benefits depend on:

    - Problem size (larger benefits more)
    - Number of steps (more steps = more benefit)
    - Hardware (GPUs benefit more than CPUs)

    First call will be slow due to compilation. Subsequent calls are fast.

    Fixed-step explicit solvers benefit most from compilation. Adaptive
    solvers may see less benefit due to dynamic control flow.
    """
    solver_name = getattr(solver, "__name__", "unknown")

    # Auto-detect dynamic requirement
    if dynamic is None:
        dynamic = solver_name in _DYNAMIC_SOLVERS

    # Build compile options
    compile_options = {
        "backend": backend,
        "fullgraph": fullgraph,
    }
    if mode is not None:
        compile_options["mode"] = mode
    if dynamic:
        compile_options["dynamic"] = dynamic

    # Compile the solver
    compiled = torch.compile(solver, **compile_options)

    # Preserve metadata
    compiled.__name__ = f"compiled({solver_name})"
    compiled.__doc__ = f"""
Compiled version of {solver_name}.

This solver has been wrapped with torch.compile for potential speedup.
First call will be slow (compilation), subsequent calls are fast.

Compilation options:
- backend: {backend}
- mode: {mode}
- dynamic: {dynamic}
- fullgraph: {fullgraph}

Original docstring:
{solver.__doc__ or "No documentation available."}
"""

    return compiled
