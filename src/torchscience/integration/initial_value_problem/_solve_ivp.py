"""Unified ODE solver API with optional sensitivity methods."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._adjoint import adjoint
from torchscience.integration.initial_value_problem._backward_euler import (
    backward_euler,
)
from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._euler import euler
from torchscience.integration.initial_value_problem._event_handling import (
    EventTracker,
    parse_events,
)
from torchscience.integration.initial_value_problem._midpoint import midpoint
from torchscience.integration.initial_value_problem._runge_kutta_4 import (
    runge_kutta_4,
)

# Method name to solver function mapping
_METHOD_MAP: Dict[str, Callable] = {
    # Dormand-Prince 5(4) adaptive method
    "dormand_prince_5": dormand_prince_5,
    "dp5": dormand_prince_5,
    # Classic 4th-order Runge-Kutta (fixed step)
    "runge_kutta_4": runge_kutta_4,
    "rk4": runge_kutta_4,
    # Forward Euler (1st order, fixed step)
    "euler": euler,
    # Explicit midpoint method (2nd order)
    "midpoint": midpoint,
    # Backward Euler (implicit, 1st order)
    "backward_euler": backward_euler,
}


@dataclass
class ODESolution:
    """
    Solution object returned by solve_ivp.

    This dataclass stores the result of an ODE integration, including the
    final state, interpolant, and solver statistics. It supports tuple
    unpacking for backwards compatibility with (y_final, interp) returns.

    Attributes
    ----------
    y_final : Tensor or TensorDict
        State at the final time t1.
    interp : callable
        Interpolant function that returns y(t) for any t in [t0, t1].
    y_eval : Tensor, optional
        States evaluated at t_eval points, if t_eval was provided.
    t_eval : Tensor, optional
        Time points where y_eval was computed.
    n_steps : int
        Number of integration steps taken.
    n_function_evals : int, optional
        Number of dynamics function evaluations (None if not available).
    success : bool
        Whether the integration completed successfully.
    message : str
        Status message describing the outcome.
    stats : dict
        Additional solver statistics (solver-specific).

    Examples
    --------
    >>> result = solve_ivp(f, y0, t_span)
    >>> print(result.y_final)  # Access final state
    >>> trajectory = result.interp(torch.linspace(0, 1, 100))  # Dense output

    >>> # Backwards compatible tuple unpacking
    >>> y_final, interp = solve_ivp(f, y0, t_span)
    """

    y_final: Union[torch.Tensor, TensorDict]
    interp: Callable[
        [Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]
    ]
    y_eval: Optional[Union[torch.Tensor, TensorDict]] = None
    t_eval: Optional[torch.Tensor] = None
    n_steps: int = 0
    n_function_evals: Optional[int] = None
    success: bool = True
    message: str = "Integration successful."
    stats: Optional[Dict[str, Any]] = None
    t_events: Optional[List[List[float]]] = None
    y_events: Optional[List[List[torch.Tensor]]] = None

    def __post_init__(self) -> None:
        """Initialize empty stats dict if not provided."""
        if self.stats is None:
            self.stats = {}

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking: y_final, interp = result."""
        yield self.y_final
        yield self.interp

    def __getitem__(self, index: int) -> Any:
        """Support indexing: result[0], result[1]."""
        if index == 0:
            return self.y_final
        elif index == 1:
            return self.interp
        else:
            raise IndexError(f"ODESolution index {index} out of range (0-1)")

    def __len__(self) -> int:
        """Return 2 for backwards compatibility with (y_final, interp) tuple."""
        return 2


def solve_ivp(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    t_eval: Optional[torch.Tensor] = None,
    method: str = "dormand_prince_5",
    sensitivity: Optional[str] = None,
    params: Optional[List[torch.Tensor]] = None,
    checkpoints: Optional[int] = None,
    adjoint_options: Optional[Dict[str, Any]] = None,
    events: Optional[List[Callable]] = None,
    **solver_kwargs: Any,
) -> ODESolution:
    """
    Solve an initial value problem for a system of ODEs.

    This is a unified interface that wraps all available ODE solvers with
    optional sensitivity methods for efficient gradient computation.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state at t0.
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    t_eval : Tensor, optional
        Times at which to evaluate the solution. If None, only the final
        state is returned via y_final; y_eval will be None.
    method : str
        Integration method. Available options:
        - 'dormand_prince_5' or 'dp5': Adaptive 5(4) method (default)
        - 'runge_kutta_4' or 'rk4': Classic 4th-order RK (fixed step, requires dt)
        - 'euler': Forward Euler (1st order, fixed step, requires dt)
    sensitivity : str, optional
        Gradient computation method:
        - None: Use standard PyTorch autograd (default)
        - 'adjoint': Use continuous adjoint method for O(1) memory
        - 'checkpoint': Use checkpointed adjoint (requires checkpoints parameter)
        - 'binomial': Use binomial checkpointing (Revolve algorithm) for
          O(log n) memory with O(n log n) recomputation
    params : list of Tensor, optional
        List of parameters requiring gradients. Required when sensitivity is
        'adjoint' or 'checkpoint'. If None, parameters are extracted from
        the dynamics function's closure (may be unreliable).
    checkpoints : int, optional
        Number of checkpoints for checkpointed adjoint. Required when
        sensitivity='checkpoint'. Ignored otherwise.
    adjoint_options : dict, optional
        Options for adjoint integration. Supported keys:
        - 'method': str, integration method ('rk4' or 'euler'). Default: 'rk4'
        - 'n_steps': int, number of integration steps. Default: 100
    events : list of callable, optional
        Event functions g(t, y) to track during integration. Events are detected
        when g changes sign (zero-crossing). Each function may have attributes:
        - terminal: bool, if True, stop integration when event occurs
        - direction: int, 0=both directions, 1=positive, -1=negative
    **solver_kwargs
        Additional keyword arguments passed to the solver. Common options:
        - dt: float, step size for fixed-step methods (euler, runge_kutta_4)
        - rtol: float, relative tolerance for adaptive methods
        - atol: float, absolute tolerance for adaptive methods
        - max_steps: int, maximum number of steps
        - throw: bool, whether to raise on solver failures

    Returns
    -------
    result : ODESolution
        Solution object with attributes:
        - y_final: Final state at t1
        - interp: Interpolant function for dense output
        - y_eval: States at t_eval points (if t_eval provided)
        - t_eval: Time points (if t_eval provided)
        - n_steps: Number of integration steps
        - n_function_evals: Number of f evaluations
        - success: Whether integration succeeded
        - message: Status message
        - stats: Additional solver statistics
        - t_events: List of event times for each event function
        - y_events: List of states at event times for each event function

    Examples
    --------
    Basic usage with adaptive solver:

    >>> def decay(t, y):
    ...     return -y
    >>> result = solve_ivp(decay, torch.tensor([1.0]), t_span=(0.0, 5.0))
    >>> trajectory = result.interp(torch.linspace(0, 5, 100))

    With learnable parameters (Neural ODE style):

    >>> theta = torch.tensor([1.5], requires_grad=True)
    >>> def dynamics(t, y):
    ...     return -theta * y
    >>> result = solve_ivp(dynamics, y0, t_span=(0.0, 1.0))
    >>> result.y_final.sum().backward()

    Memory-efficient gradients with adjoint method:

    >>> result = solve_ivp(
    ...     dynamics, y0, t_span=(0.0, 100.0),
    ...     sensitivity='adjoint', params=[theta]
    ... )
    >>> result.y_final.sum().backward()  # Uses O(1) memory

    With checkpointed adjoint for very long integrations:

    >>> result = solve_ivp(
    ...     dynamics, y0, t_span=(0.0, 1000.0),
    ...     sensitivity='checkpoint', params=[theta], checkpoints=10
    ... )

    Tuple unpacking (backwards compatible):

    >>> y_final, interp = solve_ivp(f, y0, t_span)

    See Also
    --------
    dormand_prince_5 : Adaptive solver with dense output
    euler : Simple fixed-step solver
    runge_kutta_4 : Classic 4th-order fixed-step solver
    adjoint : Wrapper for adjoint sensitivity method
    """
    # Validate method
    method_lower = method.lower()
    if method_lower not in _METHOD_MAP:
        available = ", ".join(sorted(_METHOD_MAP.keys()))
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {available}"
        )

    solver = _METHOD_MAP[method_lower]

    # Validate sensitivity option
    valid_sensitivities = {None, "adjoint", "checkpoint", "binomial"}
    if sensitivity not in valid_sensitivities:
        raise ValueError(
            f"Invalid sensitivity '{sensitivity}'. "
            f"Valid options: None, 'adjoint', 'checkpoint', 'binomial'"
        )

    # Validate checkpoint parameter
    if sensitivity == "checkpoint" and checkpoints is None:
        raise ValueError(
            "checkpoints parameter is required when sensitivity='checkpoint'"
        )

    # Apply sensitivity wrapper if specified
    if sensitivity == "adjoint":
        solver = adjoint(
            solver,
            checkpoints=None,
            params=params,
            adjoint_options=adjoint_options,
        )
    elif sensitivity == "checkpoint":
        solver = adjoint(
            solver,
            checkpoints=checkpoints,
            params=params,
            adjoint_options=adjoint_options,
        )
    elif sensitivity == "binomial":
        # Use binomial checkpointing with Revolve algorithm for O(log n) memory
        binomial_options = adjoint_options.copy() if adjoint_options else {}
        binomial_options["method"] = "binomial"
        solver = adjoint(
            solver,
            checkpoints=None,  # Binomial determines checkpoints automatically
            params=params,
            adjoint_options=binomial_options,
        )

    # Call the solver
    y_final, interp = solver(f, y0, t_span=t_span, **solver_kwargs)

    # Scan for events if provided
    t_events_result = None
    y_events_result = None
    if events is not None:
        event_specs = parse_events(events)
        tracker = EventTracker(event_specs)

        # Get time points from interpolant to check
        t0, t1 = t_span
        n_check_points = max(100, getattr(interp, "n_steps", 100))
        dt_check = (t1 - t0) / n_check_points

        t_prev = t0
        y_prev = interp(t0) if callable(interp) else y0

        for i in range(n_check_points):
            t_curr = t0 + (i + 1) * dt_check
            if t_curr > t1:
                t_curr = t1
            y_curr = interp(t_curr)

            result = tracker.check_and_handle(
                t_prev, y_prev, t_curr, y_curr, interp
            )

            if result is not None:  # Terminal event
                t_event, y_event = result
                y_final = y_event
                break

            t_prev = t_curr
            y_prev = y_curr

        t_events_result = tracker.t_events
        y_events_result = tracker.y_events

    # Evaluate at t_eval points if provided
    y_eval_result = None
    t_eval_result = None
    if t_eval is not None:
        t_eval_result = t_eval
        y_eval_result = interp(t_eval)

    # Extract statistics from interpolant if available
    n_steps = getattr(interp, "n_steps", 0)
    success_flag = getattr(interp, "success", None)

    # Handle success flag (may be tensor for batched problems)
    if success_flag is None:
        success = True
    elif isinstance(success_flag, torch.Tensor):
        success = success_flag.all().item()
    else:
        success = bool(success_flag)

    # Build stats dict
    stats = {
        "method": method_lower,
        "sensitivity": sensitivity,
    }
    if checkpoints is not None:
        stats["checkpoints"] = checkpoints

    # Build message
    if success:
        message = "Integration successful."
    else:
        message = "Integration may have failed for some batch elements."

    return ODESolution(
        y_final=y_final,
        interp=interp,
        y_eval=y_eval_result,
        t_eval=t_eval_result,
        n_steps=n_steps,
        n_function_evals=None,  # Not easily extractable from current solvers
        success=success,
        message=message,
        stats=stats,
        t_events=t_events_result,
        y_events=y_events_result,
    )
