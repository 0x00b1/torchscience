"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Solvers
-------
euler
    Forward Euler method (1st order, fixed step, explicit).
    Simplest method, educational baseline.

midpoint
    Explicit midpoint method (2nd order, fixed step).
    Good accuracy/cost tradeoff for smooth problems.

runge_kutta_4
    Classic 4th-order Runge-Kutta (fixed step, explicit).
    Widely used workhorse, excellent for non-stiff problems.

dormand_prince_5
    Dormand-Prince 5(4) adaptive method (explicit).
    Production-quality solver with error control.

backward_euler
    Backward Euler method (1st order, fixed step, implicit).
    A-stable, suitable for stiff problems.

Wrappers
--------
adjoint
    Wrap any solver to use the continuous adjoint method for
    memory-efficient gradients. Uses O(1) memory for the autograd
    graph instead of O(n_steps).

Exceptions
----------
ODESolverError
    Base exception for ODE solver errors.

MaxStepsExceeded
    Raised when adaptive solver exceeds max_steps.

StepSizeError
    Raised when adaptive step size falls below dt_min.

ConvergenceError
    Raised when implicit solver Newton iteration fails to converge.

Examples
--------
Basic usage with adaptive solver:

>>> import torch
>>> from torchscience.integration.initial_value_problem import dormand_prince_5
>>>
>>> def decay(t, y):
...     return -y
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))
>>> trajectory = interp(torch.linspace(0, 5, 100))

With learnable parameters (Neural ODE style):

>>> theta = torch.tensor([1.5], requires_grad=True)
>>> def dynamics(t, y):
...     return -theta * y
>>>
>>> y_final, _ = dormand_prince_5(dynamics, y0, t_span=(0.0, 1.0))
>>> loss = y_final.sum()
>>> loss.backward()
>>> print(theta.grad)  # gradient of loss w.r.t. theta

Memory-efficient gradients with adjoint method:

>>> from torchscience.integration.initial_value_problem import adjoint
>>>
>>> adjoint_solver = adjoint(dormand_prince_5)
>>> y_final, _ = adjoint_solver(dynamics, y0, t_span=(0.0, 100.0))
>>> loss = y_final.sum()
>>> loss.backward()  # Uses O(1) memory for autograd graph

With TensorDict state:

>>> from tensordict import TensorDict
>>> def robot_dynamics(t, state):
...     return TensorDict({
...         "position": state["velocity"],
...         "velocity": -state["position"],
...     })
>>>
>>> state0 = TensorDict({
...     "position": torch.tensor([1.0]),
...     "velocity": torch.tensor([0.0]),
... })
>>> state_final, interp = runge_kutta_4(
...     robot_dynamics, state0, t_span=(0.0, 10.0), dt=0.01
... )

Stiff problems with implicit solver:

>>> def stiff_decay(t, y):
...     return -1000 * y  # Stiff coefficient
>>>
>>> y_final, _ = backward_euler(
...     stiff_decay, y0, t_span=(0.0, 1.0), dt=0.1
... )
"""

from torchscience.integration.initial_value_problem._adjoint import adjoint
from torchscience.integration.initial_value_problem._backward_euler import (
    backward_euler,
)
from torchscience.integration.initial_value_problem._batched import (
    solve_ivp_batched,
)
from torchscience.integration.initial_value_problem._cnf import (
    cnf_dynamics,
    exact_trace,
    hutchinson_trace,
)
from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._euler import euler
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)
from torchscience.integration.initial_value_problem._midpoint import midpoint
from torchscience.integration.initial_value_problem._runge_kutta_4 import (
    runge_kutta_4,
)
from torchscience.integration.initial_value_problem._second_order import (
    solve_ivp_hvp,
)
from torchscience.integration.initial_value_problem._sensitivity import (
    solve_ivp_sensitivity,
)
from torchscience.integration.initial_value_problem._solve_ivp import (
    ODESolution,
    solve_ivp,
)

__all__ = [
    # Exceptions
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    # Explicit solvers (ordered by complexity)
    "euler",
    "midpoint",
    "runge_kutta_4",
    "dormand_prince_5",
    # Implicit solvers
    "backward_euler",
    # Wrappers
    "adjoint",
    # Unified API
    "ODESolution",
    "solve_ivp",
    "solve_ivp_batched",
    "solve_ivp_hvp",
    "solve_ivp_sensitivity",
    # CNF utilities
    "cnf_dynamics",
    "exact_trace",
    "hutchinson_trace",
]
