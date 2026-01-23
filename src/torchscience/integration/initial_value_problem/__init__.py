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

Stiff Solvers
-------------
bdf
    Variable-order BDF method (orders 1-5). Industry standard for
    stiff problems. Automatically adapts order based on local error.

radau
    Radau IIA method (3-stage, order 5, L-stable). Excellent for
    very stiff problems with high-frequency oscillations.

Symplectic Integrators
----------------------
stormer_verlet
    Störmer-Verlet (velocity Verlet) method. 2nd-order symplectic for
    separable Hamiltonians H = T(p) + V(q). Preserves energy over long
    integrations. Standard choice for molecular dynamics.

yoshida4
    Yoshida 4th-order symplectic integrator. Higher accuracy than Verlet
    with similar structure. Best for separable Hamiltonians requiring
    precision.

implicit_midpoint
    Implicit midpoint method. 2nd-order, both symplectic AND A-stable.
    Unique choice for stiff Hamiltonian systems or non-separable
    Hamiltonians.

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

Stiff chemical kinetics (Robertson's problem):

>>> def robertson(t, y):
...     y1, y2, y3 = y[0], y[1], y[2]
...     dy1 = -0.04 * y1 + 1e4 * y2 * y3
...     dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
...     dy3 = 3e7 * y2**2
...     return torch.stack([dy1, dy2, dy3])
>>>
>>> y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
>>> y_final, interp = bdf(robertson, y0, t_span=(0.0, 1e5))

Molecular dynamics (Lennard-Jones potential):

>>> def grad_V(t, q):  # Forces from potential
...     # Compute forces between particles
...     return forces
>>> def grad_T(t, p):  # Velocities
...     return p / mass
>>> q, p, interp = stormer_verlet(grad_V, grad_T, q0, p0, t_span, dt=0.001)

Choosing between symplectic integrators:

- Use ``stormer_verlet`` for most molecular dynamics and celestial mechanics
- Use ``yoshida4`` when higher accuracy is needed at same step size
- Use ``implicit_midpoint`` for stiff or non-separable Hamiltonians

Choosing between stiff solvers:

- Use ``bdf`` for most stiff problems (good general choice)
- Use ``radau`` for very stiff problems or when L-stability is important
- Use ``backward_euler`` for simple problems or when speed matters more than accuracy
"""

from torchscience.integration.initial_value_problem._adjoint import adjoint
from torchscience.integration.initial_value_problem._backward_euler import (
    backward_euler,
)
from torchscience.integration.initial_value_problem._batched import (
    solve_ivp_batched,
)
from torchscience.integration.initial_value_problem._bdf import bdf
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
from torchscience.integration.initial_value_problem._implicit_midpoint import (
    implicit_midpoint,
)
from torchscience.integration.initial_value_problem._midpoint import midpoint
from torchscience.integration.initial_value_problem._radau import radau
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
from torchscience.integration.initial_value_problem._stormer_verlet import (
    stormer_verlet,
)
from torchscience.integration.initial_value_problem._yoshida import yoshida4

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
    # Implicit stiff solvers
    "bdf",
    "radau",
    # Symplectic integrators
    "stormer_verlet",
    "yoshida4",
    "implicit_midpoint",
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
