from typing import NamedTuple, Optional

from torch import Tensor


class OptimizeResult(NamedTuple):
    """Result of an optimization routine.

    Parameters
    ----------
    x : Tensor
        Solution tensor. Autograd gradients flow through this field.
    converged : Tensor
        Boolean tensor indicating convergence. Scalar or batch shape ``(...,)``.
    num_iterations : Tensor
        Number of iterations performed. ``int64`` scalar.
    fun : Tensor, optional
        Objective value at the solution ``x``.
    """

    x: Tensor
    converged: Tensor
    num_iterations: Tensor
    fun: Optional[Tensor] = None
