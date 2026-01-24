"""Meta tensor support for BVP solver.

Provides shape inference without computation for torch.compile compatibility.
"""

import torch
from torch import Tensor

from torchscience.ordinary_differential_equation._bvp_solution import (
    BVPSolution,
)


def solve_bvp_meta(
    fun,
    bc,
    x: Tensor,
    y: Tensor,
    p: Tensor,
    tol: float,
    max_nodes: int,
    max_outer_iterations: int,
) -> BVPSolution:
    """Meta implementation of solve_bvp for shape inference.

    Returns a BVPSolution with meta tensors of appropriate shapes.
    The actual shapes depend on mesh refinement, so we conservatively
    return the input shape (adaptive refinement may increase it).
    """
    n_nodes = x.shape[0]
    n_components = y.shape[0]
    n_params = p.shape[0] if p.numel() > 0 else 0

    # For meta tensors, we can't know the final mesh size
    # Return input shape as the minimum
    return BVPSolution(
        x=torch.empty(n_nodes, dtype=x.dtype, device="meta"),
        y=torch.empty(n_components, n_nodes, dtype=y.dtype, device="meta"),
        yp=torch.empty(n_components, n_nodes, dtype=y.dtype, device="meta"),
        p=torch.empty(n_params, dtype=p.dtype, device="meta")
        if n_params > 0
        else torch.empty(0, dtype=x.dtype, device="meta"),
        rms_residuals=torch.empty((), dtype=x.dtype, device="meta"),
        n_iterations=0,
        success=True,
    )
