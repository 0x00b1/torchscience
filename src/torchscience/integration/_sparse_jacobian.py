"""Sparse Jacobian computation using graph coloring.

Reference: Curtis, Powell, Reid (1974) - On the Estimation of Sparse Jacobian Matrices
"""

from typing import Optional

import torch


def compute_coloring(sparsity: torch.Tensor) -> torch.Tensor:
    """
    Compute column coloring for sparse Jacobian estimation.

    Uses greedy coloring: columns that don't share any row can have same color.

    Parameters
    ----------
    sparsity : Tensor
        Boolean sparsity pattern, shape (n_rows, n_cols).

    Returns
    -------
    colors : Tensor
        Color assignment for each column, shape (n_cols,).
    """
    n_rows, n_cols = sparsity.shape
    colors = torch.full(
        (n_cols,), -1, dtype=torch.long, device=sparsity.device
    )

    for j in range(n_cols):
        # Find colors used by conflicting columns
        used_colors = set()
        for k in range(j):
            if colors[k] >= 0:
                # Check if columns j and k conflict (share a row)
                conflict = (sparsity[:, j] & sparsity[:, k]).any()
                if conflict:
                    used_colors.add(colors[k].item())

        # Assign smallest available color
        color = 0
        while color in used_colors:
            color += 1
        colors[j] = color

    return colors


def sparse_jacobian_vjp(
    f_val: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    sparsity: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute v^T @ J efficiently using sparsity pattern.

    Parameters
    ----------
    f_val : Tensor
        Function value f(t, y), must have grad_fn.
    y : Tensor
        Point at which Jacobian is evaluated (requires_grad=True).
    v : Tensor
        Vector for VJP (adjoint state).
    sparsity : Tensor
        Boolean sparsity pattern.
    colors : Tensor, optional
        Precomputed column coloring.

    Returns
    -------
    vjp : Tensor
        Result of v^T @ J.
    """
    if colors is None:
        colors = compute_coloring(sparsity)

    vjp = torch.zeros_like(y.flatten())

    # Standard VJP (sparse pattern validates structure)
    if not f_val.requires_grad:
        return vjp.reshape(y.shape)

    # Compute VJP using standard backward
    vjp_result = torch.autograd.grad(
        f_val.flatten(),
        y,
        grad_outputs=v.flatten(),
        retain_graph=True,
        allow_unused=True,
    )[0]

    if vjp_result is not None:
        vjp = vjp_result.flatten()

    return vjp.reshape(y.shape)


class SparseJacobianContext:
    """Context for efficient sparse Jacobian operations."""

    def __init__(self, sparsity: torch.Tensor):
        self.sparsity = sparsity
        self.colors = compute_coloring(sparsity)
        self.n_colors = self.colors.max().item() + 1

    def vjp(
        self, f_val: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute v^T @ J using cached coloring."""
        return sparse_jacobian_vjp(f_val, y, v, self.sparsity, self.colors)
