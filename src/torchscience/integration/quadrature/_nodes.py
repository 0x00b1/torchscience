"""Node and weight computation for quadrature rules."""

from typing import Optional, Tuple

import torch
from torch import Tensor


def gauss_legendre_nodes_weights(
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Gauss-Legendre nodes and weights on [-1, 1].

    Uses the Golub-Welsch algorithm (eigenvalues of symmetric tridiagonal matrix).

    Parameters
    ----------
    n : int
        Number of quadrature points.
    dtype : torch.dtype
        Data type for output tensors.
    device : torch.device, optional
        Device for output tensors.

    Returns
    -------
    nodes : Tensor
        Quadrature nodes, shape (n,), sorted ascending.
    weights : Tensor
        Quadrature weights, shape (n,).

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    Gauss-Legendre quadrature is exact for polynomials of degree <= 2n-1.

    The algorithm constructs the symmetric tridiagonal Jacobi matrix for
    Legendre polynomials and computes its eigenvalues (nodes) and
    eigenvectors (used to compute weights).

    References
    ----------
    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
    Mathematics of Computation, 23(106), 221-230.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    if n == 1:
        return (
            torch.tensor([0.0], dtype=dtype, device=device),
            torch.tensor([2.0], dtype=dtype, device=device),
        )

    # Build symmetric tridiagonal Jacobi matrix for Legendre polynomials
    # For Legendre: diagonal = 0, off-diagonal[k] = k / sqrt(4k^2 - 1)
    k = torch.arange(1, n, dtype=dtype, device=device)
    off_diag = k / torch.sqrt(4 * k**2 - 1)

    # Construct tridiagonal matrix
    T = torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    # Eigenvalues are nodes, first components of eigenvectors give weights
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    nodes = eigenvalues
    weights = 2 * eigenvectors[0, :] ** 2

    # Sort by nodes (should already be sorted from eigh, but ensure)
    sorted_idx = torch.argsort(nodes)
    nodes = nodes[sorted_idx]
    weights = weights[sorted_idx]

    return nodes, weights


# Pre-tabulated Gauss-Kronrod nodes and weights (high precision)
# These are computed to extended precision and stored here.
# Reference: QUADPACK (Piessens et al., 1983)
#
# For each order, we store:
# - positive_nodes: positive nodes (including 0 if present), ascending order
# - positive_k_weights: Kronrod weights for positive nodes
# - positive_g_weights: Gauss weights for the embedded Gauss rule
# - gauss_mask: which positive nodes are also Gauss nodes (True/False)
#
# The nodes are symmetric about 0, so we reconstruct the full set by reflection.

_GK15_POSITIVE_NODES = [
    0.000000000000000000000000000000000,
    0.207784955007898467600689403773245,
    0.405845151377397166906606412076961,
    0.586087235467691130294144838258730,
    0.741531185599394439863864773280788,
    0.864864423359769072789712788640926,
    0.949107912342758524526189684047851,
    0.991455371120812639206854697526329,
]

_GK15_POSITIVE_K_WEIGHTS = [
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970,
]

# Gauss weights for G7 (at odd indices: 1, 3, 5, 7 in positive nodes)
_GK15_POSITIVE_G_WEIGHTS = [
    0.417959183673469387755102040816327,
    0.381830050505118944950369775488975,
    0.279705391489276667901467771423780,
    0.129484966168869693270611432679082,
]

# Which positive nodes are Gauss nodes: G7 nodes are at 0, ±0.406, ±0.742, ±0.949
# In positive_nodes: indices 0, 2, 4, 6 are Gauss nodes
_GK15_GAUSS_MASK = [True, False, True, False, True, False, True, False]


_GK21_POSITIVE_NODES = [
    0.000000000000000000000000000000000,
    0.148874338981631210884826001129720,
    0.294392862701460198131126603103866,
    0.433395394129247190799265943165784,
    0.562757134668604683339000099272694,
    0.679409568299024406234327365114874,
    0.780817726586416897063717578345042,
    0.865063366688984510732096688423493,
    0.930157491355708226001207180059508,
    0.973906528517171720077964012084452,
    0.995657163025808080735527280689003,
]

_GK21_POSITIVE_K_WEIGHTS = [
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068,
    0.142775938577060080797094273138717,
    0.134709217311473325928054001771707,
    0.123491976262065851077958109831074,
    0.109387158802297641899210590325805,
    0.093125454583697605535065465083366,
    0.075039674810919952767043140916190,
    0.054755896574351996031381300244580,
    0.032558162307964727478818972459390,
    0.011694638867371874278064396062192,
]

# Gauss weights for G10 (at indices 1, 3, 5, 7, 9 in positive nodes)
_GK21_POSITIVE_G_WEIGHTS = [
    0.295524224714752870173892994651338,
    0.269266719309996355091226921569469,
    0.219086362515982043995534934228163,
    0.149451349150580593145776339657697,
    0.066671344308688137593568809893332,
]

# Which positive nodes are Gauss nodes
_GK21_GAUSS_MASK = [
    False,
    True,
    False,
    True,
    False,
    True,
    False,
    True,
    False,
    True,
    False,
]


_GK_DATA = {
    15: (
        _GK15_POSITIVE_NODES,
        _GK15_POSITIVE_K_WEIGHTS,
        _GK15_POSITIVE_G_WEIGHTS,
        _GK15_GAUSS_MASK,
    ),
    21: (
        _GK21_POSITIVE_NODES,
        _GK21_POSITIVE_K_WEIGHTS,
        _GK21_POSITIVE_G_WEIGHTS,
        _GK21_GAUSS_MASK,
    ),
}


def gauss_kronrod_nodes_weights(
    order: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute Gauss-Kronrod nodes and weights on [-1, 1].

    Returns both Kronrod (order points) and embedded Gauss weights for error estimation.

    Parameters
    ----------
    order : int
        Kronrod order: 15 or 21. (More orders can be added.)
    dtype : torch.dtype
        Data type for output tensors.
    device : torch.device, optional
        Device for output tensors.

    Returns
    -------
    nodes : Tensor
        Kronrod nodes, shape (order,), sorted ascending.
    kronrod_weights : Tensor
        Kronrod weights, shape (order,).
    gauss_weights : Tensor
        Gauss weights, shape (order // 2,).
    gauss_indices : Tensor
        Indices into nodes where Gauss nodes are located, shape (order // 2,).

    Raises
    ------
    ValueError
        If order is not implemented.

    Notes
    -----
    The Gauss-Kronrod pair G(n)-K(2n+1) allows error estimation by computing
    both the Gauss and Kronrod approximations. The difference gives an error estimate.

    G7-K15: 7-point Gauss embedded in 15-point Kronrod
    G10-K21: 10-point Gauss embedded in 21-point Kronrod

    References
    ----------
    Piessens, R., et al. (1983). QUADPACK: A subroutine package for automatic integration.
    """
    if order not in (15, 21):
        raise ValueError(f"order must be 15 or 21, got {order}")

    positive_nodes, positive_k_weights, positive_g_weights, gauss_mask = (
        _GK_DATA[order]
    )

    # Convert to tensors
    pos_nodes = torch.tensor(positive_nodes, dtype=dtype, device=device)
    pos_k_weights = torch.tensor(
        positive_k_weights, dtype=dtype, device=device
    )
    pos_g_weights = torch.tensor(
        positive_g_weights, dtype=dtype, device=device
    )
    gauss_mask_tensor = torch.tensor(gauss_mask, device=device)

    # Reflect nodes to get full set (nodes are symmetric about 0)
    # positive_nodes[0] = 0, so we reflect positive_nodes[1:] to negative
    if positive_nodes[0] == 0:
        # Has zero node: full = [-pos[n-1], ..., -pos[1], 0, pos[1], ..., pos[n-1]]
        negative_nodes = -pos_nodes[1:].flip(0)
        nodes = torch.cat([negative_nodes, pos_nodes])

        # Weights are also symmetric
        negative_k_weights = pos_k_weights[1:].flip(0)
        k_weights = torch.cat([negative_k_weights, pos_k_weights])

        # Build Gauss indices and weights together (keep paired, then sort)
        # In positive nodes, Gauss nodes are at indices where gauss_mask is True
        n_neg = len(negative_nodes)
        # Full array: [neg nodes (n_neg)] + [pos nodes (n_pos)]
        # Positive index i maps to full index n_neg + i
        # Negative of positive index i (for i > 0) maps to n_neg - i

        # Build paired (index, weight) and sort by index
        pairs = []
        gauss_pos_indices = [i for i, m in enumerate(gauss_mask) if m]
        for idx, w in zip(gauss_pos_indices, positive_g_weights):
            if positive_nodes[idx] == 0:
                pairs.append((n_neg, w))  # zero node
            else:
                pairs.append((n_neg - idx, w))  # negative
                pairs.append((n_neg + idx, w))  # positive

        # Sort by index to ensure correct ordering
        pairs.sort(key=lambda x: x[0])
        g_indices = torch.tensor(
            [p[0] for p in pairs], dtype=torch.long, device=device
        )
        g_weights = torch.tensor(
            [p[1] for p in pairs], dtype=dtype, device=device
        )

    else:
        # No zero node (shouldn't happen for standard GK rules)
        negative_nodes = -pos_nodes.flip(0)
        nodes = torch.cat([negative_nodes, pos_nodes])

        negative_k_weights = pos_k_weights.flip(0)
        k_weights = torch.cat([negative_k_weights, pos_k_weights])

        # All Gauss weights are doubled
        g_weights = torch.cat([pos_g_weights.flip(0), pos_g_weights])
        n_neg = len(negative_nodes)
        pos_gauss_indices = torch.where(gauss_mask_tensor)[0]
        neg_gauss_indices = n_neg - 1 - pos_gauss_indices.flip(0)
        pos_gauss_indices_full = pos_gauss_indices + n_neg
        g_indices = torch.cat([neg_gauss_indices, pos_gauss_indices_full])

    return nodes, k_weights, g_weights, g_indices
