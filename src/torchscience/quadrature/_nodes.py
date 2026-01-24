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


def gauss_hermite_nodes_weights(
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute Gauss-Hermite nodes and weights for the physicists' convention.

    Integrates functions with weight w(x) = exp(-x^2) on (-infinity, infinity):

    .. math::

        \int_{-\infty}^{\infty} f(x) e^{-x^2} dx \approx \sum_{i=1}^{n} w_i f(x_i)

    Uses the Golub-Welsch algorithm.

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
    Gauss-Hermite quadrature is exact for polynomials of degree <= 2n-1
    multiplied by the weight function exp(-x^2).

    The Jacobi matrix for Hermite (physicists') has:
    - diagonal = 0
    - off-diagonal[k] = sqrt(k/2)

    The total weight integral is sqrt(pi).

    Examples
    --------
    >>> nodes, weights = gauss_hermite_nodes_weights(5)
    >>> # Integrate exp(-x^2) from -inf to inf (should be sqrt(pi))
    >>> # integral = sum(weights) ≈ sqrt(pi)

    References
    ----------
    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
    """
    import math

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    if n == 1:
        return (
            torch.tensor([0.0], dtype=dtype, device=device),
            torch.tensor([math.sqrt(math.pi)], dtype=dtype, device=device),
        )

    # Jacobi matrix for Hermite (physicists'):
    # diagonal = 0, off-diagonal[k] = sqrt(k/2)
    k = torch.arange(1, n, dtype=dtype, device=device)
    off_diag = torch.sqrt(k / 2)

    # Construct tridiagonal matrix
    T = torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    # Eigenvalues are nodes, first components of eigenvectors give weights
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    nodes = eigenvalues
    # Weight formula: w_i = sqrt(pi) * v_{i,0}^2 where v is eigenvector
    weights = math.sqrt(math.pi) * eigenvectors[0, :] ** 2

    # Sort by nodes
    sorted_idx = torch.argsort(nodes)
    nodes = nodes[sorted_idx]
    weights = weights[sorted_idx]

    return nodes, weights


def gauss_laguerre_nodes_weights(
    n: int,
    alpha: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute Gauss-Laguerre nodes and weights.

    Integrates functions with weight w(x) = x^alpha * exp(-x) on [0, infinity):

    .. math::

        \int_{0}^{\infty} f(x) x^{\alpha} e^{-x} dx \approx \sum_{i=1}^{n} w_i f(x_i)

    Uses the Golub-Welsch algorithm.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    alpha : float
        Parameter for generalized Laguerre polynomials. Must be > -1.
        Default is 0 (standard Laguerre).
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
        If n < 1 or alpha <= -1.

    Notes
    -----
    Gauss-Laguerre quadrature is exact for polynomials of degree <= 2n-1
    multiplied by the weight function x^alpha * exp(-x).

    The Jacobi matrix for generalized Laguerre L_n^{(alpha)} has:
    - diagonal[k] = 2k + alpha + 1
    - off-diagonal[k] = sqrt(k * (k + alpha))

    The total weight integral is Gamma(alpha + 1).

    Examples
    --------
    >>> nodes, weights = gauss_laguerre_nodes_weights(5)
    >>> # Integrate exp(-x) from 0 to inf (should be 1 = Gamma(1))
    >>> # integral = sum(weights) ≈ 1

    References
    ----------
    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
    """
    import math

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if alpha <= -1:
        raise ValueError(f"alpha must be > -1, got {alpha}")

    if n == 1:
        return (
            torch.tensor([alpha + 1], dtype=dtype, device=device),
            torch.tensor([math.gamma(alpha + 1)], dtype=dtype, device=device),
        )

    # Jacobi matrix for generalized Laguerre L_n^{(alpha)}:
    # diagonal[k] = 2k + alpha + 1 (for k = 0, 1, ..., n-1)
    # off-diagonal[k] = sqrt(k * (k + alpha)) (for k = 1, ..., n-1)
    k = torch.arange(n, dtype=dtype, device=device)
    diag = 2 * k + alpha + 1

    k_off = torch.arange(1, n, dtype=dtype, device=device)
    off_diag = torch.sqrt(k_off * (k_off + alpha))

    # Construct tridiagonal matrix
    T = (
        torch.diag(diag)
        + torch.diag(off_diag, diagonal=1)
        + torch.diag(off_diag, diagonal=-1)
    )

    # Eigenvalues are nodes, first components of eigenvectors give weights
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    nodes = eigenvalues
    # Weight formula: w_i = Gamma(alpha + 1) * v_{i,0}^2
    weights = math.gamma(alpha + 1) * eigenvectors[0, :] ** 2

    # Sort by nodes
    sorted_idx = torch.argsort(nodes)
    nodes = nodes[sorted_idx]
    weights = weights[sorted_idx]

    return nodes, weights


def gauss_chebyshev_nodes_weights(
    n: int,
    kind: int = 1,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute Gauss-Chebyshev nodes and weights on [-1, 1].

    For kind=1 (Chebyshev T), integrates with weight w(x) = 1/sqrt(1-x^2):

    .. math::

        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \approx \sum_{i=1}^{n} w_i f(x_i)

    For kind=2 (Chebyshev U), integrates with weight w(x) = sqrt(1-x^2):

    .. math::

        \int_{-1}^{1} f(x) \sqrt{1-x^2} dx \approx \sum_{i=1}^{n} w_i f(x_i)

    Parameters
    ----------
    n : int
        Number of quadrature points.
    kind : int
        1 for Chebyshev of the first kind (T), 2 for second kind (U).
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
        If n < 1 or kind not in {1, 2}.

    Notes
    -----
    Gauss-Chebyshev quadrature has explicit formulas (no eigenvalue computation needed):

    For T (kind=1):
        nodes: cos((2k-1)*pi / (2n)) for k = 1, ..., n
        weights: pi / n (all equal)

    For U (kind=2):
        nodes: cos(k*pi / (n+1)) for k = 1, ..., n
        weights: pi/(n+1) * sin^2(k*pi/(n+1))

    Examples
    --------
    >>> nodes, weights = gauss_chebyshev_nodes_weights(5, kind=1)
    >>> # Integrate 1/sqrt(1-x^2) from -1 to 1 (should be pi)
    >>> # integral = sum(weights) ≈ pi

    References
    ----------
    Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions.
    """
    import math

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if kind not in (1, 2):
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    if kind == 1:
        # Chebyshev T: nodes = cos((2k-1)*pi / (2n)), weights = pi/n
        k = torch.arange(1, n + 1, dtype=dtype, device=device)
        nodes = torch.cos((2 * k - 1) * math.pi / (2 * n))
        weights = torch.full((n,), math.pi / n, dtype=dtype, device=device)
    else:
        # Chebyshev U: nodes = cos(k*pi / (n+1)), weights = pi/(n+1) * sin^2(k*pi/(n+1))
        k = torch.arange(1, n + 1, dtype=dtype, device=device)
        theta = k * math.pi / (n + 1)
        nodes = torch.cos(theta)
        weights = math.pi / (n + 1) * torch.sin(theta) ** 2

    # Sort by nodes (already descending from cos, need ascending)
    sorted_idx = torch.argsort(nodes)
    nodes = nodes[sorted_idx]
    weights = weights[sorted_idx]

    return nodes, weights


def gauss_jacobi_nodes_weights(
    n: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute Gauss-Jacobi nodes and weights on [-1, 1].

    Integrates functions with weight w(x) = (1-x)^alpha * (1+x)^beta:

    .. math::

        \int_{-1}^{1} f(x) (1-x)^{\alpha} (1+x)^{\beta} dx \approx \sum_{i=1}^{n} w_i f(x_i)

    Uses the Golub-Welsch algorithm.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    alpha : float
        Exponent for (1-x). Must be > -1.
    beta : float
        Exponent for (1+x). Must be > -1.
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
        If n < 1 or alpha <= -1 or beta <= -1.

    Notes
    -----
    Gauss-Jacobi quadrature is exact for polynomials of degree <= 2n-1
    multiplied by the weight function (1-x)^alpha * (1+x)^beta.

    Special cases:
    - alpha = beta = 0: Gauss-Legendre
    - alpha = beta = -1/2: Gauss-Chebyshev T
    - alpha = beta = 1/2: Gauss-Chebyshev U

    The Jacobi matrix for Jacobi polynomials P_n^{(alpha,beta)} has:
    - diagonal[k] = (beta^2 - alpha^2) / ((2k + alpha + beta)(2k + alpha + beta + 2))
    - off-diagonal formulas are more complex

    The total weight integral is:
        2^{alpha+beta+1} * Beta(alpha+1, beta+1)

    Examples
    --------
    >>> nodes, weights = gauss_jacobi_nodes_weights(5, 0.5, 0.5)

    References
    ----------
    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
    """
    import math

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if alpha <= -1:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    if beta <= -1:
        raise ValueError(f"beta must be > -1, got {beta}")

    # Total weight: 2^{alpha+beta+1} * B(alpha+1, beta+1)
    total_weight = (
        2 ** (alpha + beta + 1)
        * math.gamma(alpha + 1)
        * math.gamma(beta + 1)
        / math.gamma(alpha + beta + 2)
    )

    if n == 1:
        # Single node is at (beta - alpha) / (alpha + beta + 2)
        node = (beta - alpha) / (alpha + beta + 2)
        return (
            torch.tensor([node], dtype=dtype, device=device),
            torch.tensor([total_weight], dtype=dtype, device=device),
        )

    # Build Jacobi matrix for Jacobi polynomials
    # Using the standard three-term recurrence formulas
    k = torch.arange(n, dtype=dtype, device=device)

    # Diagonal elements: a_k = (beta^2 - alpha^2) / ((2k + ab)(2k + ab + 2))
    # where ab = alpha + beta
    ab = alpha + beta
    denom1 = 2 * k + ab
    denom2 = 2 * k + ab + 2

    # Avoid division by zero for k=0 when alpha+beta=0
    with torch.no_grad():
        safe_denom = denom1 * denom2
        safe_denom = torch.where(
            safe_denom == 0, torch.ones_like(safe_denom), safe_denom
        )

    diag = (beta**2 - alpha**2) / safe_denom

    # Off-diagonal elements (more complex formula)
    # b_k = 2 * sqrt(k*(k+alpha)*(k+beta)*(k+ab)) / ((2k+ab-1)*(2k+ab+1)) for k >= 1
    k_off = torch.arange(1, n, dtype=dtype, device=device)
    numerator = k_off * (k_off + alpha) * (k_off + beta) * (k_off + ab)
    denom_off = (
        (2 * k_off + ab - 1) * (2 * k_off + ab) ** 2 * (2 * k_off + ab + 1)
    )

    # Ensure positivity for sqrt
    numerator = torch.clamp(numerator, min=0)
    denom_off = torch.clamp(denom_off, min=1e-30)

    off_diag = 2 * torch.sqrt(numerator / denom_off)

    # Construct tridiagonal matrix
    T = (
        torch.diag(diag)
        + torch.diag(off_diag, diagonal=1)
        + torch.diag(off_diag, diagonal=-1)
    )

    # Eigenvalues are nodes, first components of eigenvectors give weights
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    nodes = eigenvalues
    weights = total_weight * eigenvectors[0, :] ** 2

    # Sort by nodes
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
