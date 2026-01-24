"""Kraskov-Stögbauer-Grassberger (KSG) mutual information estimator."""

from typing import Literal, Optional

import torch
from torch import Tensor


def kraskov_mutual_information(
    x_samples: Tensor,
    y_samples: Tensor,
    *,
    k: int = 3,
    algorithm: Literal[1, 2] = 1,
    base: Optional[float] = None,
) -> Tensor:
    r"""Estimate mutual information using the Kraskov-Stögbauer-Grassberger estimator.

    Uses k-nearest neighbor distances in joint and marginal spaces to estimate
    mutual information between two continuous random variables from samples.

    Mathematical Definition
    -----------------------
    For Algorithm 1:

    .. math::

        I(X;Y) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(n)

    For Algorithm 2:

    .. math::

        I(X;Y) = \psi(k) - \frac{1}{k} - \langle \psi(n_x) + \psi(n_y) \rangle + \psi(n)

    where:

    - :math:`n` is the number of samples
    - :math:`k` is the number of nearest neighbors
    - :math:`\psi` is the digamma function
    - :math:`n_x` is the number of points with :math:`\|x_i - x_j\| < \epsilon`
      in the X space
    - :math:`n_y` is the number of points with :math:`\|y_i - y_j\| < \epsilon`
      in the Y space
    - :math:`\epsilon` is the distance to the k-th nearest neighbor in the
      joint (X, Y) space using the max norm

    Parameters
    ----------
    x_samples : Tensor
        Samples from random variable X. Shape: ``(..., n_samples, n_dims_x)``
        where ``...`` represents optional batch dimensions.
    y_samples : Tensor
        Samples from random variable Y. Shape: ``(..., n_samples, n_dims_y)``.
        Must have the same batch dimensions and n_samples as x_samples.
    k : int, default=3
        Number of nearest neighbors. Higher k gives lower variance but
        potentially higher bias.
    algorithm : {1, 2}, default=1
        Which KSG algorithm to use:

        - ``1``: Uses strict inequality, adds 1 to n_x and n_y counts
        - ``2``: Uses non-strict inequality, no +1 adjustment
    base : float or None, default=None
        Logarithm base for MI calculation:

        - ``None``: Natural logarithm (MI in nats)
        - ``2``: Base-2 logarithm (MI in bits)

    Returns
    -------
    Tensor
        Estimated mutual information I(X;Y). Shape is the input shape with
        the last two dimensions removed.

    Raises
    ------
    ValueError
        If x_samples and y_samples have different batch dimensions or
        n_samples, if k >= n_samples, or if algorithm is not 1 or 2.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(42)
    >>> # Correlated Gaussians
    >>> n = 1000
    >>> x = torch.randn(n, 1)
    >>> y = 0.8 * x + 0.6 * torch.randn(n, 1)
    >>> mi = kraskov_mutual_information(x, y)
    >>> # MI should be positive for correlated variables

    >>> # Independent variables should have MI close to 0
    >>> x_ind = torch.randn(n, 1)
    >>> y_ind = torch.randn(n, 1)
    >>> mi_ind = kraskov_mutual_information(x_ind, y_ind)
    >>> # mi_ind should be close to 0

    Notes
    -----
    - Algorithm 1 is generally preferred and is the default.
    - Algorithm 2 can have slightly lower bias but higher variance.
    - The estimator uses the max (Chebyshev) norm in the joint space.
    - Supports first-order gradients.
    - For entropy estimation, use :func:`kraskov_entropy` or
      :func:`kozachenko_leonenko_entropy`.

    See Also
    --------
    mutual_information : MI from probability distributions.
    kraskov_entropy : k-NN entropy estimator.
    kozachenko_leonenko_entropy : k-NN entropy estimator.

    References
    ----------
    .. [1] Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating
           mutual information. Physical Review E, 69(6), 066138.
    """
    if not isinstance(x_samples, Tensor):
        raise TypeError(
            f"x_samples must be a Tensor, got {type(x_samples).__name__}"
        )
    if not isinstance(y_samples, Tensor):
        raise TypeError(
            f"y_samples must be a Tensor, got {type(y_samples).__name__}"
        )

    if x_samples.dim() < 2:
        raise ValueError(
            f"x_samples must have at least 2 dimensions, got {x_samples.dim()}"
        )
    if y_samples.dim() < 2:
        raise ValueError(
            f"y_samples must have at least 2 dimensions, got {y_samples.dim()}"
        )

    # Check batch dimensions and n_samples match
    if x_samples.shape[:-1] != y_samples.shape[:-1]:
        raise ValueError(
            f"x_samples and y_samples must have matching batch dimensions and "
            f"n_samples. Got x_samples shape {x_samples.shape} and y_samples "
            f"shape {y_samples.shape}"
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    if algorithm not in (1, 2):
        raise ValueError(f"algorithm must be 1 or 2, got {algorithm}")

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    n_samples = x_samples.shape[-2]
    if k >= n_samples:
        raise ValueError(
            f"k must be less than n_samples, got k={k} and n_samples={n_samples}"
        )

    batch_shape = x_samples.shape[:-2]
    n_dims_x = x_samples.shape[-1]
    n_dims_y = y_samples.shape[-1]

    # Flatten batch dimensions
    if batch_shape:
        x_flat = x_samples.reshape(-1, n_samples, n_dims_x)
        y_flat = y_samples.reshape(-1, n_samples, n_dims_y)
    else:
        x_flat = x_samples.unsqueeze(0)
        y_flat = y_samples.unsqueeze(0)

    batch_size = x_flat.shape[0]

    # Compute MI for each batch element
    mi_list = []
    for i in range(batch_size):
        x = x_flat[i]  # (n_samples, n_dims_x)
        y = y_flat[i]  # (n_samples, n_dims_y)
        mi = _compute_ksg_mi(x, y, k, algorithm, n_samples)
        mi_list.append(mi)

    result = torch.stack(mi_list)

    # Convert to specified base
    if base is not None:
        import math

        result = result / math.log(base)

    # Reshape back to batch shape
    if batch_shape:
        result = result.reshape(batch_shape)
    else:
        result = result.squeeze(0)

    return result


def _compute_ksg_mi(
    x: Tensor,
    y: Tensor,
    k: int,
    algorithm: int,
    n_samples: int,
) -> Tensor:
    """Compute KSG mutual information for a single batch.

    Parameters
    ----------
    x : Tensor
        Shape: (n_samples, n_dims_x)
    y : Tensor
        Shape: (n_samples, n_dims_y)
    k : int
        Number of nearest neighbors
    algorithm : int
        KSG algorithm (1 or 2)
    n_samples : int
        Number of samples

    Returns
    -------
    Tensor
        Scalar MI value
    """
    # Concatenate to get joint samples
    # Shape: (n_samples, n_dims_x + n_dims_y)
    joint = torch.cat([x, y], dim=-1)

    # Compute pairwise distances in joint space using max norm (Chebyshev)
    # For max norm: dist = max(|x1-x2|, |y1-y2|)
    # Compute distances separately in each marginal and take max
    dist_x = torch.cdist(x, x, p=float("inf"))
    dist_y = torch.cdist(y, y, p=float("inf"))
    dist_joint = torch.maximum(dist_x, dist_y)

    # Set diagonal to infinity to exclude self
    inf_diag = torch.diag(
        torch.full((n_samples,), float("inf"), device=x.device, dtype=x.dtype)
    )
    dist_joint = dist_joint + inf_diag
    dist_x_masked = dist_x + inf_diag
    dist_y_masked = dist_y + inf_diag

    # Find k-th nearest neighbor distance in joint space for each point
    kth_dist_joint, _ = torch.kthvalue(dist_joint, k, dim=-1)

    # Count points within epsilon distance in marginal spaces
    # epsilon = kth_dist_joint (the k-th nearest neighbor distance in joint space)
    eps = kth_dist_joint.unsqueeze(-1)  # (n_samples, 1)

    if algorithm == 1:
        # Algorithm 1: strict inequality, count points with dist < epsilon
        n_x = (dist_x_masked < eps).sum(dim=-1).float()
        n_y = (dist_y_masked < eps).sum(dim=-1).float()
    else:
        # Algorithm 2: non-strict inequality, count points with dist <= epsilon
        n_x = (dist_x_masked <= eps).sum(dim=-1).float()
        n_y = (dist_y_masked <= eps).sum(dim=-1).float()

    # Compute digamma values
    psi_k = torch.special.digamma(
        torch.tensor(float(k), device=x.device, dtype=x.dtype)
    )
    psi_n = torch.special.digamma(
        torch.tensor(float(n_samples), device=x.device, dtype=x.dtype)
    )

    if algorithm == 1:
        # Algorithm 1: I = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(n)
        # Ensure n_x and n_y are at least 0 to avoid digamma of non-positive
        n_x_plus_1 = torch.clamp(n_x + 1, min=1.0)
        n_y_plus_1 = torch.clamp(n_y + 1, min=1.0)
        psi_nx = torch.special.digamma(n_x_plus_1)
        psi_ny = torch.special.digamma(n_y_plus_1)
        mi = psi_k - torch.mean(psi_nx + psi_ny) + psi_n
    else:
        # Algorithm 2: I = psi(k) - 1/k - <psi(n_x) + psi(n_y)> + psi(n)
        # n_x and n_y should be at least 1 (the point itself in non-strict)
        n_x_clamped = torch.clamp(n_x, min=1.0)
        n_y_clamped = torch.clamp(n_y, min=1.0)
        psi_nx = torch.special.digamma(n_x_clamped)
        psi_ny = torch.special.digamma(n_y_clamped)
        mi = psi_k - 1.0 / k - torch.mean(psi_nx + psi_ny) + psi_n

    return mi
