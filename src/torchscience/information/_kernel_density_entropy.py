"""Kernel density estimation-based entropy estimation."""

import math
from typing import Literal, Optional

import torch
from torch import Tensor


def kernel_density_entropy(
    samples: Tensor,
    *,
    bandwidth: float | Literal["scott", "silverman"] = "scott",
    kernel: Literal["gaussian"] = "gaussian",
    base: Optional[float] = None,
) -> Tensor:
    r"""Estimate differential entropy using kernel density estimation.

    Uses the plug-in estimator that evaluates the KDE at sample points
    and averages the negative log density.

    Mathematical Definition
    -----------------------
    The KDE entropy estimator is:

    .. math::

        \hat{H}(X) = -\frac{1}{n} \sum_{i=1}^{n} \log \hat{p}(x_i)

    where :math:`\hat{p}(x)` is the kernel density estimate:

    .. math::

        \hat{p}(x) = \frac{1}{n h^d} \sum_{j=1}^{n} K\left(\frac{x - x_j}{h}\right)

    with :math:`K` being the kernel function, :math:`h` the bandwidth, and
    :math:`d` the dimensionality.

    Parameters
    ----------
    samples : Tensor
        Input samples. Shape: ``(..., n_samples, n_dims)`` where ``...``
        represents optional batch dimensions.
    bandwidth : float or {"scott", "silverman"}, default="scott"
        Bandwidth for the kernel. Can be:

        - A positive float specifying the bandwidth directly
        - ``"scott"``: Scott's rule: :math:`h = n^{-1/(d+4)}`
        - ``"silverman"``: Silverman's rule: :math:`h = (n(d+2)/4)^{-1/(d+4)}`
    kernel : {"gaussian"}, default="gaussian"
        Kernel function to use. Currently only Gaussian is supported.
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)

    Returns
    -------
    Tensor
        Estimated differential entropy. Shape is the input shape with
        the last two dimensions removed.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(42)
    >>> samples = torch.randn(1000, 2)  # 2D Gaussian
    >>> H = kernel_density_entropy(samples)
    >>> # For N(0, I_2), true entropy is log(2*pi*e) ~ 2.84

    >>> # With custom bandwidth
    >>> H_custom = kernel_density_entropy(samples, bandwidth=0.5)

    Notes
    -----
    - The KDE plug-in estimator is biased and the bias depends on bandwidth.
    - Too small bandwidth leads to underestimation; too large leads to
      overestimation.
    - Scott's and Silverman's rules are designed for Gaussian data and may
      not be optimal for other distributions.
    - Computational complexity is O(n^2) due to pairwise kernel evaluation.
    - Supports first-order gradients.

    See Also
    --------
    kozachenko_leonenko_entropy : k-NN entropy estimator.
    histogram_entropy : Histogram-based entropy estimation.

    References
    ----------
    .. [1] Silverman, B. W. (1986). Density Estimation for Statistics and
           Data Analysis. Chapman and Hall.
    .. [2] Scott, D. W. (1992). Multivariate Density Estimation: Theory,
           Practice, and Visualization. Wiley.
    """
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"samples must be a Tensor, got {type(samples).__name__}"
        )

    if samples.dim() < 2:
        raise ValueError(
            f"samples must have at least 2 dimensions (n_samples, n_dims), "
            f"got {samples.dim()}"
        )

    valid_bandwidth_methods = ("scott", "silverman")
    if isinstance(bandwidth, str) and bandwidth not in valid_bandwidth_methods:
        raise ValueError(
            f"bandwidth must be a float or one of {valid_bandwidth_methods}, "
            f"got '{bandwidth}'"
        )
    if isinstance(bandwidth, (int, float)) and bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    if kernel != "gaussian":
        raise ValueError(f"kernel must be 'gaussian', got '{kernel}'")

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    n_samples = samples.shape[-2]
    n_dims = samples.shape[-1]
    batch_shape = samples.shape[:-2]

    # Compute bandwidth if using a rule
    h = _compute_bandwidth(samples, bandwidth, n_samples, n_dims)

    # Flatten batch dimensions
    if batch_shape:
        samples_flat = samples.reshape(-1, n_samples, n_dims)
    else:
        samples_flat = samples.unsqueeze(0)

    batch_size = samples_flat.shape[0]

    # Compute entropy for each batch element
    entropies = []
    for i in range(batch_size):
        sample = samples_flat[i]
        if isinstance(h, Tensor) and h.dim() > 0:
            h_i = h[i] if batch_shape else h.squeeze()
        else:
            h_i = h
        entropy = _compute_kde_entropy(sample, h_i, n_samples, n_dims)
        entropies.append(entropy)

    result = torch.stack(entropies)

    # Convert to specified base
    if base is not None:
        result = result / math.log(base)

    # Reshape back to batch shape
    if batch_shape:
        result = result.reshape(batch_shape)
    else:
        result = result.squeeze(0)

    return result


def _compute_bandwidth(
    samples: Tensor,
    bandwidth: float | str,
    n_samples: int,
    n_dims: int,
) -> Tensor | float:
    """Compute bandwidth using specified method.

    Parameters
    ----------
    samples : Tensor
        Input samples
    bandwidth : float or str
        Bandwidth specification
    n_samples : int
        Number of samples
    n_dims : int
        Number of dimensions

    Returns
    -------
    Tensor or float
        Computed bandwidth
    """
    if isinstance(bandwidth, (int, float)):
        return float(bandwidth)

    n = float(n_samples)
    d = float(n_dims)

    if bandwidth == "scott":
        # Scott's rule: h = n^(-1/(d+4))
        h = n ** (-1.0 / (d + 4.0))
    elif bandwidth == "silverman":
        # Silverman's rule: h = (n*(d+2)/4)^(-1/(d+4))
        h = (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
    else:
        raise ValueError(f"Unknown bandwidth method: {bandwidth}")

    return h


def _compute_kde_entropy(
    samples: Tensor,
    h: float,
    n_samples: int,
    n_dims: int,
) -> Tensor:
    """Compute KDE entropy for a single batch.

    Parameters
    ----------
    samples : Tensor
        Shape: (n_samples, n_dims)
    h : float
        Bandwidth
    n_samples : int
        Number of samples
    n_dims : int
        Number of dimensions

    Returns
    -------
    Tensor
        Scalar entropy value
    """
    # Compute pairwise squared distances
    # (x_i - x_j)^T (x_i - x_j) for all i, j
    # Shape: (n_samples, n_samples)
    sq_dists = torch.cdist(samples, samples, p=2) ** 2

    # Compute Gaussian kernel values: exp(-||x_i - x_j||^2 / (2*h^2))
    # Shape: (n_samples, n_samples)
    h_sq = h * h
    kernel_vals = torch.exp(-sq_dists / (2.0 * h_sq))

    # Compute KDE at each sample point
    # p_kde(x_i) = (1/n) * (1/(2*pi*h^2)^(d/2)) * sum_j exp(-||x_i - x_j||^2 / (2*h^2))
    # The normalization constant for d-dimensional Gaussian kernel with bandwidth h is:
    # (2*pi)^(-d/2) * h^(-d)
    log_normalization = -0.5 * n_dims * math.log(
        2.0 * math.pi
    ) - n_dims * math.log(h)

    # Sum of kernel values for each point (including self)
    kernel_sum = kernel_vals.sum(dim=-1)  # Shape: (n_samples,)

    # Log density at each point
    # log p(x_i) = log(1/n) + log_normalization + log(sum_j K(x_i, x_j))
    log_densities = (
        -math.log(n_samples) + log_normalization + torch.log(kernel_sum)
    )

    # Entropy = -E[log p(x)] = -(1/n) * sum_i log p(x_i)
    entropy = -torch.mean(log_densities)

    return entropy
