"""Kozachenko-Leonenko k-NN based entropy estimator."""

import math
from typing import Optional

import torch
from torch import Tensor


def kozachenko_leonenko_entropy(
    samples: Tensor,
    *,
    k: int = 1,
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Kozachenko-Leonenko k-NN entropy estimate.

    Estimates the differential entropy of a continuous distribution using
    distances to k-th nearest neighbors. This is a consistent, non-parametric
    entropy estimator.

    Mathematical Definition
    -----------------------
    The Kozachenko-Leonenko estimator is:

    .. math::

        \hat{H} = \psi(n) - \psi(k) + \log(c_d) + \frac{d}{n} \sum_{i=1}^{n} \log(\rho_{k,i})

    where:

    - :math:`n` is the number of samples
    - :math:`k` is the number of nearest neighbors
    - :math:`\psi` is the digamma function
    - :math:`c_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}` is the volume of the
      d-dimensional unit ball
    - :math:`\rho_{k,i}` is the distance from sample :math:`i` to its
      :math:`k`-th nearest neighbor
    - :math:`d` is the dimensionality of the data

    Parameters
    ----------
    samples : Tensor
        Input samples. Shape: ``(..., n_samples, n_dims)`` where ``...``
        represents optional batch dimensions, ``n_samples`` is the number
        of samples, and ``n_dims`` is the dimensionality.
    k : int, default=1
        Number of nearest neighbors to use. Must be at least 1 and less than
        the number of samples. Larger k values reduce variance but may
        increase bias.
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)
        - ``10``: Base-10 logarithm (entropy in dits/hartleys)

    Returns
    -------
    Tensor
        Kozachenko-Leonenko entropy estimates. Shape is the input shape with
        the last two dimensions (n_samples, n_dims) removed.

    Raises
    ------
    TypeError
        If samples is not a Tensor.
    ValueError
        If samples has fewer than 2 dimensions, k is not a positive integer,
        k >= n_samples, or base is invalid.

    Examples
    --------
    >>> # 1D Gaussian entropy: H = 0.5 * log(2 * pi * e * sigma^2)
    >>> torch.manual_seed(42)
    >>> sigma = 1.0
    >>> samples = torch.randn(1000, 1) * sigma
    >>> H = kozachenko_leonenko_entropy(samples)
    >>> expected = 0.5 * math.log(2 * math.pi * math.e * sigma**2)
    >>> # H should be close to expected (approximately 1.42)

    >>> # 2D Gaussian entropy: H = d/2 * log(2 * pi * e * sigma^2)
    >>> samples_2d = torch.randn(1000, 2) * sigma
    >>> H_2d = kozachenko_leonenko_entropy(samples_2d)
    >>> expected_2d = 2 / 2 * math.log(2 * math.pi * math.e * sigma**2)

    >>> # Batched computation
    >>> samples_batch = torch.randn(5, 1000, 3)  # 5 batches, 3D data
    >>> H_batch = kozachenko_leonenko_entropy(samples_batch)
    >>> H_batch.shape
    torch.Size([5])

    >>> # Using different k values
    >>> H_k1 = kozachenko_leonenko_entropy(samples, k=1)
    >>> H_k5 = kozachenko_leonenko_entropy(samples, k=5)

    Notes
    -----
    - The estimator is consistent: it converges to the true entropy as
      n approaches infinity.
    - Larger k values reduce variance but may introduce bias, especially
      for small samples.
    - k=1 is the original Kozachenko-Leonenko estimator; k>1 is sometimes
      called the k-NN or Kraskov-Stoegbauer-Grassberger variant.
    - Uses Euclidean distance by default.
    - Supports first-order gradients through all operations.
    - For very small samples (n < 10), results may be unreliable.

    See Also
    --------
    histogram_entropy : Histogram-based entropy estimation.
    shannon_entropy : Shannon entropy from probability distributions.

    References
    ----------
    .. [1] Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the
           entropy of a random vector. Problems of Information Transmission,
           23(2), 95-101.
    .. [2] Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating
           mutual information. Physical Review E, 69(6), 066138.
    """
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"samples must be a Tensor, got {type(samples).__name__}"
        )

    if samples.dim() < 2:
        raise ValueError(
            f"samples must have at least 2 dimensions (n_samples, n_dims), "
            f"got {samples.dim()} dimensions"
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    # Get dimensions
    n_samples = samples.shape[-2]
    n_dims = samples.shape[-1]
    batch_shape = samples.shape[:-2]

    if k >= n_samples:
        raise ValueError(
            f"k must be less than n_samples, got k={k} and n_samples={n_samples}"
        )

    # Flatten batch dimensions for processing
    if batch_shape:
        samples_flat = samples.reshape(-1, n_samples, n_dims)
    else:
        samples_flat = samples.unsqueeze(0)

    batch_size = samples_flat.shape[0]

    # Compute pairwise distances for each batch
    # Shape: (batch_size, n_samples, n_samples)
    distances = torch.cdist(samples_flat, samples_flat, p=2)

    # Set diagonal to infinity to exclude self-distances
    # We need to find k-th nearest neighbor excluding the point itself
    inf_diag = torch.diag_embed(
        torch.full(
            (batch_size, n_samples),
            float("inf"),
            dtype=distances.dtype,
            device=distances.device,
        )
    )
    distances = distances + inf_diag

    # Find k-th smallest distance (k-th nearest neighbor)
    # kthvalue returns (values, indices), we want values
    # k=1 means 1st nearest neighbor, but kthvalue is 1-indexed
    kth_distances, _ = torch.kthvalue(distances, k, dim=-1)

    # Handle zero distances (duplicate points) - replace with small epsilon
    eps = torch.finfo(samples.dtype).eps
    kth_distances = torch.clamp(kth_distances, min=eps)

    # Compute log of unit ball volume: log(c_d) = (d/2) * log(pi) - gammaln(d/2 + 1)
    d = float(n_dims)
    log_c_d = (d / 2.0) * math.log(math.pi) - torch.special.gammaln(
        torch.tensor(d / 2.0 + 1.0, dtype=samples.dtype, device=samples.device)
    )

    # Compute digamma(k)
    psi_k = torch.special.digamma(
        torch.tensor(float(k), dtype=samples.dtype, device=samples.device)
    )

    # Compute psi(n)
    psi_n = torch.special.digamma(
        torch.tensor(
            float(n_samples), dtype=samples.dtype, device=samples.device
        )
    )

    # Compute entropy estimate:
    # H = psi(n) - psi(k) + log(c_d) + (d/n) * sum(log(dist_k_i))
    # Sum of log distances: (d/n) * sum_i log(rho_k_i)
    log_distances_sum = torch.sum(torch.log(kth_distances), dim=-1)
    distance_term = (d / n_samples) * log_distances_sum

    entropy = psi_n - psi_k + log_c_d + distance_term

    # Convert to specified base
    if base is not None:
        entropy = entropy / math.log(base)

    # Reshape back to batch shape
    if batch_shape:
        entropy = entropy.reshape(batch_shape)
    else:
        entropy = entropy.squeeze(0)

    return entropy
