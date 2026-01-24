"""Jackknife entropy estimation with variance."""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

from ._kozachenko_leonenko_entropy import kozachenko_leonenko_entropy


def jackknife_entropy(
    samples: Tensor,
    *,
    estimator: Literal[
        "kozachenko_leonenko", "kraskov"
    ] = "kozachenko_leonenko",
    k: int = 1,
    base: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Estimate differential entropy with standard error using jackknife resampling.

    Uses leave-one-out resampling to compute both an entropy estimate and
    its standard error, providing a measure of uncertainty.

    Mathematical Definition
    -----------------------
    Given samples :math:`X = \{x_1, \ldots, x_n\}`, the jackknife estimate is:

    .. math::

        \hat{H}_{\text{jackknife}} = n \hat{H} - (n-1) \bar{H}_{-i}

    where :math:`\hat{H}` is the full-sample estimate and :math:`\bar{H}_{-i}`
    is the mean of leave-one-out estimates.

    The standard error is:

    .. math::

        \text{SE} = \sqrt{\frac{n-1}{n} \sum_{i=1}^{n} (\hat{H}_{-i} - \bar{H}_{-i})^2}

    Parameters
    ----------
    samples : Tensor
        Input samples. Shape: ``(..., n_samples, n_dims)`` where ``...``
        represents optional batch dimensions.
    estimator : {"kozachenko_leonenko", "kraskov"}, default="kozachenko_leonenko"
        Base entropy estimator to use:

        - ``"kozachenko_leonenko"``: Uses kozachenko_leonenko_entropy
        - ``"kraskov"``: Alias for kozachenko_leonenko with default k=3
    k : int, default=1
        Number of nearest neighbors for the underlying estimator.
        Default is 1 for kozachenko_leonenko, but 3 is common for kraskov.
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)

    Returns
    -------
    entropy : Tensor
        Jackknife entropy estimate. Shape is the input shape with
        the last two dimensions removed.
    standard_error : Tensor
        Standard error of the estimate. Same shape as entropy.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(42)
    >>> samples = torch.randn(500, 2)
    >>> H, se = jackknife_entropy(samples)
    >>> # H is the entropy estimate, se is the standard error
    >>> print(f"Entropy: {H:.3f} ± {se:.3f}")

    >>> # With Kraskov estimator (k=3 default)
    >>> H_kraskov, se_kraskov = jackknife_entropy(samples, estimator="kraskov", k=3)

    Notes
    -----
    - Jackknife provides bias-corrected estimates and variance.
    - Computational complexity is O(n^3) as it computes n leave-one-out estimates.
    - For large n, consider subsampling or using variance from repeated runs.
    - The standard error can be used for confidence intervals:
      approximately 95% CI is H ± 1.96 * SE.

    See Also
    --------
    kozachenko_leonenko_entropy : Base k-NN entropy estimator.
    kraskov_entropy : k-NN entropy estimator with k=3 default.

    References
    ----------
    .. [1] Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
           The Annals of Statistics, 7(1), 1-26.
    """
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"samples must be a Tensor, got {type(samples).__name__}"
        )

    if samples.dim() < 2:
        raise ValueError(
            f"samples must have at least 2 dimensions, got {samples.dim()}"
        )

    valid_estimators = ("kozachenko_leonenko", "kraskov")
    if estimator not in valid_estimators:
        raise ValueError(
            f"estimator must be one of {valid_estimators}, got '{estimator}'"
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    n_samples = samples.shape[-2]
    n_dims = samples.shape[-1]
    batch_shape = samples.shape[:-2]

    # Use k=3 default for kraskov if k wasn't explicitly set
    effective_k = k
    if estimator == "kraskov" and k == 1:
        effective_k = 3

    if effective_k >= n_samples - 1:
        raise ValueError(
            f"k must be less than n_samples - 1 for jackknife, "
            f"got k={effective_k} and n_samples={n_samples}"
        )

    # Flatten batch dimensions
    if batch_shape:
        samples_flat = samples.reshape(-1, n_samples, n_dims)
    else:
        samples_flat = samples.unsqueeze(0)

    batch_size = samples_flat.shape[0]

    # Compute jackknife for each batch element
    entropies = []
    standard_errors = []

    for i in range(batch_size):
        sample = samples_flat[i]
        h, se = _compute_jackknife(sample, effective_k, n_samples, n_dims)
        entropies.append(h)
        standard_errors.append(se)

    entropy = torch.stack(entropies)
    std_error = torch.stack(standard_errors)

    # Convert to specified base
    if base is not None:
        import math

        log_base = math.log(base)
        entropy = entropy / log_base
        std_error = std_error / log_base

    # Reshape back to batch shape
    if batch_shape:
        entropy = entropy.reshape(batch_shape)
        std_error = std_error.reshape(batch_shape)
    else:
        entropy = entropy.squeeze(0)
        std_error = std_error.squeeze(0)

    return entropy, std_error


def _compute_jackknife(
    samples: Tensor,
    k: int,
    n_samples: int,
    n_dims: int,
) -> Tuple[Tensor, Tensor]:
    """Compute jackknife entropy estimate and standard error.

    Parameters
    ----------
    samples : Tensor
        Shape: (n_samples, n_dims)
    k : int
        Number of nearest neighbors
    n_samples : int
        Number of samples
    n_dims : int
        Number of dimensions

    Returns
    -------
    entropy : Tensor
        Jackknife entropy estimate (scalar)
    standard_error : Tensor
        Standard error (scalar)
    """
    n = n_samples

    # Full sample estimate
    h_full = kozachenko_leonenko_entropy(samples, k=k)

    # Leave-one-out estimates
    h_loo = []
    for i in range(n):
        # Create leave-one-out sample
        indices = torch.cat(
            [
                torch.arange(i, device=samples.device),
                torch.arange(i + 1, n, device=samples.device),
            ]
        )
        sample_loo = samples[indices]
        h_i = kozachenko_leonenko_entropy(sample_loo, k=k)
        h_loo.append(h_i)

    h_loo = torch.stack(h_loo)

    # Mean of leave-one-out estimates
    h_loo_mean = torch.mean(h_loo)

    # Jackknife entropy estimate (bias-corrected)
    h_jackknife = n * h_full - (n - 1) * h_loo_mean

    # Jackknife standard error
    # SE = sqrt((n-1)/n * sum((h_loo_i - h_loo_mean)^2))
    variance = (n - 1) / n * torch.sum((h_loo - h_loo_mean) ** 2)
    standard_error = torch.sqrt(variance)

    return h_jackknife, standard_error
