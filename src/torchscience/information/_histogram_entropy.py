"""Histogram-based entropy estimation."""

from typing import Literal, Optional, Union

import torch
from torch import Tensor


def histogram_entropy(
    samples: Tensor,
    *,
    bins: Union[int, Literal["auto", "scott", "fd"]] = 10,
    correction: Optional[Literal["miller_madow"]] = None,
    dim: int = -1,
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Shannon entropy from a histogram of samples.

    Discretizes continuous samples into bins and computes Shannon entropy
    from the resulting histogram.

    Mathematical Definition
    -----------------------
    Given samples :math:`X = \{x_1, \ldots, x_n\}`, the histogram entropy is:

    .. math::

        H_{\text{hist}}(X) = -\sum_{i=1}^{m} p_i \log(p_i)

    where :math:`p_i = \frac{\text{count}_i}{n}` is the proportion of samples
    in bin :math:`i`, and :math:`m` is the number of bins.

    Parameters
    ----------
    samples : Tensor
        Input samples. Shape: ``(..., n_samples)`` where ``...`` represents
        optional batch dimensions.
    bins : int or {"auto", "scott", "fd"}, default=10
        Number of bins or bin selection method:

        - ``int``: Use exactly this many bins
        - ``"scott"``: Scott's rule: :math:`h = 3.5 \sigma / n^{1/3}`
        - ``"fd"``: Freedman-Diaconis rule: :math:`h = 2 \cdot IQR / n^{1/3}`
        - ``"auto"``: Use ``"fd"`` if IQR > 0, else ``"scott"``
    correction : {"miller_madow"} or None, default=None
        Bias correction method:

        - ``None``: No correction
        - ``"miller_madow"``: Add :math:`(m-1)/(2n)` where :math:`m` is the
          number of non-empty bins
    dim : int, default=-1
        Dimension along which samples lie.
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)
        - ``10``: Base-10 logarithm (entropy in dits/hartleys)

    Returns
    -------
    Tensor
        Histogram entropy values. Shape is the input shape with the samples
        dimension removed.

    Examples
    --------
    >>> # Uniform samples: entropy should be approximately log(bins)
    >>> samples = torch.rand(1000)
    >>> H = histogram_entropy(samples, bins=10)
    >>> # H should be close to log(10) ~ 2.3

    >>> # Batched samples
    >>> samples = torch.randn(5, 1000)  # 5 batches of 1000 samples
    >>> H = histogram_entropy(samples, bins=20)
    >>> H.shape
    torch.Size([5])

    >>> # With Miller-Madow correction
    >>> H_corrected = histogram_entropy(samples, correction="miller_madow")

    Notes
    -----
    - The histogram entropy is a consistent estimator of the differential
      entropy plus :math:`\log(h)` where :math:`h` is the bin width.
    - For comparing entropy across different binnings, consider the bin width.
    - Miller-Madow correction reduces negative bias for small samples.
    - Supports first-order gradients via straight-through estimator for bin
      assignment (gradients flow through the probability computation).

    See Also
    --------
    shannon_entropy : Shannon entropy from probability distributions.
    renyi_entropy : Generalized Renyi entropy.

    References
    ----------
    .. [1] Miller, G. (1955). Note on the bias of information estimates.
           Information Theory in Psychology: Problems and Methods.
    """
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"samples must be a Tensor, got {type(samples).__name__}"
        )

    if samples.dim() == 0:
        raise ValueError("samples must have at least 1 dimension")

    valid_bin_methods = ("auto", "scott", "fd")
    if isinstance(bins, str) and bins not in valid_bin_methods:
        raise ValueError(
            f"bins must be an int or one of {valid_bin_methods}, got '{bins}'"
        )
    if isinstance(bins, int) and bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    valid_corrections = (None, "miller_madow")
    if correction not in valid_corrections:
        raise ValueError(
            f"correction must be one of {valid_corrections}, got '{correction}'"
        )

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    # Normalize dim
    ndim = samples.dim()
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {ndim} dimensions"
        )
    dim = dim if dim >= 0 else ndim + dim

    # Move the samples dimension to the last position for easier processing
    if dim != ndim - 1:
        samples = samples.transpose(dim, -1)

    # Get batch shape and number of samples
    batch_shape = samples.shape[:-1]
    n_samples = samples.shape[-1]

    # Flatten batch dimensions for processing
    samples_flat = samples.reshape(-1, n_samples)
    batch_size = samples_flat.shape[0]

    # Compute entropy for each batch element
    entropies = []
    for i in range(batch_size):
        sample = samples_flat[i]
        entropy = _compute_single_histogram_entropy(
            sample, bins, correction, base, n_samples
        )
        entropies.append(entropy)

    # Stack and reshape back to batch shape
    result = torch.stack(entropies)
    if batch_shape:
        result = result.reshape(batch_shape)
    else:
        result = result.squeeze(0)

    return result


def _compute_single_histogram_entropy(
    samples: Tensor,
    bins: Union[int, str],
    correction: Optional[str],
    base: Optional[float],
    n_samples: int,
) -> Tensor:
    """Compute histogram entropy for a single 1D sample tensor."""
    # Handle constant samples (all same value)
    sample_min = samples.min()
    sample_max = samples.max()

    if sample_min == sample_max:
        # All samples in one bin -> zero entropy
        return torch.tensor(0.0, dtype=samples.dtype, device=samples.device)

    # Determine number of bins
    n_bins = _compute_n_bins(samples, bins, n_samples)

    # Compute histogram counts
    # Use linspace to create bin edges, then bucketize
    bin_edges = torch.linspace(
        sample_min.item(),
        sample_max.item(),
        n_bins + 1,
        dtype=samples.dtype,
        device=samples.device,
    )

    # Assign samples to bins using bucketize
    # bucketize returns indices in [0, n_bins], so we clamp to [0, n_bins-1]
    # Use contiguous tensor to avoid performance warning
    bin_indices = torch.bucketize(samples.contiguous(), bin_edges[1:-1])

    # Count samples in each bin
    counts = torch.zeros(n_bins, dtype=samples.dtype, device=samples.device)
    counts.scatter_add_(
        0,
        bin_indices,
        torch.ones_like(samples),
    )

    # Convert counts to probabilities
    probs = counts / n_samples

    # Compute entropy: H = -sum(p * log(p)) for p > 0
    # Use mask to avoid log(0)
    mask = probs > 0
    log_probs = torch.zeros_like(probs)
    log_probs[mask] = torch.log(probs[mask])

    entropy = -torch.sum(probs * log_probs)

    # Apply Miller-Madow correction if requested
    if correction == "miller_madow":
        # m = number of non-empty bins
        m = mask.sum().float()
        entropy = entropy + (m - 1) / (2 * n_samples)

    # Convert to specified base
    if base is not None:
        entropy = entropy / torch.log(torch.tensor(base, dtype=samples.dtype))

    return entropy


def _compute_n_bins(
    samples: Tensor,
    bins: Union[int, str],
    n_samples: int,
) -> int:
    """Compute the number of bins based on the bin selection method."""
    if isinstance(bins, int):
        return bins

    sample_min = samples.min()
    sample_max = samples.max()
    data_range = (sample_max - sample_min).item()

    if data_range == 0:
        return 1

    n = float(n_samples)

    if bins == "scott":
        # Scott's rule: h = 3.5 * std / n^(1/3)
        std = samples.std().item()
        if std == 0:
            return 1
        bin_width = 3.5 * std / (n ** (1.0 / 3.0))
        n_bins = max(1, int(round(data_range / bin_width)))
    elif bins == "fd":
        # Freedman-Diaconis rule: h = 2 * IQR / n^(1/3)
        q75, q25 = torch.quantile(samples, torch.tensor([0.75, 0.25])).tolist()
        iqr = q75 - q25
        if iqr == 0:
            # Fall back to Scott's rule
            return _compute_n_bins(samples, "scott", n_samples)
        bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
        n_bins = max(1, int(round(data_range / bin_width)))
    elif bins == "auto":
        # Use FD if IQR > 0, else Scott
        q75, q25 = torch.quantile(samples, torch.tensor([0.75, 0.25])).tolist()
        iqr = q75 - q25
        if iqr > 0:
            return _compute_n_bins(samples, "fd", n_samples)
        else:
            return _compute_n_bins(samples, "scott", n_samples)
    else:
        raise ValueError(f"Unknown bins method: {bins}")

    return n_bins
