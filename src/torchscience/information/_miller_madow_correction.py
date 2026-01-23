"""Miller-Madow bias correction for entropy estimation."""

from typing import Union

import torch
from torch import Tensor


def miller_madow_correction(
    entropy: Tensor,
    num_bins: Union[Tensor, int],
    num_samples: Union[Tensor, int],
) -> Tensor:
    r"""Apply Miller-Madow bias correction to entropy estimates.

    The Miller-Madow correction compensates for the negative bias inherent
    in naive entropy estimation from finite samples. The correction adds a
    term that depends on the number of non-empty bins and the sample size.

    Mathematical Definition
    -----------------------
    Given a naive entropy estimate :math:`\hat{H}`, the corrected estimate is:

    .. math::

        H_{\text{corrected}} = \hat{H} + \frac{m - 1}{2n}

    where :math:`m` is the number of non-empty bins and :math:`n` is the
    sample size.

    Parameters
    ----------
    entropy : Tensor
        Naive entropy estimate. Can have arbitrary batch dimensions.
    num_bins : Tensor or int
        Number of non-empty bins (m). If Tensor, must be broadcastable
        with ``entropy``.
    num_samples : Tensor or int
        Sample size (n). If Tensor, must be broadcastable with ``entropy``.

    Returns
    -------
    Tensor
        Corrected entropy estimate with the same shape as ``entropy``.

    Examples
    --------
    >>> # Basic usage with scalar inputs
    >>> entropy = torch.tensor(1.5)
    >>> miller_madow_correction(entropy, num_bins=10, num_samples=100)
    tensor(1.5450)  # 1.5 + (10-1)/(2*100) = 1.5 + 0.045

    >>> # Batched entropy values
    >>> entropy = torch.tensor([1.0, 1.5, 2.0])
    >>> miller_madow_correction(entropy, num_bins=5, num_samples=50)
    tensor([1.0400, 1.5400, 2.0400])

    >>> # With tensor inputs for bins and samples
    >>> entropy = torch.tensor([1.0, 1.5])
    >>> num_bins = torch.tensor([5, 8])
    >>> num_samples = torch.tensor([100, 200])
    >>> miller_madow_correction(entropy, num_bins, num_samples)
    tensor([1.0200, 1.5175])

    Notes
    -----
    - The correction is always non-negative since :math:`m \geq 1`.
    - For large samples (large n), the correction becomes negligible.
    - The correction assumes the naive entropy estimator (plug-in estimator).
    - When ``num_samples`` is 0, returns the input entropy unchanged (no
      correction applied) to avoid division by zero.
    - When ``num_bins`` is 0, the correction term is :math:`-1/(2n)`, which
      is a small negative value. This edge case typically indicates empty data.

    See Also
    --------
    histogram_entropy : Histogram-based entropy estimation with optional
        Miller-Madow correction.

    References
    ----------
    .. [1] Miller, G. (1955). Note on the bias of information estimates.
           Information Theory in Psychology: Problems and Methods, 95-100.
    """
    if not isinstance(entropy, Tensor):
        raise TypeError(
            f"entropy must be a Tensor, got {type(entropy).__name__}"
        )

    # Convert to tensors if needed, preserving dtype and device
    if isinstance(num_bins, int):
        num_bins = torch.tensor(
            num_bins, dtype=entropy.dtype, device=entropy.device
        )
    elif isinstance(num_bins, Tensor):
        num_bins = num_bins.to(dtype=entropy.dtype, device=entropy.device)
    else:
        raise TypeError(
            f"num_bins must be a Tensor or int, got {type(num_bins).__name__}"
        )

    if isinstance(num_samples, int):
        num_samples = torch.tensor(
            num_samples, dtype=entropy.dtype, device=entropy.device
        )
    elif isinstance(num_samples, Tensor):
        num_samples = num_samples.to(
            dtype=entropy.dtype, device=entropy.device
        )
    else:
        raise TypeError(
            f"num_samples must be a Tensor or int, got {type(num_samples).__name__}"
        )

    # Handle edge case: num_samples = 0 (division by zero protection)
    # Return entropy unchanged when num_samples is 0
    safe_num_samples = torch.where(
        num_samples == 0,
        torch.ones_like(num_samples),
        num_samples,
    )

    # Compute the correction: (m - 1) / (2 * n)
    correction = (num_bins - 1) / (2 * safe_num_samples)

    # Zero out correction where num_samples was 0
    correction = torch.where(
        num_samples == 0,
        torch.zeros_like(correction),
        correction,
    )

    return entropy + correction
