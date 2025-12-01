"""One-way ANOVA (F-test)."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def f_oneway(*samples: Tensor) -> tuple[Tensor, Tensor]:
    r"""Perform one-way ANOVA F-test.

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean. The test is based on the ratio of between-group
    variance to within-group variance.

    .. math::
        F = \frac{\text{MS}_\text{between}}{\text{MS}_\text{within}}
          = \frac{SS_\text{between} / (k-1)}{SS_\text{within} / (N-k)}

    Parameters
    ----------
    *samples : Tensor
        Two or more 1D tensors of sample data. Each tensor represents a group.

    Returns
    -------
    statistic : Tensor
        The F-statistic.
    pvalue : Tensor
        The p-value for the test.

    Raises
    ------
    ValueError
        If fewer than 2 groups are provided.

    Examples
    --------
    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import f_oneway
    >>> group1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> group2 = torch.tensor([5.0, 6.0, 7.0, 8.0])
    >>> group3 = torch.tensor([9.0, 10.0, 11.0, 12.0])
    >>> statistic, pvalue = f_oneway(group1, group2, group3)
    >>> pvalue < 0.05  # Reject null hypothesis
    tensor(True)

    References
    ----------
    .. [1] Fisher, R.A. (1925). "Statistical Methods for Research Workers."
    """
    if len(samples) < 2:
        raise ValueError("f_oneway requires at least 2 groups")

    # Validate all inputs
    dtype = samples[0].dtype
    device = samples[0].device
    for i, s in enumerate(samples):
        if s.dim() != 1:
            raise ValueError(
                f"All samples must be 1D, group {i} has shape {s.shape}"
            )
        if s.dtype != dtype:
            samples = tuple(s.to(dtype) for s in samples)
            break

    # Concatenate all groups
    data = torch.cat(samples)
    group_sizes = torch.tensor(
        [s.size(0) for s in samples], dtype=torch.long, device=device
    )

    return torch.ops.torchscience.f_oneway(data, group_sizes)
