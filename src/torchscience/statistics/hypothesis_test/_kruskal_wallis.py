"""Kruskal-Wallis H test."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def kruskal_wallis(
    data: Tensor,
    group_sizes: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Perform the Kruskal-Wallis H test.

    The Kruskal-Wallis H test is a non-parametric test of the null hypothesis
    that the population medians of all groups are equal. It is an extension
    of the Mann-Whitney U test to more than two groups.

    Mathematical Definition
    -----------------------
    Given :math:`k` samples with :math:`n_i` observations each, the test first
    computes ranks of all observations in the combined sample.

    The H statistic is:

    .. math::
        H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)

    where :math:`N = \sum n_i` is the total sample size and :math:`R_i` is
    the sum of ranks for group :math:`i`.

    With tie correction:

    .. math::
        H_{corrected} = \frac{H}{1 - \sum(t^3 - t)/(N^3 - N)}

    Under the null hypothesis, :math:`H` follows a chi-squared distribution
    with :math:`k-1` degrees of freedom.

    Parameters
    ----------
    data : Tensor
        Concatenated data from all groups. Must be 1-dimensional.
    group_sizes : Tensor
        Sizes of each group. Must be 1-dimensional integer tensor.
        The sum of group sizes must equal the length of ``data``.

    Returns
    -------
    statistic : Tensor
        The H statistic (corrected for ties).
    pvalue : Tensor
        The p-value for the test using chi-squared distribution.

    Raises
    ------
    RuntimeError
        If the input tensor requires gradients. Rank-based tests are not
        differentiable.

    Examples
    --------
    Compare three groups:

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import kruskal_wallis
    >>> g1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    >>> g2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    >>> g3 = torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64)
    >>> data = torch.cat([g1, g2, g3])
    >>> group_sizes = torch.tensor([4, 4, 4], dtype=torch.int64)
    >>> stat, pvalue = kruskal_wallis(data, group_sizes)
    >>> pvalue < 0.05  # Reject null hypothesis
    tensor(True)

    Unequal group sizes:

    >>> g1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>> g2 = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    >>> data = torch.cat([g1, g2])
    >>> group_sizes = torch.tensor([3, 5], dtype=torch.int64)
    >>> stat, pvalue = kruskal_wallis(data, group_sizes)

    Notes
    -----
    - **Tie handling**: Ties are handled by assigning average ranks.
      A tie correction factor is applied to the H statistic.

    - **Not differentiable**: This function does not support autograd
      because it relies on ranks, which are not differentiable.

    - **Degrees of freedom**: The chi-squared approximation uses
      :math:`k-1` degrees of freedom, where :math:`k` is the number
      of groups.

    - **Assumptions**: The test assumes that observations are independent
      and from populations with the same shape (but not necessarily
      the same median).

    References
    ----------
    .. [1] Kruskal, W.H. and Wallis, W.A., "Use of ranks in one-criterion
           variance analysis," Journal of the American Statistical Association,
           vol. 47, no. 260, pp. 583-621, 1952.

    See Also
    --------
    mann_whitney_u : Two independent samples rank test.
    wilcoxon_signed_rank : Paired sample rank test.
    scipy.stats.kruskal : SciPy's Kruskal-Wallis H test.
    """
    return torch.ops.torchscience.kruskal_wallis(data, group_sizes)
