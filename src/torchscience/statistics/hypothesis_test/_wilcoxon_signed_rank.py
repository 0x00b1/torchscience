"""Wilcoxon signed-rank test."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def wilcoxon_signed_rank(
    x: Tensor,
    y: Tensor | None = None,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> tuple[Tensor, Tensor]:
    r"""
    Perform the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test is a non-parametric test of the null
    hypothesis that the median of a sample (or paired differences) is zero.

    Mathematical Definition
    -----------------------
    Given a sample :math:`X = \{x_1, ..., x_n\}` (or differences :math:`D_i = x_i - y_i`
    for paired samples), the test first removes zero differences (by default),
    computes absolute values, and ranks them.

    The test statistic :math:`W^+` is the sum of ranks for positive differences:

    .. math::
        W^+ = \sum_{i: d_i > 0} R_i

    Under the null hypothesis, the expected value and variance are:

    .. math::
        \mu = \frac{n_r(n_r + 1)}{4}, \quad
        \sigma^2 = \frac{n_r(n_r + 1)(2n_r + 1)}{24} \cdot T

    where :math:`n_r` is the number of non-zero differences and :math:`T`
    is the tie correction factor.

    Parameters
    ----------
    x : Tensor
        First sample. Must be 1-dimensional.
    y : Tensor, optional
        Second sample. If provided, the test is performed on the differences
        ``x - y``. Must have the same shape as ``x``.
    alternative : str, optional
        The alternative hypothesis:

        - ``"two-sided"`` (default): The median is not equal to zero.
        - ``"less"``: The median is less than zero.
        - ``"greater"``: The median is greater than zero.
    zero_method : str, optional
        How to handle zero differences:

        - ``"wilcox"`` (default): Exclude zero differences from the ranking.
        - ``"pratt"``: Include zero differences in the ranking but they
          do not contribute to the test statistic.

    Returns
    -------
    statistic : Tensor
        The W+ statistic (sum of ranks for positive differences).
    pvalue : Tensor
        The p-value for the test using asymptotic normal approximation.

    Raises
    ------
    RuntimeError
        If the input tensors require gradients. Rank-based tests are not
        differentiable.

    Examples
    --------
    One-sample test (test if median equals zero):

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import wilcoxon_signed_rank
    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    >>> stat, pvalue = wilcoxon_signed_rank(x)
    >>> pvalue < 0.05  # Reject null hypothesis (median != 0)
    tensor(True)

    Paired samples test:

    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    >>> y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    >>> stat, pvalue = wilcoxon_signed_rank(x, y)
    >>> pvalue < 0.05  # Reject null hypothesis (median difference != 0)
    tensor(True)

    Notes
    -----
    - **Asymptotic approximation**: This implementation uses the normal
      approximation for the p-value, which is accurate for moderate to
      large sample sizes. A continuity correction of 0.5 is applied.

    - **Tie handling**: Ties are handled by assigning average ranks.
      A tie correction factor is applied to the variance calculation.

    - **Zero differences**: The default ``zero_method="wilcox"`` excludes
      pairs with zero difference. Use ``zero_method="pratt"`` to include
      them in the ranking.

    - **Not differentiable**: This function does not support autograd
      because it relies on ranks, which are not differentiable.

    References
    ----------
    .. [1] Wilcoxon, F., "Individual comparisons by ranking methods,"
           Biometrics Bulletin, vol. 1, no. 6, pp. 80-83, 1945.

    .. [2] Pratt, J.W., "Remarks on zeros and ties in the Wilcoxon signed
           rank procedures," Journal of the American Statistical Association,
           vol. 54, no. 287, pp. 655-667, 1959.

    See Also
    --------
    mann_whitney_u : Two independent samples rank test.
    kruskal_wallis : Multi-group extension.
    scipy.stats.wilcoxon : SciPy's Wilcoxon signed-rank test.
    """
    return torch.ops.torchscience.wilcoxon_signed_rank(
        x, y, alternative, zero_method
    )
