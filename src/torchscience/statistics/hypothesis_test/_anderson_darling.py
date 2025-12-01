"""Anderson-Darling test for normality."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def anderson_darling(
    input: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform the Anderson-Darling test for normality.

    The Anderson-Darling test tests the null hypothesis that a sample comes
    from a normally distributed population. It gives more weight to the tails
    of the distribution than the Kolmogorov-Smirnov test.

    Mathematical Definition
    -----------------------
    Given a sample of size :math:`n` with standardized order statistics
    :math:`Y_{(1)} \leq Y_{(2)} \leq \cdots \leq Y_{(n)}`, the Anderson-Darling
    statistic is:

    .. math::
        A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} (2i-1)
              \left[\ln F(Y_{(i)}) + \ln(1 - F(Y_{(n+1-i)}))\right]

    where :math:`F` is the standard normal CDF.

    The statistic is then adjusted for sample size using Stephens (1974)
    correction:

    .. math::
        A^{2*} = A^2 \left(1 + \frac{0.75}{n} + \frac{2.25}{n^2}\right)

    Parameters
    ----------
    input : Tensor
        Input tensor containing the sample data. The test is computed over
        the last dimension. Must have at least 8 elements along the last
        dimension for reliable results.

    Returns
    -------
    statistic : Tensor
        The :math:`A^{2*}` statistic (sample-size corrected).
        Shape is ``input.shape[:-1]``.
    critical_values : Tensor
        Critical values at 5 significance levels (15%, 10%, 5%, 2.5%, 1%).
        Shape is ``input.shape[:-1] + (5,)``.
    significance_levels : Tensor
        The significance levels: ``[0.15, 0.10, 0.05, 0.025, 0.01]``.
        Shape is ``(5,)``.

    Raises
    ------
    RuntimeError
        If the input tensor requires gradients. The Anderson-Darling test is
        based on order statistics which are not differentiable.

    Examples
    --------
    Test if a sample is normally distributed:

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import anderson_darling
    >>> torch.manual_seed(42)
    >>> sample = torch.randn(100, dtype=torch.float64)
    >>> stat, cv, sl = anderson_darling(sample)
    >>> stat  # A^2 statistic
    tensor(0.2345, dtype=torch.float64)
    >>> stat < cv[2]  # Compare with 5% critical value
    tensor(True)  # Cannot reject normality

    Test a non-normal sample (uniform distribution):

    >>> uniform_sample = torch.rand(100, dtype=torch.float64)
    >>> stat, cv, sl = anderson_darling(uniform_sample)
    >>> stat > cv[2]  # Compare with 5% critical value
    tensor(True)  # Reject normality

    View critical values and significance levels:

    >>> stat, cv, sl = anderson_darling(torch.randn(50, dtype=torch.float64))
    >>> sl
    tensor([0.1500, 0.1000, 0.0500, 0.0250, 0.0100], dtype=torch.float64)
    >>> cv
    tensor([0.5760, 0.6560, 0.7870, 0.9180, 1.0920], dtype=torch.float64)

    Batched computation:

    >>> samples = torch.randn(5, 50, dtype=torch.float64)
    >>> stat, cv, sl = anderson_darling(samples)
    >>> stat.shape
    torch.Size([5])
    >>> cv.shape
    torch.Size([5, 5])

    Notes
    -----
    - **Sample size**: The test requires at least 8 samples for reliable
      results. For :math:`n < 8`, NaN is returned.

    - **Interpretation**: Compare the statistic to critical values:

      - If :math:`A^{2*} <` critical value: Cannot reject normality
      - If :math:`A^{2*} >` critical value: Reject normality at that level

    - **Critical values**: The critical values are for the normal distribution
      case and are derived from Stephens (1974):

      ======= ================
      Level   Critical Value
      ======= ================
      15%     0.576
      10%     0.656
      5%      0.787
      2.5%    0.918
      1%      1.092
      ======= ================

    - **Not differentiable**: This function does not support autograd
      because it relies on order statistics (sorting), which are not
      differentiable.

    - **Comparison with Shapiro-Wilk**: The Anderson-Darling test is
      particularly sensitive to departures in the tails of the distribution,
      while the Shapiro-Wilk test has better overall power for detecting
      non-normality in small samples.

    References
    ----------
    .. [1] Anderson, T.W. and Darling, D.A., "A test of goodness of fit,"
           Journal of the American Statistical Association, vol. 49,
           no. 268, pp. 765-769, 1954.

    .. [2] Stephens, M.A., "EDF Statistics for Goodness of Fit and Some
           Comparisons," Journal of the American Statistical Association,
           vol. 69, no. 347, pp. 730-737, 1974.

    See Also
    --------
    shapiro_wilk : Shapiro-Wilk test for normality.
    scipy.stats.anderson : SciPy's Anderson-Darling test.
    """
    return torch.ops.torchscience.anderson_darling(input)
