"""Shapiro-Wilk test for normality."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def shapiro_wilk(
    input: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that a sample comes
    from a normally distributed population.

    Mathematical Definition
    -----------------------
    Given an ordered sample :math:`x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}`,
    the Shapiro-Wilk statistic is:

    .. math::
        W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}
                 {\sum_{i=1}^{n} (x_i - \bar{x})^2}

    where:

    - :math:`x_{(i)}` are the order statistics (sorted values)
    - :math:`\bar{x}` is the sample mean
    - :math:`a_i` are coefficients derived from expected normal order statistics

    The statistic :math:`W` measures how well the sample order statistics
    match the expected order statistics from a normal distribution. Values
    of :math:`W` close to 1 indicate normality.

    Parameters
    ----------
    input : Tensor
        Input tensor containing the sample data. The test is computed over
        the last dimension. Must have at least 3 elements along the last
        dimension.

    Returns
    -------
    statistic : Tensor
        The W-statistic. Values close to 1 indicate normality.
        Shape is ``input.shape[:-1]``.
    pvalue : Tensor
        The p-value for the test. Small values (typically < 0.05) indicate
        the null hypothesis of normality should be rejected.
        Shape is ``input.shape[:-1]``.

    Raises
    ------
    RuntimeError
        If the input tensor requires gradients. The Shapiro-Wilk test is
        based on order statistics which are not differentiable.

    Examples
    --------
    Test if a sample is normally distributed:

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import shapiro_wilk
    >>> torch.manual_seed(42)
    >>> sample = torch.randn(100, dtype=torch.float64)
    >>> stat, pvalue = shapiro_wilk(sample)
    >>> stat  # Close to 1 for normal data
    tensor(0.9912, dtype=torch.float64)
    >>> pvalue > 0.05  # Cannot reject normality
    tensor(True)

    Test a non-normal sample (uniform distribution):

    >>> uniform_sample = torch.rand(100, dtype=torch.float64)
    >>> stat, pvalue = shapiro_wilk(uniform_sample)
    >>> pvalue < 0.05  # Reject normality
    tensor(True)

    Batched computation:

    >>> samples = torch.randn(5, 50, dtype=torch.float64)
    >>> stat, pvalue = shapiro_wilk(samples)
    >>> stat.shape
    torch.Size([5])

    Notes
    -----
    - **Sample size**: The test is most reliable for samples with
      :math:`3 \leq n \leq 5000`. For :math:`n < 3`, NaN is returned.

    - **Interpretation**:

      - :math:`W` close to 1: Sample is consistent with normality
      - :math:`W` significantly less than 1: Evidence against normality
      - p-value < 0.05: Reject null hypothesis of normality at 5% level

    - **P-value computation**: Uses the Royston (1992) polynomial
      approximation for the distribution of the W statistic.

    - **Not differentiable**: This function does not support autograd
      because it relies on order statistics (sorting), which are not
      differentiable.

    - **Power**: The Shapiro-Wilk test is one of the most powerful
      normality tests, especially for small sample sizes. It is
      particularly sensitive to departures from normality in the tails.

    References
    ----------
    .. [1] Shapiro, S.S. and Wilk, M.B., "An analysis of variance test
           for normality (complete samples)," Biometrika, vol. 52,
           no. 3/4, pp. 591-611, 1965.

    .. [2] Royston, P., "Approximating the Shapiro-Wilk W-test for
           non-normality," Statistics and Computing, vol. 2, no. 3,
           pp. 117-119, 1992.

    See Also
    --------
    anderson_darling : Anderson-Darling test for normality.
    scipy.stats.shapiro : SciPy's Shapiro-Wilk test.
    """
    return torch.ops.torchscience.shapiro_wilk(input)
