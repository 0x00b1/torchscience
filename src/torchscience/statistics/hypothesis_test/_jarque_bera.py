"""Jarque-Bera test for normality."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def jarque_bera(input: Tensor) -> tuple[Tensor, Tensor]:
    r"""Perform the Jarque-Bera test for normality.

    The Jarque-Bera test tests whether sample data has skewness and kurtosis
    matching a normal distribution. It is based on the sample skewness and
    sample kurtosis.

    .. math::
        JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)

    where :math:`S` is the sample skewness and :math:`K` is the sample kurtosis.

    Under the null hypothesis of normality, the test statistic follows a
    chi-squared distribution with 2 degrees of freedom asymptotically.

    Parameters
    ----------
    input : Tensor
        Input tensor containing sample data. The test is computed over the
        last dimension. For batched inputs, the test is applied independently
        to each sample along the last dimension.

    Returns
    -------
    statistic : Tensor
        The Jarque-Bera test statistic. Has shape ``input.shape[:-1]``.
    pvalue : Tensor
        The two-sided p-value for the test. Has shape ``input.shape[:-1]``.

    Notes
    -----
    - Requires at least 3 samples (``input.size(-1) >= 3``). Returns NaN for
      insufficient samples.
    - This test is only valid for large sample sizes. For small samples,
      consider the Shapiro-Wilk test.
    - Gradients are computed with respect to the statistic only. The p-value
      gradient is not propagated as it involves the chi-squared survival function.

    Examples
    --------
    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import jarque_bera
    >>> torch.manual_seed(42)
    >>> sample = torch.randn(1000)
    >>> statistic, pvalue = jarque_bera(sample)
    >>> print(f"JB statistic: {statistic:.4f}, p-value: {pvalue:.4f}")

    References
    ----------
    .. [1] Jarque, C. M., & Bera, A. K. (1980). "Efficient tests for normality,
           homoscedasticity and serial independence of regression residuals".
           Economics Letters, 6(3), 255-259.
    .. [2] Jarque, C. M., & Bera, A. K. (1987). "A test for normality of
           observations and regression residuals". International Statistical
           Review, 55(2), 163-172.
    """
    return torch.ops.torchscience.jarque_bera(input)
