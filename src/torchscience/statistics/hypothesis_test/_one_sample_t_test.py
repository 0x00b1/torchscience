"""One-sample t-test implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def one_sample_t_test(
    input: Tensor,
    popmean: float = 0.0,
    *,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform a one-sample t-test.

    The one-sample t-test tests whether the mean of a population from which
    a sample is drawn equals a specified value (the null hypothesis).

    Mathematical Definition
    -----------------------
    Given a sample :math:`x_1, x_2, \ldots, x_n` with sample mean
    :math:`\bar{x}` and sample standard deviation :math:`s`, the
    t-statistic is:

    .. math::
        t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}

    where :math:`\mu_0` is the hypothesized population mean (``popmean``),
    and :math:`s` is the sample standard deviation with Bessel's correction:

    .. math::
        s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}

    The t-statistic follows a Student's t-distribution with :math:`\nu = n - 1`
    degrees of freedom under the null hypothesis.

    Parameters
    ----------
    input : Tensor
        Input tensor containing the sample data. The test is computed over
        the last dimension. Must have at least 2 elements along the last
        dimension.
    popmean : float, default=0.0
        The hypothesized population mean under the null hypothesis.
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Specifies the alternative hypothesis:

        - ``"two-sided"``: The mean of the sample differs from ``popmean``.
        - ``"less"``: The mean of the sample is less than ``popmean``.
        - ``"greater"``: The mean of the sample is greater than ``popmean``.

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is ``input.shape[:-1]``.
    pvalue : Tensor
        The p-value for the test. Shape is ``input.shape[:-1]``.
    df : Tensor
        The degrees of freedom (:math:`n - 1`). Shape is ``input.shape[:-1]``.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of ``"two-sided"``, ``"less"``, or
        ``"greater"``.

    Examples
    --------
    Test if sample mean differs from zero:

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import one_sample_t_test
    >>> torch.manual_seed(42)
    >>> sample = torch.randn(100) + 0.5  # Mean shifted from 0
    >>> t_stat, p_value, df = one_sample_t_test(sample)
    >>> t_stat
    tensor(5.1234)  # Positive t-stat indicates mean > 0
    >>> p_value
    tensor(1.2e-06)  # Small p-value: reject null hypothesis

    Test against a specific population mean:

    >>> sample = torch.tensor([5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9])
    >>> t_stat, p_value, df = one_sample_t_test(sample, popmean=5.0)
    >>> p_value > 0.05  # Cannot reject null hypothesis
    tensor(True)

    One-sided test (greater):

    >>> sample = torch.randn(50) + 0.3
    >>> t_stat, p_value, df = one_sample_t_test(sample, alternative="greater")
    >>> p_value  # One-sided p-value for greater alternative

    Batched computation:

    >>> samples = torch.randn(3, 100)  # 3 independent samples
    >>> t_stat, p_value, df = one_sample_t_test(samples)
    >>> t_stat.shape
    torch.Size([3])
    >>> p_value.shape
    torch.Size([3])

    Notes
    -----
    - **Assumptions**: The one-sample t-test assumes that the sample is
      drawn from a normally distributed population. However, due to the
      Central Limit Theorem, the test is robust to non-normality for
      large sample sizes (typically :math:`n > 30`).

    - **Degrees of freedom**: The t-statistic has :math:`n - 1` degrees
      of freedom, where :math:`n` is the sample size.

    - **Two-sided p-value**: For the two-sided alternative, the p-value
      is computed as :math:`2 \times P(T > |t|)` where :math:`T` follows
      a t-distribution with :math:`n - 1` degrees of freedom.

    - **One-sided p-values**:

      - For ``alternative="greater"``: :math:`P(T > t)`
      - For ``alternative="less"``: :math:`P(T < t)`

    - **NaN handling**: If the sample has zero variance (all values are
      identical), the t-statistic will be ``inf`` or ``nan``, and the
      p-value will be ``nan``.

    - **Gradients**: This function supports autograd for the input tensor.

    References
    ----------
    .. [1] Student (W.S. Gosset), "The Probable Error of a Mean,"
           Biometrika, vol. 6, no. 1, pp. 1-25, 1908.

    .. [2] R.A. Fisher, "Statistical Methods for Research Workers,"
           Oliver and Boyd, Edinburgh, 1925.

    See Also
    --------
    two_sample_t_test : Compare means of two independent samples.
    paired_t_test : Compare means of two paired samples.
    scipy.stats.ttest_1samp : SciPy's one-sample t-test.
    """
    valid_alternatives = ("two-sided", "less", "greater")
    if alternative not in valid_alternatives:
        raise ValueError(
            f"one_sample_t_test: alternative must be one of {valid_alternatives}, "
            f"got '{alternative}'"
        )

    return torch.ops.torchscience.one_sample_t_test(
        input, popmean, alternative
    )
