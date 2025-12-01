"""Paired t-test implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def paired_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform a paired t-test.

    The paired t-test (also known as the dependent samples t-test or
    matched pairs t-test) tests whether the mean difference between
    paired observations is zero.

    This test is appropriate when two measurements are taken on the
    same subjects (e.g., before and after treatment) or on matched
    pairs of subjects.

    Mathematical Definition
    -----------------------
    Given paired observations :math:`(x_1, y_1), (x_2, y_2), \ldots,
    (x_n, y_n)`, the paired t-test computes the differences:

    .. math::
        d_i = x_i - y_i

    and tests whether the mean of these differences is zero. The
    t-statistic is:

    .. math::
        t = \frac{\bar{d}}{s_d / \sqrt{n}}

    where :math:`\bar{d}` is the mean of the differences and :math:`s_d`
    is the standard deviation of the differences (with Bessel's correction):

    .. math::
        \bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i

    .. math::
        s_d = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (d_i - \bar{d})^2}

    The t-statistic follows a Student's t-distribution with
    :math:`\nu = n - 1` degrees of freedom under the null hypothesis.

    .. note::
        The paired t-test is mathematically equivalent to performing
        a one-sample t-test on the differences :math:`d_i = x_i - y_i`
        with ``popmean=0``.

    Parameters
    ----------
    input1 : Tensor
        First sample tensor (e.g., pre-treatment measurements).
        The test is computed over the last dimension.
        Must have at least 2 elements along the last dimension.
    input2 : Tensor
        Second sample tensor (e.g., post-treatment measurements).
        The test is computed over the last dimension.
        Must have the same shape as ``input1``.
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Specifies the alternative hypothesis:

        - ``"two-sided"``: The mean difference is not zero.
        - ``"less"``: The mean of ``input1`` is less than the mean of
          ``input2`` (i.e., mean difference is negative).
        - ``"greater"``: The mean of ``input1`` is greater than the mean
          of ``input2`` (i.e., mean difference is positive).

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is ``input1.shape[:-1]``.
    pvalue : Tensor
        The p-value for the test. Shape is ``input1.shape[:-1]``.
    df : Tensor
        The degrees of freedom (:math:`n - 1`). Shape is ``input1.shape[:-1]``.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of ``"two-sided"``, ``"less"``, or
        ``"greater"``.

    Examples
    --------
    Test for treatment effect (before vs after):

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import paired_t_test
    >>> before = torch.tensor([120., 125., 130., 118., 140., 135., 128.])
    >>> after = torch.tensor([118., 122., 125., 115., 135., 130., 122.])
    >>> t_stat, p_value, df = paired_t_test(before, after)
    >>> t_stat  # Positive: before > after on average
    tensor(4.5826)
    >>> p_value < 0.05  # Significant treatment effect
    tensor(True)

    One-sided test (treatment reduces values):

    >>> t_stat, p_value, df = paired_t_test(before, after, alternative="greater")
    >>> p_value  # P(mean(before) > mean(after))

    Compare with one-sample t-test on differences:

    >>> from torchscience.statistics.hypothesis_test import one_sample_t_test
    >>> differences = before - after
    >>> t_stat_diff, p_value_diff, df_diff = one_sample_t_test(differences)
    >>> torch.allclose(t_stat, t_stat_diff)
    True

    Batched computation (multiple subjects with repeated measurements):

    >>> subjects = torch.randn(10, 5, 20)  # 10 subjects, 5 conditions, 20 trials
    >>> baseline = torch.randn(10, 5, 20)
    >>> t_stat, p_value, df = paired_t_test(subjects, baseline)
    >>> t_stat.shape
    torch.Size([10, 5])

    Notes
    -----
    - **When to use paired vs independent t-test**: Use the paired t-test
      when observations in the two samples are naturally paired or matched.
      Common scenarios include:

      - Pre-test/post-test designs (same subjects measured twice)
      - Matched pairs (e.g., twins, matched controls)
      - Repeated measures on the same subjects

    - **Equivalence to one-sample t-test**: The paired t-test is equivalent
      to a one-sample t-test on the differences with ``popmean=0``:

      .. code-block:: python

          paired_t_test(x, y) == one_sample_t_test(x - y, popmean=0)

    - **Assumptions**:

      - The differences :math:`d_i = x_i - y_i` are normally distributed
      - Pairs are independent of each other

    - **Advantages over independent t-test**: The paired t-test controls
      for individual differences (subject variability), which increases
      statistical power when the pairing is meaningful.

    - **NaN handling**: If all differences are identical (zero variance),
      the t-statistic will be ``inf`` or ``nan``.

    - **Gradients**: This function supports autograd for both input tensors.

    References
    ----------
    .. [1] Student (W.S. Gosset), "The Probable Error of a Mean,"
           Biometrika, vol. 6, no. 1, pp. 1-25, 1908.

    .. [2] J.L. Fleiss, "The Design and Analysis of Clinical Experiments,"
           Wiley, New York, 1986.

    See Also
    --------
    one_sample_t_test : Test if sample mean equals a specified value.
    two_sample_t_test : Compare means of two independent samples.
    scipy.stats.ttest_rel : SciPy's paired t-test.
    """
    valid_alternatives = ("two-sided", "less", "greater")
    if alternative not in valid_alternatives:
        raise ValueError(
            f"paired_t_test: alternative must be one of {valid_alternatives}, "
            f"got '{alternative}'"
        )

    return torch.ops.torchscience.paired_t_test(input1, input2, alternative)
