"""Two-sample t-test implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def two_sample_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform a two-sample t-test.

    The two-sample t-test (also known as the independent samples t-test)
    tests whether the means of two independent populations are equal.

    This function supports both **Student's t-test** (assuming equal
    variances) and **Welch's t-test** (not assuming equal variances).
    Welch's test is the default as it is more robust.

    Mathematical Definition
    -----------------------
    Given two samples :math:`x_1, \ldots, x_{n_1}` and
    :math:`y_1, \ldots, y_{n_2}` with sample means :math:`\bar{x}`,
    :math:`\bar{y}` and sample variances :math:`s_x^2`, :math:`s_y^2`:

    **Student's t-test** (``equal_var=True``):

    Assumes equal population variances. The pooled variance is:

    .. math::
        s_p^2 = \frac{(n_1 - 1) s_x^2 + (n_2 - 1) s_y^2}{n_1 + n_2 - 2}

    The t-statistic is:

    .. math::
        t = \frac{\bar{x} - \bar{y}}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}

    with degrees of freedom :math:`\nu = n_1 + n_2 - 2`.

    **Welch's t-test** (``equal_var=False``, default):

    Does not assume equal variances. The t-statistic is:

    .. math::
        t = \frac{\bar{x} - \bar{y}}{\sqrt{\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}}}

    The degrees of freedom are approximated using the Welch-Satterthwaite
    equation:

    .. math::
        \nu = \frac{\left(\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}\right)^2}
        {\frac{(s_x^2/n_1)^2}{n_1 - 1} + \frac{(s_y^2/n_2)^2}{n_2 - 1}}

    Parameters
    ----------
    input1 : Tensor
        First sample tensor. The test is computed over the last dimension.
        Must have at least 2 elements along the last dimension.
    input2 : Tensor
        Second sample tensor. The test is computed over the last dimension.
        Must have at least 2 elements along the last dimension.
        The shapes of ``input1`` and ``input2`` must be broadcastable
        except for the last dimension.
    equal_var : bool, default=False
        If ``True``, perform Student's t-test assuming equal population
        variances. If ``False`` (default), perform Welch's t-test which
        does not assume equal variances.

        .. note::
            Welch's test is the default because it is more robust and
            performs well even when variances are equal, with only
            slightly reduced power.

    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Specifies the alternative hypothesis:

        - ``"two-sided"``: The means of the two samples are different.
        - ``"less"``: The mean of ``input1`` is less than the mean of
          ``input2``.
        - ``"greater"``: The mean of ``input1`` is greater than the mean
          of ``input2``.

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is the broadcast of ``input1.shape[:-1]``
        and ``input2.shape[:-1]``.
    pvalue : Tensor
        The p-value for the test. Same shape as ``statistic``.
    df : Tensor
        The degrees of freedom. Same shape as ``statistic``. For Welch's
        test, this is typically a non-integer value.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of ``"two-sided"``, ``"less"``, or
        ``"greater"``.

    Examples
    --------
    Compare two independent samples (Welch's test, default):

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import two_sample_t_test
    >>> torch.manual_seed(42)
    >>> group1 = torch.randn(50) + 0.0
    >>> group2 = torch.randn(50) + 0.5
    >>> t_stat, p_value, df = two_sample_t_test(group1, group2)
    >>> p_value < 0.05  # Statistically significant difference
    tensor(True)

    Student's t-test (assuming equal variances):

    >>> t_stat, p_value, df = two_sample_t_test(group1, group2, equal_var=True)
    >>> df  # Integer degrees of freedom for Student's test
    tensor(98.)  # n1 + n2 - 2

    One-sided test:

    >>> t_stat, p_value, df = two_sample_t_test(
    ...     group1, group2, alternative="less"
    ... )
    >>> p_value  # P(mean1 < mean2)

    Samples with unequal sizes:

    >>> small_group = torch.randn(10)
    >>> large_group = torch.randn(100)
    >>> t_stat, p_value, df = two_sample_t_test(small_group, large_group)
    >>> df  # Welch-Satterthwaite approximation
    tensor(10.5)  # Approximately, depends on variances

    Batched computation:

    >>> samples1 = torch.randn(3, 50)
    >>> samples2 = torch.randn(3, 60)
    >>> t_stat, p_value, df = two_sample_t_test(samples1, samples2)
    >>> t_stat.shape
    torch.Size([3])

    Notes
    -----
    - **Welch's vs Student's**: Welch's t-test is recommended as the default
      because:

      - It does not assume equal variances
      - It has similar power to Student's test when variances are equal
      - It is more robust when variances are unequal

    - **Assumptions**:

      - Both samples are drawn from normally distributed populations
      - Samples are independent
      - For Student's test: populations have equal variances

    - **Central Limit Theorem**: For large samples (:math:`n > 30`), the
      test is robust to violations of normality.

    - **Degrees of freedom**: For Welch's test, the degrees of freedom
      are computed using the Welch-Satterthwaite approximation, which
      typically yields a non-integer value.

    - **NaN handling**: If either sample has zero variance (all values
      are identical), the test statistic may be ``inf`` or ``nan``.

    - **Gradients**: This function supports autograd for both input tensors.

    References
    ----------
    .. [1] B.L. Welch, "The generalization of 'Student's' problem when
           several different population variances are involved,"
           Biometrika, vol. 34, no. 1-2, pp. 28-35, 1947.

    .. [2] F.E. Satterthwaite, "An approximate distribution of estimates
           of variance components," Biometrics Bulletin, vol. 2, no. 6,
           pp. 110-114, 1946.

    .. [3] Student (W.S. Gosset), "The Probable Error of a Mean,"
           Biometrika, vol. 6, no. 1, pp. 1-25, 1908.

    See Also
    --------
    one_sample_t_test : Test if sample mean equals a specified value.
    paired_t_test : Compare means of two paired samples.
    scipy.stats.ttest_ind : SciPy's independent samples t-test.
    """
    valid_alternatives = ("two-sided", "less", "greater")
    if alternative not in valid_alternatives:
        raise ValueError(
            f"two_sample_t_test: alternative must be one of {valid_alternatives}, "
            f"got '{alternative}'"
        )

    return torch.ops.torchscience.two_sample_t_test(
        input1, input2, equal_var, alternative
    )
