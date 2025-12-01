"""Mann-Whitney U test (Wilcoxon rank-sum test)."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def mann_whitney_u(
    x: Tensor,
    y: Tensor,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor]:
    r"""
    Perform the Mann-Whitney U test (Wilcoxon rank-sum test).

    The Mann-Whitney U test is a non-parametric test of the null hypothesis
    that the distributions of two independent samples are equal (or that
    observations from one sample tend to be larger than observations from
    the other).

    Mathematical Definition
    -----------------------
    Given two independent samples :math:`X = \{x_1, ..., x_{n_1}\}` and
    :math:`Y = \{y_1, ..., y_{n_2}\}`, the test first computes ranks of
    all observations in the combined sample.

    The U statistic is:

    .. math::
        U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}

    where :math:`R_1` is the sum of ranks for sample :math:`X`.

    Under the null hypothesis, the expected value and variance are:

    .. math::
        \mu_U = \frac{n_1 n_2}{2}, \quad
        \sigma_U^2 = \frac{n_1 n_2 (n + 1)}{12} \cdot T

    where :math:`T` is the tie correction factor:

    .. math::
        T = 1 - \frac{\sum_i (t_i^3 - t_i)}{n^3 - n}

    and :math:`t_i` is the number of ties in group :math:`i`.

    Parameters
    ----------
    x : Tensor
        First sample. Must be 1-dimensional.
    y : Tensor
        Second sample. Must be 1-dimensional.
    alternative : str, optional
        The alternative hypothesis:

        - ``"two-sided"`` (default): The distributions are not equal.
        - ``"less"``: The distribution of x is stochastically less than y.
        - ``"greater"``: The distribution of x is stochastically greater than y.

    Returns
    -------
    statistic : Tensor
        The U statistic for the first sample.
    pvalue : Tensor
        The p-value for the test using asymptotic normal approximation.

    Raises
    ------
    RuntimeError
        If the input tensors require gradients. Rank-based tests are not
        differentiable.

    Examples
    --------
    Test if two samples come from the same distribution:

    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import mann_whitney_u
    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    >>> y = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)
    >>> stat, pvalue = mann_whitney_u(x, y)
    >>> pvalue < 0.05  # Reject null hypothesis
    tensor(True)

    Using alternative hypotheses:

    >>> stat, pvalue = mann_whitney_u(x, y, alternative="less")
    >>> pvalue < 0.05  # x values tend to be smaller
    tensor(True)

    Notes
    -----
    - **Asymptotic approximation**: This implementation uses the normal
      approximation for the p-value, which is accurate for moderate to
      large sample sizes. A continuity correction of 0.5 is applied.

    - **Tie handling**: Ties are handled by assigning average ranks.
      A tie correction factor is applied to the variance calculation.

    - **Not differentiable**: This function does not support autograd
      because it relies on ranks, which are not differentiable.

    - **Sample size**: The test requires at least 1 observation in each
      sample, but the normal approximation is most accurate when
      :math:`n_1, n_2 \geq 8`.

    References
    ----------
    .. [1] Mann, H.B. and Whitney, D.R., "On a test of whether one of two
           random variables is stochastically larger than the other,"
           Annals of Mathematical Statistics, vol. 18, no. 1, pp. 50-60, 1947.

    .. [2] Wilcoxon, F., "Individual comparisons by ranking methods,"
           Biometrics Bulletin, vol. 1, no. 6, pp. 80-83, 1945.

    See Also
    --------
    wilcoxon_signed_rank : Paired sample rank test.
    kruskal_wallis : Multi-group extension.
    scipy.stats.mannwhitneyu : SciPy's Mann-Whitney U test.
    """
    return torch.ops.torchscience.mann_whitney_u(x, y, alternative)
