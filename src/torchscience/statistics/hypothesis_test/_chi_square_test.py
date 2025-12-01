"""Chi-square goodness-of-fit test."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def chi_square_test(
    observed: Tensor,
    expected: Tensor | None = None,
    *,
    ddof: int = 0,
) -> tuple[Tensor, Tensor]:
    r"""Perform the chi-square goodness-of-fit test.

    The chi-square test tests whether observed frequencies differ significantly
    from expected frequencies. It computes:

    .. math::
        \chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}

    where :math:`O_i` are observed frequencies and :math:`E_i` are expected
    frequencies.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies. Must be non-negative. The test is computed over
        the last dimension.
    expected : Tensor, optional
        Expected frequencies. Must have the same shape as ``observed``.
        If not provided, a uniform distribution is assumed (each category
        has expected frequency ``sum(observed) / k``).
    ddof : int, default=0
        Delta degrees of freedom. The degrees of freedom for the test are
        ``k - 1 - ddof``, where ``k`` is the number of categories.

    Returns
    -------
    statistic : Tensor
        The chi-square test statistic. Has shape ``observed.shape[:-1]``.
    pvalue : Tensor
        The p-value for the test. Has shape ``observed.shape[:-1]``.

    Notes
    -----
    - Requires at least 2 categories with positive expected frequencies.
    - The chi-square approximation is most accurate when expected frequencies
      are at least 5. For smaller expected frequencies, consider exact tests.
    - Returns NaN if degrees of freedom are non-positive or if any expected
      frequency is zero.

    Examples
    --------
    >>> import torch
    >>> from torchscience.statistics.hypothesis_test import chi_square_test
    >>> observed = torch.tensor([16.0, 18.0, 16.0, 14.0, 12.0, 12.0])
    >>> statistic, pvalue = chi_square_test(observed)
    >>> print(f"chi2 = {statistic:.4f}, p-value = {pvalue:.4f}")

    With explicit expected frequencies:

    >>> expected = torch.tensor([16.0, 16.0, 16.0, 16.0, 12.0, 12.0])
    >>> statistic, pvalue = chi_square_test(observed, expected)

    References
    ----------
    .. [1] Pearson, K. (1900). "On the criterion that a given system of
           deviations from the probable in the case of a correlated system
           of variables is such that it can be reasonably supposed to have
           arisen from random sampling". Philosophical Magazine, 50(302),
           157-175.
    """
    return torch.ops.torchscience.chi_square_test(observed, expected, ddof)
