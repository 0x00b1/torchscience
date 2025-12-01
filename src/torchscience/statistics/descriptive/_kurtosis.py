"""Kurtosis implementation."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def kurtosis(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    fisher: bool = True,
    bias: bool = True,
) -> Tensor:
    r"""Compute the kurtosis of a tensor along specified dimensions.

    Kurtosis is a measure of the "tailedness" of a probability distribution.
    This function computes the sample kurtosis of the input tensor.

    Mathematical Definition
    -----------------------
    For a sample :math:`x_1, x_2, \ldots, x_n`:

    **Biased (sample) kurtosis** (``bias=True``):

    .. math::
        g_2 = \frac{m_4}{m_2^2} - 3 \quad \text{(excess, fisher=True)}

    .. math::
        g_2 = \frac{m_4}{m_2^2} \quad \text{(Pearson, fisher=False)}

    where:

    .. math::
        m_2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2

    .. math::
        m_4 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4

    **Unbiased kurtosis** (``bias=False``):

    .. math::
        G_2 = \frac{(n-1)}{(n-2)(n-3)} \left[ (n+1) g_2 + 6 \right]

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    dim : int or tuple of ints, optional
        The dimension or dimensions along which to compute kurtosis.
        If ``None`` (default), computes kurtosis over all elements.
    keepdim : bool, optional
        Whether the output tensor has ``dim`` retained or not.
        Default: ``False``.
    fisher : bool, optional
        If ``True`` (default), compute excess kurtosis (subtract 3).
        Normal distribution has excess kurtosis of 0.
        If ``False``, compute Pearson's kurtosis.
        Normal distribution has Pearson's kurtosis of 3.
    bias : bool, optional
        If ``True`` (default), compute the biased estimator.
        If ``False``, apply bias correction for sample kurtosis.
        Requires at least 4 elements; returns NaN otherwise.

    Returns
    -------
    Tensor
        The kurtosis of the input tensor. Shape depends on ``dim`` and
        ``keepdim`` parameters.

        For complex inputs, computes kurtosis of magnitudes (absolute values).

    Examples
    --------
    Compute kurtosis of all elements:

    >>> x = torch.randn(100)
    >>> torchscience.statistics.descriptive.kurtosis(x)
    tensor(0.1234)  # Value will vary

    Compute kurtosis along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> torchscience.statistics.descriptive.kurtosis(x, dim=1)
    tensor([0.05, -0.12, 0.23])  # Values will vary

    Compute Pearson's kurtosis (normal = 3):

    >>> x = torch.randn(1000)
    >>> torchscience.statistics.descriptive.kurtosis(x, fisher=False)
    tensor(3.05)  # Close to 3 for normal distribution

    Compute unbiased (sample) kurtosis:

    >>> x = torch.randn(50)
    >>> torchscience.statistics.descriptive.kurtosis(x, bias=False)
    tensor(-0.15)  # Bias-corrected value

    Batched computation with keepdim:

    >>> x = torch.randn(2, 3, 100)
    >>> torchscience.statistics.descriptive.kurtosis(x, dim=2, keepdim=True)
    tensor([[[ 0.12], [-0.05], [ 0.18]],
            [[ 0.03], [ 0.22], [-0.11]]])

    Notes
    -----
    - A **normal distribution** has excess kurtosis (Fisher's) of 0 and
      Pearson's kurtosis of 3.

    - **Leptokurtic** distributions (positive excess kurtosis) have heavier
      tails than normal (e.g., t-distribution, Laplace).

    - **Platykurtic** distributions (negative excess kurtosis) have lighter
      tails than normal (e.g., uniform distribution).

    - For **complex tensors**, kurtosis is computed on the magnitudes
      :math:`|z|` of the complex values.

    - **Gradients** are supported for the input tensor, enabling use in
      gradient-based optimization.

    - Returns ``NaN`` for:
        - Zero variance (all elements equal)
        - Fewer than 2 elements
        - Fewer than 4 elements when ``bias=False``

    Warnings
    --------
    - Kurtosis is sensitive to outliers. A few extreme values can
      significantly affect the result.

    - For small sample sizes, the bias correction (``bias=False``) may
      produce highly variable estimates.

    - Numerical precision may be reduced for half-precision (float16)
      inputs due to the fourth power computation.

    References
    ----------
    .. [1] D.N. Joanes and C.A. Gill, "Comparing Measures of Sample Skewness
           and Kurtosis," The Statistician, vol. 47, no. 1, pp. 183-189, 1998.

    .. [2] R.A. Fisher, "Statistical Methods for Research Workers,"
           Oliver and Boyd, 1925.

    .. [3] SciPy documentation for scipy.stats.kurtosis:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

    See Also
    --------
    scipy.stats.kurtosis : SciPy's kurtosis function.
    torch.var : Variance (second central moment).
    torch.std : Standard deviation.
    """
    # Handle complex inputs by computing kurtosis of magnitudes
    if input.is_complex():
        input = input.abs()

    # Handle dim parameter
    if dim is None:
        dim_list = None
    elif isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)

    return torch.ops.torchscience.kurtosis(
        input,
        dim_list,
        keepdim,
        fisher,
        bias,
    )
