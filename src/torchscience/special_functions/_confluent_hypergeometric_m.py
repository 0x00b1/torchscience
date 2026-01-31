import torch
from torch import Tensor


def confluent_hypergeometric_m(a: Tensor, b: Tensor, z: Tensor) -> Tensor:
    r"""
    Confluent hypergeometric function M(a, b, z), also known as Kummer's
    function of the first kind or 1F1(a; b; z).

    Mathematical Definition
    -----------------------
    The confluent hypergeometric function M is defined by the series:

    .. math::

       M(a, b, z) = {}_1F_1(a; b; z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b)_n \, n!} z^n

    where :math:`(x)_n = x(x+1)\cdots(x+n-1)` is the Pochhammer symbol
    (rising factorial), with :math:`(x)_0 = 1`.

    This series converges for all finite z (entire function in z).

    Domain
    ------
    - a: any real or complex value
    - b: must not be a non-positive integer (poles at b = 0, -1, -2, ...)
    - z: any real or complex value (entire function)

    Special Values
    --------------
    - M(a, b, 0) = 1
    - M(0, b, z) = 1
    - M(a, a, z) = exp(z)
    - M(a, b, z) = exp(z) * M(b-a, b, -z)  (Kummer transformation)

    Parameters
    ----------
    a : Tensor
        Numerator parameter. Broadcasting with b and z is supported.
    b : Tensor
        Denominator parameter. Must not be a non-positive integer.
        Broadcasting with a and z is supported.
    z : Tensor
        Input value. Any real or complex value is valid.
        Broadcasting with a and b is supported.

    Returns
    -------
    Tensor
        The confluent hypergeometric function M(a, b, z) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([2.0])
    >>> z = torch.tensor([0.5])
    >>> confluent_hypergeometric_m(a, b, z)
    tensor([1.2974])

    Special case M(a, a, z) = exp(z):

    >>> a = torch.tensor([2.0])
    >>> z = torch.tensor([1.0])
    >>> result = confluent_hypergeometric_m(a, a, z)
    >>> expected = torch.exp(z)
    >>> torch.allclose(result, expected)
    True

    See Also
    --------
    hypergeometric_2_f_1 : Gauss hypergeometric function
    gamma : Gamma function
    """
    return torch.ops.torchscience.confluent_hypergeometric_m(a, b, z)
