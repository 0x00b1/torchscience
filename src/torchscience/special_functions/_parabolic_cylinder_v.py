import torch
from torch import Tensor


def parabolic_cylinder_v(a: Tensor, z: Tensor) -> Tensor:
    r"""
    Parabolic cylinder function V(a, z).

    Computes the parabolic cylinder function V(a, z), the standard
    solution to Weber's differential equation that is dominant
    (grows) as z → +∞.

    Mathematical Definition
    -----------------------
    V(a, z) satisfies Weber's differential equation:

    .. math::

       \frac{d^2 w}{dz^2} - \left(\frac{z^2}{4} + a\right) w = 0

    Parameters
    ----------
    a : Tensor
        Order parameter. Can be any real or complex number.
    z : Tensor
        Argument at which to evaluate the function.

    Returns
    -------
    Tensor
        The parabolic cylinder function V(a, z) evaluated at the input values.

    See Also
    --------
    parabolic_cylinder_u : Recessive solution U(a, z)

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Chapter 12.
           https://dlmf.nist.gov/12
    """
    return torch.ops.torchscience.parabolic_cylinder_v(a, z)
