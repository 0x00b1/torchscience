import torch
from torch import Tensor


def laguerre_polynomial_l(n: Tensor, alpha: Tensor, z: Tensor) -> Tensor:
    r"""
    Generalized Laguerre polynomial.

    Computes the generalized (associated) Laguerre polynomial :math:`L_n^\alpha(z)`.

    Mathematical Definition
    -----------------------
    The generalized Laguerre polynomial is defined as:

    .. math::

       L_n^\alpha(z) = \frac{\Gamma(n+\alpha+1)}{\Gamma(\alpha+1) \Gamma(n+1)}
                       \, {}_1F_1(-n; \alpha+1; z)

    where :math:`{}_1F_1(a; b; z)` is the confluent hypergeometric function
    (Kummer's function of the first kind).

    Alternatively, using the explicit formula:

    .. math::

       L_n^\alpha(z) = \sum_{k=0}^{n} \binom{n+\alpha}{n-k} \frac{(-z)^k}{k!}

    Special Values
    --------------
    - :math:`L_0^\alpha(z) = 1`
    - :math:`L_1^\alpha(z) = 1 + \alpha - z`
    - :math:`L_2^\alpha(z) = \frac{1}{2}[(1+\alpha)(2+\alpha) - 2(2+\alpha)z + z^2]`
    - :math:`L_n^0(z) = L_n(z)` (ordinary Laguerre polynomial)

    Recurrence Relation
    -------------------
    The generalized Laguerre polynomials satisfy the three-term recurrence:

    .. math::

       (n+1) L_{n+1}^\alpha(z) = (2n + \alpha + 1 - z) L_n^\alpha(z)
                                - (n + \alpha) L_{n-1}^\alpha(z)

    Orthogonality
    -------------
    The polynomials are orthogonal on :math:`[0, \infty)` with weight
    :math:`w(z) = z^\alpha e^{-z}`:

    .. math::

       \int_0^\infty z^\alpha e^{-z} L_m^\alpha(z) L_n^\alpha(z) \, dz
       = \frac{\Gamma(n+\alpha+1)}{n!} \delta_{mn}

    Applications
    ------------
    - **Quantum mechanics**: Radial wave functions of the hydrogen atom are
      expressed in terms of generalized Laguerre polynomials. The radial
      part of the hydrogen wave function is:

      .. math::

         R_{nl}(r) \propto r^l e^{-r/(na_0)} L_{n-l-1}^{2l+1}(2r/(na_0))

    - **Signal processing**: Used in filter design
    - **Numerical integration**: Gauss-Laguerre quadrature nodes and weights

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Can be any real (or complex) number,
        though the polynomial interpretation holds for non-negative integers.
        Broadcasting with alpha and z is supported.
    alpha : Tensor
        Order (generalization) parameter. Must satisfy :math:`\alpha > -1`
        for the orthogonality relation to hold, though the function is
        defined for general alpha. Broadcasting with n and z is supported.
    z : Tensor
        Input values at which to evaluate the polynomial.
        Broadcasting with n and alpha is supported.

    Returns
    -------
    Tensor
        The generalized Laguerre polynomial :math:`L_n^\alpha(z)` evaluated
        at the input values.

    Examples
    --------
    Basic usage with small degrees:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> alpha = torch.tensor([0.0])
    >>> z = torch.tensor([1.0])
    >>> laguerre_polynomial_l(n, alpha, z)
    tensor([ 1.0000,  0.0000, -0.5000])

    Generalized Laguerre polynomial with alpha=1:

    >>> n = torch.tensor([2.0])
    >>> alpha = torch.tensor([1.0])
    >>> z = torch.tensor([0.0, 1.0, 2.0])
    >>> laguerre_polynomial_l(n, alpha, z)
    tensor([3.0000, 1.5000, 0.5000])

    Special case L_0^alpha(z) = 1:

    >>> n = torch.tensor([0.0])
    >>> alpha = torch.tensor([2.5])
    >>> z = torch.tensor([0.0, 1.0, 5.0])
    >>> laguerre_polynomial_l(n, alpha, z)
    tensor([1., 1., 1.])

    .. warning:: Reduced precision for large n

       For large polynomial degrees (n > 20), the hypergeometric series
       may converge slowly or have reduced accuracy. For small integer n,
       the recurrence relation is used for better numerical stability.

    .. warning:: Second-order gradients use finite differences

       The gradients with respect to n and alpha, and all second-order
       gradients, are computed using finite differences and may have
       reduced accuracy compared to analytical gradients.

    Notes
    -----
    - The implementation uses recurrence relations for small non-negative
      integer n (n <= 20) and the hypergeometric representation otherwise.
    - The ordinary Laguerre polynomials are obtained by setting alpha = 0.
    - The associated Laguerre polynomials used in quantum mechanics
      sometimes use a different normalization convention.

    See Also
    --------
    confluent_hypergeometric_m : Kummer's function 1F1 (used internally)
    legendre_polynomial_p : Legendre polynomial of the first kind
    hermite_polynomial_h : Hermite polynomial (physicists' convention)
    """
    return torch.ops.torchscience.laguerre_polynomial_l(n, alpha, z)
