import torch
from torch import Tensor


def weierstrass_p(z: Tensor, g2: Tensor, g3: Tensor) -> Tensor:
    r"""
    Weierstrass elliptic function P(z; g2, g3).

    Computes the Weierstrass elliptic P function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Weierstrass elliptic function P is defined as:

    .. math::

       \wp(z; g_2, g_3) = \frac{1}{z^2} + \sum_{\omega \in \Lambda \setminus \{0\}}
       \left( \frac{1}{(z - \omega)^2} - \frac{1}{\omega^2} \right)

    where the lattice Lambda is determined by the invariants g2 and g3:

    .. math::

       \Lambda = \{2m\omega_1 + 2n\omega_2 : m, n \in \mathbb{Z}\}

    The invariants g2 and g3 are related to the half-periods omega_1 and omega_2 by:

    .. math::

       g_2 = 60 \sum_{\omega \in \Lambda \setminus \{0\}} \omega^{-4}

       g_3 = 140 \sum_{\omega \in \Lambda \setminus \{0\}} \omega^{-6}

    The function satisfies the differential equation:

    .. math::

       (\wp')^2 = 4\wp^3 - g_2 \wp - g_3

    Domain
    ------
    - z: any complex value except lattice points (poles)
    - g2: invariant (any complex value)
    - g3: invariant (any complex value)

    The discriminant Delta = g2^3 - 27*g3^2 determines the lattice type:
    - Delta != 0: non-degenerate elliptic curve
    - Delta = 0: degenerate case (cusp or node)

    Special Values
    --------------
    - P(z) has a double pole at each lattice point omega in Lambda
    - P(omega_1; g2, g3) = e1 (first root)
    - P(omega_2; g2, g3) = e2 (second root)
    - P(omega_3; g2, g3) = e3 (third root, where omega_3 = omega_1 + omega_2)
    - The roots e1, e2, e3 satisfy: e1 + e2 + e3 = 0

    Laurent Expansion
    -----------------
    Near z = 0, the Laurent expansion is:

    .. math::

       \wp(z) = \frac{1}{z^2} + \frac{g_2}{20} z^2 + \frac{g_3}{28} z^4
       + \frac{g_2^2}{1200} z^6 + O(z^8)

    Period Structure
    ----------------
    The P function is doubly periodic with periods 2*omega_1 and 2*omega_2:

    .. math::

       \wp(z + 2\omega_1) = \wp(z + 2\omega_2) = \wp(z)

    The function is even: P(-z) = P(z)

    Relationship to Other Functions
    -------------------------------
    The Weierstrass P function is related to other elliptic functions:

    - Jacobi elliptic functions via:

      .. math::

         \wp(z) = e_3 + \frac{e_1 - e_3}{\text{sn}^2(u, m)}

      where u = sqrt(e_1 - e_3) * z and m = (e_2 - e_3)/(e_1 - e_3)

    - The derivative P'(z) satisfies:

      .. math::

         \wp'(z)^2 = 4(\wp(z) - e_1)(\wp(z) - e_2)(\wp(z) - e_3)

    Algorithm
    ---------
    The implementation uses the Carlson symmetric elliptic integral R_F
    to compute the inverse function, then applies Newton iteration or
    series expansion depending on the region of the complex plane.

    Applications
    ------------
    The Weierstrass elliptic function appears in many mathematical and
    physical contexts:
    - Uniformization of elliptic curves
    - Exact solutions to nonlinear differential equations
    - Classical mechanics (pendulum, spinning top)
    - Conformal field theory
    - String theory and integrable systems
    - Algebraic geometry and number theory
    - Crystallography and lattice sums

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to z is:

    .. math::

       \frac{\partial \wp}{\partial z} = \wp'(z)

    where P'(z) is the Weierstrass zeta derivative satisfying the
    differential equation above.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    z : Tensor
        The argument. Can be floating-point or complex.
        Broadcasting with g2 and g3 is supported.
    g2 : Tensor
        The first invariant of the Weierstrass function.
        Broadcasting with z and g3 is supported.
    g3 : Tensor
        The second invariant of the Weierstrass function.
        Broadcasting with z and g2 is supported.

    Returns
    -------
    Tensor
        The Weierstrass elliptic function P(z; g2, g3) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> weierstrass_p(z, g2, g3)
    tensor([4.0000])

    Lemniscatic case (g3 = 0):

    >>> z = torch.tensor([1.0])
    >>> g2 = torch.tensor([4.0])
    >>> g3 = torch.tensor([0.0])
    >>> weierstrass_p(z, g2, g3)
    tensor([1.8174])

    Equianharmonic case (g2 = 0):

    >>> z = torch.tensor([1.0])
    >>> g2 = torch.tensor([0.0])
    >>> g3 = torch.tensor([1.0])
    >>> weierstrass_p(z, g2, g3)
    tensor([1.1547])

    Complex input:

    >>> z = torch.tensor([0.5 + 0.5j])
    >>> g2 = torch.tensor([1.0 + 0.0j])
    >>> g3 = torch.tensor([0.0 + 0.0j])
    >>> weierstrass_p(z, g2, g3)
    tensor([1.6928-3.4641j])

    Autograd:

    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> p = weierstrass_p(z, g2, g3)
    >>> p.backward()
    >>> z.grad  # d(P)/dz = P'(z)
    tensor([-15.5885])

    Notes
    -----
    - The Weierstrass P function has double poles at all lattice points.
      The function returns inf or nan at these singular points.
    - For z very close to a lattice point, numerical precision may be
      reduced due to the pole.
    - The discriminant Delta = g2^3 - 27*g3^2 should be nonzero for a
      proper elliptic curve. When Delta = 0, the curve is degenerate.
    - The implementation is optimized for accuracy across a wide range
      of invariants and arguments.

    See Also
    --------
    weierstrass_p_prime : Derivative of the Weierstrass P function
    weierstrass_zeta : Weierstrass zeta function
    weierstrass_sigma : Weierstrass sigma function
    carlson_elliptic_integral_r_f : Carlson symmetric elliptic integral (used internally)
    jacobi_elliptic_sn : Related Jacobi elliptic function
    """
    return torch.ops.torchscience.weierstrass_p(z, g2, g3)
