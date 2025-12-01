import torch
from torch import Tensor


def gamma(z: Tensor) -> Tensor:
    r"""
    Gamma function.

    Computes the gamma function evaluated at each element of the input tensor.

    Mathematical Definition
    -----------------------
    For real z > 0, the gamma function is defined as the integral:

    .. math::

       \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt

    For other values (except non-positive integers), the function is extended
    via analytic continuation using the reflection formula:

    .. math::

       \Gamma(z) \Gamma(1-z) = \frac{\pi}{\sin(\pi z)}

    Special Values
    --------------
    - Gamma(n) = (n-1)! for positive integers n
    - Gamma(1) = 1
    - Gamma(1/2) = sqrt(pi)
    - Gamma(z) has poles (returns inf or nan) at z = 0, -1, -2, -3, ...

    Domain
    ------
    - z: any real or complex value except non-positive integers
    - Poles at z = 0, -1, -2, -3, ... where the function returns inf
    - For complex z, poles occur only when Im(z) = 0 and Re(z) is a
      non-positive integer

    Algorithm
    ---------
    - Uses the Lanczos approximation (g=7, n=9) for all types
    - Provides consistent results across CPU and CUDA devices
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The gamma function appears in many mathematical and scientific contexts:
    - Factorials and combinatorics: Gamma(n) = (n-1)! extends factorials to non-integers
    - Probability distributions: normalization constants for gamma, beta, Student's t, F, and chi-squared distributions
    - Bayesian inference: prior and posterior distributions
    - Physics: solutions to differential equations, quantum mechanics, statistical mechanics
    - Machine learning: regularization terms, loss functions, variational inference

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the digamma function:

    .. math::

       \frac{d}{dz} \Gamma(z) = \Gamma(z) \psi(z)

    where :math:`\psi(z) = \frac{d}{dz} \ln \Gamma(z)` is the digamma function
    (logarithmic derivative of the gamma function).

    Second-order derivatives (gradgradcheck) are also supported, computed
    using the trigamma function:

    .. math::

       \frac{d^2}{dz^2} \Gamma(z) = \Gamma(z) \left[ \psi(z)^2 + \psi'(z) \right]

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The gamma function evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Factorial via gamma function (Gamma(n) = (n-1)!):

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> gamma(z)
    tensor([ 1.,  1.,  2.,  6., 24.])

    Half-integer arguments:

    >>> z = torch.tensor([0.5, 1.5, 2.5])
    >>> gamma(z)
    tensor([1.7725, 0.8862, 1.3293])  # sqrt(pi), sqrt(pi)/2, 3*sqrt(pi)/4

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j])
    >>> gamma(z)
    tensor([0.4980-0.1549j, 0.8182-0.7633j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = gamma(z)
    >>> y.backward()
    >>> z.grad  # Gamma(2) * digamma(2) = 1 * (1 - gamma_euler) approx 0.4228
    tensor([0.4228])

    .. warning:: Overflow for large arguments

       The gamma function grows extremely fast - faster than exponential.
       For reference:

       - Gamma(20) ≈ 1.2e17
       - Gamma(100) ≈ 9.3e155
       - Gamma(171) > 1.7e308 (overflows float64)
       - Gamma(35) > 6.5e37 (overflows float32)

       For arguments where overflow is a concern, use ``torch.special.gammaln``
       (log-gamma) instead and exponentiate only when needed:

       >>> # Instead of: result = gamma(large_z)
       >>> # Use: log_result = torch.special.gammaln(large_z)
       >>> # Then: result = torch.exp(log_result)  # only if you need the actual value

       For ratios of gamma functions, use the log-difference:

       >>> # gamma(a) / gamma(b) = exp(gammaln(a) - gammaln(b))

    .. warning:: Poles at non-positive integers

       The function returns inf at z = 0, -1, -2, ... (poles of gamma).
       Gradients at these points return NaN since the derivative (digamma)
       is undefined there.

    Notes
    -----
    - Values very close to poles (within floating-point tolerance) are
      treated as poles and return inf. The detection tolerance scales
      with the magnitude of the value to handle large negative integers
      correctly.
    - For complex inputs, poles occur at z = n + 0j for non-positive
      integers n. Complex values with nonzero imaginary part (even very
      small) are not poles.

    See Also
    --------
    torch.special.gammaln : Natural logarithm of the gamma function (preferred for large arguments)
    torch.special.digamma : Logarithmic derivative of the gamma function
    """
    return torch.ops.torchscience.gamma(z)
