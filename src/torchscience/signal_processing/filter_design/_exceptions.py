"""Exceptions for filter design module."""


class FilterDesignError(Exception):
    """Base exception for filter design errors."""

    pass


class InvalidOrderError(FilterDesignError):
    """Raised when filter order is invalid.

    This occurs when:
    - Order is not a positive integer
    - Order exceeds implementation limits
    - Order is incompatible with requested filter type
    """

    pass


class InvalidCutoffError(FilterDesignError):
    """Raised when cutoff frequency is invalid.

    This occurs when:
    - Cutoff is outside valid range [0, Nyquist]
    - For bandpass/bandstop, low >= high frequency
    """

    pass


class InvalidNumTapsError(FilterDesignError):
    """Raised when FIR filter tap count is invalid for the requested filter type.

    This occurs when:
    - num_taps is not positive
    - Even num_taps used with highpass/bandstop (Type II filter constraint)
    - Odd num_taps required for certain filter types
    """

    pass


class FrequencyOrderError(FilterDesignError):
    """Raised when frequency band edges are not in ascending order.

    This occurs when frequency arrays for multi-band filters are not
    monotonically increasing.
    """

    pass


class NyquistViolationError(FilterDesignError):
    """Raised when frequency exceeds the Nyquist frequency.

    This occurs when:
    - Cutoff >= sampling_frequency / 2
    - Band edges reach or exceed Nyquist
    """

    pass


class SpecificationError(FilterDesignError):
    """Raised when filter specifications are contradictory or impossible to meet.

    This occurs when:
    - Passband ripple > stopband attenuation (makes no physical sense)
    - Transition band is too narrow for achievable filter order
    - Specifications cannot be met with any finite order
    """

    pass


class ConvergenceError(FilterDesignError):
    """Raised when iterative algorithms fail to converge.

    This occurs in:
    - Parks-McClellan (Remez) algorithm exceeding max iterations
    - Elliptic function calculations not converging
    - Optimization-based filter design methods
    """

    pass


class SOSNormalizationError(FilterDesignError):
    """Raised when second-order section normalization fails.

    This occurs when:
    - a0 coefficient is zero (cannot normalize)
    - Numerical issues prevent proper gain distribution
    """

    pass
