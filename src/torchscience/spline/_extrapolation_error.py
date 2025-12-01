from ._spline_error import SplineError


class ExtrapolationError(SplineError):
    """Raised when query point is outside spline domain with extrapolate='error'."""

    pass
