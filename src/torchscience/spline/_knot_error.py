from ._spline_error import SplineError


class KnotError(SplineError):
    """Raised for invalid knot vectors (non-monotonic, insufficient knots)."""

    pass
