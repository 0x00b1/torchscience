"""Euler angle conventions.

This module defines the 12 standard Euler angle conventions:
- 6 Tait-Bryan angles (three different axes): XYZ, XZY, YXZ, YZX, ZXY, ZYX
- 6 Proper Euler angles (first and third axes same): XYX, XZX, YXY, YZY, ZXZ, ZYZ
"""

from typing import Tuple

# Tait-Bryan angles (three different axes)
# Also known as Cardan angles or nautical angles
TAIT_BRYAN_CONVENTIONS = frozenset({"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"})

# Proper Euler angles (first and third axes same)
# Also known as classic Euler angles
PROPER_EULER_CONVENTIONS = frozenset(
    {"XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"}
)

# All valid Euler angle conventions
ALL_CONVENTIONS = TAIT_BRYAN_CONVENTIONS | PROPER_EULER_CONVENTIONS


def validate_convention(convention: str) -> None:
    """Validate Euler angle convention string.

    Parameters
    ----------
    convention : str
        Euler angle convention string (e.g., "XYZ", "ZYX", "ZXZ").

    Raises
    ------
    ValueError
        If convention is not one of the 12 valid conventions.

    Examples
    --------
    >>> validate_convention("XYZ")  # No error
    >>> validate_convention("ZYX")  # No error
    >>> validate_convention("ABC")  # Raises ValueError
    """
    if convention not in ALL_CONVENTIONS:
        raise ValueError(
            f"Invalid Euler angle convention '{convention}'. "
            f"Valid: {sorted(ALL_CONVENTIONS)}"
        )


def get_axis_indices(convention: str) -> Tuple[int, int, int]:
    """Get axis indices for convention (0=X, 1=Y, 2=Z).

    Parameters
    ----------
    convention : str
        Euler angle convention string (e.g., "XYZ", "ZYX", "ZXZ").

    Returns
    -------
    Tuple[int, int, int]
        Axis indices (i, j, k) where 0=X, 1=Y, 2=Z.

    Examples
    --------
    >>> get_axis_indices("XYZ")
    (0, 1, 2)
    >>> get_axis_indices("ZYX")
    (2, 1, 0)
    >>> get_axis_indices("ZXZ")
    (2, 0, 2)
    """
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    return (
        axis_map[convention[0]],
        axis_map[convention[1]],
        axis_map[convention[2]],
    )
