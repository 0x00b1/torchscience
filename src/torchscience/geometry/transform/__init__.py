"""
Geometric transformation
========================
"""

from torchscience.geometry.transform._axis_angle import (
    AxisAngle,
    axis_angle,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_axis_angle,
)
from torchscience.geometry.transform._quaternion import (
    Quaternion,
    matrix_to_quaternion,
    quaternion,
    quaternion_apply,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_slerp,
    quaternion_to_matrix,
)
from torchscience.geometry.transform._reflect import reflect
from torchscience.geometry.transform._refract import refract
from torchscience.geometry.transform._rotation_matrix import (
    RotationMatrix,
    rotation_matrix,
)

__all__ = [
    "AxisAngle",
    "Quaternion",
    "RotationMatrix",
    "axis_angle",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "quaternion",
    "quaternion_apply",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_slerp",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "reflect",
    "refract",
    "rotation_matrix",
]
