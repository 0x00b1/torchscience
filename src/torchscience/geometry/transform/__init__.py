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
from torchscience.geometry.transform._euler_angles import (
    EulerAngles,
    euler_angles,
    euler_angles_to_matrix,
    euler_angles_to_quaternion,
    matrix_to_euler_angles,
    quaternion_to_euler_angles,
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
from torchscience.geometry.transform._rotation_6d import (
    Rotation6D,
    matrix_to_rotation_6d,
    quaternion_to_rotation_6d,
    rotation_6d,
    rotation_6d_to_matrix,
    rotation_6d_to_quaternion,
)
from torchscience.geometry.transform._rotation_matrix import (
    RotationMatrix,
    rotation_matrix,
)
from torchscience.geometry.transform._rotation_vector import (
    RotationVector,
    matrix_to_rotation_vector,
    quaternion_to_rotation_vector,
    rotation_vector,
    rotation_vector_to_matrix,
    rotation_vector_to_quaternion,
)

__all__ = [
    "AxisAngle",
    "EulerAngles",
    "Quaternion",
    "Rotation6D",
    "RotationMatrix",
    "RotationVector",
    "axis_angle",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "euler_angles",
    "euler_angles_to_matrix",
    "euler_angles_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_euler_angles",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "matrix_to_rotation_vector",
    "quaternion",
    "quaternion_apply",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_slerp",
    "quaternion_to_axis_angle",
    "quaternion_to_euler_angles",
    "quaternion_to_matrix",
    "quaternion_to_rotation_6d",
    "quaternion_to_rotation_vector",
    "reflect",
    "refract",
    "rotation_6d",
    "rotation_6d_to_matrix",
    "rotation_6d_to_quaternion",
    "rotation_matrix",
    "rotation_vector",
    "rotation_vector_to_matrix",
    "rotation_vector_to_quaternion",
]
