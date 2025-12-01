"""Test mixins for PyTorch operators."""

from ._autocast_mixin import AutocastMixin
from ._autograd_mixin import AutogradMixin
from ._broadcasting_mixin import BroadcastingMixin
from ._device_mixin import DeviceMixin
from ._dtype_mixin import DtypeMixin
from ._identity_mixin import IdentityMixin
from ._low_precision_mixin import LowPrecisionMixin
from ._meta_tensor_mixin import MetaTensorMixin
from ._nan_inf_mixin import NanInfMixin
from ._quantized_mixin import QuantizedMixin
from ._recurrence_mixin import RecurrenceMixin
from ._singularity_mixin import SingularityMixin
from ._sparse_mixin import SparseMixin
from ._special_value_mixin import SpecialValueMixin
from ._sympy_reference_mixin import SymPyReferenceMixin
from ._torch_compile_mixin import TorchCompileMixin
from ._vmap_mixin import VmapMixin

__all__ = [
    "AutocastMixin",
    "AutogradMixin",
    "BroadcastingMixin",
    "DeviceMixin",
    "DtypeMixin",
    "IdentityMixin",
    "LowPrecisionMixin",
    "MetaTensorMixin",
    "NanInfMixin",
    "QuantizedMixin",
    "RecurrenceMixin",
    "SingularityMixin",
    "SparseMixin",
    "SpecialValueMixin",
    "SymPyReferenceMixin",
    "TorchCompileMixin",
    "VmapMixin",
]
