"""Filter design functions for IIR and FIR filters."""

from ._ba_to_sos import ba_to_sos
from ._ba_to_zpk import ba_to_zpk
from ._bessel_design import bessel_design
from ._bessel_prototype import bessel_prototype
from ._bilinear_transform_ba import bilinear_transform_ba
from ._bilinear_transform_zpk import bilinear_transform_zpk
from ._butterworth_design import butterworth_design
from ._butterworth_minimum_order import butterworth_minimum_order
from ._butterworth_prototype import butterworth_prototype
from ._cascade_sos import cascade_sos
from ._chebyshev_type_1_design import chebyshev_type_1_design
from ._chebyshev_type_1_minimum_order import chebyshev_type_1_minimum_order
from ._chebyshev_type_1_prototype import chebyshev_type_1_prototype
from ._chebyshev_type_2_design import chebyshev_type_2_design
from ._chebyshev_type_2_minimum_order import chebyshev_type_2_minimum_order
from ._chebyshev_type_2_prototype import chebyshev_type_2_prototype
from ._constants import (
    Q_BUTTERWORTH,
    Q_MEDIUM,
    Q_NARROW,
    Q_WIDE,
)
from ._elliptic_design import elliptic_design
from ._elliptic_minimum_order import elliptic_minimum_order
from ._elliptic_prototype import elliptic_prototype
from ._exceptions import (
    ConvergenceError,
    FilterDesignError,
    FrequencyOrderError,
    InvalidCutoffError,
    InvalidNumTapsError,
    InvalidOrderError,
    NyquistViolationError,
    SOSNormalizationError,
    SpecificationError,
)
from ._firwin import firwin
from ._firwin2 import firwin2
from ._iirnotch import iirnotch
from ._iirpeak import iirpeak
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._savgol_coeffs import savgol_coeffs
from ._sos_normalize import sos_normalize
from ._sos_sections_count import sos_sections_count
from ._sos_to_ba import sos_to_ba
from ._sos_to_zpk import sos_to_zpk
from ._zpk_to_ba import zpk_to_ba
from ._zpk_to_sos import zpk_to_sos

__all__ = [
    # Design functions
    "bessel_design",
    "bessel_prototype",
    "butterworth_design",
    "butterworth_minimum_order",
    "butterworth_prototype",
    "chebyshev_type_1_design",
    "chebyshev_type_1_minimum_order",
    "chebyshev_type_1_prototype",
    "chebyshev_type_2_design",
    "chebyshev_type_2_minimum_order",
    "chebyshev_type_2_prototype",
    "elliptic_design",
    "elliptic_minimum_order",
    "elliptic_prototype",
    "firwin",
    "firwin2",
    "iirnotch",
    "iirpeak",
    "savgol_coeffs",
    # Transforms
    "bilinear_transform_ba",
    "bilinear_transform_zpk",
    "lowpass_to_bandpass_zpk",
    "lowpass_to_bandstop_zpk",
    "lowpass_to_highpass_zpk",
    "lowpass_to_lowpass_zpk",
    # Conversions
    "ba_to_sos",
    "ba_to_zpk",
    "sos_to_ba",
    "sos_to_zpk",
    "zpk_to_ba",
    "zpk_to_sos",
    # SOS utilities
    "cascade_sos",
    "sos_normalize",
    "sos_sections_count",
    # Constants
    "Q_BUTTERWORTH",
    "Q_MEDIUM",
    "Q_NARROW",
    "Q_WIDE",
    # Exceptions
    "ConvergenceError",
    "FilterDesignError",
    "FrequencyOrderError",
    "InvalidCutoffError",
    "InvalidNumTapsError",
    "InvalidOrderError",
    "NyquistViolationError",
    "SOSNormalizationError",
    "SpecificationError",
]
