import importlib.util
import sys
import types
from pathlib import Path
import math

import torch


REPO_SRC = Path(__file__).resolve().parents[3] / "src"
PKG_DIR = REPO_SRC / "torchscience"
SP_DIR = PKG_DIR / "special_functions"


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


# Create minimal package entries to satisfy relative imports
sys.modules.setdefault("torchscience", types.ModuleType("torchscience"))
sys.modules["torchscience"].__path__ = [str(PKG_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault(
    "torchscience.special_functions", types.ModuleType("torchscience.special_functions")
)
sys.modules["torchscience.special_functions"].__path__ = [  # type: ignore[attr-defined]
    str(SP_DIR)
]

_airy_ai = _load_module(
    "torchscience.special_functions._airy_ai",
    SP_DIR / "_airy_ai.py",
)
_airy_bi = _load_module(
    "torchscience.special_functions._airy_bi",
    SP_DIR / "_airy_bi.py",
)
_airy_ai_prime = _load_module(
    "torchscience.special_functions._airy_ai_prime",
    SP_DIR / "_airy_ai_prime.py",
)
_airy_bi_prime = _load_module(
    "torchscience.special_functions._airy_bi_prime",
    SP_DIR / "_airy_bi_prime.py",
)

airy_ai = _airy_ai.airy_ai
airy_bi = _airy_bi.airy_bi
airy_ai_prime = _airy_ai_prime.airy_ai_prime
airy_bi_prime = _airy_bi_prime.airy_bi_prime


def _constants():
    ln3 = math.log(3.0)
    lg23 = math.lgamma(2.0 / 3.0)
    lg13 = math.lgamma(1.0 / 3.0)
    c_ai = math.exp(-(2.0 / 3.0) * ln3 - lg23)
    c_aip = -math.exp(-(1.0 / 3.0) * ln3 - lg13)
    c_bi = math.exp(-(1.0 / 6.0) * ln3 - lg23)
    # Bi'(0) = 3^(1/6) / Γ(1/3)
    c_bip = math.exp((1.0 / 6.0) * ln3 - lg13)
    return c_ai, c_aip, c_bi, c_bip


def test_airy_values_at_zero():
    dtype = torch.float64
    device = torch.device("cpu")
    x0 = torch.tensor(0.0, dtype=dtype, device=device)
    ai0 = float(airy_ai(x0).item())
    bi0 = float(airy_bi(x0).item())
    aip0 = float(airy_ai_prime(x0).item())
    bip0 = float(airy_bi_prime(x0).item())

    c_ai, c_aip, c_bi, c_bip = _constants()

    assert abs(ai0 - c_ai) < 1e-13
    assert abs(bi0 - c_bi) < 1e-13
    assert abs(aip0 - c_aip) < 1e-13
    assert abs(bip0 - c_bip) < 1e-13


def test_airy_wronskian_identity_near_zero():
    # Ai*Bi' - Ai'*Bi = 1/pi
    dtype = torch.float64
    device = torch.device("cpu")
    x = torch.linspace(-0.5, 0.5, steps=7, dtype=dtype, device=device)
    ai = airy_ai(x)
    bi = airy_bi(x)
    aip = airy_ai_prime(x)
    bip = airy_bi_prime(x)
    wr = ai * bip - aip * bi
    target = torch.full_like(wr, 1.0 / math.pi)
    assert torch.allclose(wr, target, rtol=1e-10, atol=1e-12)

