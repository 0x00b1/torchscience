import importlib.util
import sys
import types
from pathlib import Path

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

_core = _load_module(
    "torchscience.special_functions._hypergeometric_core",
    SP_DIR / "_hypergeometric_core.py",
)
_m = _load_module(
    "torchscience.special_functions._hypergeometric_1_f_1",
    SP_DIR / "_hypergeometric_1_f_1.py",
)
_u = _load_module(
    "torchscience.special_functions._hypergeometric_u",
    SP_DIR / "_hypergeometric_u.py",
)

_pfq_series = _core._pfq_series
hypergeometric_1_f_1 = _m.hypergeometric_1_f_1
hypergeometric_u = _u.hypergeometric_u


def test_1f1_wrapper_matches_core_small_z():
    dtype = torch.float64
    a = torch.tensor(0.75, dtype=dtype)
    b = torch.tensor(2.25, dtype=dtype)
    z = torch.linspace(-1e-3, 1e-3, steps=7, dtype=dtype)
    out = hypergeometric_1_f_1(a, b, z, tol=1e-16, max_terms=3000)
    ref = _pfq_series(a_params=a.expand(z.shape).unsqueeze(-1), b_params=b.expand(z.shape).unsqueeze(-1), z=z, tol=1e-16, max_terms=3000)
    assert torch.allclose(out, ref, rtol=1e-14, atol=1e-14)


def test_1f1_terminating_series():
    dtype = torch.float64
    z = torch.linspace(-2.0, 2.0, steps=11, dtype=dtype)
    a = torch.tensor(-3.0, dtype=dtype)  # terminates at k=3
    b = torch.tensor(2.5, dtype=dtype)
    # M(-3,b,z) = 1 + (-3/b)z + [(-3)(-2)/(b(b+1))] z^2/2! + [(-3)(-2)(-1)/(b(b+1)(b+2))] z^3/3!
    expected = (
        1.0
        + (-3.0 / b) * z
        + ((6.0) / (b * (b + 1.0))) * (z * z) / 2.0
        + ((-6.0) / (b * (b + 1.0) * (b + 2.0))) * (z * z * z) / 6.0
    )
    out = hypergeometric_1_f_1(a, b, z, tol=1e-16, max_terms=100)
    assert torch.allclose(out, expected, rtol=1e-14, atol=1e-14)


def test_u_satisfies_kummer_ode_residual_small():
    # z y'' + (b - z) y' - a y = 0
    dtype = torch.float64
    a = torch.tensor(1.3, dtype=dtype)
    b = torch.tensor(2.7, dtype=dtype)
    z = torch.linspace(0.2, 1.0, steps=9, dtype=dtype)
    y = hypergeometric_u(a, b, z, tol=1e-16, max_terms=4000)

    # Finite differences for derivatives
    h = 1e-4
    zp = z + h
    zm = z - h
    yp = hypergeometric_u(a, b, zp, tol=1e-16, max_terms=4000)
    ym = hypergeometric_u(a, b, zm, tol=1e-16, max_terms=4000)
    y1 = (yp - ym) / (2 * h)
    y2 = (yp - 2 * y + ym) / (h * h)

    residual = z * y2 + (b - z) * y1 - a * y
    # Relative tolerance based on scale of y; allow small absolute slack
    scale = torch.clamp(y.abs(), min=1e-12)
    rel = (residual.abs() / scale).max().item()
    assert rel < 1e-5
