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

_hcore = _load_module(
    "torchscience.special_functions._hypergeometric_core",
    SP_DIR / "_hypergeometric_core.py",
)
sys.modules["torchscience.special_functions._hypergeometric_core"] = _hcore

_f0 = _load_module(
    "torchscience.special_functions._hypergeometric_0_f_1",
    SP_DIR / "_hypergeometric_0_f_1.py",
)

_pfq_series = _hcore._pfq_series
hypergeometric_0_f_1 = _f0.hypergeometric_0_f_1


def test_0f1_matches_direct_series_small_z():
    # 0F1(;b;z) = sum_{k>=0} z^k / ((b)_k k!)
    dtype = torch.float64
    device = torch.device("cpu")
    z = torch.tensor([0.0, 1e-8, -1e-8, 1e-5, -1e-5], dtype=dtype, device=device)
    b = torch.tensor(1.75, dtype=dtype, device=device)

    def direct_0f1(b_val: float, z_tensor: torch.Tensor, terms: int = 60) -> torch.Tensor:
        out = torch.zeros_like(z_tensor)
        for idx in range(z_tensor.numel()):
            z_i = z_tensor.flatten()[idx]
            s = torch.tensor(0.0, dtype=dtype)
            poch = torch.tensor(1.0, dtype=dtype)
            fact = torch.tensor(1.0, dtype=dtype)
            zpow = torch.tensor(1.0, dtype=dtype)
            for k in range(terms):
                if k > 0:
                    fact = fact * k
                    poch = poch * (b_val + (k - 1))
                    zpow = zpow * z_i
                s = s + zpow / (fact * poch)
            out.view(-1)[idx] = s
        return out

    val_fn = hypergeometric_0_f_1(b, z, tol=1e-18, max_terms=3000)
    val_direct = direct_0f1(float(b.item()), z, terms=60)

    assert torch.allclose(val_fn, val_direct, rtol=1e-12, atol=1e-14)


def test_0f1_wrapper_matches_core():
    dtype = torch.float64
    device = torch.device("cpu")
    z = torch.linspace(-1e-1, 1e-1, steps=7, dtype=dtype, device=device)
    b = torch.tensor(2.25, dtype=dtype, device=device)

    val_fn = hypergeometric_0_f_1(b, z, tol=1e-16, max_terms=2000)
    val_core = _pfq_series(a_params=None, b_params=(b,), z=z, tol=1e-16, max_terms=2000)

    assert torch.allclose(val_fn, val_core, rtol=1e-14, atol=1e-14)


def test_0f1_broadcast_and_dtype():
    z = torch.linspace(0, 1e-4, steps=5, dtype=torch.float32)
    b = torch.tensor([1.5, 2.5], dtype=torch.float32)
    # Broadcast: b (2,), z (5,) -> (2,5)
    out = hypergeometric_0_f_1(b[:, None], z[None, :], tol=1e-7, max_terms=200)
    assert out.shape == (2, 5)
    assert out.dtype == torch.float32
