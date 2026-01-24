# Brainstorm: SOTA Improvements for Differential Equation Modules

## Problem Statement

**User's Goal**: Make differential equation solving in torchscience state-of-the-art (SOTA).

**Current State**: `torchscience.ordinary_differential_equation` has solid ODE/BVP solver coverage (41 Python files, 49 test files) with:
- 14 ODE solvers (explicit, implicit, symplectic, adaptive)
- BVP solver with Lobatto IIIA collocation
- Adjoint methods for memory-efficient backpropagation
- Neural ODE optimizations (reversible_heun, asynchronous_leapfrog)
- TensorDict support, event handling, batched solving

**Related Modules**:
- `torchscience.quadrature` - Already exists with adaptive/fixed quadrature methods
- `torchscience.differentiation` - Spatial operators including PDE building blocks (`diffuse`, `advect`, `wave_operator`)

---

## Refined Module Structure

Based on project patterns (`differentiation` and `quadrature` use flat structures with ~50 and ~8 files respectively), the refactoring will split `integration` by mathematical framework:

```
torchscience.ordinary_differential_equation/     # Flat, ~45 files
    # IVP solvers: euler, midpoint, runge_kutta_4, dormand_prince_5,
    #              dop853, adams, backward_euler, bdf, radau,
    #              implicit_midpoint, stormer_verlet, yoshida4,
    #              reversible_heun, asynchronous_leapfrog
    # BVP solvers: solve_bvp, collocation, mesh refinement
    # DAE support: extends implicit solvers (bdf, radau)
    # DDE support: method_of_steps wrapping IVP solvers
    # Infrastructure: newton, interpolation, adjoint, sensitivity
    # Unified API: solve_ivp, solve_bvp, recommend_solver

torchscience.stochastic_differential_equation/   # Flat, ~15 files (new)
    # SDE solvers: euler_maruyama, milstein, sriw1, stratonovich_heun
    # Noise types: diagonal, general, additive, multiplicative
    # SDE adjoints: pathwise, reparameterization gradient
    # Unified API: solve_sde

torchscience.controlled_differential_equation/   # Flat, ~10 files (new)
    # CDE solvers: solve_cde
    # Path interpolation: linear, cubic, rectilinear
    # Rough paths: log_signature, signature
    # Neural CDE/RDE utilities
    # Unified API: solve_cde
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **BVP stays with ODE** | Same underlying math, different boundary conditions |
| **DAE stays with ODE** | Extends implicit solvers (BDF, Radau) |
| **DDE stays with ODE** | Method of steps wraps existing IVP solvers |
| **SDE is separate** | Fundamentally different: Itô calculus, noise terms |
| **CDE is separate** | Different theory: rough paths, signatures |
| **Flat structure** | Matches `differentiation` pattern (~50 files, no subdirs) |
| **PDE spatial ops stay in `differentiation`** | They compute RHS; time-stepping uses ODE solvers |

### Migration Path

```python
# Old (deprecated)
from torchscience.ordinary_differential_equation import solve_ivp, dormand_prince_5

# New
from torchscience.ordinary_differential_equation import solve_ivp, dormand_prince_5
```

The `integration` module will re-export from new modules during deprecation period.

---

## Gap Analysis vs. SOTA Libraries

| Feature | torchscience | diffrax (JAX) | DifferentialEquations.jl | torchdiffeq |
|---------|-------------|---------------|-------------------------|-------------|
| ODEs | 14 solvers | ~20+ solvers | 100+ solvers | 6 solvers |
| SDEs | **None** | Yes | Yes | No |
| DDEs | **None** | Yes | Yes | No |
| DAEs | **None** | No | Yes | No |
| CDEs | **None** | No | No | torchcde |
| Quadrature | `torchscience.quadrature` | No | Yes | No |
| PDE spatial ops | `torchscience.differentiation` | No | Yes (limited) | No |
| Parallel batch | Partial | Yes | Yes | No |
| torch.compile | Partial | N/A (JAX) | N/A | No |
| CUDA kernels | None | N/A (JAX) | N/A | No |

---

## New Equation Types to Add

### 1. Stochastic Differential Equations (SDEs)

**Module**: `torchscience.stochastic_differential_equation`

**Why SOTA needs this**: SDEs are fundamental for:
- Finance: Black-Scholes, interest rate models
- Physics: Langevin dynamics, fluctuating hydrodynamics
- Generative AI: Diffusion models, score-based generative models

**Specific additions**:
- `euler_maruyama` - Basic SDE solver (Itô interpretation, strong order 0.5)
- `milstein` - Higher-order SDE solver (strong order 1.0)
- `sriw1` - Adaptive SDE solver from DifferentialEquations.jl
- `stratonovich_heun` - Stratonovich interpretation (preserves geometric structure)
- `srk2` - Stochastic Runge-Kutta (strong order 1.5 for additive noise)
- Noise types: diagonal, scalar, general, commutative
- Adjoint sensitivity: pathwise gradients, reparameterization trick

**Differentiator**: Native torch.compile support for SDE solvers would be unique in PyTorch ecosystem.

---

### 2. Delay Differential Equations (DDEs)

**Module**: `torchscience.ordinary_differential_equation` (extends ODE infrastructure)

**Why SOTA needs this**: DDEs model systems with memory/feedback delays:
- Epidemiology: SEIR with incubation periods
- Control systems: feedback with communication delays
- Biology: gene regulatory networks, population dynamics

**Specific additions**:
- `solve_dde` - Unified DDE API
- `method_of_steps` - Classic DDE approach wrapping existing ODE solvers
- History function interpolation (hermite, linear)
- Constant delays, time-dependent delays, state-dependent delays
- Neutral DDEs (derivatives of delayed terms appear)

**Challenge**: Adjoint methods for DDEs are complex; discrete sensitivity analysis initially.

---

### 3. Differential-Algebraic Equations (DAEs)

**Module**: `torchscience.ordinary_differential_equation` (extends implicit solvers)

**Why SOTA needs this**: DAEs arise naturally in:
- Constrained mechanical systems (robotics, multibody dynamics)
- Circuit simulation (Kirchhoff's laws as constraints)
- Chemical process modeling (equilibrium constraints)

**Specific additions**:
- `solve_dae` - Unified DAE API
- Index-1 DAE solver (extend `bdf`, `radau`)
- Fully implicit form: F(t, y, y') = 0
- Consistent initialization detection and correction
- Index reduction utilities (Pantelides algorithm)

**Current foundation**: `bdf` and `radau` already handle stiff problems; DAE is natural extension.

---

### 4. Controlled Differential Equations (CDEs)

**Module**: `torchscience.controlled_differential_equation`

**Why SOTA needs this**: CDEs/Neural CDEs are SOTA for:
- Irregular time series (medical records, financial data)
- Long sequences via rough path theory
- Missing data handling without imputation

**Specific additions**:
- `solve_cde` - CDE solver with path interpolation
- Path construction: `linear_interpolation`, `cubic_interpolation`, `rectilinear_interpolation`
- `log_signature` - Log-signature computation for rough paths
- `signature` - Signature computation
- Neural RDE support (log-ODE method for efficiency)
- Adjoint backpropagation through CDEs

**Reference**: [torchcde](https://github.com/patrick-kidger/torchcde) by Patrick Kidger informs design.

---

## Enhancements to Existing ODE Infrastructure

### 5. Native CUDA/C++ Kernels for Hot Paths

**Why SOTA needs this**: Current solvers are pure Python. Key bottlenecks:
- Jacobian computation for implicit solvers
- RK stage evaluations in inner loops
- Newton iteration convergence checks

**Specific additions**:
- C++ kernels for `dormand_prince_5`, `runge_kutta_4` step functions
- CUDA fused kernels for batched RK stages
- Sparse Jacobian assembly on GPU
- Compile with `torch.compile` for JIT optimization

**Impact**: 2-10x speedup for large systems.

---

### 6. Enhanced Parallel/Batched Solving

**Why SOTA needs this**: `solve_ivp_batched` only supports synchronized stepping. [torchode](https://arxiv.org/pdf/2210.12375) shows 4x speedup with async batching.

**Specific additions**:
- Asynchronous batched stepping (each trajectory adapts independently)
- Dynamic load balancing for variable-length trajectories
- Multi-GPU distribution for ensemble simulations
- Early termination for converged trajectories

---

### 7. Improved Adjoint Methods

**Why SOTA needs this**: Current adjoint has code duplication (noted TODO in `_ivp_adjoint.py:130`).

**Specific additions**:
- Unified adjoint infrastructure (refactor shared code into base class)
- Interpolation-free adjoint (discrete adjoints avoid interpolation errors)
- Implicit function theorem-based differentiation for implicit solvers
- Piggyback differentiation for steady-state problems
- Higher-order sensitivities (Hessians via forward-over-reverse)

**Reference**: [ImplicitAD.jl](https://github.com/byuflowlab/ImplicitAD.jl) patterns.

---

### 8. Physics-Informed Constraints

**Why SOTA needs this**: Many physical systems have known invariants.

**Specific additions**:
- Energy-preserving integrators (discrete gradient methods)
- Projection methods for constraint manifolds
- Lie group integrators (SO(3), SE(3) for rigid body dynamics)
- Symplectic integrators for non-separable Hamiltonians (implicit midpoint extension)

---

### 9. Enhanced Neural ODE Features

**Why SOTA needs this**: Neural ODEs are a key use case.

**Specific additions**:
- Seminorm-based error control (for augmented state)
- Regularization: Jacobian norm, kinetic energy penalties
- CNF helpers already exist (`_cnf.py`); enhance with exact trace option
- Latent ODE/SDE utilities for variational inference

---

## PDE Workflow (No Changes Needed)

The current architecture correctly separates concerns:

```python
# Spatial discretization (torchscience.differentiation)
from torchscience.differentiation import diffuse, advect, laplacian

# Time stepping (torchscience.ordinary_differential_equation)
from torchscience.ordinary_differential_equation import solve_ivp

# Heat equation example
def heat_rhs(t, u):
    return diffuse(u, diffusivity=alpha, dx=0.01)

solution = solve_ivp(heat_rhs, y0=initial, t_span=(0, 1), method="bdf")
```

**Existing PDE spatial operators in `differentiation`**:
- `diffuse` - Diffusion term: div(D grad f)
- `advect` - Advection term: (v · grad)f
- `wave_operator` - Wave equation spatial part: c² laplacian(f)
- `material_derivative` - Lagrangian derivative: Df/Dt

These stay in `differentiation` as they compose with core spatial operators.

---

## Priority Ranking (Impact vs. Effort)

| Priority | Feature | Impact | Effort | Rationale |
|----------|---------|--------|--------|-----------|
| 1 | **SDEs** | High | High | Essential for diffusion models, unique in PyTorch |
| 2 | **Module refactor** | Medium | Medium | Clean architecture enables future growth |
| 3 | **DDEs** | Medium | Medium | Unique in PyTorch ecosystem, reuses ODE infra |
| 4 | **DAEs** | Medium | Medium | Natural extension of implicit solvers |
| 5 | **CDEs** | Medium | Medium | SOTA for irregular time series |
| 6 | **CUDA kernels** | High | High | Major performance differentiator |
| 7 | **Enhanced batching** | Medium | Medium | torchode proves value |
| 8 | **Adjoint refactor** | Medium | Low | Cleanup improves maintainability |
| 9 | **Physics constraints** | Low | High | Niche but valuable |

---

## Implementation Phases

### Phase 1: Module Restructure
- Create `torchscience.ordinary_differential_equation` with current ODE/BVP code
- Move code from old `integration` module (already completed)
- Update imports across codebase and tests

### Phase 2: SDE Module
- Create `torchscience.stochastic_differential_equation`
- Implement `euler_maruyama`, `milstein`
- Add noise abstractions and SDE adjoint

### Phase 3: DDE/DAE Extensions
- Add `solve_dde` with method of steps to ODE module
- Add `solve_dae` extending BDF/Radau

### Phase 4: CDE Module
- Create `torchscience.controlled_differential_equation`
- Implement path interpolation and `solve_cde`
- Add signature/log-signature computation

### Phase 5: Performance
- CUDA kernels for core solvers
- Async batched solving
- torch.compile optimization

---

## References

- [diffrax](https://docs.kidger.site/diffrax/) - JAX differential equation solvers
- [DifferentialEquations.jl](https://diffeq.sciml.ai/) - Julia's comprehensive DE suite
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) - Original PyTorch neural ODE library
- [torchcde](https://github.com/patrick-kidger/torchcde) - Controlled differential equations
- [torchode](https://arxiv.org/pdf/2210.12375) - Parallel ODE solving paper
- [Log-NCDEs](https://arxiv.org/abs/2402.18512) - SOTA for long time series
- [ImplicitAD.jl](https://github.com/byuflowlab/ImplicitAD.jl) - Implicit differentiation patterns

---

*Updated after brainstorming discussion. Ready for implementation planning.*
