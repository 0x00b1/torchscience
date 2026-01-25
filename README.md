# torchscience

A PyTorch standard library for scientists, mathematicians, and engineers.

There should be one—and preferably only one—obvious way to do something. torchscience aims to be that obvious way for core scientific and mathematical operators in PyTorch.

## What Is torchscience?

torchscience is a collection of carefully implemented operators that:

- Are broadly useful across scientific and engineering domains.
- Bridge PyTorch to downstream areas like vision, NLP, and recommender systems by providing reliable, reusable building blocks.
- Tackle challenging algorithms where “rolling your own” often leads to subtle bugs and inconsistent behavior.

Rather than introducing domain‑specific data structures, torchscience favors general, mathematical ones and leans on PyTorch tensors, broadcasting, and batching. Public data structures are kept minimal and reserved for mathematical concepts (for example, `Polynomial`, `Quaternion`).

## Philosophy

- One obvious way: Prefer a single, well‑documented operator over many competing variants.
- Cross‑domain first: Focus on primitives that are useful in multiple fields.
- Reduce reinvention: Implement tricky operators once, test them well, and make them easy to adopt.
- Minimal surface area: Avoid specialized container types unless they encode core mathematics.

## Design and Compatibility

torchscience is orthogonal to PyTorch’s deep‑learning layers and optimizers—it does not ship models or `nn.Module` building blocks. Instead, it provides tensor operators that:

- Follow PyTorch’s broadcasting and batching semantics.
- Are differentiable whenever possible, with analytic backward and double‑backward where relevant.
- Integrate smoothly with `torch.compile` and `torch.func`.

Where feasible, operators are implemented from scratch for tight integration with PyTorch mechanics. Implementations are guided by standard references and informed by testing strategies seen in the ecosystem.

### Runtime Support Goals

Except in rare cases, operators target:

- Autograd and autocast
- CPU, CUDA, and Meta devices
- Where relevant: sparse CSR/COO, quantized CPU/CUDA

Many operators are implemented in C++ for performance; Python is used when algorithmic flexibility (e.g., Python callables in root‑finding) is crucial.

## Scope Overview

The library spans common scientific and engineering areas. Below is a non‑exhaustive map of packages and subpackages:

- combinatorics: combinatorial utilities
- cryptography: core cryptographic primitives
- game: game theory (sequential, simultaneous)
- geometry: computational geometry (e.g., intersection, meshing, transform)
- graph: graph and network theory
- graphics: color, lighting, shading, projection, texture mapping, tone mapping
- information: coding theory, compression
- ordinary_differential_equation: boundary-value problems, initial-value problems
- linear_algebra: matrix ops and decompositions
- morphology: mathematical morphology
- pad: padding utilities
- optimization: convex and nonconvex (LP, QP, SOCP, SDP, integer, global/local search, combinatorial, curve fitting, test functions)
- polynomial: algebra over polynomials; related structures
- privacy: privacy-preserving utilities
- probability: distributions and stochastic processes
- root_finding: scalar and multivariate methods
- signal_processing: filters, filter analysis, filter design, noise, spectra,
  waveforms, window functions, transform (Fourier, Laplace, Hilbert, Mellin,
  Hankel, Radon, convolution), integral_transform
- distance and similarity: distance metrics and similarity measures
- space_partitioning: spatial indexing and partitioning
- special_functions: special mathematical functions
- spline: B‑splines, cubic splines, Bézier, Catmull–Rom, Hermite spline, PCHIP,
  radial basis functions, smoothing splines, tensor-product splines
- statistics: descriptive stats, regression, and hypothesis testing
- wavelet: wavelet transforms and utilities
- window_function: standard analysis windows

Additional topics under exploration include numerical methods (FDM/FEM/DEM), PDEs (acoustics, EM, fluids, heat/mass transfer, solid mechanics), integration, time series, queueing theory, and computer algebra (e.g., finite fields).

## Status

This project is an active work‑in‑progress. Feedback, issue reports, and proposals are welcome.

## Installation

```bash
git clone https://github.com/torchscience/torchscience.git
cd torchscience
pip install -e ".[dev]"
```
