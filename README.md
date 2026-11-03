# EulerianFSI.jl

A two-dimensional, fully Eulerian solver for fluid–structure interaction with hyperelastic solids, written in Julia. The original incompressible FSI configuration is preserved; optional growth introduces a prescribed nonzero velocity divergence.

The solver uses a staggered MAC grid, a fractional-step IMEX time integrator, and conservative diffuse-interface transport. Three phase-field discretizations are available: **CDI**, **ACDI**, and **UWCDI**.

## Installation

Julia 1.12 is recommended. From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Examples

Lid-driven cavity with a deformable disk:

```bash
julia --project=. scripts/benchmarks/run_sugiyama_lid_driven_cavity.jl --N=128 --T=20 --phase-field=acdi
```

Wall-supported filament under Kolmogorov forcing:

```bash
julia --project=. scripts/benchmarks/run_wall_supported_filament_kolmogorov_flow.jl --N=128 --T=10 --phase-field=uwcdi
```

Select `--phase-field=current_cdi`, `acdi`, or `uwcdi`. Use `--N`, `--T`, and the optional method-specific flags to change the configuration. Results, diagnostics, and figures are saved under `outputs/`.

For refinement studies, use `scripts/benchmarks/analyze_convergence.jl` on a completed cavity study directory.

## Layout

- `src/` — FSI solver and numerical methods
- `scripts/benchmarks/` — example simulations and convergence analysis
- `data/benchmarks/` — reference cavity trajectory
- `test/` — numerical checks

Run the tests with:

```bash
julia --project=. test/runtests.jl
```

The benchmark drivers retain their configured defaults unless overridden from the command line.

## Optional volumetric growth and resource transport (development)

The solver additionally supports a single-velocity growth formulation with
`∇·u = G(ϕ,c)` and a kinematic growth tensor `K_g` satisfying
`tr(K_g) = G`. Available closures include constant and Monod-type volumetric
sources, isotropic or directionally prescribed growth, resource diffusion,
conservative resource advection, and saturating consumption.

All original FTaC benchmark scripts remain unchanged. The legacy six-array
`FSIState(U,V,phi,sigma_xx,sigma_yy,sigma_xy)` constructor automatically
creates a zero resource field; the five-argument `FSIBoundaryConditions`
constructor reuses phase-field boundary conditions for the unused resource.
`FSIModels()` defaults to `NoGrowth()`, `NoDiffusion()`, and
`NoResourceConsumption()`.

New simulations should explicitly construct:

```julia
state = FSIState(U, V, phi, c, sigma_xx, sigma_yy, sigma_xy)
bcs = FSIBoundaryConditions(bc_U, bc_V, bc_correction, bc_phi, bc_c, bc_sigma)
models = FSIModels(
    growth_source=SaturatingResourceGrowth(0.1, 0.5),
    growth_tensor=IsotropicGrowthTensor(),
    diffusivity=ConstantDiffusivity(0.01),
    resource_consumption=SaturatingResourceConsumption(0.05, 0.5),
)
```

Then build an `IMEXContext` and use `make_velocity_compatible!` before the
first `integrate_step!`. The boundary conditions **must** permit the net flux
required by the growth source:

```math
\int_\Omega G\,dV = \int_{\partial\Omega} \mathbf{u}\cdot\mathbf{n}\,dS.
```

Consequently, the original sealed/periodic benchmark domains cannot simply
be run with everywhere-positive growth. Configure an open-flow problem or a
compatible volume source and verify the boundary treatment before use.

**Status:** This integration includes constitutive and discrete-projection
checks, but does not yet establish grid/time convergence or physical validation
of a complete nonzero-growth FSI benchmark. Do not interpret this extension
as a numerically validated research result without such experiments.
See [`GROWTH_INTEGRATION_NOTES.md`](GROWTH_INTEGRATION_NOTES.md).

### Tests

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The GitHub Actions workflow also runs these checks for branches and PRs.
