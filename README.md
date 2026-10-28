# EulerianFSI.jl

A two-dimensional, fully Eulerian solver for incompressible fluid–structure interaction with hyperelastic solids, written in Julia.

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
