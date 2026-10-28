# Wall-supported hyperelastic filament under Kolmogorov forcing.

using Dates
using LinearAlgebra
using Printf

using EulerianFSI
using EulerianFSI.BoundaryConditions
using EulerianFSI.MeshGeneration
using EulerianFSI.Operators
using EulerianFSI.InterfaceInitialization
using EulerianFSI.StructureTensor
import EulerianFSI.TransportConstitutive: AbstractBodyForce, body_force!

"""Benchmark-local deterministic Kolmogorov forcing."""
struct DeterministicKolmogorovForce <: AbstractBodyForce
    amplitude::Float64
    wavenumber::Float64
    phase::Float64
end

function body_force!(
    Fx,
    Fy,
    model::DeterministicKolmogorovForce,
    t::Real,
    metrics::MeshMetrics,
)
    size(Fx) == (metrics.Nx + 1, metrics.Ny) || throw(DimensionMismatch(
        "Fx has size $(size(Fx)); expected $((metrics.Nx + 1, metrics.Ny))."
    ))
    size(Fy) == (metrics.Nx, metrics.Ny + 1) || throw(DimensionMismatch(
        "Fy has size $(size(Fy)); expected $((metrics.Nx, metrics.Ny + 1))."
    ))
    isfinite(t) || throw(ArgumentError("Body-force evaluation time must be finite."))

    fill!(Fy, 0.0)
    y_center = 0.0
    @inbounds for j in 1:metrics.Ny
        y_center += 0.5 * metrics.dy_c[j]
        fj = model.amplitude * sin(model.wavenumber * y_center + model.phase)
        @views fill!(Fx[:, j], fj)
        y_center += 0.5 * metrics.dy_c[j]
    end
    return Fx, Fy
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
try
    @eval using Plots
catch err
    error(
        "This benchmark generates PNG/PDF figures and requires Plots.jl. " *
        "Install the project environment with: julia --project=. -e 'using Pkg; Pkg.instantiate()'. " *
        "Original error: $(sprint(showerror, err))"
    )
end

gr()

include(joinpath(@__DIR__, "BenchmarkUtilities.jl"))
using .BenchmarkUtilities

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# CONFIGURATION

Base.@kwdef mutable struct FilamentBenchmarkConfig
    # Domain and grid. --N sets Ny=N and Nx=3N so cells remain square.
    Lx::Float64 = 6.0
    Ly::Float64 = 2.0
    Nx::Int = 384
    Ny::Int = 128

    # Wall-supported filament geometry. `filament_height` is the emerged
    # height measured from the top of the support layer to the cap tip.
    support_height::Float64 = 0.20
    center_x::Float64 = 3.0
    filament_height::Float64 = 1.00
    cap_radius::Float64 = 0.10

    # Same constitutive model as the LDC; only parameter values / forcing differ.
    U_ref::Float64 = 1.0
    density::Float64 = 1.0
    viscosity::Float64 = 0.01
    elastic_modulus::Float64 = 0.05

    # Deterministic Kolmogorov forcing and smooth divergence-free seed.
    kolmogorov_waves::Int = 4
    kolmogorov_phase::Float64 = 0.0
    perturbation_amplitude::Float64 = 0.02
    perturbation_mode_x::Int = 1

    # Diffuse-interface settings, matched to the lid-driven-cavity campaign.
    Pe_gamma::Float64 = 0.25
    interface_epsilon::Float64 = NaN
    epsilon_over_h::Float64 = 2.0
    normal_regularization::Float64 = 1e-6

    phase_field::Symbol = :current_cdi
    acdi_epsilon_over_h::Float64 = 0.51
    acdi_gamma::Float64 = 1.0
    acdi_dynamic_gamma::Bool = false
    uwcdi_alpha::Float64 = 1.0
    uwcdi_normal::Symbol = :standard
    uwcdi_init::Symbol = :sharp
    uwcdi_init_epsilon_over_h::Float64 = 1.0

    # Numerical extension of sigma (same as the lid-driven cavity).
    sigma_extension_rate::Float64 = 20.0
    sigma_extension_exponent::Float64 = 4.0
    sigma_safety_cutoff::Float64 = 1e-6
    sigma_core_cutoff::Float64 = 0.90
    sigma_interface_cutoff::Float64 = 0.10

    # Time integration.
    T::Float64 = 20.0
    cfl_hyperbolic::Float64 = 0.10
    cfl_interface_compression::Float64 = 0.25
    cfl_interface_diffusion::Float64 = 0.50
    dt_max::Float64 = 5e-3
    dt_min_physical::Float64 = 1e-8

    # Outputs.
    trajectory_interval::Float64 = 0.02
    diagnostics_interval::Float64 = 0.10
    terminal_update_seconds::Float64 = 5.0
    snapshot_times::Vector{Float64} = [
        0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0,
    ]
    snapshot_speed_scale::Float64 = 1.0

    # Linear solvers.
    velocity_reltol::Float64 = 1e-8
    velocity_maxiter::Int = 800
    velocity_restart::Int = 100
    projection_reltol::Float64 = 1e-11
    projection_maxiter::Int = 3000
    projection_restart::Int = 50
    use_projection_amg::Bool = true

    output_root::String = joinpath(
        ROOT, "outputs", "benchmarks", "wall_supported_filament_kolmogorov_flow"
    )
end

struct FilamentSnapshot
    time::Float64
    phi::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    speed::Matrix{Float64}
    sigma_xx::Matrix{Float64}
    sigma_yy::Matrix{Float64}
    sigma_xy::Matrix{Float64}
    x_centroid::Float64
    y_centroid::Float64
end

@inline fluid_viscosity_coefficient(cfg::FilamentBenchmarkConfig) = cfg.viscosity

@inline elastic_stress_coefficient(cfg::FilamentBenchmarkConfig) = cfg.elastic_modulus

@inline kolmogorov_wavenumber(cfg::FilamentBenchmarkConfig) =
    2.0 * pi * cfg.kolmogorov_waves / cfg.Ly

# For an empty domain, nu*laplacian(u)+f=0 admits
# u(y)=Uref*sin(k_y*y) when A=nu*k_y^2*Uref.
@inline kolmogorov_force_amplitude(cfg::FilamentBenchmarkConfig) =
    cfg.viscosity * kolmogorov_wavenumber(cfg)^2 * cfg.U_ref

@inline filament_width(cfg::FilamentBenchmarkConfig) = 2.0 * cfg.cap_radius

@inline protrusion_geometric_area(cfg::FilamentBenchmarkConfig) =
    2.0 * cfg.cap_radius * (cfg.filament_height - cfg.cap_radius) +
    0.5 * pi * cfg.cap_radius^2

@inline total_structural_geometric_area(cfg::FilamentBenchmarkConfig) =
    cfg.Lx * cfg.support_height + protrusion_geometric_area(cfg)

# Exclude the support layer and the immediate root region from the moment-based
# deformation response. This cutoff is geometric and identical for all
# phase-field methods.
@inline protrusion_response_cutoff_y(cfg::FilamentBenchmarkConfig) =
    cfg.support_height + cfg.cap_radius

# COMMAND-LINE OVERRIDES

function apply_command_line_overrides!(cfg::FilamentBenchmarkConfig, args)
    for arg in args
        if startswith(arg, "--N=")
            N = parse(Int, split(arg, "=", limit=2)[2])
            cfg.Ny = N
            cfg.Nx = 3N
        elseif startswith(arg, "--T=")
            cfg.T = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--Uref=")
            cfg.U_ref = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nu=")
            cfg.viscosity = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--G=")
            cfg.elastic_modulus = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--waves=")
            cfg.kolmogorov_waves = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--kolmogorov-phase=")
            cfg.kolmogorov_phase = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--perturbation=")
            cfg.perturbation_amplitude = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--perturbation-mode-x=")
            cfg.perturbation_mode_x = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--height=")
            cfg.filament_height = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--radius=")
            cfg.cap_radius = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--support-height=")
            cfg.support_height = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--center-x=")
            cfg.center_x = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--phase-field=")
            cfg.phase_field = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--epsilon=")
            cfg.interface_epsilon = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--epsilon-over-h=")
            cfg.epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
            cfg.interface_epsilon = NaN
        elseif startswith(arg, "--acdi-epsilon-over-h=")
            cfg.acdi_epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--acdi-gamma=")
            cfg.acdi_gamma = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--acdi-gamma-mode=")
            mode = lowercase(split(arg, "=", limit=2)[2])
            mode in ("fixed", "dynamic") || throw(ArgumentError(
                "--acdi-gamma-mode must be fixed or dynamic."
            ))
            cfg.acdi_dynamic_gamma = (mode == "dynamic")
        elseif startswith(arg, "--uwcdi-alpha=")
            cfg.uwcdi_alpha = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--uwcdi-normal=")
            cfg.uwcdi_normal = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--uwcdi-init=")
            cfg.uwcdi_init = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--uwcdi-init-epsilon-over-h=")
            cfg.uwcdi_init_epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-root=")
            cfg.output_root = abspath(split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("Unknown command-line option: $arg"))
        end
    end
    return cfg
end

function validate_config(cfg::FilamentBenchmarkConfig)
    cfg.Lx > 0 || throw(ArgumentError("Lx must be positive."))
    cfg.Ly > 0 || throw(ArgumentError("Ly must be positive."))
    cfg.Nx >= 12 || throw(ArgumentError("Nx must be at least 12."))
    cfg.Ny >= 8 || throw(ArgumentError("Ny must be at least 8."))
    isapprox(cfg.Nx / cfg.Ny, cfg.Lx / cfg.Ly; atol=1e-12, rtol=1e-12) || throw(ArgumentError(
        "The benchmark requires square cells: Nx/Ny must equal Lx/Ly."
    ))
    cfg.T > 0 || throw(ArgumentError("T must be positive."))
    cfg.U_ref > 0 || throw(ArgumentError("U_ref must be positive."))
    cfg.density > 0 || throw(ArgumentError("density must be positive."))
    cfg.viscosity > 0 || throw(ArgumentError("viscosity must be positive."))
    cfg.elastic_modulus > 0 || throw(ArgumentError("elastic_modulus must be positive."))
    cfg.kolmogorov_waves >= 1 || throw(ArgumentError("kolmogorov_waves must be >= 1."))
    isfinite(cfg.kolmogorov_phase) || throw(ArgumentError("kolmogorov_phase must be finite."))
    cfg.perturbation_amplitude >= 0 || throw(ArgumentError("perturbation_amplitude must be non-negative."))
    cfg.perturbation_mode_x >= 1 || throw(ArgumentError("perturbation_mode_x must be >= 1."))
    cfg.cap_radius > 0 || throw(ArgumentError("cap_radius must be positive."))
    cfg.filament_height > cfg.cap_radius || throw(ArgumentError(
        "filament_height must exceed cap_radius."
    ))
    cfg.support_height > 0 || throw(ArgumentError("support_height must be positive."))
    cfg.support_height + cfg.filament_height < cfg.Ly || throw(ArgumentError(
        "The support layer plus emerged filament height must remain below the upper wall."
    ))
    0.0 < cfg.center_x < cfg.Lx || throw(ArgumentError("center_x must lie inside the periodic domain."))
    cfg.phase_field in (:current_cdi, :acdi, :uwcdi) || throw(ArgumentError(
        "phase_field must be :current_cdi, :acdi, or :uwcdi."
    ))
    cfg.Pe_gamma > 0 || throw(ArgumentError("Pe_gamma must be positive."))
    cfg.acdi_epsilon_over_h > 0.5 || throw(ArgumentError("acdi_epsilon_over_h must exceed 0.5."))
    cfg.acdi_gamma >= 0 || throw(ArgumentError("acdi_gamma must be non-negative."))
    cfg.uwcdi_alpha >= 1 || throw(ArgumentError("uwcdi_alpha must be >= 1."))
    cfg.uwcdi_normal in (:standard, :youngs, :smoothed9) || throw(ArgumentError(
        "uwcdi_normal must be :standard, :youngs, or :smoothed9."
    ))
    cfg.uwcdi_init in (:sharp, :tanh) || throw(ArgumentError("uwcdi_init must be :sharp or :tanh."))
    cfg.uwcdi_init_epsilon_over_h > 0 || throw(ArgumentError("uwcdi_init_epsilon_over_h must be positive."))
    0.0 < cfg.sigma_interface_cutoff < cfg.sigma_core_cutoff <= 1.0 || throw(ArgumentError(
        "Require 0 < sigma_interface_cutoff < sigma_core_cutoff <= 1."
    ))
    cfg.sigma_safety_cutoff < cfg.sigma_interface_cutoff || throw(ArgumentError(
        "sigma_safety_cutoff must be smaller than sigma_interface_cutoff."
    ))
    return nothing
end

# PHASE-FIELD SETUP AND OUTPUT LAYOUT

function phase_field_setup(cfg::FilamentBenchmarkConfig, h_interface::Float64)
    cdi_info = resolve_interface_epsilon(
        cfg.interface_epsilon, cfg.epsilon_over_h, h_interface
    )

    if cfg.phase_field === :current_cdi
        return (
            initialization_mode=:equilibrium_tanh,
            initialization_info=cdi_info,
            cdi_epsilon=cdi_info.epsilon,
            acdi_epsilon=cfg.acdi_epsilon_over_h * h_interface,
        )
    elseif cfg.phase_field === :acdi
        acdi_epsilon = cfg.acdi_epsilon_over_h * h_interface
        initialization_info = resolve_interface_epsilon(
            NaN, 2.0 * cfg.acdi_epsilon_over_h, h_interface
        )
        return (
            initialization_mode=:equilibrium_tanh,
            initialization_info=initialization_info,
            cdi_epsilon=cdi_info.epsilon,
            acdi_epsilon=acdi_epsilon,
        )
    elseif cfg.phase_field === :uwcdi
        if cfg.uwcdi_init === :sharp
            initialization_info = (
                epsilon=0.0,
                mode=:sharp,
                epsilon_over_h=0.0,
                width_5_95=0.0,
                cells_5_95=0.0,
            )
            initialization_mode = :sharp_indicator
        else
            initialization_info = resolve_interface_epsilon(
                NaN, cfg.uwcdi_init_epsilon_over_h, h_interface
            )
            initialization_mode = :equilibrium_tanh
        end
        return (
            initialization_mode=initialization_mode,
            initialization_info=initialization_info,
            cdi_epsilon=cdi_info.epsilon,
            acdi_epsilon=cfg.acdi_epsilon_over_h * h_interface,
        )
    end

    error("Unsupported phase-field method: $(cfg.phase_field)")
end

function prepare_output_directory(cfg::FilamentBenchmarkConfig, interface_info)
    method_root = joinpath(cfg.output_root, String(cfg.phase_field))
    subgroup = nothing
    if cfg.phase_field === :uwcdi
        tags = String[]
        cfg.uwcdi_normal !== :standard && push!(tags, "normal_" * String(cfg.uwcdi_normal))
        cfg.uwcdi_init !== :sharp && push!(tags, "init_" * String(cfg.uwcdi_init))
        subgroup = isempty(tags) ? nothing : join(tags, "__")
    end
    return prepare_study_output_directory(
        method_root,
        interface_info,
        @sprintf("N%d", cfg.Ny);
        subgroup=subgroup,
    )
end

# BOUNDARY CONDITIONS, MODELS, AND INITIAL STATE

function build_boundary_conditions(::FilamentBenchmarkConfig)
    periodic_x = Periodic()
    wall = Dirichlet(0.0)

    # The structural support occupies the complete lower wall, whereas the
    # upper wall is pure fluid.  These phase values are therefore known
    bc_phi = BoundaryConditionSetup(
        periodic_x, periodic_x, Dirichlet(1.0), Dirichlet(0.0)
    )

    noflux_y = BoundaryConditionSetup(
        periodic_x, periodic_x, Neumann(0.0), Neumann(0.0)
    )

    bc_U = BoundaryConditionSetup(periodic_x, periodic_x, wall, wall)
    bc_V = BoundaryConditionSetup(periodic_x, periodic_x, wall, wall)
    bc_correction = BoundaryConditionSetup(
        periodic_x, periodic_x, Neumann(0.0), Neumann(0.0)
    )
    # The structure tensor is advected with a first-order upwind flux.  Even
    # though the physical wall-normal velocity is zero, an IMEX stage can
    bc_sigma_xx = BoundaryConditionSetup(
        periodic_x, periodic_x, Dirichlet(1.0), Dirichlet(1.0)
    )
    bc_sigma_yy = BoundaryConditionSetup(
        periodic_x, periodic_x, Dirichlet(1.0), Dirichlet(1.0)
    )
    bc_sigma_xy = BoundaryConditionSetup(
        periodic_x, periodic_x, Dirichlet(0.0), Dirichlet(0.0)
    )
    bc_sigma = (xx=bc_sigma_xx, yy=bc_sigma_yy, xy=bc_sigma_xy)

    return FSIBoundaryConditions(
        bc_U,
        bc_V,
        bc_correction,
        bc_phi,
        bc_sigma,
    )
end

function build_models(cfg::FilamentBenchmarkConfig, setup)
    phase_method = if cfg.phase_field === :current_cdi
        CurrentCDIPhaseField(
            1.0 / cfg.Pe_gamma,
            setup.cdi_epsilon;
            delta_normal=cfg.normal_regularization,
        )
    elseif cfg.phase_field === :acdi
        JainACDIPhaseField(
            cfg.acdi_gamma,
            setup.acdi_epsilon;
            delta_normal=1e-14,
            dynamic_gamma=cfg.acdi_dynamic_gamma,
        )
    elseif cfg.phase_field === :uwcdi
        PirozzoliUWCDIPhaseField(
            cfg.uwcdi_alpha;
            delta_normal=1e-14,
            normal_reconstruction=cfg.uwcdi_normal,
        )
    else
        error("Unsupported phase-field method: $(cfg.phase_field)")
    end

    return FSIModels(
        viscosity=ConstantViscosity(cfg.viscosity),
        tensor_relaxation=NoRelaxation(),
        elastic_stress=LinearInternalTensorStress(cfg.elastic_modulus),
        capillarity=NoCapillarity(),
        phase_field_method=phase_method,
        body_force=DeterministicKolmogorovForce(
            kolmogorov_force_amplitude(cfg),
            kolmogorov_wavenumber(cfg),
            cfg.kolmogorov_phase,
        ),
    )
end

function build_numerics(cfg::FilamentBenchmarkConfig)
    linear_solvers = LinearSolverOptions(
        velocity_method=:gmres,
        reltol=cfg.velocity_reltol,
        maxiter=cfg.velocity_maxiter,
        restart=cfg.velocity_restart,
        direct_fallback=true,
        fallback_reltol=1e-6,
        verbose=false,
    )

    return IMEXNumerics(
        linear_solvers=linear_solvers,
        structure_extension=GeometricStructureExtension(
            cfg.sigma_extension_rate,
            cfg.sigma_extension_exponent;
            safety_cutoff=cfg.sigma_safety_cutoff,
        ),
        projection_reltol=cfg.projection_reltol,
        projection_maxiter=cfg.projection_maxiter,
        projection_restart=cfg.projection_restart,
        stage1_constraint_atol=5e-8,
        stage1_constraint_rtol=1e-8,
        check_stage1_constraint=true,
    )
end

@inline function periodic_dx(x::Float64, cx::Float64, Lx::Float64)
    return mod(x - cx + 0.5 * Lx, Lx) - 0.5 * Lx
end

function sharp_wall_supported_filament(cfg::FilamentBenchmarkConfig, grid::MACGrid)
    phi = zeros(Float64, cfg.Nx, cfg.Ny)
    base = cfg.support_height
    cap_center_y = base + cfg.filament_height - cfg.cap_radius
    stem_bottom = base - cfg.cap_radius

    @inbounds for j in 1:cfg.Ny, i in 1:cfg.Nx
        x = grid.x_p[i]
        y = grid.y_p[j]
        dx = periodic_dx(x, cfg.center_x, cfg.Lx)

        inside_support = y <= base
        inside_stem = abs(dx) <= cfg.cap_radius &&
                      y <= cap_center_y &&
                      y >= stem_bottom
        inside_cap = hypot(dx, y - cap_center_y) <= cfg.cap_radius
        phi[i, j] = (inside_support || inside_stem || inside_cap) ? 1.0 : 0.0
    end
    return phi
end

function initial_state(cfg::FilamentBenchmarkConfig, grid::MACGrid, setup)
    phi = if setup.initialization_mode === :sharp_indicator
        sharp_wall_supported_filament(cfg, grid)
    else
        initialize_filament(
            grid,
            cfg.support_height,
            cfg.filament_height,
            cfg.cap_radius,
            cfg.center_x,
            setup.initialization_info.epsilon;
            periodic_x=true,
        )
    end

    # Small deterministic x-dependent perturbation built from a streamfunction
    # on MAC corners. The discrete divergence cancels identically at cell centers,
    # and psi=0 at y=0,Ly makes the wall-normal velocity zero at both walls.
    U = zeros(Float64, cfg.Nx + 1, cfg.Ny)
    V = zeros(Float64, cfg.Nx, cfg.Ny + 1)

    if cfg.perturbation_amplitude > 0.0
        kx = 2.0 * pi * cfg.perturbation_mode_x / cfg.Lx
        psi_amp = cfg.perturbation_amplitude / kx
        psi = Matrix{Float64}(undef, cfg.Nx + 1, cfg.Ny + 1)

        @inbounds for j in 1:(cfg.Ny + 1), i in 1:(cfg.Nx + 1)
            x = grid.x_u[i]
            y = grid.y_v[j]
            psi[i, j] = psi_amp * cos(kx * x) * sin(pi * y / cfg.Ly)^2
        end

        @inbounds for j in 1:cfg.Ny, i in 1:(cfg.Nx + 1)
            U[i, j] = (psi[i, j + 1] - psi[i, j]) /
                      (grid.y_v[j + 1] - grid.y_v[j])
        end
        @inbounds for j in 1:(cfg.Ny + 1), i in 1:cfg.Nx
            V[i, j] = -(psi[i + 1, j] - psi[i, j]) /
                       (grid.x_u[i + 1] - grid.x_u[i])
        end
    end

    return FSIState(
        U,
        V,
        phi,
        ones(Float64, cfg.Nx, cfg.Ny),
        ones(Float64, cfg.Nx, cfg.Ny),
        zeros(Float64, cfg.Nx, cfg.Ny),
    )
end

# DIAGNOSTICS

@inline cell_area(metrics::MeshMetrics, i::Int, j::Int) =
    metrics.dx_c[i] * metrics.dy_c[j]

function phase_integral(phi, metrics::MeshMetrics)
    total = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        total += phi[i, j] * cell_area(metrics, i, j)
    end
    return total
end

function phase_centroid(phi, grid::MACGrid, metrics::MeshMetrics)
    mass = phase_integral(phi, metrics)
    mass > eps(Float64) || error("Phase integral is too small to define a centroid.")
    xm = 0.0
    ym = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        w = phi[i, j] * cell_area(metrics, i, j)
        xm += grid.x_p[i] * w
        ym += grid.y_p[j] * w
    end
    return xm / mass, ym / mass, mass
end

function cell_center_velocity(state::FSIState)
    Nx, Ny = size(state.phi)
    u = Matrix{Float64}(undef, Nx, Ny)
    v = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        u[i, j] = 0.5 * (state.U[i, j] + state.U[i + 1, j])
        v[i, j] = 0.5 * (state.V[i, j] + state.V[i, j + 1])
    end
    return u, v
end

function capture_snapshot(
    t::Real,
    state::FSIState,
    grid::MACGrid,
    metrics::MeshMetrics,
)
    xc, yc, _ = phase_centroid(state.phi, grid, metrics)
    u, v = cell_center_velocity(state)
    return FilamentSnapshot(
        Float64(t),
        copy(state.phi),
        u,
        v,
        hypot.(u, v),
        copy(state.sigma_xx),
        copy(state.sigma_yy),
        copy(state.sigma_xy),
        xc,
        yc,
    )
end

function phase_center_gradient(phi, metrics::MeshMetrics, bc_phi)
    gx_face, gy_face = scalar_gradient(phi, metrics, bc_phi)
    gx = zeros(Float64, metrics.Nx, metrics.Ny)
    gy = zeros(Float64, metrics.Nx, metrics.Ny)
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        gx[i, j] = metrics.wL_f2cx[i] * gx_face[i, j] +
                   metrics.wR_f2cx[i] * gx_face[i + 1, j]
        gy[i, j] = metrics.wB_f2cy[j] * gy_face[i, j] +
                   metrics.wT_f2cy[j] * gy_face[i, j + 1]
    end
    return gx, gy
end

function phase_geometry_diagnostics(
    state::FSIState,
    grid::MACGrid,
    metrics::MeshMetrics,
    bc_phi,
    h_interface::Float64,
)
    xc, yc, mass = phase_centroid(state.phi, grid, metrics)
    gx, gy = phase_center_gradient(state.phi, metrics, bc_phi)

    perimeter_proxy = 0.0
    mixed_layer_measure = 0.0
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        area = cell_area(metrics, i, j)
        phi = state.phi[i, j]
        phi_bounded = clamp(phi, 0.0, 1.0)
        perimeter_proxy += area * hypot(gx[i, j], gy[i, j])
        mixed_layer_measure += area * 4.0 * phi_bounded * (1.0 - phi_bounded)

        dx = grid.x_p[i] - xc
        dy = grid.y_p[j] - yc
        w = area * phi
        cxx += w * dx * dx
        cyy += w * dy * dy
        cxy += w * dx * dy
    end

    cxx /= mass
    cyy /= mass
    cxy /= mass
    gap = hypot(cxx - cyy, 2.0 * cxy)
    covariance_major = 0.5 * (cxx + cyy + gap)
    covariance_minor = 0.5 * (cxx + cyy - gap)
    aspect_ratio = covariance_minor > 0.0 ? sqrt(covariance_major / covariance_minor) : Inf
    orientation = gap > 100.0 * eps(Float64) ?
        0.5 * atan(2.0 * cxy, cxx - cyy) : 0.0
    axis_orientation = mod(orientation, pi)
    tilt_from_vertical = 0.5 * pi - axis_orientation
    effective_width = perimeter_proxy > eps(Float64) ?
        mixed_layer_measure / perimeter_proxy : NaN

    return (
        x_centroid=xc,
        y_centroid=yc,
        phase_integral=mass,
        perimeter_proxy=perimeter_proxy,
        mixed_layer_measure=mixed_layer_measure,
        effective_interface_width=effective_width,
        effective_interface_cells=effective_width / h_interface,
        covariance_major=covariance_major,
        covariance_minor=covariance_minor,
        aspect_ratio=aspect_ratio,
        orientation_radians=axis_orientation,
        tilt_from_vertical_radians=tilt_from_vertical,
    )
end

function protrusion_geometry_diagnostics(
    state::FSIState,
    grid::MACGrid,
    metrics::MeshMetrics,
    cfg::FilamentBenchmarkConfig,
)
    ycut = protrusion_response_cutoff_y(cfg)
    mass = 0.0
    xm = 0.0
    ym = 0.0

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        grid.y_p[j] >= ycut || continue
        w = state.phi[i, j] * cell_area(metrics, i, j)
        mass += w
        xm += grid.x_p[i] * w
        ym += grid.y_p[j] * w
    end
    mass > eps(Float64) || error(
        "The protrusion response region contains too little phase mass to define moments."
    )

    xc = xm / mass
    yc = ym / mass
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        grid.y_p[j] >= ycut || continue
        w = state.phi[i, j] * cell_area(metrics, i, j)
        dx = grid.x_p[i] - xc
        dy = grid.y_p[j] - yc
        cxx += w * dx * dx
        cyy += w * dy * dy
        cxy += w * dx * dy
    end
    cxx /= mass
    cyy /= mass
    cxy /= mass

    gap = hypot(cxx - cyy, 2.0 * cxy)
    covariance_major = 0.5 * (cxx + cyy + gap)
    covariance_minor = 0.5 * (cxx + cyy - gap)
    aspect_ratio = covariance_minor > 0.0 ? sqrt(covariance_major / covariance_minor) : Inf
    orientation = gap > 100.0 * eps(Float64) ?
        0.5 * atan(2.0 * cxy, cxx - cyy) : 0.0
    axis_orientation = mod(orientation, pi)
    tilt_from_vertical = 0.5 * pi - axis_orientation

    return (
        phase_mass=mass,
        x_centroid=xc,
        y_centroid=yc,
        covariance_major=covariance_major,
        covariance_minor=covariance_minor,
        aspect_ratio=aspect_ratio,
        orientation_radians=axis_orientation,
        tilt_from_vertical_radians=tilt_from_vertical,
        cutoff_y=ycut,
    )
end

function phase_bound_diagnostics(phi)
    phi_min = minimum(phi)
    phi_max = maximum(phi)
    return (
        minimum=phi_min,
        maximum=phi_max,
        lower_violation=max(0.0, -phi_min),
        upper_violation=max(0.0, phi_max - 1.0),
        out_of_bounds_cells=count(value -> value < 0.0 || value > 1.0, phi),
    )
end

function flow_diagnostics(state::FSIState, metrics::MeshMetrics, density::Float64)
    u, v = cell_center_velocity(state)
    kinetic_energy = 0.0
    max_speed = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        speed2 = u[i, j]^2 + v[i, j]^2
        max_speed = max(max_speed, sqrt(speed2))
        kinetic_energy += 0.5 * density * speed2 * cell_area(metrics, i, j)
    end

    div = divergence(state.U, state.V, metrics)
    linf = maximum(abs, div)
    l2sq = 0.0
    volume = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        area = cell_area(metrics, i, j)
        l2sq += area * div[i, j]^2
        volume += area
    end
    return (
        maximum_speed=max_speed,
        kinetic_energy=kinetic_energy,
        divergence_linf=linf,
        divergence_l2=sqrt(l2sq / volume),
    )
end

function tensor_region_diagnostics(
    state::FSIState,
    metrics::MeshMetrics,
    cfg::FilamentBenchmarkConfig,
    selector,
)
    area_total = 0.0
    cell_count = 0
    non_spd_count = 0
    determinant_min = Inf
    determinant_max = -Inf
    max_abs_det_error = 0.0
    mean_abs_det_error_integral = 0.0
    lambda_min = Inf
    lambda_max = -Inf
    max_stress = 0.0
    mean_stress_integral = 0.0

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        phi = state.phi[i, j]
        selector(phi) || continue
        area = cell_area(metrics, i, j)
        tensor = tensor_cell_diagnostics_2d(
            state.sigma_xx[i, j],
            state.sigma_yy[i, j],
            state.sigma_xy[i, j],
            phi,
            elastic_stress_coefficient(cfg),
        )
        area_total += area
        cell_count += 1
        determinant_min = min(determinant_min, tensor.determinant)
        determinant_max = max(determinant_max, tensor.determinant)
        det_error = abs(tensor.determinant_error)
        max_abs_det_error = max(max_abs_det_error, det_error)
        mean_abs_det_error_integral += area * det_error
        lambda_min = min(lambda_min, tensor.lambda_min)
        lambda_max = max(lambda_max, tensor.lambda_max)
        !(isfinite(tensor.lambda_min) && tensor.lambda_min > 0.0) && (non_spd_count += 1)
        max_stress = max(max_stress, tensor.effective_stress_frobenius)
        mean_stress_integral += area * tensor.effective_stress_frobenius
    end

    if cell_count == 0 || area_total == 0.0
        return (
            cells=0, area=0.0, determinant_min=NaN, determinant_max=NaN,
            max_abs_determinant_error=NaN, mean_abs_determinant_error=NaN,
            lambda_min=NaN, lambda_max=NaN, non_spd_cells=0,
            max_effective_stress=NaN, mean_effective_stress=NaN,
        )
    end

    return (
        cells=cell_count,
        area=area_total,
        determinant_min=determinant_min,
        determinant_max=determinant_max,
        max_abs_determinant_error=max_abs_det_error,
        mean_abs_determinant_error=mean_abs_det_error_integral / area_total,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        non_spd_cells=non_spd_count,
        max_effective_stress=max_stress,
        mean_effective_stress=mean_stress_integral / area_total,
    )
end

function tensor_diagnostics_detailed(
    state::FSIState,
    metrics::MeshMetrics,
    cfg::FilamentBenchmarkConfig,
)
    core = tensor_region_diagnostics(state, metrics, cfg, phi -> phi >= cfg.sigma_core_cutoff)
    interface = tensor_region_diagnostics(
        state, metrics, cfg,
        phi -> cfg.sigma_interface_cutoff < phi < cfg.sigma_core_cutoff,
    )
    tail = tensor_region_diagnostics(
        state, metrics, cfg,
        phi -> cfg.sigma_safety_cutoff < phi <= cfg.sigma_interface_cutoff,
    )
    global_region = tensor_region_diagnostics(state, metrics, cfg, phi -> true)
    return (global_region=global_region, core=core, interface=interface, tail=tail)
end

# COMMON TIMESTEP ENVELOPE

function comparison_timestep_bounds(
    cfg::FilamentBenchmarkConfig,
    state::FSIState,
    metrics::MeshMetrics,
    setup,
)
    hx = minimum(metrics.dx_c)
    hy = minimum(metrics.dy_c)
    hmin = min(hx, hy)

    # Match PhaseFieldMethods._acdi_effective_gamma when dynamic gamma is selected.
    instantaneous_component_speed = max(maximum(abs, state.U), maximum(abs, state.V))
    max_advection_speed = max(cfg.U_ref, instantaneous_component_speed)
    elastic_wave_speed = sqrt(cfg.elastic_modulus / cfg.density)
    hyperbolic_speed = max(max_advection_speed, elastic_wave_speed, eps(Float64))
    dt_hyperbolic = cfg.cfl_hyperbolic * hmin / hyperbolic_speed

    # Same common CDI/ACDI/UWCDI stability envelope used by the LDC comparison.
    inverse_pe_gamma = 1.0 / cfg.Pe_gamma
    dt_cdi_compression = cfg.cfl_interface_compression * hmin / inverse_pe_gamma
    nu_cdi = 0.5 * setup.cdi_epsilon * inverse_pe_gamma
    dt_cdi_diffusion_limit = 1.0 / (
        2.0 * nu_cdi * (1.0 / hx^2 + 1.0 / hy^2)
    )
    dt_cdi_diffusion = cfg.cfl_interface_diffusion * dt_cdi_diffusion_limit

    gamma_acdi = cfg.acdi_dynamic_gamma ?
        cfg.acdi_gamma * instantaneous_component_speed : cfg.acdi_gamma
    dt_acdi_compression = cfg.cfl_interface_compression * hmin /
        max(gamma_acdi, eps(Float64))
    nu_acdi = gamma_acdi * setup.acdi_epsilon
    dt_acdi_diffusion = nu_acdi > 0.0 ?
        cfg.cfl_interface_diffusion / (
            2.0 * nu_acdi * (1.0 / hx^2 + 1.0 / hy^2)
        ) : Inf

    uwcdi_cfl_limit = 1.0 / (1.0 + 1.0 / cfg.uwcdi_alpha)
    dt_uwcdi = uwcdi_cfl_limit * hmin /
        max(max_advection_speed, eps(Float64))

    labels = (
        "hyperbolic",
        "current_cdi_compression",
        "current_cdi_diffusion",
        "acdi_compression",
        "acdi_diffusion",
        "uwcdi",
        "dt_max",
    )
    values = (
        dt_hyperbolic,
        dt_cdi_compression,
        dt_cdi_diffusion,
        dt_acdi_compression,
        dt_acdi_diffusion,
        dt_uwcdi,
        cfg.dt_max,
    )
    selected, index = findmin(values)

    return (
        hyperbolic=dt_hyperbolic,
        cdi_compression=dt_cdi_compression,
        cdi_diffusion=dt_cdi_diffusion,
        acdi_compression=dt_acdi_compression,
        acdi_diffusion=dt_acdi_diffusion,
        uwcdi=dt_uwcdi,
        selected=selected,
        limiting_constraint=labels[index],
        max_advection_speed=max_advection_speed,
        elastic_wave_speed=elastic_wave_speed,
        gamma_acdi=gamma_acdi,
    )
end

# CSV OUTPUT

function write_trajectory_header(io)
    println(io, "time,x_centroid,y_centroid,delta_x_centroid,delta_y_centroid,phase_integral,phase_integral_ratio")
end

function write_trajectory_row(io, t, geometry, xc0, yc0, mass0)
    @printf(
        io,
        "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
        t,
        geometry.x_centroid,
        geometry.y_centroid,
        geometry.x_centroid - xc0,
        geometry.y_centroid - yc0,
        geometry.phase_integral,
        geometry.phase_integral / mass0,
    )
    flush(io)
end

function write_phase_diagnostics_header(io)
    println(
        io,
        "time,phase_integral,relative_phase_integral_error,geometric_area_reference," *
        "phase_area_bias,relative_phase_area_bias,phi_min,phi_max," *
        "lower_bound_violation,upper_bound_violation,out_of_bounds_cells," *
        "perimeter_proxy,mixed_layer_measure,effective_interface_width," *
        "effective_interface_cells,aspect_ratio,orientation_radians," *
        "tilt_from_vertical_radians,covariance_major,covariance_minor",
    )
end

function write_phase_diagnostics_row(io, t, geometry, bounds, mass0, geometric_area)
    area_bias = geometry.phase_integral - geometric_area
    @printf(
        io,
        "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
        t,
        geometry.phase_integral,
        (geometry.phase_integral - mass0) / mass0,
        geometric_area,
        area_bias,
        area_bias / geometric_area,
        bounds.minimum,
        bounds.maximum,
        bounds.lower_violation,
        bounds.upper_violation,
        bounds.out_of_bounds_cells,
        geometry.perimeter_proxy,
        geometry.mixed_layer_measure,
        geometry.effective_interface_width,
        geometry.effective_interface_cells,
        geometry.aspect_ratio,
        geometry.orientation_radians,
        geometry.tilt_from_vertical_radians,
        geometry.covariance_major,
        geometry.covariance_minor,
    )
    flush(io)
end

function write_response_header(io)
    println(
        io,
        "time,delta_x_protrusion_centroid,delta_y_protrusion_centroid," *
        "x_protrusion_centroid,y_protrusion_centroid,orientation_radians," *
        "tilt_from_vertical_radians,aspect_ratio,protrusion_phase_mass,cutoff_y"
    )
end

function write_response_row(io, t, geometry, xc0, yc0)
    @printf(
        io,
        "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
        t,
        geometry.x_centroid - xc0,
        geometry.y_centroid - yc0,
        geometry.x_centroid,
        geometry.y_centroid,
        geometry.orientation_radians,
        geometry.tilt_from_vertical_radians,
        geometry.aspect_ratio,
        geometry.phase_mass,
        geometry.cutoff_y,
    )
    flush(io)
end

const TENSOR_REGION_FIELDS = (
    :cells,
    :area,
    :determinant_min,
    :determinant_max,
    :max_abs_determinant_error,
    :mean_abs_determinant_error,
    :lambda_min,
    :lambda_max,
    :non_spd_cells,
    :max_effective_stress,
    :mean_effective_stress,
)

function write_tensor_diagnostics_header(io)
    names = String["time"]
    for region in ("global", "core", "interface", "tail")
        for field in TENSOR_REGION_FIELDS
            push!(names, region * "_" * String(field))
        end
    end
    println(io, join(names, ','))
end

@inline function tensor_region_csv_values(region)
    return String[
        string(region.cells),
        @sprintf("%.16e", region.area),
        @sprintf("%.16e", region.determinant_min),
        @sprintf("%.16e", region.determinant_max),
        @sprintf("%.16e", region.max_abs_determinant_error),
        @sprintf("%.16e", region.mean_abs_determinant_error),
        @sprintf("%.16e", region.lambda_min),
        @sprintf("%.16e", region.lambda_max),
        string(region.non_spd_cells),
        @sprintf("%.16e", region.max_effective_stress),
        @sprintf("%.16e", region.mean_effective_stress),
    ]
end

function write_tensor_diagnostics_row(io, t, tensor)
    values = String[@sprintf("%.16e", t)]
    for region in (tensor.global_region, tensor.core, tensor.interface, tensor.tail)
        append!(values, tensor_region_csv_values(region))
    end
    println(io, join(values, ','))
    flush(io)
end

function write_flow_diagnostics_header(io)
    println(io, "time,max_speed,kinetic_energy,divergence_linf,divergence_l2")
end

function write_flow_diagnostics_row(io, t, flow)
    @printf(
        io,
        "%.16e,%.16e,%.16e,%.16e,%.16e\n",
        t,
        flow.maximum_speed,
        flow.kinetic_energy,
        flow.divergence_linf,
        flow.divergence_l2,
    )
    flush(io)
end

function write_numerics_diagnostics_header(io)
    println(
        io,
        "step,time,dt_actual,dt_stability,limiting_constraint," *
        "dt_hyperbolic,dt_cdi_compression,dt_cdi_diffusion," *
        "dt_acdi_compression,dt_acdi_diffusion,dt_uwcdi," *
        "max_advection_speed,elastic_wave_speed,gamma_acdi,projection_constraint_linf",
    )
end

function write_numerics_diagnostics_row(io, step, t, dt, timestep, projection_constraint)
    @printf(
        io,
        "%d,%.16e,%.16e,%.16e,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
        step,
        t,
        dt,
        timestep.selected,
        timestep.limiting_constraint,
        timestep.hyperbolic,
        timestep.cdi_compression,
        timestep.cdi_diffusion,
        timestep.acdi_compression,
        timestep.acdi_diffusion,
        timestep.uwcdi,
        timestep.max_advection_speed,
        timestep.elastic_wave_speed,
        timestep.gamma_acdi,
        projection_constraint,
    )
    flush(io)
end

function write_snapshot_index_header(io)
    println(io, "target_time,actual_time,x_centroid,y_centroid,fields_csv,png_file,pdf_file")
end

function write_snapshot_fields_csv(
    path::AbstractString,
    record::FilamentSnapshot,
    grid::MACGrid,
    cfg::FilamentBenchmarkConfig,
)
    open(path, "w") do io
        println(
            io,
            "x,y,phi,u,v,speed,sigma_xx,sigma_yy,sigma_xy,sigma_trace,det_sigma," *
            "det_sigma_error,lambda_min_sigma,lambda_max_sigma," *
            "effective_tau_xx,effective_tau_yy,effective_tau_xy," *
            "effective_elastic_stress_frobenius,phase_mixing_indicator",
        )
        Nx, Ny = size(record.phi)
        @inbounds for j in 1:Ny, i in 1:Nx
            tensor = tensor_cell_diagnostics_2d(
                record.sigma_xx[i, j],
                record.sigma_yy[i, j],
                record.sigma_xy[i, j],
                record.phi[i, j],
                elastic_stress_coefficient(cfg),
            )
            effective_tau_xx = record.phi[i, j] * elastic_stress_coefficient(cfg) * (record.sigma_xx[i, j] - 1.0)
            effective_tau_yy = record.phi[i, j] * elastic_stress_coefficient(cfg) * (record.sigma_yy[i, j] - 1.0)
            effective_tau_xy = record.phi[i, j] * elastic_stress_coefficient(cfg) * record.sigma_xy[i, j]
            @printf(
                io,
                "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                grid.x_p[i],
                grid.y_p[j],
                record.phi[i, j],
                record.u[i, j],
                record.v[i, j],
                record.speed[i, j],
                record.sigma_xx[i, j],
                record.sigma_yy[i, j],
                record.sigma_xy[i, j],
                record.sigma_xx[i, j] + record.sigma_yy[i, j],
                tensor.determinant,
                tensor.determinant_error,
                tensor.lambda_min,
                tensor.lambda_max,
                effective_tau_xx,
                effective_tau_yy,
                effective_tau_xy,
                tensor.effective_stress_frobenius,
                4.0 * clamp(record.phi[i, j], 0.0, 1.0) * (1.0 - clamp(record.phi[i, j], 0.0, 1.0)),
            )
        end
    end
    return path
end

# FIGURES

function snapshot_plot(
    record::FilamentSnapshot,
    cfg::FilamentBenchmarkConfig,
    grid::MACGrid;
    colorbar::Bool=true,
    compact_title::Bool=false,
    panel_label::Union{Nothing,String}=nothing,
    vmax_override::Union{Nothing,Float64}=nothing,
)
    vmax_actual = maximum(record.speed)
    vmax = vmax_override === nothing ?
        max(cfg.snapshot_speed_scale, vmax_actual, 1e-12) : max(vmax_override, 1e-12)
    levels = range(0.0, vmax; length=41)
    prefix = panel_label === nothing ? "" : panel_label * "   "
    title_text = compact_title ? prefix * @sprintf("t = %.2f", record.time) :
        @sprintf("%dx%d   |   t = %.2f", cfg.Nx, cfg.Ny, record.time)

    p = contourf(
        grid.x_p,
        grid.y_p,
        permutedims(record.speed);
        levels=levels,
        c=:viridis,
        clims=(0.0, vmax),
        colorbar=colorbar,
        colorbar_title="|u|",
        xlabel="x",
        ylabel="y",
        title=title_text,
        aspect_ratio=:equal,
        xlims=(0.0, cfg.Lx),
        ylims=(0.0, cfg.Ly),
        framestyle=:box,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color_subplot=:black,
        fontfamily="Times",
        guidefontsize=11,
        tickfontsize=9,
        titlefontsize=11,
        tick_direction=:in,
    )
    contour!(
        p,
        grid.x_p,
        grid.y_p,
        permutedims(record.phi);
        levels=[0.5],
        linewidth=3.2,
        color=:white,
        label=false,
    )
    contour!(
        p,
        grid.x_p,
        grid.y_p,
        permutedims(record.phi);
        levels=[0.5],
        linewidth=1.5,
        color=:black,
        label=false,
    )
    return p
end

function save_snapshot(
    record::FilamentSnapshot,
    cfg::FilamentBenchmarkConfig,
    grid::MACGrid,
    snapshot_dir::AbstractString,
)
    tag = replace(@sprintf("%05.2f", record.time), "." => "p")
    stem = "snapshot_t_" * tag
    png_path = joinpath(snapshot_dir, stem * ".png")
    pdf_path = joinpath(snapshot_dir, stem * ".pdf")
    fields_path = joinpath(snapshot_dir, stem * "_fields.csv")
    write_snapshot_fields_csv(fields_path, record, grid, cfg)
    p = snapshot_plot(record, cfg, grid; colorbar=true)
    plot!(p; size=(1150, 430), dpi=300)
    savefig(p, png_path)
    savefig(p, pdf_path)
    return png_path, pdf_path, fields_path
end

function save_snapshot_summary(
    records::Vector{FilamentSnapshot},
    cfg::FilamentBenchmarkConfig,
    grid::MACGrid,
    snapshot_dir::AbstractString,
)
    isempty(records) && return nothing
    selected = length(records) <= 6 ? records : records[round.(Int, range(1, length(records), length=6))]
    global_vmax = max(
        cfg.snapshot_speed_scale,
        maximum(maximum(record.speed) for record in selected),
        1e-12,
    )
    panels = Plots.Plot[]
    for (k, record) in enumerate(selected)
        push!(
            panels,
            snapshot_plot(
                record,
                cfg,
                grid;
                colorbar=false,
                compact_title=true,
                panel_label="(" * string(Char(Int('a') + k - 1)) * ")",
                vmax_override=global_vmax,
            ),
        )
    end
    summary = plot(
        panels...;
        layout=(length(panels), 1),
        size=(1150, 330 * length(panels)),
        dpi=250,
        background_color=:white,
        foreground_color=:black,
        fontfamily="Times",
    )
    png_path = joinpath(snapshot_dir, "snapshots_summary.png")
    pdf_path = joinpath(snapshot_dir, "snapshots_summary.pdf")
    savefig(summary, png_path)
    savefig(summary, pdf_path)
    return png_path, pdf_path
end

# RUN METADATA

function write_run_parameters(
    path::AbstractString,
    cfg::FilamentBenchmarkConfig,
    grid::MACGrid,
    metrics::MeshMetrics,
    setup,
    initial_bounds,
    initial_projection,
)
    open(path, "w") do io
        println(io, "Wall-supported hyperelastic filament in deterministic Kolmogorov flow")
        println(io, "==================================================")
        println(io, "Created: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Julia version: ", VERSION)
        println(io, "Julia threads: ", Threads.nthreads())
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        println(io, "Command: ", PROGRAM_FILE, isempty(ARGS) ? "" : " " * join(ARGS, " "))
        println(io)

        println(io, "Geometry and flow")
        println(io, "-----------------")
        @printf(io, "Domain: %.8g x %.8g\n", cfg.Lx, cfg.Ly)
        @printf(io, "Grid: %d x %d uniform MAC cells\n", cfg.Nx, cfg.Ny)
        println(io, "x boundary: periodic")
        println(io, "y boundaries: no-slip")
        println(io, "phi y-boundaries: bottom Dirichlet(1), top Dirichlet(0)")
        @printf(io, "Kolmogorov reference speed Uref: %.8g\n", cfg.U_ref)
        @printf(io, "Fluid viscosity nu: %.8g\n", cfg.viscosity)
        @printf(io, "Effective Re based on filament height: %.8g\n", cfg.U_ref * cfg.filament_height / cfg.viscosity)
        @printf(io, "Kolmogorov waves / ky: %d / %.10e\n", cfg.kolmogorov_waves, kolmogorov_wavenumber(cfg))
        @printf(io, "Kolmogorov force amplitude: %.10e\n", kolmogorov_force_amplitude(cfg))
        println(io, "Viscosity model: ConstantViscosity")
        @printf(io, "Elastic modulus G: %.8g\n", cfg.elastic_modulus)
        println(io, "Physical tensor relaxation: OFF")
        @printf(io, "Perturbation amplitude / x-mode: %.8g / %d\n", cfg.perturbation_amplitude, cfg.perturbation_mode_x)
        
        
        println(io, "Capillarity: OFF")
        
        println(io, "Initial velocity: deterministic discrete-divergence-free perturbation")
        println(io, "Empty-domain laminar target: u=Uref*sin(ky*y), v=0")
        @printf(io, "Initial projection correction Linf: %.10e\n", initial_projection.constraint_linf)
        println(io)

        println(io, "Wall-supported filament")
        println(io, "------------------------")
        @printf(io, "Support-layer thickness: %.8g\n", cfg.support_height)
        @printf(io, "Center x: %.8g\n", cfg.center_x)
        @printf(io, "Emerged filament height: %.8g\n", cfg.filament_height)
        @printf(io, "Filament width: %.8g\n", filament_width(cfg))
        @printf(io, "Cap radius: %.8g\n", cfg.cap_radius)
        @printf(io, "Slenderness height/width: %.8g\n", cfg.filament_height / filament_width(cfg))
        @printf(io, "Support thickness/width: %.8g\n", cfg.support_height / filament_width(cfg))
        @printf(io, "Protrusion geometric area: %.10e\n", protrusion_geometric_area(cfg))
        @printf(io, "Total structural geometric area: %.10e\n", total_structural_geometric_area(cfg))
        @printf(io, "Protrusion-response cutoff y: %.8g\n", protrusion_response_cutoff_y(cfg))
        @printf(io, "Elastic stress coefficient: %.8g\n", elastic_stress_coefficient(cfg))
        println(io, "Tensor relaxation: OFF")
        println(io, "Capillarity: OFF")
        println(io)

        println(io, "Phase-field numerics")
        println(io, "--------------------")
        println(io, "Method: ", cfg.phase_field)
        @printf(io, "CDI Pe_Gamma: %.8g\n", cfg.Pe_gamma)
        @printf(io, "CDI epsilon/h: %.8g\n", setup.cdi_epsilon / conservative_mesh_spacing(metrics))
        @printf(io, "ACDI epsilon_A/h: %.8g\n", cfg.acdi_epsilon_over_h)
        @printf(io, "ACDI Gamma parameter: %.8g (%s)\n", cfg.acdi_gamma,
            cfg.acdi_dynamic_gamma ? "scale * instantaneous max component speed" : "fixed")
        @printf(io, "UWCDI alpha: %.8g\n", cfg.uwcdi_alpha)
        println(io, "UWCDI normal: ", cfg.uwcdi_normal)
        println(io, "UWCDI initialization: ", cfg.uwcdi_init)
        println(io, "Selected initialization mode: ", setup.initialization_mode)
        @printf(io, "Selected initialization epsilon/h: %.8g\n", setup.initialization_info.epsilon_over_h)
        @printf(io, "sigma extension kappa0: %.8g\n", cfg.sigma_extension_rate)
        @printf(io, "sigma extension exponent m: %.8g\n", cfg.sigma_extension_exponent)
        @printf(io, "sigma far-fluid cutoff: %.8g\n", cfg.sigma_safety_cutoff)
        println(io)

        println(io, "Time integration")
        println(io, "----------------")
        @printf(io, "Final time: %.8g\n", cfg.T)
        @printf(io, "CFL hyperbolic: %.8g\n", cfg.cfl_hyperbolic)
        @printf(io, "CFL interface compression: %.8g\n", cfg.cfl_interface_compression)
        @printf(io, "CFL interface diffusion: %.8g\n", cfg.cfl_interface_diffusion)
        
        @printf(io, "dt_max: %.10e\n", cfg.dt_max)
        @printf(io, "Initial selected dt: %.10e (%s)\n", initial_bounds.selected, initial_bounds.limiting_constraint)
        println(io, "Common stability envelope: CDI + ACDI + UWCDI + hyperbolic")
        println(io, "Snapshot times: ", join(cfg.snapshot_times, ", "))
    end
end

# MAIN SIMULATION

function run_benchmark(cfg::FilamentBenchmarkConfig)
    validate_config(cfg)
    LinearAlgebra.BLAS.set_num_threads(1)

    grid = generate_grid(
        cfg.Lx,
        cfg.Ly,
        cfg.Nx,
        cfg.Ny;
        stretch_x=:uniform,
        stretch_y=:uniform,
    )
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=false)
    h_interface = conservative_mesh_spacing(metrics)
    setup = phase_field_setup(cfg, h_interface)

    bcs = build_boundary_conditions(cfg)
    models = build_models(cfg, setup)
    numerics = build_numerics(cfg)
    context = build_imex_context(
        metrics,
        bcs,
        models;
        numerics=numerics,
        use_projection_amg=cfg.use_projection_amg,
    )
    work = allocate_imex_workspace(metrics)
    state = initial_state(cfg, grid, setup)

    # The streamfunction construction is already discretely divergence free,
    # so no initialization Poisson solve is required. Keep the same diagnostic
    # quantity for the run metadata and let the normal stage projections operate
    # unchanged during time integration.
    initial_divergence = divergence(state.U, state.V, metrics)
    init_projection = (constraint_linf=maximum(abs, initial_divergence),)

    output_dir = prepare_output_directory(cfg, setup.initialization_info)
    snapshot_dir = joinpath(output_dir, "snapshots")
    mkpath(snapshot_dir)


    initial_bounds = comparison_timestep_bounds(cfg, state, metrics, setup)
    write_run_parameters(
        joinpath(output_dir, "run_parameters.txt"),
        cfg,
        grid,
        metrics,
        setup,
        initial_bounds,
        init_projection,
    )

    trajectory_path = joinpath(output_dir, "centroid_trajectory.csv")
    response_path = joinpath(output_dir, "protrusion_response.csv")
    phase_path = joinpath(output_dir, "phase_diagnostics.csv")
    tensor_path = joinpath(output_dir, "tensor_diagnostics.csv")
    flow_path = joinpath(output_dir, "flow_diagnostics.csv")
    numerics_path = joinpath(output_dir, "numerics_diagnostics.csv")
    snapshot_index_path = joinpath(output_dir, "snapshot_index.csv")

    trajectory_io = open(trajectory_path, "w")
    response_io = open(response_path, "w")
    phase_io = open(phase_path, "w")
    tensor_io = open(tensor_path, "w")
    flow_io = open(flow_path, "w")
    numerics_io = open(numerics_path, "w")
    snapshot_index_io = open(snapshot_index_path, "w")

    write_trajectory_header(trajectory_io)
    write_response_header(response_io)
    write_phase_diagnostics_header(phase_io)
    write_tensor_diagnostics_header(tensor_io)
    write_flow_diagnostics_header(flow_io)
    write_numerics_diagnostics_header(numerics_io)
    write_snapshot_index_header(snapshot_index_io)

    geometry0 = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
    response0 = protrusion_geometry_diagnostics(state, grid, metrics, cfg)
    bounds0 = phase_bound_diagnostics(state.phi)
    tensor0 = tensor_diagnostics_detailed(state, metrics, cfg)
    flow0 = flow_diagnostics(state, metrics, cfg.density)
    xc0 = geometry0.x_centroid
    yc0 = geometry0.y_centroid
    response_xc0 = response0.x_centroid
    response_yc0 = response0.y_centroid
    mass0 = geometry0.phase_integral
    geometric_area = total_structural_geometric_area(cfg)

    write_trajectory_row(trajectory_io, 0.0, geometry0, xc0, yc0, mass0)
    write_response_row(response_io, 0.0, response0, response_xc0, response_yc0)
    write_phase_diagnostics_row(phase_io, 0.0, geometry0, bounds0, mass0, geometric_area)
    write_tensor_diagnostics_row(tensor_io, 0.0, tensor0)
    write_flow_diagnostics_row(flow_io, 0.0, flow0)
    write_numerics_diagnostics_row(
        numerics_io, 0, 0.0, 0.0, initial_bounds, init_projection.constraint_linf
    )

    active_snapshot_times = sort(unique(filter(t -> 0.0 <= t <= cfg.T + 1e-12, cfg.snapshot_times)))
    snapshots = FilamentSnapshot[]
    snapshot_index = 1

    while snapshot_index <= length(active_snapshot_times) && active_snapshot_times[snapshot_index] <= 1e-12
        record = capture_snapshot(0.0, state, grid, metrics)
        push!(snapshots, record)
        png_path, pdf_path, fields_path = save_snapshot(record, cfg, grid, snapshot_dir)
        @printf(
            snapshot_index_io,
            "%.10e,%.10e,%.10e,%.10e,%s,%s,%s\n",
            active_snapshot_times[snapshot_index],
            0.0,
            geometry0.x_centroid,
            geometry0.y_centroid,
            relpath(fields_path, output_dir),
            relpath(png_path, output_dir),
            relpath(pdf_path, output_dir),
        )
        flush(snapshot_index_io)
        snapshot_index += 1
    end

    println()
    println("Wall-supported hyperelastic filament in deterministic Kolmogorov flow")
    println("-------------------------------------------------------------")
    @printf("Grid                 : %d x %d\n", cfg.Nx, cfg.Ny)
    @printf("Uref / nu / Re_eff   : %.4g / %.4g / %.4g\n", cfg.U_ref, cfg.viscosity, cfg.U_ref * cfg.filament_height / cfg.viscosity)
    @printf("Kolmogorov n / ky    : %d / %.6g\n", cfg.kolmogorov_waves, kolmogorov_wavenumber(cfg))
    @printf("Kolmogorov amplitude : %.6g\n", kolmogorov_force_amplitude(cfg))
    println("Initial flow         : deterministic divergence-free perturbation")
    @printf("Support thickness     : %.4g\n", cfg.support_height)
    @printf("Filament H x W        : %.4g x %.4g\n", cfg.filament_height, filament_width(cfg))
    println("Viscosity model      : constant")
    println("Relaxation / capill. : OFF / OFF")
    @printf("Elastic modulus G    : %.6g\n", cfg.elastic_modulus)
    println("Phase-field method    : ", cfg.phase_field)
    println("UWCDI normal/init     : ", cfg.uwcdi_normal, " / ", cfg.uwcdi_init)
    @printf("Initial phase area    : %.10e\n", mass0)
    @printf("Geometric area        : %.10e\n", geometric_area)
    @printf("Initial area bias     : %+.4e (%.3f%%)\n", mass0-geometric_area, 100*(mass0-geometric_area)/geometric_area)
    @printf("Initial global centroid: (%.8f, %.8f)\n", xc0, yc0)
    @printf("Initial protrusion ctr : (%.8f, %.8f)\n", response_xc0, response_yc0)
    @printf("Initial dt bound      : %.6e (%s)\n", initial_bounds.selected, initial_bounds.limiting_constraint)
    println("Output directory      : $output_dir")
    println()

    t = 0.0
    step = 0
    next_trajectory_time = cfg.trajectory_interval
    next_diagnostics_time = cfg.diagnostics_interval
    last_trajectory_written = 0.0
    last_diagnostics_written = 0.0
    last_terminal_update = time()
    last_report = nothing
    event_tol = 5e-12

    try
        while t < cfg.T - event_tol
            timestep = comparison_timestep_bounds(cfg, state, metrics, setup)
            dt_physical = timestep.selected
            dt_physical >= cfg.dt_min_physical || error(
                @sprintf(
                    "Physical timestep bound %.6e is below dt_min_physical %.6e at t=%.6e.",
                    dt_physical,
                    cfg.dt_min_physical,
                    t,
                )
            )

            next_snapshot_time = snapshot_index <= length(active_snapshot_times) ?
                active_snapshot_times[snapshot_index] : Inf
            next_event_time = min(
                cfg.T,
                next_trajectory_time,
                next_diagnostics_time,
                next_snapshot_time,
            )
            dt = min(dt_physical, next_event_time - t)

            if dt <= event_tol
                t = next_event_time
            else
                last_report = integrate_step!(state, t, dt, context, work)
                t += dt
                step += 1
                abs(t - next_event_time) <= event_tol && (t = next_event_time)
            end

            if t >= next_trajectory_time - event_tol
                geometry = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
                response = protrusion_geometry_diagnostics(state, grid, metrics, cfg)
                write_trajectory_row(trajectory_io, t, geometry, xc0, yc0, mass0)
                write_response_row(response_io, t, response, response_xc0, response_yc0)
                last_trajectory_written = t
                next_trajectory_time += cfg.trajectory_interval
            end

            if t >= next_diagnostics_time - event_tol
                geometry = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
                bounds = phase_bound_diagnostics(state.phi)
                tensor = tensor_diagnostics_detailed(state, metrics, cfg)
                flow = flow_diagnostics(state, metrics, cfg.density)
                projection_constraint = last_report === nothing ?
                    init_projection.constraint_linf : last_report.maximum_projection_constraint_linf
                write_phase_diagnostics_row(phase_io, t, geometry, bounds, mass0, geometric_area)
                write_tensor_diagnostics_row(tensor_io, t, tensor)
                write_flow_diagnostics_row(flow_io, t, flow)
                write_numerics_diagnostics_row(
                    numerics_io, step, t, dt, timestep, projection_constraint
                )
                last_diagnostics_written = t
                next_diagnostics_time += cfg.diagnostics_interval
            end

            if snapshot_index <= length(active_snapshot_times) &&
               t >= active_snapshot_times[snapshot_index] - event_tol
                target_time = active_snapshot_times[snapshot_index]
                record = capture_snapshot(t, state, grid, metrics)
                push!(snapshots, record)
                png_path, pdf_path, fields_path = save_snapshot(record, cfg, grid, snapshot_dir)
                @printf(
                    snapshot_index_io,
                    "%.10e,%.10e,%.10e,%.10e,%s,%s,%s\n",
                    target_time,
                    t,
                    record.x_centroid,
                    record.y_centroid,
                    relpath(fields_path, output_dir),
                    relpath(png_path, output_dir),
                    relpath(pdf_path, output_dir),
                )
                flush(snapshot_index_io)
                @printf("Saved snapshot t = %.2f\n", target_time)
                snapshot_index += 1
            end

            wall_now = time()
            if wall_now - last_terminal_update >= cfg.terminal_update_seconds
                geometry = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
                response = protrusion_geometry_diagnostics(state, grid, metrics, cfg)
                sigma_min = last_report === nothing ? tensor0.global_region.lambda_min :
                    last_report.minimum_sigma_eigenvalue
                non_spd = last_report === nothing ? 0 : last_report.non_spd_cells
                @printf(
                    "[%6.2f%%] step=%d  t=%.6f  dt=%.3e  dx_c=%+.5f  tilt=%+.3fdeg  mass_err=%+.3e  min_lambda=%.3e  nonSPD=%d\n",
                    100.0 * t / cfg.T,
                    step,
                    t,
                    dt,
                    response.x_centroid - response_xc0,
                    rad2deg(response.tilt_from_vertical_radians),
                    (geometry.phase_integral - mass0) / mass0,
                    sigma_min,
                    non_spd,
                )
                last_terminal_update = wall_now
            end
        end

        abs(t - cfg.T) <= 1e-10 && (t = cfg.T)
        final_geometry = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
        final_response = protrusion_geometry_diagnostics(state, grid, metrics, cfg)
        if abs(last_trajectory_written - t) > event_tol
            write_trajectory_row(trajectory_io, t, final_geometry, xc0, yc0, mass0)
            write_response_row(response_io, t, final_response, response_xc0, response_yc0)
        end
        if abs(last_diagnostics_written - t) > event_tol
            bounds = phase_bound_diagnostics(state.phi)
            tensor = tensor_diagnostics_detailed(state, metrics, cfg)
            flow = flow_diagnostics(state, metrics, cfg.density)
            timestep = comparison_timestep_bounds(cfg, state, metrics, setup)
            projection_constraint = last_report === nothing ?
                init_projection.constraint_linf : last_report.maximum_projection_constraint_linf
            write_phase_diagnostics_row(phase_io, t, final_geometry, bounds, mass0, geometric_area)
            write_tensor_diagnostics_row(tensor_io, t, tensor)
            write_flow_diagnostics_row(flow_io, t, flow)
            write_numerics_diagnostics_row(
                numerics_io, step, t, 0.0, timestep, projection_constraint
            )
        end
    finally
        close(trajectory_io)
        close(response_io)
        close(phase_io)
        close(tensor_io)
        close(flow_io)
        close(numerics_io)
        close(snapshot_index_io)
    end

    save_snapshot_summary(snapshots, cfg, grid, snapshot_dir)

    final_geometry = phase_geometry_diagnostics(state, grid, metrics, bcs.phi, h_interface)
    final_response = protrusion_geometry_diagnostics(state, grid, metrics, cfg)
    final_record = capture_snapshot(t, state, grid, metrics)
    write_snapshot_fields_csv(joinpath(output_dir, "final_fields.csv"), final_record, grid, cfg)
    final_sigma = structure_tensor_diagnostics(state.sigma_xx, state.sigma_yy, state.sigma_xy)

    println()
    println("Benchmark completed")
    println("-------------------")
    @printf("Steps                    : %d\n", step)
    @printf("Final time               : %.8f\n", t)
    @printf("Final global centroid    : (%.8f, %.8f)\n", final_geometry.x_centroid, final_geometry.y_centroid)
    @printf("Final protrusion centroid: (%.8f, %.8f)\n", final_response.x_centroid, final_response.y_centroid)
    @printf("Final downstream shift   : %+.8e\n", final_response.x_centroid - response_xc0)
    @printf("Final tilt from vertical : %+.6f deg\n", rad2deg(final_response.tilt_from_vertical_radians))
    @printf("Relative phase error     : %+.6e\n", (final_geometry.phase_integral - mass0) / mass0)
    @printf("Final min eig(sigma)     : %.6e\n", final_sigma.minimum_eigenvalue)
    @printf("Final max |det(sigma)-1| : %.6e\n", final_sigma.maximum_abs_determinant_error)
    @printf("Final non-SPD cells      : %d\n", final_sigma.non_spd_count)
    println("Response CSV             : $response_path")
    println("Phase diagnostics        : $phase_path")
    println("Tensor diagnostics       : $tensor_path")
    println("Flow diagnostics         : $flow_path")
    println("Numerics diagnostics     : $numerics_path")
    println("Snapshots                : $snapshot_dir")
    println("Run metadata             : ", joinpath(output_dir, "run_parameters.txt"))

    return (
        output_dir=output_dir,
        trajectory_path=trajectory_path,
        response_path=response_path,
        phase_diagnostics_path=phase_path,
        tensor_diagnostics_path=tensor_path,
        flow_diagnostics_path=flow_path,
        numerics_diagnostics_path=numerics_path,
        snapshot_dir=snapshot_dir,
        final_state=state,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = FilamentBenchmarkConfig()
    apply_command_line_overrides!(cfg, ARGS)
    run_benchmark(cfg)
end
