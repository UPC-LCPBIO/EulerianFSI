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

# GR must remain headless on remote/HPC nodes.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
try
    @eval using Plots
catch err
    error(
        "This benchmark generates PNG/PDF figures and requires Plots.jl. " *
        "Install it once with: julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'. " *
        "Original error: $(sprint(showerror, err))"
    )
end

gr()

include(joinpath(@__DIR__, "BenchmarkUtilities.jl"))
using .BenchmarkUtilities

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# CONFIGURATION

Base.@kwdef mutable struct BenchmarkConfig
    # Geometry and mesh.
    Lx::Float64 = 1.0
    Ly::Float64 = 1.0
    Nx::Int = 128
    Ny::Int = 128

    # Sugiyama circular solid.
    center_x::Float64 = 0.60
    center_y::Float64 = 0.50
    radius::Float64 = 0.20

    # Lid-driven cavity and material parameters.
    U_lid::Float64 = 1.0
    density::Float64 = 1.0
    Reynolds::Float64 = 100.0
    viscosity::Float64 = 0.01             # mu/rho = 1/Re for rho=U=L=1
    elastic_modulus::Float64 = 0.10        # G = 2*c1 = 0.1

    # Diffuse-interface numerical parameters. Current CDI uses a grid-scaled
    # interface by default. Set interface_epsilon
    # to a positive value, or pass --epsilon=..., for a fixed-epsilon mesh study.
    Pe_gamma::Float64 = 0.25
    interface_epsilon::Float64 = NaN
    epsilon_over_h::Float64 = 2.0
    normal_regularization::Float64 = 1e-6

    # Phase-field strategy. `current_cdi` is the historical EulerianFSI baseline.
    # ACDI and UWCDI use the spatial discretisations from Jain (2022) and
    # Pirozzoli et al. (2025), respectively, but are advanced by the same
    # EulerianFSI IMEX RK scheme in coupled runs.
    phase_field::Symbol = :current_cdi
    acdi_epsilon_over_h::Float64 = 0.51
    acdi_gamma::Float64 = 1.0
    uwcdi_alpha::Float64 = 1.0
    uwcdi_normal::Symbol = :standard
    # UWCDI normally starts from a sharp indicator in the present comparison.
    # `:tanh` is provided as a controlled sensitivity test reproducing the
    # diffuse initialization used by the legacy lid-driven-cavity solver:
    # phi = 0.5 * [1 - tanh(d/(alpha*h))], with alpha=1 by default.
    uwcdi_init::Symbol = :sharp
    uwcdi_init_epsilon_over_h::Float64 = 1.0

    # Numerical extension of the structural tensor. This is distinct from the
    # physical constitutive relaxation, which remains OFF in this benchmark.
    sigma_extension_rate::Float64 = 20.0
    sigma_extension_exponent::Float64 = 4.0
    sigma_safety_cutoff::Float64 = 1e-6

    # Time interval.
    T::Float64 = 20.0

    # Explicit stability restrictions.
    # Sugiyama uses CFL <= 0.1 based on max(advection speed, elastic wave speed).
    cfl_hyperbolic::Float64 = 0.10
    # These two values are the same conservative restrictions used in the
    # repository Pe_Gamma reversible-vortex analysis.
    cfl_interface_compression::Float64 = 0.25
    cfl_interface_diffusion::Float64 = 0.50
    dt_max::Float64 = 5e-3
    dt_min_physical::Float64 = 1e-8

    # Output cadence.
    trajectory_interval::Float64 = 0.02
    diagnostics_interval::Float64 = 0.10
    terminal_update_seconds::Float64 = 5.0

    # Exact times from Sugiyama Fig. 10, followed by additional late-time
    # states used for post-processing without rerunning the benchmark.
    snapshot_times::Vector{Float64} = [
        0.00, 1.17, 2.34, 3.52, 4.69, 5.86, 7.03, 8.20,
        10.00, 12.00, 14.00, 16.00, 18.00, 20.00,
    ]

    # Diagnostic definitions.
    deformation_mode_max::Int = 12
    sigma_core_cutoff::Float64 = 0.90
    sigma_interface_cutoff::Float64 = 0.10

    # Plot appearance. The value is only a common scale target; if a snapshot
    # exceeds it the script expands the upper color limit rather than clipping.
    snapshot_speed_scale::Float64 = 1.0

    # Linear solvers.
    velocity_reltol::Float64 = 1e-8
    velocity_maxiter::Int = 800
    velocity_restart::Int = 100
    projection_reltol::Float64 = 1e-10
    projection_maxiter::Int = 3000
    projection_restart::Int = 50
    use_projection_amg::Bool = true

    # Files.
    output_root::String = joinpath(
        ROOT, "outputs", "benchmarks", "sugiyama_lid_driven_cavity"
    )
    reference_csv::String = joinpath(
        ROOT, "data", "benchmarks", "sugiyama_lid_driven_cavity_1024_digitized.csv"
    )
end

struct SnapshotRecord
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

# COMMAND-LINE OVERRIDES

function apply_command_line_overrides!(cfg::BenchmarkConfig, args)
    for arg in args
        if startswith(arg, "--N=")
            value = parse(Int, split(arg, "=", limit=2)[2])
            cfg.Nx = value
            cfg.Ny = value
        elseif startswith(arg, "--T=")
            cfg.T = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--epsilon=")
            cfg.interface_epsilon = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--epsilon-over-h=")
            cfg.epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
            cfg.interface_epsilon = NaN
        elseif startswith(arg, "--phase-field=")
            cfg.phase_field = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--acdi-epsilon-over-h=")
            cfg.acdi_epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--acdi-gamma=")
            cfg.acdi_gamma = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--uwcdi-alpha=")
            cfg.uwcdi_alpha = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--uwcdi-normal=")
            cfg.uwcdi_normal = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--uwcdi-init=")
            cfg.uwcdi_init = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--uwcdi-init-epsilon-over-h=")
            cfg.uwcdi_init_epsilon_over_h = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sigma-extension-rate=")
            cfg.sigma_extension_rate = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sigma-extension-exponent=")
            cfg.sigma_extension_exponent = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sigma-safety-cutoff=")
            cfg.sigma_safety_cutoff = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--reference=")
            cfg.reference_csv = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-root=")
            cfg.output_root = abspath(split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("Unknown command-line option: $arg"))
        end
    end
    return cfg
end

function validate_config(cfg::BenchmarkConfig)
    cfg.Lx > 0 || throw(ArgumentError("Lx must be positive."))
    cfg.Ly > 0 || throw(ArgumentError("Ly must be positive."))
    cfg.Nx >= 8 || throw(ArgumentError("Nx must be at least 8."))
    cfg.Ny >= 8 || throw(ArgumentError("Ny must be at least 8."))
    cfg.T > 0 || throw(ArgumentError("T must be positive."))
    cfg.radius > 0 || throw(ArgumentError("radius must be positive."))
    cfg.Reynolds > 0 || throw(ArgumentError("Reynolds must be positive."))
    cfg.viscosity > 0 || throw(ArgumentError("viscosity must be positive."))
    cfg.elastic_modulus >= 0 || throw(ArgumentError("elastic_modulus must be non-negative."))
    cfg.Pe_gamma > 0 || throw(ArgumentError("Pe_gamma must be positive."))
    cfg.phase_field in (:current_cdi, :acdi, :uwcdi) || throw(ArgumentError(
        "phase_field must be :current_cdi, :acdi, or :uwcdi."
    ))
    cfg.acdi_epsilon_over_h > 0.5 || throw(ArgumentError(
        "acdi_epsilon_over_h must be > 0.5 for the Jain ACDI setting."
    ))
    cfg.acdi_gamma >= 0 || throw(ArgumentError("acdi_gamma must be non-negative."))
    cfg.uwcdi_alpha >= 1 || throw(ArgumentError("uwcdi_alpha must be >= 1."))
    cfg.uwcdi_normal in (:standard, :youngs, :smoothed9) || throw(ArgumentError(
        "uwcdi_normal must be :standard, :youngs, or :smoothed9."
    ))
    cfg.uwcdi_init in (:sharp, :tanh) || throw(ArgumentError(
        "uwcdi_init must be :sharp or :tanh."
    ))
    cfg.uwcdi_init_epsilon_over_h > 0 || throw(ArgumentError(
        "uwcdi_init_epsilon_over_h must be positive."
    ))
    (isfinite(cfg.interface_epsilon) ? cfg.interface_epsilon > 0 : cfg.epsilon_over_h > 0) || throw(ArgumentError(
        "Use a positive --epsilon or --epsilon-over-h."
    ))
    cfg.sigma_extension_rate >= 0 || throw(ArgumentError("sigma_extension_rate must be non-negative."))
    cfg.sigma_extension_exponent > 0 || throw(ArgumentError("sigma_extension_exponent must be positive."))
    0 <= cfg.sigma_safety_cutoff < 1 || throw(ArgumentError("sigma_safety_cutoff must lie in [0,1)."))
    cfg.trajectory_interval > 0 || throw(ArgumentError("trajectory_interval must be positive."))
    cfg.diagnostics_interval > 0 || throw(ArgumentError("diagnostics_interval must be positive."))
    cfg.deformation_mode_max >= 0 || throw(ArgumentError("deformation_mode_max must be non-negative."))
    0.0 < cfg.sigma_interface_cutoff < cfg.sigma_core_cutoff <= 1.0 || throw(ArgumentError(
        "Require 0 < sigma_interface_cutoff < sigma_core_cutoff <= 1."
    ))
    cfg.sigma_safety_cutoff < cfg.sigma_interface_cutoff || throw(ArgumentError(
        "sigma_safety_cutoff must be smaller than sigma_interface_cutoff."
    ))
    return nothing
end

# OUTPUT DIRECTORY AND REFERENCE DATA

function prepare_output_directory(cfg::BenchmarkConfig, interface_info)
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
        @sprintf("N%d", cfg.Nx);
        subgroup=subgroup,
    )
end

function resolve_reference_path(cfg::BenchmarkConfig)
    candidates = String[
        cfg.reference_csv,
        joinpath(ROOT, "sugiyama_lid_driven_cavity_1024_digitized.csv"),
        joinpath(ROOT, "sugiyama_digitized.csv"),
        joinpath(ROOT, "sugiyama_digitized(1).csv"),
        joinpath(ROOT, "data", "benchmarks", "sugiyama_digitized(1).csv"),
    ]

    for path in candidates
        isempty(path) && continue
        isfile(path) && return abspath(path)
    end
    return nothing
end

function read_xy_reference(path::AbstractString)
    x = Float64[]
    y = Float64[]

    for (line_number, line) in enumerate(eachline(path))
        stripped = strip(line)
        isempty(stripped) && continue

        columns = split(stripped, ',')
        length(columns) >= 2 || continue

        x_value = tryparse(Float64, strip(columns[1]))
        y_value = tryparse(Float64, strip(columns[2]))

        # Header lines are skipped naturally because tryparse returns nothing.
        if x_value !== nothing && y_value !== nothing
            push!(x, x_value)
            push!(y, y_value)
        elseif line_number > 1
            @warn "Skipping non-numeric reference row" line_number stripped
        end
    end

    isempty(x) && error("No numeric (x,y) data were found in reference file: $path")
    return x, y
end

# BOUNDARY CONDITIONS, MODELS, AND INITIAL STATE

zero_neumann_bc() = BoundaryConditionSetup(
    Neumann(0.0), Neumann(0.0), Neumann(0.0), Neumann(0.0)
)

function build_boundary_conditions(cfg::BenchmarkConfig)
    wall = Dirichlet(0.0)
    lid = Dirichlet(cfg.U_lid)
    noflux = zero_neumann_bc()

    # Velocity boundary order: left, right, bottom, top.
    bc_U = BoundaryConditionSetup(wall, wall, wall, lid)
    bc_V = BoundaryConditionSetup(wall, wall, wall, wall)

    # The correction potential uses homogeneous Neumann boundaries. The
    # projection module fixes one algebraic degree of freedom to remove the
    # constant nullspace while leaving the physical operator unchanged.
    bc_correction = noflux

    bc_phi = noflux
    bc_sigma = (xx=noflux, yy=noflux, xy=noflux)

    return FSIBoundaryConditions(
        bc_U,
        bc_V,
        bc_correction,
        bc_phi,
        bc_sigma,
    )
end

function phase_field_setup(cfg::BenchmarkConfig, h_interface::Float64)
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
            # Legacy diffuse initialization: phi = 0.5*(1-tanh(d/epsilon)),
            # with epsilon/h = 1 for the original alpha=1 setting.
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

function build_models(
    cfg::BenchmarkConfig,
    setup,
)
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
            dynamic_gamma=false,
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
        body_force=NoBodyForce(),
    )
end
function build_numerics(cfg::BenchmarkConfig)
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

function sharp_circle_indicator(cfg::BenchmarkConfig, grid::MACGrid)
    phi = zeros(Float64, cfg.Nx, cfg.Ny)
    @inbounds for j in 1:cfg.Ny, i in 1:cfg.Nx
        radius = hypot(grid.x_p[i] - cfg.center_x, grid.y_p[j] - cfg.center_y)
        phi[i, j] = radius <= cfg.radius ? 1.0 : 0.0
    end
    return phi
end

function initial_state(cfg::BenchmarkConfig, grid::MACGrid, setup)
    phi = if setup.initialization_mode === :sharp_indicator
        sharp_circle_indicator(cfg, grid)
    else
        initialize_circle(
            grid,
            (cfg.center_x, cfg.center_y),
            cfg.radius,
            setup.initialization_info.epsilon;
            periodic_x=false,
            periodic_y=false,
        )
    end

    return FSIState(
        zeros(Float64, cfg.Nx + 1, cfg.Ny),
        zeros(Float64, cfg.Nx, cfg.Ny + 1),
        phi,
        ones(Float64, cfg.Nx, cfg.Ny),        # sigma_xx = 1
        ones(Float64, cfg.Nx, cfg.Ny),        # sigma_yy = 1
        zeros(Float64, cfg.Nx, cfg.Ny),       # sigma_xy = 0
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

    x_moment = 0.0
    y_moment = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        weight = phi[i, j] * cell_area(metrics, i, j)
        x_moment += grid.x_p[i] * weight
        y_moment += grid.y_p[j] * weight
    end

    return x_moment / mass, y_moment / mass, mass
end

function cell_center_velocity(state::FSIState)
    Nx, Ny = size(state.phi)
    Uc = Matrix{Float64}(undef, Nx, Ny)
    Vc = Matrix{Float64}(undef, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        Uc[i, j] = 0.5 * (state.U[i, j] + state.U[i + 1, j])
        Vc[i, j] = 0.5 * (state.V[i, j] + state.V[i, j + 1])
    end
    return Uc, Vc
end

function speed_field(state::FSIState)
    Uc, Vc = cell_center_velocity(state)
    return sqrt.(Uc .^ 2 .+ Vc .^ 2)
end

function capture_snapshot_record(
    t::Real,
    state::FSIState,
    grid::MACGrid,
    metrics::MeshMetrics,
)
    xc, yc, _ = phase_centroid(state.phi, grid, metrics)
    u, v = cell_center_velocity(state)
    return SnapshotRecord(
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

function write_snapshot_fields_csv(
    path::AbstractString,
    record::SnapshotRecord,
    grid::MACGrid,
    cfg::BenchmarkConfig,
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
                cfg.elastic_modulus,
            )
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
                record.phi[i, j] * cfg.elastic_modulus * (record.sigma_xx[i, j] - 1.0),
                record.phi[i, j] * cfg.elastic_modulus * (record.sigma_yy[i, j] - 1.0),
                record.phi[i, j] * cfg.elastic_modulus * record.sigma_xy[i, j],
                tensor.effective_stress_frobenius,
                4.0 * clamp(record.phi[i, j], 0.0, 1.0) * (1.0 - clamp(record.phi[i, j], 0.0, 1.0)),
            )
        end
    end
    return nothing
end

function maximum_speed(state::FSIState)
    Uc, Vc = cell_center_velocity(state)
    return maximum(sqrt.(Uc .^ 2 .+ Vc .^ 2))
end

function current_divergence_linf(state::FSIState, metrics::MeshMetrics)
    return maximum(abs, divergence(state.U, state.V, metrics))
end

function initial_sigma_diagnostics(state::FSIState)
    return structure_tensor_diagnostics(
        state.sigma_xx,
        state.sigma_yy,
        state.sigma_xy,
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
    nmax::Int,
)
    xc, yc, mass = phase_centroid(state.phi, grid, metrics)
    gx, gy = phase_center_gradient(state.phi, metrics, bc_phi)

    perimeter_proxy = 0.0
    mixed_layer_measure = 0.0
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    rc = zeros(Float64, nmax)
    rs = zeros(Float64, nmax)

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        area = cell_area(metrics, i, j)
        phi = state.phi[i, j]
        phi_bounded = clamp(phi, 0.0, 1.0)
        gradmag = hypot(gx[i, j], gy[i, j])
        interface_weight = area * gradmag

        perimeter_proxy += interface_weight
        mixed_layer_measure += area * 4.0 * phi_bounded * (1.0 - phi_bounded)

        dx = grid.x_p[i] - xc
        dy = grid.y_p[j] - yc
        mass_weight = area * phi
        cxx += mass_weight * dx * dx
        cyy += mass_weight * dy * dy
        cxy += mass_weight * dx * dy

        theta = atan(dy, dx)
        for n in 1:nmax
            rc[n] += interface_weight * cos(n * theta) / pi
            rs[n] += interface_weight * sin(n * theta) / pi
        end
    end

    cxx /= mass
    cyy /= mass
    cxy /= mass
    covariance_gap = hypot(cxx - cyy, 2.0 * cxy)
    covariance_major = 0.5 * (cxx + cyy + covariance_gap)
    covariance_minor = 0.5 * (cxx + cyy - covariance_gap)
    aspect_ratio = covariance_minor > 0.0 ? sqrt(covariance_major / covariance_minor) : Inf
    orientation = covariance_gap > 100.0 * eps(Float64) ?
        0.5 * atan(2.0 * cxy, cxx - cyy) : 0.0

    r0 = perimeter_proxy / (2.0 * pi)
    amplitudes = hypot.(rc, rs)
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
        orientation_radians=orientation,
        R0=r0,
        Rc=rc,
        Rs=rs,
        amplitudes=amplitudes,
    )
end

function phase_bound_diagnostics(phi)
    phi_min = minimum(phi)
    phi_max = maximum(phi)
    lower_violation = max(0.0, -phi_min)
    upper_violation = max(0.0, phi_max - 1.0)
    out_of_bounds = count(value -> value < 0.0 || value > 1.0, phi)
    return (
        minimum=phi_min,
        maximum=phi_max,
        lower_violation=lower_violation,
        upper_violation=upper_violation,
        out_of_bounds_cells=out_of_bounds,
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
    divergence_linf = maximum(abs, div)
    divergence_l2_squared = 0.0
    volume = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        area = cell_area(metrics, i, j)
        divergence_l2_squared += area * div[i, j]^2
        volume += area
    end
    divergence_l2 = sqrt(divergence_l2_squared / volume)

    return (
        maximum_speed=max_speed,
        kinetic_energy=kinetic_energy,
        divergence_linf=divergence_linf,
        divergence_l2=divergence_l2,
    )
end

function tensor_region_diagnostics(
    state::FSIState,
    metrics::MeshMetrics,
    cfg::BenchmarkConfig,
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
            cfg.elastic_modulus,
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
        if !(isfinite(tensor.lambda_min) && tensor.lambda_min > 0.0)
            non_spd_count += 1
        end
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
    cfg::BenchmarkConfig,
)
    core = tensor_region_diagnostics(
        state, metrics, cfg, phi -> phi >= cfg.sigma_core_cutoff
    )
    interface = tensor_region_diagnostics(
        state, metrics, cfg,
        phi -> cfg.sigma_interface_cutoff < phi < cfg.sigma_core_cutoff,
    )
    tail = tensor_region_diagnostics(
        state, metrics, cfg,
        phi -> cfg.sigma_safety_cutoff < phi <= cfg.sigma_interface_cutoff,
    )
    global_region = tensor_region_diagnostics(state, metrics, cfg, phi -> true)

    return (
        global_region=global_region,
        core=core,
        interface=interface,
        tail=tail,
    )
end

# TIMESTEP SELECTION

function comparison_timestep_bounds(
    cfg::BenchmarkConfig,
    state::FSIState,
    metrics::MeshMetrics,
    setup,
)
    hx = minimum(metrics.dx_c)
    hy = minimum(metrics.dy_c)
    hmin = min(hx, hy)

    max_advection_speed = max(
        cfg.U_lid,
        maximum(abs, state.U),
        maximum(abs, state.V),
    )
    elastic_wave_speed = sqrt(cfg.elastic_modulus / cfg.density)
    hyperbolic_speed = max(max_advection_speed, elastic_wave_speed, eps(Float64))
    dt_hyperbolic = cfg.cfl_hyperbolic * hmin / hyperbolic_speed

    # Current CDI bound. This is included for every run so that the principal
    # comparison uses the same stability envelope for all phase-field methods.
    inverse_pe_gamma = 1.0 / cfg.Pe_gamma
    dt_cdi_compression = cfg.cfl_interface_compression * hmin / inverse_pe_gamma
    nu_cdi = 0.5 * setup.cdi_epsilon * inverse_pe_gamma
    dt_cdi_diffusion_limit = 1.0 / (
        2.0 * nu_cdi * (1.0 / hx^2 + 1.0 / hy^2)
    )
    dt_cdi_diffusion = cfg.cfl_interface_diffusion * dt_cdi_diffusion_limit

    # ACDI bound with Gamma(t)=gamma_scale*|u|_max.
    gamma_acdi = cfg.acdi_gamma
    dt_acdi_compression = cfg.cfl_interface_compression * hmin /
        max(gamma_acdi, eps(Float64))
    nu_acdi = gamma_acdi * setup.acdi_epsilon
    dt_acdi_diffusion = if nu_acdi > 0.0
        cfg.cfl_interface_diffusion / (
            2.0 * nu_acdi * (1.0 / hx^2 + 1.0 / hy^2)
        )
    else
        Inf
    end

    # Published UWCDI explicit-Euler monotonicity bound. It is retained as a
    # conservative spatial stability restriction although coupled runs use the
    # common EulerianFSI Runge--Kutta time integrator.
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
    println(io, "time,x_centroid,y_centroid,phase_integral,phase_integral_ratio")
end

function write_trajectory_row(io, t, xc, yc, mass, mass0)
    @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e\n", t, xc, yc, mass, mass / mass0)
    flush(io)
end

function write_diagnostics_header(io)
    println(
        io,
        "step,time,dt,x_centroid,y_centroid,phase_integral,relative_phase_integral_error," *
        "phi_min,phi_max,max_speed,divergence_linf,min_sigma_eigenvalue," *
        "max_abs_sigma_det_error,non_spd_cells,projection_constraint_linf"
    )
end

function write_diagnostics_row(
    io,
    step::Int,
    t::Float64,
    dt::Float64,
    xc::Float64,
    yc::Float64,
    mass::Float64,
    mass0::Float64,
    phi_min::Float64,
    phi_max::Float64,
    max_speed::Float64,
    divergence_linf::Float64,
    min_sigma_eigenvalue::Float64,
    max_abs_sigma_det_error::Float64,
    non_spd_cells::Int,
    projection_constraint_linf::Float64,
)
    relative_mass_error = (mass - mass0) / mass0
    @printf(
        io,
        "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d,%.10e\n",
        step,
        t,
        dt,
        xc,
        yc,
        mass,
        relative_mass_error,
        phi_min,
        phi_max,
        max_speed,
        divergence_linf,
        min_sigma_eigenvalue,
        max_abs_sigma_det_error,
        non_spd_cells,
        projection_constraint_linf,
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
        "covariance_major,covariance_minor",
    )
end

function write_phase_diagnostics_row(io, t, geometry, bounds, mass0, geometric_area)
    area_bias = geometry.phase_integral - geometric_area
    relative_area_bias = area_bias / geometric_area
    @printf(
        io,
        "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
        t,
        geometry.phase_integral,
        (geometry.phase_integral - mass0) / mass0,
        geometric_area,
        area_bias,
        relative_area_bias,
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
        geometry.covariance_major,
        geometry.covariance_minor,
    )
    flush(io)
end

function write_deformation_modes_header(io, nmax::Int)
    names = String["time", "R0"]
    for n in 1:nmax
        append!(names, ["Rc$(n)", "Rs$(n)", "R$(n)"])
    end
    append!(names, ["high_mode_amplitude_n6_to_nmax", "high_mode_ratio_n6_to_nmax"])
    println(io, join(names, ','))
end

function write_deformation_modes_row(io, t, geometry, nmax::Int)
    values = String[@sprintf("%.16e", t), @sprintf("%.16e", geometry.R0)]
    for n in 1:nmax
        append!(
            values,
            [
                @sprintf("%.16e", geometry.Rc[n]),
                @sprintf("%.16e", geometry.Rs[n]),
                @sprintf("%.16e", geometry.amplitudes[n]),
            ],
        )
    end

    high_mode_amplitude = nmax >= 6 ? sqrt(sum(abs2, @view geometry.amplitudes[6:nmax])) : 0.0
    high_mode_ratio = abs(geometry.R0) > eps(Float64) ?
        high_mode_amplitude / abs(geometry.R0) : NaN
    append!(
        values,
        [@sprintf("%.16e", high_mode_amplitude), @sprintf("%.16e", high_mode_ratio)],
    )

    println(io, join(values, ','))
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
        "max_advection_speed,elastic_wave_speed,gamma_acdi_scale,gamma_acdi_effective,projection_constraint_linf",
    )
end

function write_numerics_diagnostics_row(io, step, t, dt, timestep, projection_constraint, cfg::BenchmarkConfig)
    @printf(
        io,
        "%d,%.16e,%.16e,%.16e,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
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
        cfg.acdi_gamma,
        timestep.gamma_acdi,
        projection_constraint,
    )
    flush(io)
end

function write_snapshot_index_header(io)
    println(io, "target_time,actual_time,x_centroid,y_centroid,fields_csv,png_file,pdf_file")
end

# FIGURES

function snapshot_plot(
    record::SnapshotRecord,
    cfg::BenchmarkConfig,
    grid::MACGrid;
    colorbar::Bool=true,
    compact_title::Bool=false,
    panel_label::Union{Nothing,String}=nothing,
    show_xlabel::Bool=true,
    show_ylabel::Bool=true,
    show_xticks::Bool=true,
    show_yticks::Bool=true,
    show_centroid::Bool=true,
    vmax_override::Union{Nothing,Float64}=nothing,
)
    vmax_actual = maximum(record.speed)
    vmax = vmax_override === nothing ?
        max(cfg.snapshot_speed_scale, vmax_actual, 1e-12) :
        max(vmax_override, 1e-12)
    levels = range(0.0, vmax; length=41)

    if compact_title
        prefix = panel_label === nothing ? "" : panel_label * "   "
        title_text = prefix * @sprintf("t = %.2f", record.time)
    else
        title_text = @sprintf("N = %d   |   t = %.2f", cfg.Nx, record.time)
    end

    p = contourf(
        grid.x_p,
        grid.y_p,
        permutedims(record.speed);
        levels=levels,
        c=:viridis,
        clims=(0.0, vmax),
        colorbar=colorbar,
        colorbar_title="|v|",
        xlabel=show_xlabel ? "x" : "",
        ylabel=show_ylabel ? "y" : "",
        xticks=show_xticks ? :auto : false,
        yticks=show_yticks ? :auto : false,
        title=title_text,
        aspect_ratio=:equal,
        xlims=(0.0, cfg.Lx),
        ylims=(0.0, cfg.Ly),
        framestyle=:box,
        grid=false,
        minorgrid=false,
        legend=false,
        background_color=:white,
        foreground_color_subplot=:black,
        fontfamily="Times",
        guidefontsize=12,
        tickfontsize=10,
        titlefontsize=12,
        colorbar_titlefontsize=11,
        colorbar_tickfontsize=10,
        tick_direction=:in,
    )

    # Two-layer interface curve: the white outer stroke guarantees contrast
    # against the velocity colormap, while the black inner stroke remains
    # crisp in print and in grayscale reproductions.
    contour!(
        p,
        grid.x_p,
        grid.y_p,
        permutedims(record.phi);
        levels=[0.5],
        linewidth=4.0,
        color=:white,
        label=false,
    )
    contour!(
        p,
        grid.x_p,
        grid.y_p,
        permutedims(record.phi);
        levels=[0.5],
        linewidth=2.0,
        color=:black,
        label=false,
    )

    if show_centroid
        scatter!(
            p,
            [record.x_centroid],
            [record.y_centroid];
            marker=:circle,
            markersize=4.5,
            markercolor=:white,
            markerstrokecolor=:black,
            markerstrokewidth=1.1,
            label=false,
        )
    end

    return p
end

function save_snapshot(
    record::SnapshotRecord,
    cfg::BenchmarkConfig,
    grid::MACGrid,
    snapshot_dir::AbstractString,
)
    tag = replace(@sprintf("%05.2f", record.time), "." => "p")
    stem = "snapshot_t_" * tag
    png_path = joinpath(snapshot_dir, stem * ".png")
    pdf_path = joinpath(snapshot_dir, stem * ".pdf")
    fields_path = joinpath(snapshot_dir, stem * "_fields.csv")
    write_snapshot_fields_csv(fields_path, record, grid, cfg)

    p = snapshot_plot(
        record,
        cfg,
        grid;
        colorbar=true,
        compact_title=false,
        show_centroid=true,
    )
    plot!(
        p;
        size=(780, 700),
        dpi=300,
    )
    savefig(p, png_path)
    savefig(p, pdf_path)
    return png_path, pdf_path, fields_path
end

function save_snapshot_summary(
    records::Vector{SnapshotRecord},
    cfg::BenchmarkConfig,
    grid::MACGrid,
    snapshot_dir::AbstractString,
)
    isempty(records) && return nothing

    n = length(records)
    ncols = min(4, n)
    nrows = ceil(Int, n / ncols)

    # All panels use one common velocity scale so that colors are directly
    # comparable in time. Individual snapshots retain their own colorbar.
    global_vmax = max(
        cfg.snapshot_speed_scale,
        maximum(maximum(record.speed) for record in records),
        1e-12,
    )

    panels = Plots.Plot[]
    for (k, record) in enumerate(records)
        row = div(k - 1, ncols) + 1
        col = mod(k - 1, ncols) + 1
        panel_label = "(" * string(Char(Int('a') + k - 1)) * ")"

        push!(
            panels,
            snapshot_plot(
                record,
                cfg,
                grid;
                colorbar=false,
                compact_title=true,
                panel_label=panel_label,
                show_xlabel=(row == nrows),
                show_ylabel=(col == 1),
                show_xticks=(row == nrows),
                show_yticks=(col == 1),
                show_centroid=false,
                vmax_override=global_vmax,
            ),
        )
    end

    scale_text = @sprintf(
        "Velocity magnitude |v|   (common scale: 0 to %.2f)   |   interface: phi = 0.5",
        global_vmax,
    )

    summary = plot(
        panels...;
        layout=(nrows, ncols),
        size=(360 * ncols, 350 * nrows),
        dpi=300,
        plot_title=scale_text,
        plot_titlefontsize=13,
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

function read_simulation_trajectory(path::AbstractString)
    t = Float64[]
    x = Float64[]
    y = Float64[]

    for line in Iterators.drop(eachline(path), 1)
        columns = split(strip(line), ',')
        length(columns) >= 3 || continue
        push!(t, parse(Float64, columns[1]))
        push!(x, parse(Float64, columns[2]))
        push!(y, parse(Float64, columns[3]))
    end
    return t, x, y
end

function save_centroid_comparison(
    trajectory_path::AbstractString,
    reference_path::AbstractString,
    output_dir::AbstractString,
    cfg::BenchmarkConfig,
)
    _, x_sim, y_sim = read_simulation_trajectory(trajectory_path)
    x_ref, y_ref = read_xy_reference(reference_path)

    p = plot(
        x_ref,
        y_ref;
        linewidth=2.2,
        label="Sugiyama 1024x1024 (digitized)",
        xlabel="x_c",
        ylabel="y_c",
        title=@sprintf("Centroid trajectory comparison | N=%d", cfg.Nx),
        aspect_ratio=:equal,
        framestyle=:box,
        legend=:best,
        size=(820, 720),
    )
    plot!(
        p,
        x_sim,
        y_sim;
        linewidth=2.2,
        label=@sprintf("Present model %dx%d", cfg.Nx, cfg.Ny),
    )
    scatter!(p, [x_sim[1]], [y_sim[1]]; marker=:circle, markersize=5, label="initial")

    png_path = joinpath(output_dir, "centroid_comparison.png")
    pdf_path = joinpath(output_dir, "centroid_comparison.pdf")
    savefig(p, png_path)
    savefig(p, pdf_path)
    return png_path, pdf_path
end

# RUN METADATA

function write_run_parameters(
    path::AbstractString,
    cfg::BenchmarkConfig,
    grid::MACGrid,
    metrics::MeshMetrics,
    setup,
    reference_path,
    initial_bounds,
)
    open(path, "w") do io
        println(io, "Sugiyama lid-driven cavity benchmark")
        println(io, "====================================")
        println(io, "Created: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Julia version: ", VERSION)
        println(io, "Julia threads: ", Threads.nthreads())
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        println(io, "Command: ", PROGRAM_FILE, isempty(ARGS) ? "" : " " * join(ARGS, " "))
        println(io)

        println(io, "Physical benchmark")
        println(io, "------------------")
        @printf(io, "Domain: %.8g x %.8g\n", cfg.Lx, cfg.Ly)
        @printf(io, "Grid: %d x %d uniform MAC cells\n", cfg.Nx, cfg.Ny)
        @printf(io, "Initial center: (%.8g, %.8g)\n", cfg.center_x, cfg.center_y)
        @printf(io, "Initial radius: %.8g\n", cfg.radius)
        @printf(io, "Lid speed: %.8g\n", cfg.U_lid)
        @printf(io, "Density: %.8g\n", cfg.density)
        @printf(io, "Reynolds: %.8g\n", cfg.Reynolds)
        @printf(io, "Viscosity coefficient: %.8g\n", cfg.viscosity)
        @printf(io, "Neo-Hookean shear modulus G: %.8g\n", cfg.elastic_modulus)
        println(io, "Tensor relaxation: OFF (pure elastic limit)")
        println(io, "Capillarity: OFF")
        println(io)

        println(io, "Diffuse-interface numerics")
        println(io, "--------------------------")
        println(io, "phase-field method: ", cfg.phase_field)
        @printf(io, "Current CDI Pe_Gamma: %.8g\n", cfg.Pe_gamma)
        @printf(io, "Current CDI epsilon/h: %.8g\n", setup.cdi_epsilon / conservative_mesh_spacing(metrics))
        @printf(io, "Current CDI epsilon: %.10e\n", setup.cdi_epsilon)
        @printf(io, "ACDI epsilon_Jain/h: %.8g\n", cfg.acdi_epsilon_over_h)
        @printf(io, "ACDI epsilon_Jain: %.10e\n", setup.acdi_epsilon)
        @printf(io, "ACDI Gamma (fixed): %.8g\n", cfg.acdi_gamma)
        @printf(io, "UWCDI alpha: %.8g\n", cfg.uwcdi_alpha)
        println(io, "UWCDI normal treatment: ", cfg.uwcdi_normal)
        println(io, "UWCDI initialization: ", cfg.uwcdi_init)
        @printf(io, "UWCDI initialization epsilon/h (when tanh): %.8g\n", cfg.uwcdi_init_epsilon_over_h)
        println(io, "Initial geometry policy: common phi=0.5 radius; selected method-specific interface profile")
        println(io, "Phase-volume matching: OFF (initial integral bias is documented, not corrected)")
        println(io, "Selected initialization mode: ", setup.initialization_mode)
        println(io, "Selected initialization interface mode: ", setup.initialization_info.mode)
        @printf(io, "Selected initialization epsilon/h: %.8g\n", setup.initialization_info.epsilon_over_h)
        @printf(io, "Selected initialization epsilon: %.10e\n", setup.initialization_info.epsilon)
        @printf(io, "Selected initialization 5-95 width: %.10e\n", setup.initialization_info.width_5_95)
        @printf(io, "Selected initialization 5-95 cells: %.4f\n", setup.initialization_info.cells_5_95)
        @printf(io, "sigma extension rate kappa0: %.8g\n", cfg.sigma_extension_rate)
        @printf(io, "sigma extension exponent m: %.8g\n", cfg.sigma_extension_exponent)
        @printf(io, "sigma far-fluid safety cutoff: %.8g\n", cfg.sigma_safety_cutoff)
        @printf(io, "normal regularization delta_n: %.8g\n", cfg.normal_regularization)
        @printf(io, "sigma core cutoff: %.8g\n", cfg.sigma_core_cutoff)
        @printf(io, "sigma interface lower cutoff: %.8g\n", cfg.sigma_interface_cutoff)
        @printf(io, "deformation modes retained: 0..%d\n", cfg.deformation_mode_max)
        println(io)

        println(io, "Time integration")
        println(io, "----------------")
        @printf(io, "Final time: %.8g\n", cfg.T)
        @printf(io, "CFL hyperbolic: %.8g\n", cfg.cfl_hyperbolic)
        @printf(io, "CFL interface compression: %.8g\n", cfg.cfl_interface_compression)
        @printf(io, "CFL interface diffusion: %.8g\n", cfg.cfl_interface_diffusion)
        @printf(io, "dt_max: %.10e\n", cfg.dt_max)
        println(io, "Comparison timestep policy: common stability envelope for CDI, ACDI and UWCDI")
        @printf(io, "Initial dt hyperbolic bound: %.10e\n", initial_bounds.hyperbolic)
        @printf(io, "Initial CDI compression bound: %.10e\n", initial_bounds.cdi_compression)
        @printf(io, "Initial CDI diffusion bound: %.10e\n", initial_bounds.cdi_diffusion)
        @printf(io, "Initial ACDI compression bound: %.10e\n", initial_bounds.acdi_compression)
        @printf(io, "Initial ACDI diffusion bound: %.10e\n", initial_bounds.acdi_diffusion)
        @printf(io, "Initial UWCDI bound: %.10e\n", initial_bounds.uwcdi)
        @printf(io, "Initial selected dt bound: %.10e\n", initial_bounds.selected)
        println(io, "Initial limiting constraint: ", initial_bounds.limiting_constraint)
        println(io)

        println(io, "Mesh")
        println(io, "----")
        @printf(io, "min(dx): %.10e\n", minimum(metrics.dx_c))
        @printf(io, "min(dy): %.10e\n", minimum(metrics.dy_c))
        @printf(io, "nominal dx: %.10e\n", grid.dx_nominal)
        @printf(io, "nominal dy: %.10e\n", grid.dy_nominal)
        println(io)

        println(io, "Output")
        println(io, "------")
        @printf(io, "Trajectory interval: %.8g\n", cfg.trajectory_interval)
        @printf(io, "Diagnostics interval: %.8g\n", cfg.diagnostics_interval)
        println(io, "Snapshot times: ", join(cfg.snapshot_times, ", "))
        println(io, "Reference CSV: ", reference_path === nothing ? "NOT FOUND" : reference_path)
    end
end

# MAIN SIMULATION

function run_benchmark(cfg::BenchmarkConfig)
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
    metrics = build_mesh_metrics(grid; periodic_x=false, periodic_y=false)

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

    # The initial rest state is already divergence free, but use the common
    # initialization path so the benchmark exercises the same constraint logic
    # for additional configurations.
    init_projection = make_velocity_compatible!(state, context, work)

    output_dir = prepare_output_directory(cfg, setup.initialization_info)
    snapshot_dir = joinpath(output_dir, "snapshots")
    mkpath(snapshot_dir)
    reference_path = resolve_reference_path(cfg)
    if reference_path !== nothing
        cp(
            reference_path,
            joinpath(output_dir, "reference_sugiyama_1024_digitized.csv");
            force=true,
        )
    end

    initial_bounds = comparison_timestep_bounds(cfg, state, metrics, setup)
    write_run_parameters(
        joinpath(output_dir, "run_parameters.txt"),
        cfg,
        grid,
        metrics,
        setup,
        reference_path,
        initial_bounds,
    )

    trajectory_path = joinpath(output_dir, "centroid_trajectory.csv")
    diagnostics_path = joinpath(output_dir, "diagnostics.csv")
    phase_diagnostics_path = joinpath(output_dir, "phase_diagnostics.csv")
    deformation_modes_path = joinpath(output_dir, "deformation_modes.csv")
    tensor_diagnostics_path = joinpath(output_dir, "tensor_diagnostics.csv")
    flow_diagnostics_path = joinpath(output_dir, "flow_diagnostics.csv")
    numerics_diagnostics_path = joinpath(output_dir, "numerics_diagnostics.csv")
    snapshot_index_path = joinpath(output_dir, "snapshot_index.csv")

    trajectory_io = open(trajectory_path, "w")
    diagnostics_io = open(diagnostics_path, "w")
    phase_diagnostics_io = open(phase_diagnostics_path, "w")
    deformation_modes_io = open(deformation_modes_path, "w")
    tensor_diagnostics_io = open(tensor_diagnostics_path, "w")
    flow_diagnostics_io = open(flow_diagnostics_path, "w")
    numerics_diagnostics_io = open(numerics_diagnostics_path, "w")
    snapshot_index_io = open(snapshot_index_path, "w")

    write_trajectory_header(trajectory_io)
    write_diagnostics_header(diagnostics_io)
    write_phase_diagnostics_header(phase_diagnostics_io)
    write_deformation_modes_header(deformation_modes_io, cfg.deformation_mode_max)
    write_tensor_diagnostics_header(tensor_diagnostics_io)
    write_flow_diagnostics_header(flow_diagnostics_io)
    write_numerics_diagnostics_header(numerics_diagnostics_io)
    write_snapshot_index_header(snapshot_index_io)

    xc0, yc0, mass0 = phase_centroid(state.phi, grid, metrics)
    geometric_area = pi * cfg.radius^2
    sigma0 = initial_sigma_diagnostics(state)
    div0 = current_divergence_linf(state, metrics)
    geometry0 = phase_geometry_diagnostics(
        state, grid, metrics, bcs.phi, h_interface, cfg.deformation_mode_max
    )
    phase_bounds0 = phase_bound_diagnostics(state.phi)
    tensor0 = tensor_diagnostics_detailed(state, metrics, cfg)
    flow0 = flow_diagnostics(state, metrics, cfg.density)

    write_trajectory_row(trajectory_io, 0.0, xc0, yc0, mass0, mass0)
    write_diagnostics_row(
        diagnostics_io,
        0,
        0.0,
        0.0,
        xc0,
        yc0,
        mass0,
        mass0,
        minimum(state.phi),
        maximum(state.phi),
        maximum_speed(state),
        div0,
        sigma0.minimum_eigenvalue,
        sigma0.maximum_abs_determinant_error,
        sigma0.non_spd_count,
        init_projection.constraint_linf,
    )
    write_phase_diagnostics_row(phase_diagnostics_io, 0.0, geometry0, phase_bounds0, mass0, geometric_area)
    write_deformation_modes_row(
        deformation_modes_io, 0.0, geometry0, cfg.deformation_mode_max
    )
    write_tensor_diagnostics_row(tensor_diagnostics_io, 0.0, tensor0)
    write_flow_diagnostics_row(flow_diagnostics_io, 0.0, flow0)
    write_numerics_diagnostics_row(
        numerics_diagnostics_io, 0, 0.0, 0.0, initial_bounds, init_projection.constraint_linf, cfg
    )

    active_snapshot_times = sort(unique(filter(t -> 0.0 <= t <= cfg.T + 1e-12, cfg.snapshot_times)))
    snapshot_records = SnapshotRecord[]
    snapshot_index = 1

    # Capture any requested t=0 snapshot immediately.
    while snapshot_index <= length(active_snapshot_times) &&
          active_snapshot_times[snapshot_index] <= 1e-12
        record = capture_snapshot_record(0.0, state, grid, metrics)
        push!(snapshot_records, record)
        png_path, pdf_path, fields_path = save_snapshot(record, cfg, grid, snapshot_dir)
        @printf(
            snapshot_index_io,
            "%.10e,%.10e,%.10e,%.10e,%s,%s,%s\n",
            active_snapshot_times[snapshot_index],
            0.0,
            xc0,
            yc0,
            relpath(fields_path, output_dir),
            relpath(png_path, output_dir),
            relpath(pdf_path, output_dir),
        )
        flush(snapshot_index_io)
        snapshot_index += 1
    end

    println()
    println("Sugiyama lid-driven cavity benchmark")
    println("------------------------------------")
    @printf("Grid                 : %d x %d\n", cfg.Nx, cfg.Ny)
    @printf("Pe_Gamma             : %.6g\n", cfg.Pe_gamma)
    println("Initialization        : ", setup.initialization_mode)
    @printf("Initialization epsilon: %.6e\n", setup.initialization_info.epsilon)
    @printf("Geometric disk area   : %.10e\n", geometric_area)
    @printf("Initial phase integral: %.10e\n", mass0)
    @printf("Initial area bias      : %+.6e (%.4f%%)\n", mass0 - geometric_area, 100.0 * (mass0 - geometric_area) / geometric_area)
    @printf("Initial centroid      : (%.8f, %.8f)\n", xc0, yc0)
    @printf("Initial dt bound      : %.6e\n", initial_bounds.selected)
    @printf("  hyperbolic          : %.6e\n", initial_bounds.hyperbolic)
    @printf("  CDI compression     : %.6e\n", initial_bounds.cdi_compression)
    @printf("  CDI diffusion       : %.6e\n", initial_bounds.cdi_diffusion)
    @printf("  ACDI compression    : %.6e\n", initial_bounds.acdi_compression)
    @printf("  ACDI diffusion      : %.6e\n", initial_bounds.acdi_diffusion)
    @printf("  UWCDI               : %.6e\n", initial_bounds.uwcdi)
    println("  limiting constraint : ", initial_bounds.limiting_constraint)
    println("Output directory      : $output_dir")
    if reference_path === nothing
        println("Reference trajectory  : not found; comparison plot will be skipped")
    else
        println("Reference trajectory  : $reference_path")
    end
    println()

    t = 0.0
    step = 0
    next_trajectory_time = cfg.trajectory_interval
    next_diagnostics_time = cfg.diagnostics_interval
    last_trajectory_written = 0.0
    last_diagnostics_written = 0.0
    last_terminal_update = time()
    last_report = nothing

    # Floating-point tolerance for exact output events.
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

            dt_to_event = next_event_time - t
            dt = min(dt_physical, dt_to_event)

            # If a previous roundoff put us exactly on an event, process it below
            # instead of trying to advance by a zero timestep.
            if dt <= event_tol
                t = next_event_time
            else
                report = integrate_step!(state, t, dt, context, work)
                last_report = report
                t += dt
                step += 1

                # Snap to the intended event time when the last operation differs
                # only by floating-point roundoff.
                if abs(t - next_event_time) <= event_tol
                    t = next_event_time
                end
            end

            # Trajectory output.
            if t >= next_trajectory_time - event_tol
                xc, yc, mass = phase_centroid(state.phi, grid, metrics)
                write_trajectory_row(trajectory_io, t, xc, yc, mass, mass0)
                last_trajectory_written = t
                next_trajectory_time += cfg.trajectory_interval
            end

            # Diagnostics output.
            if t >= next_diagnostics_time - event_tol
                geometry = phase_geometry_diagnostics(
                    state, grid, metrics, bcs.phi, h_interface, cfg.deformation_mode_max
                )
                phase_bounds = phase_bound_diagnostics(state.phi)
                tensor = tensor_diagnostics_detailed(state, metrics, cfg)
                flow = flow_diagnostics(state, metrics, cfg.density)
                projection_constraint = last_report === nothing ?
                    init_projection.constraint_linf :
                    last_report.maximum_projection_constraint_linf

                write_diagnostics_row(
                    diagnostics_io,
                    step,
                    t,
                    dt,
                    geometry.x_centroid,
                    geometry.y_centroid,
                    geometry.phase_integral,
                    mass0,
                    phase_bounds.minimum,
                    phase_bounds.maximum,
                    flow.maximum_speed,
                    flow.divergence_linf,
                    tensor.global_region.lambda_min,
                    tensor.global_region.max_abs_determinant_error,
                    tensor.global_region.non_spd_cells,
                    projection_constraint,
                )
                write_phase_diagnostics_row(
                    phase_diagnostics_io, t, geometry, phase_bounds, mass0, geometric_area
                )
                write_deformation_modes_row(
                    deformation_modes_io, t, geometry, cfg.deformation_mode_max
                )
                write_tensor_diagnostics_row(tensor_diagnostics_io, t, tensor)
                write_flow_diagnostics_row(flow_diagnostics_io, t, flow)
                write_numerics_diagnostics_row(
                    numerics_diagnostics_io, step, t, dt, timestep, projection_constraint, cfg
                )

                last_diagnostics_written = t
                next_diagnostics_time += cfg.diagnostics_interval
            end

            # Exact Sugiyama snapshots.
            if snapshot_index <= length(active_snapshot_times) &&
               t >= active_snapshot_times[snapshot_index] - event_tol
                target_time = active_snapshot_times[snapshot_index]
                xc, yc, _ = phase_centroid(state.phi, grid, metrics)
                record = capture_snapshot_record(t, state, grid, metrics)
                push!(snapshot_records, record)
                png_path, pdf_path, fields_path = save_snapshot(record, cfg, grid, snapshot_dir)

                @printf(
                    snapshot_index_io,
                    "%.10e,%.10e,%.10e,%.10e,%s,%s,%s\n",
                    target_time,
                    t,
                    xc,
                    yc,
                    relpath(fields_path, output_dir),
                    relpath(png_path, output_dir),
                    relpath(pdf_path, output_dir),
                )
                flush(snapshot_index_io)
                @printf("Saved snapshot t = %.2f\n", target_time)
                snapshot_index += 1
            end

            # Terminal status.
            wall_now = time()
            if wall_now - last_terminal_update >= cfg.terminal_update_seconds
                xc, yc, mass = phase_centroid(state.phi, grid, metrics)
                sigma_min = last_report === nothing ?
                    sigma0.minimum_eigenvalue :
                    last_report.minimum_sigma_eigenvalue
                non_spd = last_report === nothing ? 0 : last_report.non_spd_cells

                @printf(
                    "[%6.2f%%] step=%d  t=%.6f  dt=%.3e  xc=%.5f  yc=%.5f  mass_err=%+.3e  min_lambda_sigma=%.3e  nonSPD=%d\n",
                    100.0 * t / cfg.T,
                    step,
                    t,
                    dt,
                    xc,
                    yc,
                    (mass - mass0) / mass0,
                    sigma_min,
                    non_spd,
                )
                last_terminal_update = wall_now
            end
        end

        # Guarantee a final trajectory row at T even if T is not an integer
        # multiple of trajectory_interval.
        if abs(t - cfg.T) <= 1e-10
            t = cfg.T
        end
        xc, yc, mass = phase_centroid(state.phi, grid, metrics)
        if abs(last_trajectory_written - t) > event_tol
            write_trajectory_row(trajectory_io, t, xc, yc, mass, mass0)
            last_trajectory_written = t
        end

        if abs(last_diagnostics_written - t) > event_tol
            geometry = phase_geometry_diagnostics(
                state, grid, metrics, bcs.phi, h_interface, cfg.deformation_mode_max
            )
            phase_bounds = phase_bound_diagnostics(state.phi)
            tensor = tensor_diagnostics_detailed(state, metrics, cfg)
            flow = flow_diagnostics(state, metrics, cfg.density)
            timestep = comparison_timestep_bounds(cfg, state, metrics, setup)
            projection_constraint = last_report === nothing ?
                init_projection.constraint_linf :
                last_report.maximum_projection_constraint_linf

            write_diagnostics_row(
                diagnostics_io,
                step,
                t,
                0.0,
                geometry.x_centroid,
                geometry.y_centroid,
                geometry.phase_integral,
                mass0,
                phase_bounds.minimum,
                phase_bounds.maximum,
                flow.maximum_speed,
                flow.divergence_linf,
                tensor.global_region.lambda_min,
                tensor.global_region.max_abs_determinant_error,
                tensor.global_region.non_spd_cells,
                projection_constraint,
            )
            write_phase_diagnostics_row(
                phase_diagnostics_io, t, geometry, phase_bounds, mass0, geometric_area
            )
            write_deformation_modes_row(
                deformation_modes_io, t, geometry, cfg.deformation_mode_max
            )
            write_tensor_diagnostics_row(tensor_diagnostics_io, t, tensor)
            write_flow_diagnostics_row(flow_diagnostics_io, t, flow)
            write_numerics_diagnostics_row(
                numerics_diagnostics_io, step, t, 0.0, timestep, projection_constraint, cfg
            )
            last_diagnostics_written = t
        end

    finally
        close(trajectory_io)
        close(diagnostics_io)
        close(phase_diagnostics_io)
        close(deformation_modes_io)
        close(tensor_diagnostics_io)
        close(flow_diagnostics_io)
        close(numerics_diagnostics_io)
        close(snapshot_index_io)
    end

    # Composite qualitative benchmark figure.
    save_snapshot_summary(snapshot_records, cfg, grid, snapshot_dir)

    # Quantitative trajectory comparison. The digitized Sugiyama CSV contains
    # spatial path coordinates only (x_c,y_c), not time stamps; consequently the
    # plot is a geometric trajectory comparison, not a pointwise-in-time error.
    if reference_path !== nothing
        save_centroid_comparison(
            trajectory_path,
            reference_path,
            output_dir,
            cfg,
        )
    end

    xc_final, yc_final, mass_final = phase_centroid(state.phi, grid, metrics)
    final_record = capture_snapshot_record(t, state, grid, metrics)
    write_snapshot_fields_csv(
        joinpath(output_dir, "final_fields.csv"),
        final_record,
        grid,
        cfg,
    )

    final_sigma = structure_tensor_diagnostics(
        state.sigma_xx,
        state.sigma_yy,
        state.sigma_xy,
    )

    println()
    println("Benchmark completed")
    println("-------------------")
    @printf("Steps                 : %d\n", step)
    @printf("Final time            : %.8f\n", t)
    @printf("Final centroid        : (%.8f, %.8f)\n", xc_final, yc_final)
    @printf("Relative phase error  : %+.6e\n", (mass_final - mass0) / mass0)
    @printf("Final min eig(sigma)  : %.6e\n", final_sigma.minimum_eigenvalue)
    @printf("Final max |det(sigma)-1|: %.6e\n", final_sigma.maximum_abs_determinant_error)
    @printf("Final non-SPD cells       : %d\n", final_sigma.non_spd_count)
    println("Trajectory CSV        : $trajectory_path")
    println("Diagnostics CSV       : $diagnostics_path")
    println("Phase diagnostics     : $phase_diagnostics_path")
    println("Deformation modes     : $deformation_modes_path")
    println("Tensor diagnostics    : $tensor_diagnostics_path")
    println("Flow diagnostics      : $flow_diagnostics_path")
    println("Numerics diagnostics  : $numerics_diagnostics_path")
    println("Snapshots             : $snapshot_dir")
    if reference_path !== nothing
        println("Centroid comparison   : ", joinpath(output_dir, "centroid_comparison.png"))
    end
    println("Run metadata          : ", joinpath(output_dir, "run_parameters.txt"))

    return (
        output_dir=output_dir,
        trajectory_path=trajectory_path,
        diagnostics_path=diagnostics_path,
        phase_diagnostics_path=phase_diagnostics_path,
        deformation_modes_path=deformation_modes_path,
        tensor_diagnostics_path=tensor_diagnostics_path,
        flow_diagnostics_path=flow_diagnostics_path,
        numerics_diagnostics_path=numerics_diagnostics_path,
        snapshot_dir=snapshot_dir,
        final_state=state,
    )
end

# ENTRY POINT

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = BenchmarkConfig()
    apply_command_line_overrides!(cfg, ARGS)
    run_benchmark(cfg)
end
