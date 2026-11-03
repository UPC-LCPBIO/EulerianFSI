raw"""
    TimeIntegrationIMEX

Four-stage IMEX fractional-step time integration for the one-velocity Eulerian
FSI-growth formulation.

The dynamic state contains only

(\mathbf{v},\phi,c,\sigma).

At every stage, the scalar growth source and growth tensor are algebraic fields
constructed from the current stage state,

G_s = \mathcal{G}(\phi_s,c_s),
\qquad
K_{g,s} = \mathcal{K}(G_s),

and the velocity projection enforces

\nabla_h\cdot\mathbf{v}_s = G_s.

There is no auxiliary growth potential, expansion velocity, or second material
velocity. The correction produced by the projection module is retained as a
numerical correction potential `q`; this module does not identify it with the
physical pressure without a separate pressure-recovery derivation.

The explicit tableau is third order, while the implicit tableau used by the
inherited algorithm is only second order. In addition, velocity is advanced by
a fractional-step projection. The method is therefore documented here as a
four-stage IMEX fractional-step scheme without claiming a global formal order.
"""
module TimeIntegrationIMEX

using LinearAlgebra
using SparseArrays
using IterativeSolvers

using ..MeshGeneration
using ..BoundaryConditions
using ..Operators
using ..Projection
using ..MomentumTransport
using ..ResourceTransport
using ..InterfaceTransport
using ..InterfaceRegularization
using ..PhaseFieldMethods
using ..Capillarity
using ..StructuralConstitutive
using ..StructureTensor
using ..GrowthModels
using ..TransportConstitutive

export FSIState,
       FSIBoundaryConditions,
       FSIModels,
       LinearSolverOptions,
       FourStageIMEXFractionalStep,
       IMEXNumerics,
       IMEXContext,
       IMEXWorkspace,
       StepReport,
       build_imex_context,
       allocate_imex_workspace,
       make_velocity_compatible!,
       integrate_step!

# ==============================================================================
# TIME TABLEAU
# ==============================================================================

"""Marker type for the inherited four-stage IMEX fractional-step tableau."""
struct FourStageIMEXFractionalStep end

const A_EXPLICIT = (
    (0.0, 0.0, 0.0, 0.0),
    (8 / 15, 0.0, 0.0, 0.0),
    (1 / 4, 5 / 12, 0.0, 0.0),
    (1 / 4, 0.0, 3 / 4, 0.0),
)

const A_IMPLICIT = (
    (0.0, 0.0, 0.0, 0.0),
    (4 / 15, 4 / 15, 0.0, 0.0),
    (4 / 15, 1 / 3, 1 / 15, 0.0),
    (4 / 15, 1 / 3, 7 / 30, 1 / 6),
)

const STAGE_ABSCISSA = (0.0, 8 / 15, 2 / 3, 1.0)
const NSTAGES = 4
const NHISTORY = 3

# ==============================================================================
# PUBLIC DATA STRUCTURES
# ==============================================================================

"""
    FSIState(U, V, phi, resource, sigma_xx, sigma_yy, sigma_xy)

Dynamic variables advanced by the time integrator. All arrays are stored as
`Float64` matrices. `U` and `V` are the full MAC velocity arrays; the remaining
fields are cell centred.
"""
mutable struct FSIState
    U::Matrix{Float64}
    V::Matrix{Float64}
    phi::Matrix{Float64}
    resource::Matrix{Float64}
    sigma_xx::Matrix{Float64}
    sigma_yy::Matrix{Float64}
    sigma_xy::Matrix{Float64}
end

# Backward compatibility: the 6-array state used by the FTaC benchmarks has
# no transported resource. It is initialized to zero; this state is only
# suitable for configurations without resource-dependent growth.
function FSIState(U, V, phi, sigma_xx, sigma_yy, sigma_xy)
    return FSIState(U, V, phi, zeros(Float64, size(phi)), sigma_xx, sigma_yy, sigma_xy)
end

"""
    FSIBoundaryConditions(U, V, correction, phi, resource, sigma)

Boundary conditions consumed by the one-velocity solver. `correction` refers to
the projection correction potential, not to a recovered physical pressure.
`sigma` is a named tuple with fields `xx`, `yy`, and `xy`.
"""
struct FSIBoundaryConditions{BU,BV,BQ,BP,BC,BS}
    U::BU
    V::BV
    correction::BQ
    phi::BP
    resource::BC
    sigma::BS

    function FSIBoundaryConditions(U::BU, V::BV, correction::BQ, phi::BP, resource::BC, sigma::BS) where {BU,BV,BQ,BP,BC,BS}
        hasproperty(sigma, :xx) || throw(ArgumentError("sigma boundary data require an `xx` entry."))
        hasproperty(sigma, :yy) || throw(ArgumentError("sigma boundary data require a `yy` entry."))
        hasproperty(sigma, :xy) || throw(ArgumentError("sigma boundary data require an `xy` entry."))
        return new{BU,BV,BQ,BP,BC,BS}(U, V, correction, phi, resource, sigma)
    end
end

# Backward compatibility: for no-resource FTaC cases, reuse the phase-field
# boundary setup. For simulations transporting c, always specify its BCs
# explicitly using the 6-argument constructor above.
FSIBoundaryConditions(U, V, correction, phi, sigma) =
    FSIBoundaryConditions(U, V, correction, phi, phi, sigma)

"""
    FSIModels(; kwargs...)

Constitutive and forcing models coordinated by the time integrator. The default
constructor represents a no-growth, inviscid, non-diffusive, non-reactive,
non-elastic and non-capillary system.
"""
struct FSIModels{GS,GK,MU,DI,RC,TR,ES,CAP,PF,BF}
    growth_source::GS
    growth_tensor::GK
    viscosity::MU
    diffusivity::DI
    resource_consumption::RC
    tensor_relaxation::TR
    elastic_stress::ES
    capillarity::CAP
    phase_field_method::PF
    body_force::BF
end

function FSIModels(;
    growth_source = NoGrowth(),
    growth_tensor = IsotropicGrowthTensor(),
    viscosity = NoViscosity(),
    diffusivity = NoDiffusion(),
    resource_consumption = NoResourceConsumption(),
    tensor_relaxation = NoRelaxation(),
    elastic_stress = NoElasticStress(),
    capillarity = NoCapillarity(),
    phase_field_method = nothing,
    interface_flux = NoInterfaceFlux(),
    body_force = NoBodyForce(),
)
    # Compatibility with the original `interface_flux` API. New configurations
    # should select the complete operator through `phase_field_method`.
    pf_method = if phase_field_method === nothing
        LegacySplitPhaseFieldMethod(interface_flux)
    else
        interface_flux isa NoInterfaceFlux || throw(ArgumentError(
            "Specify either phase_field_method or the legacy interface_flux keyword, not both."
        ))
        phase_field_method
    end
    pf_method isa AbstractPhaseFieldMethod || throw(ArgumentError(
        "phase_field_method must be an AbstractPhaseFieldMethod."
    ))

    return FSIModels(
        growth_source,
        growth_tensor,
        viscosity,
        diffusivity,
        resource_consumption,
        tensor_relaxation,
        elastic_stress,
        capillarity,
        pf_method,
        body_force,
    )
end

"""Linear solver configuration for the stage velocity and resource systems."""
struct LinearSolverOptions
    velocity_method::Symbol
    resource_method::Symbol
    reltol::Float64
    maxiter::Int
    restart::Int
    direct_fallback::Bool
    fallback_reltol::Float64
    verbose::Bool
end

function LinearSolverOptions(;
    velocity_method::Symbol = :direct,
    resource_method::Symbol = :gmres,
    reltol::Real = 1e-8,
    maxiter::Integer = 800,
    restart::Integer = 100,
    direct_fallback::Bool = true,
    fallback_reltol::Real = 1e-6,
    verbose::Bool = false,
)
    velocity_method in (:direct, :gmres) || throw(ArgumentError(
        "velocity_method must be :direct or :gmres."
    ))
    resource_method in (:direct, :gmres) || throw(ArgumentError(
        "resource_method must be :direct or :gmres."
    ))
    reltol > 0 || throw(ArgumentError("Linear-solver reltol must be positive."))
    maxiter > 0 || throw(ArgumentError("Linear-solver maxiter must be positive."))
    restart > 0 || throw(ArgumentError("Linear-solver restart must be positive."))
    fallback_reltol > 0 || throw(ArgumentError("fallback_reltol must be positive."))
    return LinearSolverOptions(
        velocity_method,
        resource_method,
        Float64(reltol),
        Int(maxiter),
        Int(restart),
        direct_fallback,
        Float64(fallback_reltol),
        verbose,
    )
end

"""
    IMEXNumerics(; kwargs...)

Numerical choices that are not constitutive physics. No phase clipping or FCT
policy is hidden in this object. Phase-interface regularization is supplied
through `FSIModels.phase_field_method`, while structural-tensor extension through
the diffuse interface is supplied by `structure_extension`.
"""
struct IMEXNumerics{S,LS,SE}
    scheme::S
    linear_solvers::LS
    structure_extension::SE
    projection_reltol::Float64
    projection_maxiter::Int
    projection_restart::Int
    stage1_constraint_atol::Float64
    stage1_constraint_rtol::Float64
    check_stage1_constraint::Bool
end

function IMEXNumerics(;
    scheme = FourStageIMEXFractionalStep(),
    linear_solvers = LinearSolverOptions(),
    structure_extension = NoStructureExtension(),
    projection_reltol::Real = 1e-10,
    projection_maxiter::Integer = 3000,
    projection_restart::Integer = 50,
    stage1_constraint_atol::Real = 5e-8,
    stage1_constraint_rtol::Real = 1e-8,
    check_stage1_constraint::Bool = true,
)
    projection_reltol > 0 || throw(ArgumentError("projection_reltol must be positive."))
    projection_maxiter > 0 || throw(ArgumentError("projection_maxiter must be positive."))
    projection_restart > 0 || throw(ArgumentError("projection_restart must be positive."))
    stage1_constraint_atol >= 0 || throw(ArgumentError("stage1_constraint_atol must be non-negative."))
    stage1_constraint_rtol >= 0 || throw(ArgumentError("stage1_constraint_rtol must be non-negative."))
    return IMEXNumerics(
        scheme,
        linear_solvers,
        structure_extension,
        Float64(projection_reltol),
        Int(projection_maxiter),
        Int(projection_restart),
        Float64(stage1_constraint_atol),
        Float64(stage1_constraint_rtol),
        check_stage1_constraint,
    )
end

"""Preassembled geometry and solver context that does not change within a step."""
struct IMEXContext{M,B,MO,N,P,VG}
    metrics::M
    bcs::B
    models::MO
    numerics::N
    projection_system::P
    velocity_gradients::VG
end

"""
    StepReport

Numerical diagnostics accumulated over one full IMEX step.
"""
struct StepReport
    maximum_projection_constraint_linf::Float64
    maximum_growth_trace_error::Float64
    minimum_phi::Float64
    maximum_phi::Float64
    minimum_resource::Float64
    minimum_sigma_eigenvalue::Float64
    maximum_abs_sigma_determinant_error::Float64
    non_spd_cells::Int
end

# ==============================================================================
# WORKSPACE
# ==============================================================================

mutable struct IMEXWorkspace
    # Time-n base state used by every stage.
    uv_n::Vector{Float64}
    c_n::Vector{Float64}
    phi_n::Matrix{Float64}
    sigma_xx_n::Matrix{Float64}
    sigma_yy_n::Matrix{Float64}
    sigma_xy_n::Matrix{Float64}

    # Current stage state and predictor vectors.
    U_stage::Matrix{Float64}
    V_stage::Matrix{Float64}
    phi_stage::Matrix{Float64}
    c_stage::Matrix{Float64}
    sigma_xx_stage::Matrix{Float64}
    sigma_yy_stage::Matrix{Float64}
    sigma_xy_stage::Matrix{Float64}
    uv_stage::Vector{Float64}
    uv_predictor::Vector{Float64}
    rhs_uv::Vector{Float64}
    rhs_c::Vector{Float64}
    c_predictor::Vector{Float64}

    # Stage histories required by stages 2:4.
    E_uv::Vector{Vector{Float64}}
    I_uv::Vector{Vector{Float64}}
    E_c::Vector{Vector{Float64}}
    I_c::Vector{Vector{Float64}}
    E_phi::Vector{Matrix{Float64}}
    E_sigma_xx::Vector{Matrix{Float64}}
    E_sigma_yy::Vector{Matrix{Float64}}
    E_sigma_xy::Vector{Matrix{Float64}}

    # Algebraic fields shared consistently within a stage.
    G::Matrix{Float64}
    Kg_xx::Matrix{Float64}
    Kg_xy::Matrix{Float64}
    Kg_yx::Matrix{Float64}
    Kg_yy::Matrix{Float64}
    viscosity::Matrix{Float64}
    diffusivity::Matrix{Float64}
    resource_sink::Matrix{Float64}

    # Momentum and constitutive scratch fields.
    force_x::Matrix{Float64}
    force_y::Matrix{Float64}
    scratch_force_x::Matrix{Float64}
    scratch_force_y::Matrix{Float64}
    stress_xx::Matrix{Float64}
    stress_yy::Matrix{Float64}
    stress_xy::Matrix{Float64}

    # Projection correction warm start.
    correction::Matrix{Float64}
end

function allocate_imex_workspace(metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    return IMEXWorkspace(
        zeros(2N),
        zeros(N),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),

        zeros(Nx + 1, Ny),
        zeros(Nx, Ny + 1),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(2N),
        zeros(2N),
        zeros(2N),
        zeros(N),
        zeros(N),

        [zeros(2N) for _ in 1:NHISTORY],
        [zeros(2N) for _ in 1:NHISTORY],
        [zeros(N) for _ in 1:NHISTORY],
        [zeros(N) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],

        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),

        zeros(Nx + 1, Ny),
        zeros(Nx, Ny + 1),
        zeros(Nx + 1, Ny),
        zeros(Nx, Ny + 1),
        zeros(Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny),

        zeros(Nx, Ny),
    )
end

# ==============================================================================
# CONTEXT CONSTRUCTION AND VALIDATION
# ==============================================================================

function _validate_periodicity(metrics::MeshMetrics, bcs::FSIBoundaryConditions)
    for (name, bc) in (
        ("U", bcs.U),
        ("V", bcs.V),
        ("correction", bcs.correction),
        ("phi", bcs.phi),
        ("resource", bcs.resource),
        ("sigma_xx", bcs.sigma.xx),
        ("sigma_yy", bcs.sigma.yy),
        ("sigma_xy", bcs.sigma.xy),
    )
        metrics.periodic_x == is_periodic_x(bc) || throw(ArgumentError(
            "$name boundary conditions do not match MeshMetrics periodicity in x."
        ))
        metrics.periodic_y == is_periodic_y(bc) || throw(ArgumentError(
            "$name boundary conditions do not match MeshMetrics periodicity in y."
        ))
    end
    return nothing
end

function build_imex_context(
    metrics::MeshMetrics,
    bcs::FSIBoundaryConditions,
    models::FSIModels;
    numerics::IMEXNumerics = IMEXNumerics(),
    use_projection_amg::Bool = true,
)
    _validate_periodicity(metrics, bcs)
    projection_system = build_projection_system(
        metrics,
        bcs.correction;
        use_amg=use_projection_amg,
    )

    velocity_gradients = viscosity_active(models.viscosity) ?
        build_velocity_gradient_operators(metrics, bcs.U, bcs.V) : nothing

    return IMEXContext(
        metrics,
        bcs,
        models,
        numerics,
        projection_system,
        velocity_gradients,
    )
end

function _validate_state(state::FSIState, metrics::MeshMetrics)
    expected_U = (metrics.Nx + 1, metrics.Ny)
    expected_V = (metrics.Nx, metrics.Ny + 1)
    expected_C = (metrics.Nx, metrics.Ny)

    size(state.U) == expected_U || throw(DimensionMismatch(
        "state.U must have size $expected_U; received $(size(state.U))."
    ))
    size(state.V) == expected_V || throw(DimensionMismatch(
        "state.V must have size $expected_V; received $(size(state.V))."
    ))

    for (name, field) in (
        ("state.phi", state.phi),
        ("state.resource", state.resource),
        ("state.sigma_xx", state.sigma_xx),
        ("state.sigma_yy", state.sigma_yy),
        ("state.sigma_xy", state.sigma_xy),
    )
        size(field) == expected_C || throw(DimensionMismatch(
            "$name must have size $expected_C; received $(size(field))."
        ))
        all(isfinite, field) || throw(ArgumentError("$name contains non-finite values."))
    end

    all(isfinite, state.U) || throw(ArgumentError("state.U contains non-finite values."))
    all(isfinite, state.V) || throw(ArgumentError("state.V contains non-finite values."))
    return nothing
end

# ==============================================================================
# COMPACT MAC STORAGE HELPERS
# ==============================================================================

@inline _cell_index(i, j, Nx) = i + (j - 1) * Nx

function _extract_compact_velocity!(output, U, V, metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    length(output) == 2N || throw(DimensionMismatch("Compact velocity vector has incorrect length."))
    @inbounds for j in 1:Ny, i in 1:Nx
        k = _cell_index(i, j, Nx)
        output[k] = U[i, j]
        output[N + k] = V[i, j]
    end
    return output
end

function _inject_compact_predictor!(U, V, compact, metrics::MeshMetrics, bcs::FSIBoundaryConditions)
    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    length(compact) == 2N || throw(DimensionMismatch("Compact velocity vector has incorrect length."))

    @inbounds for j in 1:Ny, i in 1:Nx
        k = _cell_index(i, j, Nx)
        U[i, j] = compact[k]
        V[i, j] = compact[N + k]
    end

    if metrics.periodic_x
        U[Nx + 1, :] .= U[1, :]
    elseif bcs.U.right isa Dirichlet
        @inbounds for j in 1:Ny
            U[Nx + 1, j] = Float64(boundary_value(bcs.U.right, j))
        end
    else
        throw(ArgumentError(
            "The compact predictor currently requires a Dirichlet or Periodic U-right boundary."
        ))
    end

    if metrics.periodic_y
        V[:, Ny + 1] .= V[:, 1]
    elseif bcs.V.top isa Dirichlet
        @inbounds for i in 1:Nx
            V[i, Ny + 1] = Float64(boundary_value(bcs.V.top, i))
        end
    else
        throw(ArgumentError(
            "The compact predictor currently requires a Dirichlet or Periodic V-top boundary."
        ))
    end

    return nothing
end

function _copy_cell_to_vector!(output, field)
    length(output) == length(field) || throw(DimensionMismatch("Cell vector has incorrect length."))
    copyto!(output, vec(field))
    return output
end

function _copy_vector_to_cell!(field, input)
    length(input) == length(field) || throw(DimensionMismatch("Cell vector has incorrect length."))
    copyto!(vec(field), input)
    return field
end

# ==============================================================================
# LINEAR ALGEBRA HELPERS
# ==============================================================================

function _relative_residual(A, x, b)
    scale = max(norm(b), eps(Float64))
    return norm(A * x - b) / scale
end

function _solve_linear_system!(x, A, b, x0, options::LinearSolverOptions, method::Symbol, name::AbstractString)
    if method === :direct
        copyto!(x, A \ b)
        return x
    end

    method === :gmres || throw(ArgumentError("Unsupported linear solver method $method."))
    copyto!(x, x0)

    try
        gmres!(
            x,
            A,
            b;
            restart=options.restart,
            reltol=options.reltol,
            maxiter=options.maxiter,
        )
    catch err
        if options.direct_fallback
            @warn "$name GMRES failed; falling back to a sparse direct solve." exception=(err, catch_backtrace())
            copyto!(x, A \ b)
            return x
        end
        rethrow()
    end

    rr = _relative_residual(A, x, b)
    options.verbose && println("$name relative residual = ", rr)

    if !isfinite(rr) || rr > options.fallback_reltol
        if options.direct_fallback
            @warn "$name GMRES residual $rr exceeds fallback tolerance; using direct solve."
            copyto!(x, A \ b)
        else
            throw(ErrorException("$name GMRES residual is too large: $rr"))
        end
    end
    return x
end

function _shifted_identity(matrix::SparseMatrixCSC{Float64,Int}, alpha::Float64)
    n = size(matrix, 1)
    size(matrix, 2) == n || throw(DimensionMismatch("Implicit operator must be square."))
    return spdiagm(0 => ones(Float64, n)) - alpha * matrix
end

function _impose_row_dirichlet(A::SparseMatrixCSC{Float64,Int}, rhs, indices, values)
    isempty(indices) && return A, rhs
    length(indices) == length(values) || throw(DimensionMismatch(
        "Dirichlet index/value lists must have equal length."
    ))

    n = size(A, 1)
    constrained = falses(n)
    @inbounds for index in indices
        1 <= index <= n || throw(BoundsError(constrained, index))
        constrained[index] = true
    end

    rows, cols, vals = findnz(A)
    keep = .!constrained[rows]
    constrained_matrix = sparse(rows[keep], cols[keep], vals[keep], n, n)
    constrained_rhs = copy(rhs)

    @inbounds for (index, value) in zip(indices, values)
        constrained_matrix[index, index] = 1.0
        constrained_rhs[index] = value
    end
    return constrained_matrix, constrained_rhs
end

function _enforce_compact_dirichlet_values!(compact, metrics::MeshMetrics, bcs::FSIBoundaryConditions)
    indices, values = velocity_dirichlet_data(metrics, bcs.U, bcs.V)
    @inbounds for (index, value) in zip(indices, values)
        compact[index] = value
    end
    return compact
end

# ==============================================================================
# STAGE FIELD CONSTRUCTION
# ==============================================================================

function _copy_time_n_state!(work::IMEXWorkspace, state::FSIState, context::IMEXContext)
    _extract_compact_velocity!(work.uv_n, state.U, state.V, context.metrics)
    _copy_cell_to_vector!(work.c_n, state.resource)
    copyto!(work.phi_n, state.phi)
    copyto!(work.sigma_xx_n, state.sigma_xx)
    copyto!(work.sigma_yy_n, state.sigma_yy)
    copyto!(work.sigma_xy_n, state.sigma_xy)

    apply_structure_extension!(
        context.numerics.structure_extension,
        work.sigma_xx_n,
        work.sigma_yy_n,
        work.sigma_xy_n,
        work.phi_n,
    )
    return nothing
end

function _set_stage_one!(work::IMEXWorkspace, state::FSIState)
    copyto!(work.U_stage, state.U)
    copyto!(work.V_stage, state.V)
    copyto!(work.phi_stage, work.phi_n)
    _copy_vector_to_cell!(work.c_stage, work.c_n)
    copyto!(work.sigma_xx_stage, work.sigma_xx_n)
    copyto!(work.sigma_yy_stage, work.sigma_yy_n)
    copyto!(work.sigma_xy_stage, work.sigma_xy_n)
    copyto!(work.uv_stage, work.uv_n)
    return nothing
end

function _build_explicit_stage!(work::IMEXWorkspace, s::Int, dt::Float64, context::IMEXContext)
    copyto!(work.phi_stage, work.phi_n)
    copyto!(work.sigma_xx_stage, work.sigma_xx_n)
    copyto!(work.sigma_yy_stage, work.sigma_yy_n)
    copyto!(work.sigma_xy_stage, work.sigma_xy_n)

    for j in 1:(s - 1)
        a = Float64(A_EXPLICIT[s][j])
        if a != 0.0
            @. work.phi_stage += dt * a * work.E_phi[j]
            @. work.sigma_xx_stage += dt * a * work.E_sigma_xx[j]
            @. work.sigma_yy_stage += dt * a * work.E_sigma_yy[j]
            @. work.sigma_xy_stage += dt * a * work.E_sigma_xy[j]
        end
    end

    apply_structure_extension!(
        context.numerics.structure_extension,
        work.sigma_xx_stage,
        work.sigma_yy_stage,
        work.sigma_xy_stage,
        work.phi_stage,
    )
    return nothing
end

function _build_resource_stage!(work::IMEXWorkspace, s::Int, dt::Float64, context::IMEXContext, diffusion_operator)
    copyto!(work.rhs_c, work.c_n)
    for j in 1:(s - 1)
        a_exp = Float64(A_EXPLICIT[s][j])
        a_imp = Float64(A_IMPLICIT[s][j])
        if a_exp != 0.0 || a_imp != 0.0
            @. work.rhs_c += dt * (a_exp * work.E_c[j] + a_imp * work.I_c[j])
        end
    end

    if diffusion_operator === nothing
        _copy_vector_to_cell!(work.c_stage, work.rhs_c)
        return nothing
    end

    a_ss = Float64(A_IMPLICIT[s][s])
    alpha = dt * a_ss
    @. work.rhs_c += alpha * diffusion_operator.boundary
    lhs = _shifted_identity(diffusion_operator.matrix, alpha)

    options = context.numerics.linear_solvers
    _solve_linear_system!(
        work.c_predictor,
        lhs,
        work.rhs_c,
        work.c_n,
        options,
        options.resource_method,
        "resource stage $s",
    )
    _copy_vector_to_cell!(work.c_stage, work.c_predictor)
    return nothing
end

function _build_velocity_stage!(work::IMEXWorkspace, s::Int, dt::Float64, context::IMEXContext, viscous_operator)
    copyto!(work.rhs_uv, work.uv_n)
    for j in 1:(s - 1)
        a_exp = Float64(A_EXPLICIT[s][j])
        a_imp = Float64(A_IMPLICIT[s][j])
        if a_exp != 0.0 || a_imp != 0.0
            @. work.rhs_uv += dt * (a_exp * work.E_uv[j] + a_imp * work.I_uv[j])
        end
    end

    if viscous_operator === nothing
        copyto!(work.uv_predictor, work.rhs_uv)
        _enforce_compact_dirichlet_values!(work.uv_predictor, context.metrics, context.bcs)
    else
        a_ss = Float64(A_IMPLICIT[s][s])
        alpha = dt * a_ss
        @. work.rhs_uv += alpha * viscous_operator.boundary
        lhs = _shifted_identity(viscous_operator.matrix, alpha)

        indices, values = velocity_dirichlet_data(
            context.metrics,
            context.bcs.U,
            context.bcs.V,
        )
        lhs_bc, rhs_bc = _impose_row_dirichlet(lhs, work.rhs_uv, indices, values)

        options = context.numerics.linear_solvers
        _solve_linear_system!(
            work.uv_predictor,
            lhs_bc,
            rhs_bc,
            work.uv_n,
            options,
            options.velocity_method,
            "velocity stage $s",
        )
    end

    _inject_compact_predictor!(
        work.U_stage,
        work.V_stage,
        work.uv_predictor,
        context.metrics,
        context.bcs,
    )

    result = project!(
        work.U_stage,
        work.V_stage,
        work.G,
        context.projection_system,
        context.metrics;
        correction_guess=work.correction,
        reltol=context.numerics.projection_reltol,
        maxiter=context.numerics.projection_maxiter,
        restart=context.numerics.projection_restart,
    )
    copyto!(work.correction, result.correction)
    _extract_compact_velocity!(work.uv_stage, work.U_stage, work.V_stage, context.metrics)
    return result
end

# ==============================================================================
# STAGE CONSTITUTIVE FIELDS
# ==============================================================================

function _evaluate_growth!(work::IMEXWorkspace, context::IMEXContext)
    growth_source!(
        work.G,
        context.models.growth_source,
        work.phi_stage,
        work.c_stage,
    )
    growth_tensor!(
        work.Kg_xx,
        work.Kg_xy,
        work.Kg_yx,
        work.Kg_yy,
        context.models.growth_tensor,
        work.G,
    )
    diagnostics = growth_diagnostics(
        work.G,
        work.Kg_xx,
        work.Kg_xy,
        work.Kg_yx,
        work.Kg_yy,
    )
    return diagnostics.maximum_trace_error
end

function _build_diffusion_operator!(work::IMEXWorkspace, context::IMEXContext)
    diffusivity_field!(
        work.diffusivity,
        context.models.diffusivity,
        work.phi_stage,
        context.metrics,
    )
    diffusion_active(context.models.diffusivity) || return nothing
    return build_diffusion_operator(
        context.metrics,
        work.diffusivity,
        context.bcs.resource,
    )
end

function _build_viscous_operator!(work::IMEXWorkspace, context::IMEXContext)
    viscosity_field!(
        work.viscosity,
        context.models.viscosity,
        work.phi_stage,
        context.metrics,
    )
    viscosity_active(context.models.viscosity) || return nothing
    return build_viscous_operator(
        context.metrics,
        work.viscosity,
        context.velocity_gradients,
        context.bcs.U,
        context.bcs.V,
    )
end

# ==============================================================================
# EXPLICIT AND IMPLICIT STAGE TENDENCIES
# ==============================================================================

function _accumulate_elastic_force!(work::IMEXWorkspace, context::IMEXContext)
    tau_xx, tau_yy, tau_xy = elastic_stress(
        context.models.elastic_stress,
        work.sigma_xx_stage,
        work.sigma_yy_stage,
        work.sigma_xy_stage,
    )

    @. work.stress_xx = work.phi_stage * tau_xx
    @. work.stress_yy = work.phi_stage * tau_yy
    @. work.stress_xy = work.phi_stage * tau_xy

    Fx, Fy = symmetric_stress_divergence(
        work.stress_xx,
        work.stress_yy,
        work.stress_xy,
        context.metrics,
    )
    work.force_x .+= Fx
    work.force_y .+= Fy
    return nothing
end

function _accumulate_growth_viscous_force!(work::IMEXWorkspace, context::IMEXContext)
    viscosity_active(context.models.viscosity) || return nothing

    @. work.stress_xx = -2.0 * work.viscosity * work.Kg_xx
    @. work.stress_yy = -2.0 * work.viscosity * work.Kg_yy
    @. work.stress_xy = -work.viscosity * (work.Kg_xy + work.Kg_yx)

    Fx, Fy = symmetric_stress_divergence(
        work.stress_xx,
        work.stress_yy,
        work.stress_xy,
        context.metrics,
    )
    work.force_x .+= Fx
    work.force_y .+= Fy
    return nothing
end

function _evaluate_stage_tendencies!(
    work::IMEXWorkspace,
    s::Int,
    t_stage::Float64,
    context::IMEXContext,
    viscous_operator,
    diffusion_operator,
)
    metrics = context.metrics
    bcs = context.bcs
    models = context.models

    # ------------------------------------------------------------------
    # Momentum explicit terms.
    # ------------------------------------------------------------------
    Cu, Cv = material_convection(
        work.U_stage,
        work.V_stage,
        work.G,
        metrics,
        bcs.U,
        bcs.V,
    )
    @. work.force_x = -Cu
    @. work.force_y = -Cv

    body_force!(
        work.scratch_force_x,
        work.scratch_force_y,
        models.body_force,
        t_stage,
        metrics,
    )
    work.force_x .+= work.scratch_force_x
    work.force_y .+= work.scratch_force_y

    Fx_cap, Fy_cap = capillary_force(models.capillarity, work.phi_stage, metrics, bcs.phi)
    work.force_x .+= Fx_cap
    work.force_y .+= Fy_cap

    _accumulate_elastic_force!(work, context)
    _accumulate_growth_viscous_force!(work, context)
    _extract_compact_velocity!(work.E_uv[s], work.force_x, work.force_y, metrics)

    # ------------------------------------------------------------------
    # Interface transport.
    # ------------------------------------------------------------------
    work.E_phi[s] .= phase_field_rhs(
        models.phase_field_method,
        work.U_stage,
        work.V_stage,
        work.phi_stage,
        work.G,
        metrics,
        bcs.phi,
    )

    # ------------------------------------------------------------------
    # Resource transport and consumption.
    # ------------------------------------------------------------------
    if models.growth_source isa NoGrowth &&
       models.diffusivity isa NoDiffusion &&
       models.resource_consumption isa NoResourceConsumption
        # Resource is absent from the original FTaC system. Avoid imposing
        # scalar inflow BCs on a field that has no physical role in that case.
        fill!(work.E_c[s], 0.0)
    else
        resource_sink!(
            work.resource_sink,
            models.resource_consumption,
            work.phi_stage,
            work.c_stage,
            metrics,
        )
        E_c_matrix = resource_explicit_rhs(
            work.U_stage,
            work.V_stage,
            work.c_stage,
            work.resource_sink,
            metrics,
            bcs.resource,
        )
        _copy_cell_to_vector!(work.E_c[s], E_c_matrix)
    end

    # ------------------------------------------------------------------
    # Structural tensor.
    # ------------------------------------------------------------------
    rhs_xx, rhs_yy, rhs_xy = structure_tensor_rhs(
        work.sigma_xx_stage,
        work.sigma_yy_stage,
        work.sigma_xy_stage,
        work.U_stage,
        work.V_stage,
        work.Kg_xx,
        work.Kg_xy,
        work.Kg_yx,
        work.Kg_yy,
        models.tensor_relaxation,
        metrics,
        bcs.sigma,
        bcs.U,
        bcs.V;
        phi=work.phi_stage,
        structure_extension=context.numerics.structure_extension,
    )
    copyto!(work.E_sigma_xx[s], rhs_xx)
    copyto!(work.E_sigma_yy[s], rhs_yy)
    copyto!(work.E_sigma_xy[s], rhs_xy)

    # ------------------------------------------------------------------
    # Implicit histories evaluated on the projected stage state.
    # ------------------------------------------------------------------
    if viscous_operator === nothing
        fill!(work.I_uv[s], 0.0)
    else
        work.I_uv[s] .= apply_operator(viscous_operator, work.uv_stage)
    end

    if diffusion_operator === nothing
        fill!(work.I_c[s], 0.0)
    else
        _copy_cell_to_vector!(work.rhs_c, work.c_stage)
        work.I_c[s] .= apply_operator(diffusion_operator, work.rhs_c)
    end

    return nothing
end

# ==============================================================================
# CONSTRAINT INITIALIZATION
# ==============================================================================

raw"""
    make_velocity_compatible!(state, context, workspace)

Evaluate the growth source from the current `phi` and resource fields and
project the supplied velocity onto

\nabla_h\cdot\mathbf{v}=G.

This operation is intended for simulation initialization. The projected
velocity is written directly into `state`.
"""
function make_velocity_compatible!(
    state::FSIState,
    context::IMEXContext,
    work::IMEXWorkspace,
)
    _validate_state(state, context.metrics)
    growth_source!(work.G, context.models.growth_source, state.phi, state.resource)

    result = project!(
        state.U,
        state.V,
        work.G,
        context.projection_system,
        context.metrics;
        correction_guess=work.correction,
        reltol=context.numerics.projection_reltol,
        maxiter=context.numerics.projection_maxiter,
        restart=context.numerics.projection_restart,
    )
    copyto!(work.correction, result.correction)
    return result
end

function _stage_one_constraint_linf(work::IMEXWorkspace, context::IMEXContext)
    residual = divergence(work.U_stage, work.V_stage, context.metrics) .- work.G
    return maximum(abs, residual)
end

function _check_stage_one_constraint!(linf::Float64, work::IMEXWorkspace, context::IMEXContext)
    context.numerics.check_stage1_constraint || return nothing
    scale = max(maximum(abs, work.G), 1.0)
    tolerance = context.numerics.stage1_constraint_atol +
                context.numerics.stage1_constraint_rtol * scale
    linf <= tolerance || throw(ArgumentError(
        "The velocity entering stage 1 does not satisfy the prescribed divergence constraint. " *
        "Maximum |div(v)-G| = $linf, tolerance = $tolerance. " *
        "Initialize the state with make_velocity_compatible! or provide a compatible velocity."
    ))
    return nothing
end

# ==============================================================================
# MAIN INTEGRATION STEP
# ==============================================================================

"""
    integrate_step!(state, t, dt, context, workspace)

Advance `state` from `t` to `t + dt` with the four-stage IMEX fractional-step
scheme.

For stages `s > 1`, the operation order is

phi_s, sigma_s
      -> resource implicit solve c_s
      -> G_s
      -> K_g,s
      -> viscous velocity predictor
      -> projection: div(v_s) = G_s
      -> explicit/implicit stage tendencies.

The same stage growth field `G_s` is used by projection, material momentum
convection, interface transport, and growth kinematics.
"""
function integrate_step!(
    state::FSIState,
    t::Real,
    dt::Real,
    context::IMEXContext,
    work::IMEXWorkspace,
)
    _validate_state(state, context.metrics)
    isfinite(t) || throw(ArgumentError("t must be finite."))
    isfinite(dt) || throw(ArgumentError("dt must be finite."))
    dt > 0 || throw(ArgumentError("dt must be strictly positive."))
    context.numerics.scheme isa FourStageIMEXFractionalStep || throw(ArgumentError(
        "Unsupported IMEX scheme $(typeof(context.numerics.scheme))."
    ))

    dt_f = Float64(dt)
    t_f = Float64(t)
    _copy_time_n_state!(work, state, context)

    maximum_projection_constraint_linf = 0.0
    maximum_growth_trace_error = 0.0

    for s in 1:NSTAGES
        if s == 1
            _set_stage_one!(work, state)
        else
            _build_explicit_stage!(work, s, dt_f, context)
        end

        # Resource state first: G_s may depend on c_s and must be known before
        # the velocity projection.
        diffusion_operator = _build_diffusion_operator!(work, context)
        if s > 1
            _build_resource_stage!(work, s, dt_f, context, diffusion_operator)
        end

        trace_error = _evaluate_growth!(work, context)
        maximum_growth_trace_error = max(maximum_growth_trace_error, trace_error)

        # Viscosity is evaluated from the same phi_s used to construct K_g,s.
        viscous_operator = _build_viscous_operator!(work, context)

        if s == 1
            _extract_compact_velocity!(work.uv_stage, work.U_stage, work.V_stage, context.metrics)
            constraint_linf = _stage_one_constraint_linf(work, context)
            _check_stage_one_constraint!(constraint_linf, work, context)
            maximum_projection_constraint_linf = max(
                maximum_projection_constraint_linf,
                constraint_linf,
            )
        else
            projection_result = _build_velocity_stage!(
                work,
                s,
                dt_f,
                context,
                viscous_operator,
            )
            maximum_projection_constraint_linf = max(
                maximum_projection_constraint_linf,
                projection_result.constraint_linf,
            )
        end

        # Only stages 1:3 are required as histories by this stiffly accurate
        # tableau. Stage 4 is already the final state.
        if s <= NHISTORY
            t_stage = t_f + Float64(STAGE_ABSCISSA[s]) * dt_f
            _evaluate_stage_tendencies!(
                work,
                s,
                t_stage,
                context,
                viscous_operator,
                diffusion_operator,
            )
        end
    end

    # The fourth stage is the n+1 state.
    copyto!(state.U, work.U_stage)
    copyto!(state.V, work.V_stage)
    copyto!(state.phi, work.phi_stage)
    copyto!(state.resource, work.c_stage)
    copyto!(state.sigma_xx, work.sigma_xx_stage)
    copyto!(state.sigma_yy, work.sigma_yy_stage)
    copyto!(state.sigma_xy, work.sigma_xy_stage)

    sigma_diagnostics = structure_tensor_diagnostics(
        state.sigma_xx,
        state.sigma_yy,
        state.sigma_xy,
    )

    return StepReport(
        maximum_projection_constraint_linf,
        maximum_growth_trace_error,
        minimum(state.phi),
        maximum(state.phi),
        minimum(state.resource),
        sigma_diagnostics.minimum_eigenvalue,
        sigma_diagnostics.maximum_abs_determinant_error,
        sigma_diagnostics.non_spd_count,
    )
end

end # module TimeIntegrationIMEX
