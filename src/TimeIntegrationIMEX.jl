"""Four-stage IMEX fractional-step method for incompressible Eulerian FSI."""
module TimeIntegrationIMEX

using LinearAlgebra
using SparseArrays
using IterativeSolvers

using ..MeshGeneration
using ..BoundaryConditions
using ..Operators
using ..Projection
using ..MomentumTransport
using ..InterfaceTransport
using ..InterfaceRegularization
using ..PhaseFieldMethods
using ..Capillarity
using ..StructuralConstitutive
using ..StructureTensor
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

# TIME TABLEAU

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

# PUBLIC DATA STRUCTURES

"""MAC velocity, phase indicator, and symmetric structural tensor."""
mutable struct FSIState
    U::Matrix{Float64}
    V::Matrix{Float64}
    phi::Matrix{Float64}
    sigma_xx::Matrix{Float64}
    sigma_yy::Matrix{Float64}
    sigma_xy::Matrix{Float64}
end

"""Boundary conditions for velocity, projection potential, interface and tensor."""
struct FSIBoundaryConditions{BU,BV,BQ,BP,BS}
    U::BU
    V::BV
    correction::BQ
    phi::BP
    sigma::BS

    function FSIBoundaryConditions(U::BU, V::BV, correction::BQ, phi::BP, sigma::BS) where {BU,BV,BQ,BP,BS}
        hasproperty(sigma, :xx) || throw(ArgumentError("sigma boundary data require an `xx` entry."))
        hasproperty(sigma, :yy) || throw(ArgumentError("sigma boundary data require a `yy` entry."))
        hasproperty(sigma, :xy) || throw(ArgumentError("sigma boundary data require an `xy` entry."))
        return new{BU,BV,BQ,BP,BS}(U, V, correction, phi, sigma)
    end
end

"""Constitutive and forcing models for incompressible flow and elasticity."""
struct FSIModels{MU,TR,ES,CAP,PF,BF}
    viscosity::MU
    tensor_relaxation::TR
    elastic_stress::ES
    capillarity::CAP
    phase_field_method::PF
    body_force::BF
end

function FSIModels(;
    viscosity = NoViscosity(),
    tensor_relaxation = NoRelaxation(),
    elastic_stress = NoElasticStress(),
    capillarity = NoCapillarity(),
    phase_field_method = nothing,
    interface_flux = NoInterfaceFlux(),
    body_force = NoBodyForce(),
)
    pf_method = if phase_field_method === nothing
        LegacySplitPhaseFieldMethod(interface_flux)
    else
        interface_flux isa NoInterfaceFlux || throw(ArgumentError(
            "Specify either phase_field_method or interface_flux, not both."
        ))
        phase_field_method
    end
    pf_method isa AbstractPhaseFieldMethod || throw(ArgumentError(
        "phase_field_method must be an AbstractPhaseFieldMethod."
    ))
    return FSIModels(viscosity, tensor_relaxation, elastic_stress,
                     capillarity, pf_method, body_force)
end

"""Linear solver configuration for the implicit velocity systems."""
struct LinearSolverOptions
    velocity_method::Symbol
    reltol::Float64
    maxiter::Int
    restart::Int
    direct_fallback::Bool
    fallback_reltol::Float64
    verbose::Bool
end

function LinearSolverOptions(;
    velocity_method::Symbol = :direct,
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
    reltol > 0 || throw(ArgumentError("Linear-solver reltol must be positive."))
    maxiter > 0 || throw(ArgumentError("Linear-solver maxiter must be positive."))
    restart > 0 || throw(ArgumentError("Linear-solver restart must be positive."))
    fallback_reltol > 0 || throw(ArgumentError("fallback_reltol must be positive."))
    return LinearSolverOptions(velocity_method, Float64(reltol), Int(maxiter),
        Int(restart), direct_fallback, Float64(fallback_reltol), verbose)
end

"""Numerical choices that are not constitutive physics."""
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

"""Diagnostics accumulated over one complete IMEX step."""
struct StepReport
    maximum_projection_constraint_linf::Float64
    minimum_phi::Float64
    maximum_phi::Float64
    minimum_sigma_eigenvalue::Float64
    maximum_abs_sigma_determinant_error::Float64
    non_spd_cells::Int
end

# WORKSPACE

mutable struct IMEXWorkspace
    uv_n::Vector{Float64}
    phi_n::Matrix{Float64}
    sigma_xx_n::Matrix{Float64}
    sigma_yy_n::Matrix{Float64}
    sigma_xy_n::Matrix{Float64}

    U_stage::Matrix{Float64}
    V_stage::Matrix{Float64}
    phi_stage::Matrix{Float64}
    sigma_xx_stage::Matrix{Float64}
    sigma_yy_stage::Matrix{Float64}
    sigma_xy_stage::Matrix{Float64}
    uv_stage::Vector{Float64}
    uv_predictor::Vector{Float64}
    rhs_uv::Vector{Float64}

    E_uv::Vector{Vector{Float64}}
    I_uv::Vector{Vector{Float64}}
    E_phi::Vector{Matrix{Float64}}
    E_sigma_xx::Vector{Matrix{Float64}}
    E_sigma_yy::Vector{Matrix{Float64}}
    E_sigma_xy::Vector{Matrix{Float64}}

    # Fixed zero target for the incompressible projection and transport terms.
    zero_divergence::Matrix{Float64}
    viscosity::Matrix{Float64}

    force_x::Matrix{Float64}
    force_y::Matrix{Float64}
    scratch_force_x::Matrix{Float64}
    scratch_force_y::Matrix{Float64}
    stress_xx::Matrix{Float64}
    stress_yy::Matrix{Float64}
    stress_xy::Matrix{Float64}
    correction::Matrix{Float64}
end

function allocate_imex_workspace(metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    return IMEXWorkspace(
        zeros(2N),
        zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny),
        zeros(Nx + 1, Ny), zeros(Nx, Ny + 1),
        zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny),
        zeros(2N), zeros(2N), zeros(2N),
        [zeros(2N) for _ in 1:NHISTORY],
        [zeros(2N) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        [zeros(Nx, Ny) for _ in 1:NHISTORY],
        zeros(Nx, Ny), zeros(Nx, Ny),
        zeros(Nx + 1, Ny), zeros(Nx, Ny + 1),
        zeros(Nx + 1, Ny), zeros(Nx, Ny + 1),
        zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny),
        zeros(Nx, Ny),
    )
end

# CONTEXT CONSTRUCTION AND VALIDATION

function _validate_periodicity(metrics::MeshMetrics, bcs::FSIBoundaryConditions)
    for (name, bc) in (
        ("U", bcs.U),
        ("V", bcs.V),
        ("correction", bcs.correction),
        ("phi", bcs.phi),
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

# COMPACT MAC STORAGE HELPERS

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

# STAGE FIELD CONSTRUCTION

function _copy_time_n_state!(work::IMEXWorkspace, state::FSIState, context::IMEXContext)
    _extract_compact_velocity!(work.uv_n, state.U, state.V, context.metrics)
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
        work.zero_divergence,
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

# STAGE CONSTITUTIVE FIELDS

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

# EXPLICIT AND IMPLICIT STAGE TENDENCIES

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

function _evaluate_stage_tendencies!(
    work::IMEXWorkspace,
    s::Int,
    t_stage::Float64,
    context::IMEXContext,
    viscous_operator,
)
    metrics = context.metrics
    bcs = context.bcs
    models = context.models

    # Momentum explicit terms.
    Cu, Cv = material_convection(
        work.U_stage,
        work.V_stage,
        work.zero_divergence,
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
    _extract_compact_velocity!(work.E_uv[s], work.force_x, work.force_y, metrics)

    # Interface transport.
    work.E_phi[s] .= phase_field_rhs(
        models.phase_field_method,
        work.U_stage,
        work.V_stage,
        work.phi_stage,
        work.zero_divergence,
        metrics,
        bcs.phi,
    )

    # Structural tensor.
    rhs_xx, rhs_yy, rhs_xy = structure_tensor_rhs(
        work.sigma_xx_stage,
        work.sigma_yy_stage,
        work.sigma_xy_stage,
        work.U_stage,
        work.V_stage,
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

    # Implicit histories evaluated on the projected stage state.
    if viscous_operator === nothing
        fill!(work.I_uv[s], 0.0)
    else
        work.I_uv[s] .= apply_operator(viscous_operator, work.uv_stage)
    end

    return nothing
end

# CONSTRAINT INITIALIZATION

"""Project the initial MAC velocity onto the incompressible constraint."""
function make_velocity_compatible!(
    state::FSIState,
    context::IMEXContext,
    work::IMEXWorkspace,
)
    _validate_state(state, context.metrics)

    result = project!(
        state.U,
        state.V,
        work.zero_divergence,
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
    residual = divergence(work.U_stage, work.V_stage, context.metrics) .- work.zero_divergence
    return maximum(abs, residual)
end

function _check_stage_one_constraint!(linf::Float64, work::IMEXWorkspace, context::IMEXContext)
    context.numerics.check_stage1_constraint || return nothing
    scale = 1.0
    tolerance = context.numerics.stage1_constraint_atol +
                context.numerics.stage1_constraint_rtol * scale
    linf <= tolerance || throw(ArgumentError(
        "The velocity entering stage 1 does not satisfy the prescribed divergence constraint. " *
        "Maximum |div(v)| = $linf, tolerance = $tolerance. " *
        "Initialize the state with make_velocity_compatible! or provide a compatible velocity."
    ))
    return nothing
end

# MAIN INTEGRATION STEP

"""Advance incompressible velocity, phase indicator and structural tensor."""
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

    for s in 1:NSTAGES
        if s == 1
            _set_stage_one!(work, state)
        else
            _build_explicit_stage!(work, s, dt_f, context)
        end

        # Viscosity is evaluated at the current interface stage.
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
            )
        end
    end

    # The fourth stage is the n+1 state.
    copyto!(state.U, work.U_stage)
    copyto!(state.V, work.V_stage)
    copyto!(state.phi, work.phi_stage)
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
        minimum(state.phi),
        maximum(state.phi),
        sigma_diagnostics.minimum_eigenvalue,
        sigma_diagnostics.maximum_abs_determinant_error,
        sigma_diagnostics.non_spd_count,
    )
end

end # module TimeIntegrationIMEX
