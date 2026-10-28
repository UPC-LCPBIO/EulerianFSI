"""Finite-difference projection of a staggered velocity field onto a prescribed cell-centered divergence constraint."""
module Projection

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using AlgebraicMultigrid

using ..MeshGeneration
using ..BoundaryConditions
using ..Operators

export ProjectionSystem,
       ProjectionResult,
       build_projection_system,
       build_preconditioner,
       project!

# DATA STRUCTURES

"""Preassembled linear system for the correction-potential solve."""
struct ProjectionSystem{P,BC<:BoundaryConditionSetup}
    operator::AffineOperator
    solve_matrix::SparseMatrixCSC{Float64,Int}
    preconditioner::P
    bc_correction::BC
    has_constant_nullspace::Bool
end

"""Diagnostics returned by the projection solver."""
struct ProjectionResult
    correction::Matrix{Float64}
    linear_relative_residual::Float64
    constraint_linf::Float64
    constraint_l2::Float64
    compatibility_defect::Float64
end

# INTERNAL UTILITIES

function _validate_velocity_arrays(U, V, metrics::MeshMetrics)
    expected_U = (metrics.Nx + 1, metrics.Ny)
    expected_V = (metrics.Nx, metrics.Ny + 1)

    size(U) == expected_U || throw(DimensionMismatch(
        "U must have size $expected_U; received $(size(U))."
    ))
    size(V) == expected_V || throw(DimensionMismatch(
        "V must have size $expected_V; received $(size(V))."
    ))

    all(isfinite, U) || throw(ArgumentError("U contains non-finite values."))
    all(isfinite, V) || throw(ArgumentError("V contains non-finite values."))
    return nothing
end

function _target_divergence_field(target, metrics::MeshMetrics)
    if target isa Number
        value = Float64(target)
        isfinite(value) || throw(ArgumentError("target_divergence must be finite."))
        return fill(value, metrics.Nx, metrics.Ny)
    end

    expected = (metrics.Nx, metrics.Ny)
    size(target) == expected || throw(DimensionMismatch(
        "target_divergence must have size $expected; received $(size(target))."
    ))
    all(isfinite, target) || throw(ArgumentError(
        "target_divergence contains non-finite values."
    ))
    return Matrix{Float64}(target)
end

@inline function _cell_area(metrics::MeshMetrics, i::Int, j::Int)
    return metrics.dx_c[i] * metrics.dy_c[j]
end

function _weighted_integral(field, metrics::MeshMetrics)
    total = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        total += field[i, j] * _cell_area(metrics, i, j)
    end
    return total
end

function _weighted_mean(field, metrics::MeshMetrics)
    return _weighted_integral(field, metrics) / (metrics.Lx * metrics.Ly)
end

function _weighted_l2(field, metrics::MeshMetrics)
    total = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        total += abs2(field[i, j]) * _cell_area(metrics, i, j)
    end
    return sqrt(total / (metrics.Lx * metrics.Ly))
end

function _compatibility_scale(rhs_field, metrics::MeshMetrics)
    total = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        total += abs(rhs_field[i, j]) * _cell_area(metrics, i, j)
    end
    return total
end

function _gauge_fixed_matrix(matrix::SparseMatrixCSC{Float64,Int})
    pinned = copy(matrix)
    pinned[1, :] .= 0.0
    pinned[:, 1] .= 0.0
    pinned[1, 1] = 1.0
    dropzeros!(pinned)
    return pinned
end

function _prepare_initial_guess(correction_guess, metrics::MeshMetrics)
    N = metrics.Nx * metrics.Ny

    if correction_guess === nothing
        return zeros(Float64, N)
    elseif correction_guess isa AbstractVector
        length(correction_guess) == N || throw(DimensionMismatch(
            "correction_guess vector must have length $N; received $(length(correction_guess))."
        ))
        all(isfinite, correction_guess) || throw(ArgumentError(
            "correction_guess contains non-finite values."
        ))
        return Float64.(correction_guess)
    else
        expected = (metrics.Nx, metrics.Ny)
        size(correction_guess) == expected || throw(DimensionMismatch(
            "correction_guess must have size $expected; received $(size(correction_guess))."
        ))
        all(isfinite, correction_guess) || throw(ArgumentError(
            "correction_guess contains non-finite values."
        ))
        return vec(Matrix{Float64}(correction_guess))
    end
end

# PRECONDITIONER AND SYSTEM CONSTRUCTION

"""Build a Ruge-Stüben algebraic-multigrid preconditioner for a sparse correction matrix."""
function build_preconditioner(matrix::SparseMatrixCSC{Float64,Int})
    hierarchy = ruge_stuben(matrix)
    return aspreconditioner(hierarchy)
end

"""Preassemble the finite-difference correction-potential system."""
function build_projection_system(
    metrics::MeshMetrics,
    bc_correction::BoundaryConditionSetup;
    use_amg::Bool = true,
)
    operator = Operators.build_pressure_laplacian(metrics, bc_correction)
    has_nullspace = !has_dirichlet(bc_correction)
    solve_matrix = has_nullspace ? _gauge_fixed_matrix(operator.matrix) : copy(operator.matrix)
    preconditioner = use_amg ? build_preconditioner(solve_matrix) : nothing

    return ProjectionSystem(
        operator,
        solve_matrix,
        preconditioner,
        bc_correction,
        has_nullspace,
    )
end

# PROJECTION

function project!(
    U,
    V,
    target_divergence,
    system::ProjectionSystem,
    metrics::MeshMetrics;
    correction_guess = nothing,
    reltol::Real = 1e-10,
    maxiter::Integer = 3000,
    restart::Integer = 50,
    compatibility_atol::Real = 1e-12,
    compatibility_rtol::Real = 1e-10,
    check_residual::Bool = true,
    linear_residual_rtol::Real = 1e-8,
    constraint_atol::Real = 1e-12,
    constraint_rtol::Real = 1e-8,
)
    _validate_velocity_arrays(U, V, metrics)

    reltol > 0 || throw(ArgumentError("reltol must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    restart > 0 || throw(ArgumentError("restart must be positive."))
    compatibility_atol >= 0 || throw(ArgumentError("compatibility_atol must be non-negative."))
    compatibility_rtol >= 0 || throw(ArgumentError("compatibility_rtol must be non-negative."))
    linear_residual_rtol > 0 || throw(ArgumentError("linear_residual_rtol must be positive."))
    constraint_atol >= 0 || throw(ArgumentError("constraint_atol must be non-negative."))
    constraint_rtol >= 0 || throw(ArgumentError("constraint_rtol must be non-negative."))

    target = _target_divergence_field(target_divergence, metrics)
    predictor_divergence = Operators.divergence(U, V, metrics)

    # Aq + b_bc = div(v†) - S  ->  Aq = div(v†) - S - b_bc.
    physical_rhs = predictor_divergence .- target
    rhs = vec(physical_rhs) .- system.operator.boundary
    rhs_field = reshape(rhs, metrics.Nx, metrics.Ny)

    compatibility_defect = 0.0
    if system.has_constant_nullspace
        compatibility_defect = _weighted_integral(rhs_field, metrics)
        scale = _compatibility_scale(rhs_field, metrics)
        tolerance = Float64(compatibility_atol) +
                    Float64(compatibility_rtol) * max(scale, eps(Float64))

        abs(compatibility_defect) <= tolerance || throw(ArgumentError(
            "The prescribed divergence constraint is incompatible with the " *
            "Neumann/periodic correction boundaries. The volume integral of " *
            "the Poisson right-hand side is $(compatibility_defect), while the " *
            "compatibility tolerance is $(tolerance). Adjust the target " *
            "divergence or the boundary flux; the solver will not alter the " *
            "constraint automatically."
        ))
    end

    x = _prepare_initial_guess(correction_guess, metrics)
    rhs_solve = copy(rhs)

    if system.has_constant_nullspace
        # The first equation is replaced by q[1] = 0 only to fix the gauge.
        x[1] = 0.0
        rhs_solve[1] = 0.0
    end

    if system.preconditioner === nothing
        gmres!(
            x,
            system.solve_matrix,
            rhs_solve;
            restart=Int(restart),
            reltol=Float64(reltol),
            maxiter=Int(maxiter),
        )
    else
        gmres!(
            x,
            system.solve_matrix,
            rhs_solve;
            restart=Int(restart),
            reltol=Float64(reltol),
            maxiter=Int(maxiter),
            Pl=system.preconditioner,
        )
    end

    correction = reshape(copy(x), metrics.Nx, metrics.Ny)

    if system.has_constant_nullspace
        # Select the zero-volume-mean representative of q. This changes neither
        # its gradient nor the projected velocity.
        correction .-= _weighted_mean(correction, metrics)
    end

    # Residual of the original unpinned affine Poisson equation.
    linear_residual = system.operator.matrix * vec(correction) - rhs
    linear_scale = max(norm(rhs), eps(Float64))
    linear_relative_residual = norm(linear_residual) / linear_scale

    grad_q_x, grad_q_y = Operators.scalar_gradient(
        correction,
        metrics,
        system.bc_correction,
    )

    U .-= grad_q_x
    V .-= grad_q_y

    constraint_residual = Operators.divergence(U, V, metrics) .- target
    constraint_linf = maximum(abs, constraint_residual)
    constraint_l2 = _weighted_l2(constraint_residual, metrics)

    if check_residual
        linear_tolerance = Float64(linear_residual_rtol)
        linear_relative_residual <= linear_tolerance || throw(ErrorException(
            "Projection Poisson solve did not reach the required true-residual accuracy. " *
            "Relative residual = $(linear_relative_residual), tolerance = $(linear_tolerance)."
        ))

        constraint_scale = max(
            _weighted_l2(predictor_divergence, metrics),
            _weighted_l2(target, metrics),
            1.0,
        )
        constraint_tolerance = Float64(constraint_atol) +
                               Float64(constraint_rtol) * constraint_scale
        constraint_l2 <= constraint_tolerance || throw(ErrorException(
            "Projection did not satisfy the prescribed divergence constraint. " *
            "Weighted L2 residual = $(constraint_l2), tolerance = " *
            "$(constraint_tolerance)."
        ))
    end

    return ProjectionResult(
        correction,
        linear_relative_residual,
        constraint_linf,
        constraint_l2,
        compatibility_defect,
    )
end

end # module Projection
