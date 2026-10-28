"""Boundary conditions for Cartesian finite-difference grids."""
module BoundaryConditions

export AbstractBoundaryCondition,
       Periodic,
       Dirichlet,
       Neumann,
       MixedBoundary,
       AdvectiveOutflow,
       BoundaryConditionSetup,
       boundary_value,
       mixed_is_dirichlet,
       mixed_dirichlet_value,
       mixed_neumann_value,
       backflow_value,
       is_periodic_x,
       is_periodic_y,
       has_dirichlet

# BOUNDARY TYPES

"""Abstract supertype for all boundary-condition specifications."""
abstract type AbstractBoundaryCondition end

"""Periodic boundary condition."""
struct Periodic <: AbstractBoundaryCondition end

"""Prescribe a Dirichlet boundary value."""
struct Dirichlet{T} <: AbstractBoundaryCondition
    value::T
end

struct Neumann{T} <: AbstractBoundaryCondition
    value::T
end

struct MixedBoundary{M<:AbstractVector{Bool},D,N} <: AbstractBoundaryCondition
    dirichlet_mask::M
    dirichlet_value::D
    neumann_value::N
end

"""Open boundary condition for an advected-diffused scalar."""
struct AdvectiveOutflow{T} <: AbstractBoundaryCondition
    backflow_value::T
end

# FOUR-SIDED BOUNDARY CONTAINER

"""Boundary conditions on the four sides of a rectangular two-dimensional domain."""
struct BoundaryConditionSetup{
    L<:AbstractBoundaryCondition,
    R<:AbstractBoundaryCondition,
    B<:AbstractBoundaryCondition,
    Top<:AbstractBoundaryCondition,
}
    left::L
    right::R
    bottom::B
    top::Top

    function BoundaryConditionSetup(left::L, right::R, bottom::B, top::Top) where {
        L<:AbstractBoundaryCondition,
        R<:AbstractBoundaryCondition,
        B<:AbstractBoundaryCondition,
        Top<:AbstractBoundaryCondition,
    }
        _validate_periodic_pair(left, right, "x")
        _validate_periodic_pair(bottom, top, "y")
        return new{L,R,B,Top}(left, right, bottom, top)
    end
end

function _validate_periodic_pair(first, second, direction::AbstractString)
    first_periodic = first isa Periodic
    second_periodic = second isa Periodic

    first_periodic == second_periodic || throw(ArgumentError(
        "Periodic boundaries in the $direction-direction must be specified as an opposing pair."
    ))

    return nothing
end

"""Return `true` when the boundary setup is periodic in the x-direction."""
is_periodic_x(bc::BoundaryConditionSetup) = bc.left isa Periodic

"""Return `true` when the boundary setup is periodic in the y-direction."""
is_periodic_y(bc::BoundaryConditionSetup) = bc.bottom isa Periodic

"""Return `true` when at least one side carries a Dirichlet condition."""
_boundary_has_dirichlet(::AbstractBoundaryCondition) = false
_boundary_has_dirichlet(::Dirichlet) = true
_boundary_has_dirichlet(bc::MixedBoundary) = any(bc.dirichlet_mask)

has_dirichlet(bc::BoundaryConditionSetup) =
    _boundary_has_dirichlet(bc.left) || _boundary_has_dirichlet(bc.right) ||
    _boundary_has_dirichlet(bc.bottom) || _boundary_has_dirichlet(bc.top)

# BOUNDARY-DATA EVALUATION

"""Evaluate a Dirichlet or Neumann boundary datum at boundary-point `index`."""
boundary_value(bc::Union{Dirichlet,Neumann}, index::Integer) =
    _evaluate_boundary_data(bc.value, index)

"""Return whether a point on a [`MixedBoundary`](@ref) uses Dirichlet data."""
function mixed_is_dirichlet(bc::MixedBoundary, index::Integer)
    checkbounds(bc.dirichlet_mask, index)
    return bc.dirichlet_mask[index]
end

"""Evaluate the Dirichlet datum of a [`MixedBoundary`](@ref)."""
mixed_dirichlet_value(bc::MixedBoundary, index::Integer) =
    _evaluate_boundary_data(bc.dirichlet_value, index)

"""Evaluate the Neumann datum of a [`MixedBoundary`](@ref)."""
mixed_neumann_value(bc::MixedBoundary, index::Integer) =
    _evaluate_boundary_data(bc.neumann_value, index)

"""Evaluate the exterior scalar state used when an open boundary backflows."""
backflow_value(bc::AdvectiveOutflow, index::Integer) =
    _evaluate_boundary_data(bc.backflow_value, index)

_evaluate_boundary_data(value::Number, ::Integer) = value

function _evaluate_boundary_data(value::AbstractVector, index::Integer)
    checkbounds(value, index)
    return value[index]
end

function _evaluate_boundary_data(value, index::Integer)
    applicable(value, index) || throw(ArgumentError(
        "Boundary data must be a scalar, an AbstractVector, or callable as value(index)."
    ))
    return value(index)
end

end # module BoundaryConditions
