raw"""
    ResourceTransport

Explicit transport terms for a cell-centered scalar resource field on a
staggered MAC finite-difference grid.

The module discretizes the conservative advective term

-\nabla\cdot(c\mathbf{v})

with first-order upwinding at velocity faces. A pre-evaluated cell-centered
sink field may be supplied separately, giving the explicit contribution

\mathcal{R}_c = -\nabla\cdot(c\mathbf{v}) - \mathcal{C}.

Diffusion is intentionally not assembled here. It is represented by the
variable-coefficient diffusion operator in `Operators.jl` and may therefore be
treated implicitly by the time integrator.

`ResourceTransport` contains no constitutive consumption law, diffusivity law,
growth law, phase-field dependence, or time integration. In particular, the
same material velocity used by the rest of the FSI solver is passed directly to
this module.

Boundary conditions require special care because advective inflow and diffusive
boundary data are not interchangeable. For first-order upwinding:

- periodic faces wrap to the opposite cell;
- outflow faces use the adjacent interior value;
- Dirichlet inflow faces use the prescribed boundary value directly;
- Neumann data do not determine an incoming advected state, so a nonzero
  Neumann inflow is rejected rather than silently extrapolating an exterior
  concentration.

This prevents the centered ghost construction commonly used for diffusion from
being incorrectly reused as an upwind inflow state.
"""
module ResourceTransport

using ..BoundaryConditions
using ..MeshGeneration

export resource_advection,
       resource_explicit_rhs

# ==============================================================================
# VALIDATION AND FIELD NORMALIZATION
# ==============================================================================

function _validate_scalar_field(c, metrics::MeshMetrics)
    expected = (metrics.Nx, metrics.Ny)
    size(c) == expected || throw(DimensionMismatch(
        "Resource field must have size $expected; received $(size(c))."
    ))
    return nothing
end

function _validate_velocity_fields(U, V, metrics::MeshMetrics)
    expected_U = (metrics.Nx + 1, metrics.Ny)
    expected_V = (metrics.Nx, metrics.Ny + 1)

    size(U) == expected_U || throw(DimensionMismatch(
        "U must have size $expected_U; received $(size(U))."
    ))
    size(V) == expected_V || throw(DimensionMismatch(
        "V must have size $expected_V; received $(size(V))."
    ))

    return nothing
end

function _validate_scalar_periodicity(
    metrics::MeshMetrics,
    bc_c::BoundaryConditionSetup,
)
    metrics.periodic_x == is_periodic_x(bc_c) || throw(ArgumentError(
        "Resource boundary conditions do not match MeshMetrics periodicity in x."
    ))
    metrics.periodic_y == is_periodic_y(bc_c) || throw(ArgumentError(
        "Resource boundary conditions do not match MeshMetrics periodicity in y."
    ))

    return nothing
end

function _sink_field(sink, metrics::MeshMetrics)
    if sink isa Number
        return fill(Float64(sink), metrics.Nx, metrics.Ny)
    end

    expected = (metrics.Nx, metrics.Ny)
    size(sink) == expected || throw(DimensionMismatch(
        "sink must be a scalar or have size $expected; received $(size(sink))."
    ))

    return Float64.(sink)
end

# ==============================================================================
# ADVECTIVE BOUNDARY STATES
# ==============================================================================

@inline function _x_boundary_state(
    c,
    velocity::Real,
    side::Symbol,
    j::Int,
    bc_c::BoundaryConditionSetup,
)
    Nx = size(c, 1)

    if side === :left
        bc = bc_c.left
        interior = c[1, j]
        inflow = velocity > 0.0
    elseif side === :right
        bc = bc_c.right
        interior = c[Nx, j]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("x-boundary side must be :left or :right."))
    end

    # At zero normal speed the advective flux vanishes independently of the
    # selected state. Returning the interior value keeps this helper benign.
    velocity == 0.0 && return interior

    # Upwind outflow always originates from the adjacent interior cell.
    !inflow && return interior

    if bc isa Dirichlet
        return Float64(boundary_value(bc, j))
    elseif bc isa AdvectiveOutflow
        return Float64(backflow_value(bc, j))
    elseif bc isa Neumann
        throw(ArgumentError(
            "A Neumann resource condition on the $side boundary does not define " *
            "the incoming advected state. Prescribe Dirichlet data for advective inflow " *
            "or use AdvectiveOutflow(backflow_value) for a backflow-safe open boundary."
        ))
    elseif bc isa Periodic
        throw(ArgumentError(
            "Periodic x-boundary states must be handled by periodic wrapping."
        ))
    end

    error("Unsupported resource boundary condition $(typeof(bc)).")
end

@inline function _y_boundary_state(
    c,
    velocity::Real,
    side::Symbol,
    i::Int,
    bc_c::BoundaryConditionSetup,
)
    Ny = size(c, 2)

    if side === :bottom
        bc = bc_c.bottom
        interior = c[i, 1]
        inflow = velocity > 0.0
    elseif side === :top
        bc = bc_c.top
        interior = c[i, Ny]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("y-boundary side must be :bottom or :top."))
    end

    velocity == 0.0 && return interior
    !inflow && return interior

    if bc isa Dirichlet
        return Float64(boundary_value(bc, i))
    elseif bc isa AdvectiveOutflow
        return Float64(backflow_value(bc, i))
    elseif bc isa Neumann
        throw(ArgumentError(
            "A Neumann resource condition on the $side boundary does not define " *
            "the incoming advected state. Prescribe Dirichlet data for advective inflow " *
            "or use AdvectiveOutflow(backflow_value) for a backflow-safe open boundary."
        ))
    elseif bc isa Periodic
        throw(ArgumentError(
            "Periodic y-boundary states must be handled by periodic wrapping."
        ))
    end

    error("Unsupported resource boundary condition $(typeof(bc)).")
end

# ==============================================================================
# FIRST-ORDER UPWIND FACE FLUXES
# ==============================================================================

function _x_advective_fluxes(U, c, metrics::MeshMetrics, bc_c::BoundaryConditionSetup)
    Nx, Ny = metrics.Nx, metrics.Ny
    flux_x = zeros(Float64, Nx + 1, Ny)

    @inbounds for j in 1:Ny
        if metrics.periodic_x
            u_face = U[1, j]
            c_left = c[Nx, j]
            c_right = c[1, j]
            flux_x[1, j] = u_face * (u_face >= 0.0 ? c_left : c_right)
        else
            u_face = U[1, j]
            boundary_state = _x_boundary_state(c, u_face, :left, j, bc_c)
            interior_state = c[1, j]
            flux_x[1, j] = u_face * (u_face >= 0.0 ? boundary_state : interior_state)
        end

        for i in 2:Nx
            u_face = U[i, j]
            c_left = c[i - 1, j]
            c_right = c[i, j]
            flux_x[i, j] = u_face * (u_face >= 0.0 ? c_left : c_right)
        end

        if metrics.periodic_x
            # The first and last MAC faces represent the same physical periodic
            # interface. Copying the flux enforces exact discrete telescoping.
            flux_x[Nx + 1, j] = flux_x[1, j]
        else
            u_face = U[Nx + 1, j]
            interior_state = c[Nx, j]
            boundary_state = _x_boundary_state(c, u_face, :right, j, bc_c)
            flux_x[Nx + 1, j] = u_face * (u_face >= 0.0 ? interior_state : boundary_state)
        end
    end

    return flux_x
end

function _y_advective_fluxes(V, c, metrics::MeshMetrics, bc_c::BoundaryConditionSetup)
    Nx, Ny = metrics.Nx, metrics.Ny
    flux_y = zeros(Float64, Nx, Ny + 1)

    @inbounds for i in 1:Nx
        if metrics.periodic_y
            v_face = V[i, 1]
            c_bottom = c[i, Ny]
            c_top = c[i, 1]
            flux_y[i, 1] = v_face * (v_face >= 0.0 ? c_bottom : c_top)
        else
            v_face = V[i, 1]
            boundary_state = _y_boundary_state(c, v_face, :bottom, i, bc_c)
            interior_state = c[i, 1]
            flux_y[i, 1] = v_face * (v_face >= 0.0 ? boundary_state : interior_state)
        end

        for j in 2:Ny
            v_face = V[i, j]
            c_bottom = c[i, j - 1]
            c_top = c[i, j]
            flux_y[i, j] = v_face * (v_face >= 0.0 ? c_bottom : c_top)
        end

        if metrics.periodic_y
            flux_y[i, Ny + 1] = flux_y[i, 1]
        else
            v_face = V[i, Ny + 1]
            interior_state = c[i, Ny]
            boundary_state = _y_boundary_state(c, v_face, :top, i, bc_c)
            flux_y[i, Ny + 1] = v_face * (v_face >= 0.0 ? interior_state : boundary_state)
        end
    end

    return flux_y
end

# ==============================================================================
# PUBLIC OPERATORS
# ==============================================================================

raw"""
    resource_advection(U, V, c, metrics, bc_c)

Return the explicit conservative advection contribution

-\nabla_h\cdot(c\mathbf{v})

for a cell-centered scalar `c` transported by face-centered MAC velocities
`U` and `V`.

The flux on each face is `v_n * c_upwind`, with first-order upwinding. At an
advective inflow, a `Dirichlet` boundary value is used directly as the incoming
state. `Neumann` inflow is rejected because a normal derivative does not specify
what scalar value enters the domain.
"""
function resource_advection(
    U,
    V,
    c,
    metrics::MeshMetrics,
    bc_c::BoundaryConditionSetup,
)
    _validate_velocity_fields(U, V, metrics)
    _validate_scalar_field(c, metrics)
    _validate_scalar_periodicity(metrics, bc_c)

    flux_x = _x_advective_fluxes(U, c, metrics, bc_c)
    flux_y = _y_advective_fluxes(V, c, metrics, bc_c)

    Nx, Ny = metrics.Nx, metrics.Ny
    rhs = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        div_flux = (flux_x[i + 1, j] - flux_x[i, j]) / metrics.dx_c[i] +
                   (flux_y[i, j + 1] - flux_y[i, j]) / metrics.dy_c[j]
        rhs[i, j] = -div_flux
    end

    return rhs
end

raw"""
    resource_explicit_rhs(U, V, c, sink, metrics, bc_c)

Return the explicit resource contribution

-\nabla_h\cdot(c\mathbf{v}) - \mathcal{C},

where `sink` is a pre-evaluated cell-centered field `\mathcal{C}`. `sink` may
also be a scalar, in which case it is applied uniformly.

No constitutive law is evaluated in this function. Diffusion is likewise left
to `Operators.build_diffusion_operator` and the time integrator.
"""
function resource_explicit_rhs(
    U,
    V,
    c,
    sink,
    metrics::MeshMetrics,
    bc_c::BoundaryConditionSetup,
)
    advective_rhs = resource_advection(U, V, c, metrics, bc_c)
    sink_field = _sink_field(sink, metrics)
    return advective_rhs .- sink_field
end

end # module ResourceTransport
