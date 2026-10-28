"""Conservative transport of the cell-centered diffuse structural indicator `phi` on a staggered MAC finite-difference grid."""
module InterfaceTransport

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators

export interface_advection,
       interface_explicit_rhs

# VALIDATION

function _validate_fields(U, V, phi, metrics::MeshMetrics)
    size(phi) == (metrics.Nx, metrics.Ny) || throw(DimensionMismatch(
        "phi must have size $((metrics.Nx, metrics.Ny)); received $(size(phi))."
    ))
    size(U) == (metrics.Nx + 1, metrics.Ny) || throw(DimensionMismatch(
        "U must have size $((metrics.Nx + 1, metrics.Ny)); received $(size(U))."
    ))
    size(V) == (metrics.Nx, metrics.Ny + 1) || throw(DimensionMismatch(
        "V must have size $((metrics.Nx, metrics.Ny + 1)); received $(size(V))."
    ))
    return nothing
end

function _validate_periodicity(metrics::MeshMetrics, bc_phi::BoundaryConditionSetup)
    metrics.periodic_x == is_periodic_x(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in x."
    ))
    metrics.periodic_y == is_periodic_y(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in y."
    ))
    return nothing
end

function _cell_field(value, metrics::MeshMetrics, name::AbstractString)
    if value isa Number
        value_f = Float64(value)
        isfinite(value_f) || throw(ArgumentError("$name must be finite."))
        return fill(value_f, metrics.Nx, metrics.Ny)
    end

    expected = (metrics.Nx, metrics.Ny)
    size(value) == expected || throw(DimensionMismatch(
        "$name must be a scalar or have size $expected; received $(size(value))."
    ))
    all(isfinite, value) || throw(ArgumentError("$name must contain only finite values."))
    return Float64.(value)
end

function _validate_interface_flux(jx, jy, metrics::MeshMetrics)
    expected_x = (metrics.Nx + 1, metrics.Ny)
    expected_y = (metrics.Nx, metrics.Ny + 1)
    size(jx) == expected_x || throw(DimensionMismatch(
        "jx must have size $expected_x; received $(size(jx))."
    ))
    size(jy) == expected_y || throw(DimensionMismatch(
        "jy must have size $expected_y; received $(size(jy))."
    ))
    all(isfinite, jx) || throw(ArgumentError("jx must contain only finite values."))
    all(isfinite, jy) || throw(ArgumentError("jy must contain only finite values."))

    if metrics.periodic_x
        scale = max(maximum(abs, jx), 1.0)
        mismatch = maximum(abs.(jx[end, :] .- jx[1, :]))
        mismatch <= 100 * eps(Float64) * scale || throw(ArgumentError(
            "Periodic x-interface flux must satisfy jx[end,:] == jx[1,:]."
        ))
    end
    if metrics.periodic_y
        scale = max(maximum(abs, jy), 1.0)
        mismatch = maximum(abs.(jy[:, end] .- jy[:, 1]))
        mismatch <= 100 * eps(Float64) * scale || throw(ArgumentError(
            "Periodic y-interface flux must satisfy jy[:,end] == jy[:,1]."
        ))
    end

    return nothing
end

# ADVECTIVE BOUNDARY STATES

@inline function _x_boundary_state(
    phi,
    velocity::Real,
    side::Symbol,
    j::Int,
    bc_phi::BoundaryConditionSetup,
)
    Nx = size(phi, 1)

    if side === :left
        bc = bc_phi.left
        interior = phi[1, j]
        inflow = velocity > 0.0
    elseif side === :right
        bc = bc_phi.right
        interior = phi[Nx, j]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("x-boundary side must be :left or :right."))
    end

    velocity == 0.0 && return interior
    !inflow && return interior

    if bc isa Dirichlet
        return Float64(boundary_value(bc, j))
    elseif bc isa Neumann
        throw(ArgumentError(
            "A Neumann phi condition on the $side boundary does not define " *
            "the incoming advected phase. Prescribe Dirichlet data for advective inflow."
        ))
    elseif bc isa Periodic
        throw(ArgumentError("Periodic x-boundary states must be handled by wrapping."))
    end

    error("Unsupported phi boundary condition $(typeof(bc)).")
end

@inline function _y_boundary_state(
    phi,
    velocity::Real,
    side::Symbol,
    i::Int,
    bc_phi::BoundaryConditionSetup,
)
    Ny = size(phi, 2)

    if side === :bottom
        bc = bc_phi.bottom
        interior = phi[i, 1]
        inflow = velocity > 0.0
    elseif side === :top
        bc = bc_phi.top
        interior = phi[i, Ny]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("y-boundary side must be :bottom or :top."))
    end

    velocity == 0.0 && return interior
    !inflow && return interior

    if bc isa Dirichlet
        return Float64(boundary_value(bc, i))
    elseif bc isa Neumann
        throw(ArgumentError(
            "A Neumann phi condition on the $side boundary does not define " *
            "the incoming advected phase. Prescribe Dirichlet data for advective inflow."
        ))
    elseif bc isa Periodic
        throw(ArgumentError("Periodic y-boundary states must be handled by wrapping."))
    end

    error("Unsupported phi boundary condition $(typeof(bc)).")
end

# FIRST-ORDER UPWIND MATERIAL FLUX

function _x_advective_fluxes(U, phi, metrics::MeshMetrics, bc_phi::BoundaryConditionSetup)
    Nx, Ny = metrics.Nx, metrics.Ny
    flux_x = zeros(Float64, Nx + 1, Ny)

    @inbounds for j in 1:Ny
        if metrics.periodic_x
            u = U[1, j]
            left = phi[Nx, j]
            right = phi[1, j]
            flux_x[1, j] = u * (u >= 0.0 ? left : right)
        else
            u = U[1, j]
            boundary = _x_boundary_state(phi, u, :left, j, bc_phi)
            interior = phi[1, j]
            flux_x[1, j] = u * (u >= 0.0 ? boundary : interior)
        end

        for i in 2:Nx
            u = U[i, j]
            left = phi[i - 1, j]
            right = phi[i, j]
            flux_x[i, j] = u * (u >= 0.0 ? left : right)
        end

        if metrics.periodic_x
            flux_x[Nx + 1, j] = flux_x[1, j]
        else
            u = U[Nx + 1, j]
            interior = phi[Nx, j]
            boundary = _x_boundary_state(phi, u, :right, j, bc_phi)
            flux_x[Nx + 1, j] = u * (u >= 0.0 ? interior : boundary)
        end
    end

    return flux_x
end

function _y_advective_fluxes(V, phi, metrics::MeshMetrics, bc_phi::BoundaryConditionSetup)
    Nx, Ny = metrics.Nx, metrics.Ny
    flux_y = zeros(Float64, Nx, Ny + 1)

    @inbounds for i in 1:Nx
        if metrics.periodic_y
            v = V[i, 1]
            bottom = phi[i, Ny]
            top = phi[i, 1]
            flux_y[i, 1] = v * (v >= 0.0 ? bottom : top)
        else
            v = V[i, 1]
            boundary = _y_boundary_state(phi, v, :bottom, i, bc_phi)
            interior = phi[i, 1]
            flux_y[i, 1] = v * (v >= 0.0 ? boundary : interior)
        end

        for j in 2:Ny
            v = V[i, j]
            bottom = phi[i, j - 1]
            top = phi[i, j]
            flux_y[i, j] = v * (v >= 0.0 ? bottom : top)
        end

        if metrics.periodic_y
            flux_y[i, Ny + 1] = flux_y[i, 1]
        else
            v = V[i, Ny + 1]
            interior = phi[i, Ny]
            boundary = _y_boundary_state(phi, v, :top, i, bc_phi)
            flux_y[i, Ny + 1] = v * (v >= 0.0 ? interior : boundary)
        end
    end

    return flux_y
end

# PUBLIC OPERATORS

function interface_advection(
    U,
    V,
    phi,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(U, V, phi, metrics)
    _validate_periodicity(metrics, bc_phi)

    flux_x = _x_advective_fluxes(U, phi, metrics, bc_phi)
    flux_y = _y_advective_fluxes(V, phi, metrics, bc_phi)

    rhs = zeros(Float64, metrics.Nx, metrics.Ny)
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        div_flux = (flux_x[i + 1, j] - flux_x[i, j]) / metrics.dx_c[i] +
                   (flux_y[i, j + 1] - flux_y[i, j]) / metrics.dy_c[j]
        rhs[i, j] = -div_flux
    end

    return rhs
end

function interface_explicit_rhs(
    U,
    V,
    phi,
    target_divergence,
    jx,
    jy,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    advective_rhs = interface_advection(U, V, phi, metrics, bc_phi)
    S = _cell_field(target_divergence, metrics, "target_divergence")
    _validate_interface_flux(jx, jy, metrics)

    regularization_divergence = divergence(jx, jy, metrics)
    return advective_rhs .- regularization_divergence .+ Float64.(phi) .* S
end

end # module InterfaceTransport
