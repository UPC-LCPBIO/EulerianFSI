"""Convective momentum transport for a two-dimensional finite-difference solver on an explicitly staggered MAC grid."""
module MomentumTransport

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators: interpolate_center_to_x_faces, interpolate_center_to_y_faces

export conservative_convection,
       material_convection

# VALIDATION

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

function _validate_velocity_periodicity(
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    for (name, bc) in (("U", bc_U), ("V", bc_V))
        metrics.periodic_x == is_periodic_x(bc) || throw(ArgumentError(
            "$name boundary conditions do not match MeshMetrics periodicity in x."
        ))
        metrics.periodic_y == is_periodic_y(bc) || throw(ArgumentError(
            "$name boundary conditions do not match MeshMetrics periodicity in y."
        ))
    end

    return nothing
end

function _target_divergence_field(target_divergence, metrics::MeshMetrics)
    if target_divergence isa Number
        return fill(Float64(target_divergence), metrics.Nx, metrics.Ny)
    end

    expected = (metrics.Nx, metrics.Ny)
    size(target_divergence) == expected || throw(DimensionMismatch(
        "target_divergence must be a scalar or have size $expected; received " *
        "$(size(target_divergence))."
    ))

    return Float64.(target_divergence)
end

# BOUNDARY VALUES USED BY TRANSVERSE CORNER INTERPOLATION

@inline function _canonical_xface_index(i::Int, metrics::MeshMetrics)
    return metrics.periodic_x && i == metrics.Nx + 1 ? 1 : i
end

@inline function _canonical_yface_index(j::Int, metrics::MeshMetrics)
    return metrics.periodic_y && j == metrics.Ny + 1 ? 1 : j
end

function _tangential_u_boundary_value(
    U,
    i::Int,
    side::Symbol,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
)
    ii = _canonical_xface_index(i, metrics)

    if side === :bottom
        bc = bc_U.bottom
        adjacent = U[ii, 1]
        distance = metrics.dy_f[1] / 2.0
    elseif side === :top
        bc = bc_U.top
        adjacent = U[ii, metrics.Ny]
        distance = metrics.dy_f[end] / 2.0
    else
        throw(ArgumentError("U tangential boundary side must be :bottom or :top."))
    end

    if bc isa Dirichlet
        return Float64(boundary_value(bc, ii))
    elseif bc isa Neumann
        # `Neumann(g)` stores the outward normal derivative. Linear extension
        # from the adjacent staggered point gives q_wall = q_adjacent + g*d on
        # either side of a Cartesian axis.
        return adjacent + Float64(boundary_value(bc, ii)) * distance
    elseif bc isa Periodic
        throw(ArgumentError(
            "Periodic transverse U values must be handled through periodic interpolation."
        ))
    end

    error("Unsupported U boundary condition $(typeof(bc)).")
end

function _tangential_v_boundary_value(
    V,
    j::Int,
    side::Symbol,
    metrics::MeshMetrics,
    bc_V::BoundaryConditionSetup,
)
    jj = _canonical_yface_index(j, metrics)

    if side === :left
        bc = bc_V.left
        adjacent = V[1, jj]
        distance = metrics.dx_f[1] / 2.0
    elseif side === :right
        bc = bc_V.right
        adjacent = V[metrics.Nx, jj]
        distance = metrics.dx_f[end] / 2.0
    else
        throw(ArgumentError("V tangential boundary side must be :left or :right."))
    end

    if bc isa Dirichlet
        return Float64(boundary_value(bc, jj))
    elseif bc isa Neumann
        return adjacent + Float64(boundary_value(bc, jj)) * distance
    elseif bc isa Periodic
        throw(ArgumentError(
            "Periodic transverse V values must be handled through periodic interpolation."
        ))
    end

    error("Unsupported V boundary condition $(typeof(bc)).")
end

# STAGGERED INTERPOLATION OF MOMENTUM-FLUX COMPONENTS

function _normal_flux_squares(U, V, metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    u2_center = zeros(Float64, Nx, Ny)
    v2_center = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        u_right = (metrics.periodic_x && i == Nx) ? U[1, j] : U[i + 1, j]
        u_center = metrics.wL_f2cx[i] * U[i, j] +
                   metrics.wR_f2cx[i] * u_right
        u2_center[i, j] = u_center^2

        v_top = (metrics.periodic_y && j == Ny) ? V[i, 1] : V[i, j + 1]
        v_center = metrics.wB_f2cy[j] * V[i, j] +
                   metrics.wT_f2cy[j] * v_top
        v2_center[i, j] = v_center^2
    end

    return u2_center, v2_center
end

function _u_at_corner(
    U,
    i::Int,
    j::Int,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    ii = _canonical_xface_index(i, metrics)

    if metrics.periodic_y
        if j == 1 || j == Ny + 1
            return metrics.wB_c2fy[1] * U[ii, Ny] +
                   metrics.wT_c2fy[1] * U[ii, 1]
        end

        return metrics.wB_c2fy[j] * U[ii, j - 1] +
               metrics.wT_c2fy[j] * U[ii, j]
    end

    if j == 1
        return _tangential_u_boundary_value(U, ii, :bottom, metrics, bc_U)
    elseif j == Ny + 1
        return _tangential_u_boundary_value(U, ii, :top, metrics, bc_U)
    end

    return metrics.wB_c2fy[j] * U[ii, j - 1] +
           metrics.wT_c2fy[j] * U[ii, j]
end

function _v_at_corner(
    V,
    i::Int,
    j::Int,
    metrics::MeshMetrics,
    bc_V::BoundaryConditionSetup,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    jj = _canonical_yface_index(j, metrics)

    if metrics.periodic_x
        if i == 1 || i == Nx + 1
            return metrics.wL_c2fx[1] * V[Nx, jj] +
                   metrics.wR_c2fx[1] * V[1, jj]
        end

        return metrics.wL_c2fx[i] * V[i - 1, jj] +
               metrics.wR_c2fx[i] * V[i, jj]
    end

    if i == 1
        return _tangential_v_boundary_value(V, jj, :left, metrics, bc_V)
    elseif i == Nx + 1
        return _tangential_v_boundary_value(V, jj, :right, metrics, bc_V)
    end

    return metrics.wL_c2fx[i] * V[i - 1, jj] +
           metrics.wR_c2fx[i] * V[i, jj]
end

function _cross_flux_corners(
    U,
    V,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    uv_corner = zeros(Float64, Nx + 1, Ny + 1)

    @inbounds for j in 1:(Ny + 1), i in 1:(Nx + 1)
        u_corner = _u_at_corner(U, i, j, metrics, bc_U)
        v_corner = _v_at_corner(V, i, j, metrics, bc_V)
        uv_corner[i, j] = u_corner * v_corner
    end

    # Enforce exact equality of duplicated periodic boundary locations. This
    # avoids propagating stale duplicate face values supplied by a caller.
    if metrics.periodic_x
        uv_corner[Nx + 1, :] .= uv_corner[1, :]
    end
    if metrics.periodic_y
        uv_corner[:, Ny + 1] .= uv_corner[:, 1]
    end

    return uv_corner
end

# CONSERVATIVE CONVECTION

raw"""Evaluate the conservative staggered finite-difference momentum convection \nabla_h\cdot(\mathbf{v}\otimes\mathbf{v})."""
function conservative_convection(
    U,
    V,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    _validate_velocity_fields(U, V, metrics)
    _validate_velocity_periodicity(metrics, bc_U, bc_V)

    Nx, Ny = metrics.Nx, metrics.Ny
    u2_center, v2_center = _normal_flux_squares(U, V, metrics)
    uv_corner = _cross_flux_corners(U, V, metrics, bc_U, bc_V)

    Cu = zeros(Float64, Nx + 1, Ny)
    Cv = zeros(Float64, Nx, Ny + 1)

    # x-momentum: d(u^2)/dx + d(uv)/dy at U locations.
    @inbounds for j in 1:Ny
        if metrics.periodic_x
            Cu[1, j] = (u2_center[1, j] - u2_center[Nx, j]) / metrics.dx_f[1] +
                       (uv_corner[1, j + 1] - uv_corner[1, j]) / metrics.dy_c[j]
            Cu[Nx + 1, j] = Cu[1, j]
        end

        for i in 2:Nx
            Cu[i, j] = (u2_center[i, j] - u2_center[i - 1, j]) / metrics.dx_f[i] +
                       (uv_corner[i, j + 1] - uv_corner[i, j]) / metrics.dy_c[j]
        end
    end

    # y-momentum: d(uv)/dx + d(v^2)/dy at V locations.
    @inbounds for i in 1:Nx
        if metrics.periodic_y
            Cv[i, 1] = (uv_corner[i + 1, 1] - uv_corner[i, 1]) / metrics.dx_c[i] +
                       (v2_center[i, 1] - v2_center[i, Ny]) / metrics.dy_f[1]
            Cv[i, Ny + 1] = Cv[i, 1]
        end

        for j in 2:Ny
            Cv[i, j] = (uv_corner[i + 1, j] - uv_corner[i, j]) / metrics.dx_c[i] +
                       (v2_center[i, j] - v2_center[i, j - 1]) / metrics.dy_f[j]
        end
    end

    return Cu, Cv
end

# MATERIAL CONVECTION FOR PRESCRIBED DIVERGENCE

raw"""Evaluate the material convection operator using the conservative identity (\mathbf{v}\cdot\nabla)\mathbf{v}."""
function material_convection(
    U,
    V,
    target_divergence,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    Cu, Cv = conservative_convection(U, V, metrics, bc_U, bc_V)
    S = _target_divergence_field(target_divergence, metrics)

    Sx = interpolate_center_to_x_faces(S, metrics)
    Sy = interpolate_center_to_y_faces(S, metrics)

    Cu .-= Sx .* U
    Cv .-= Sy .* V

    # Physical normal-velocity boundary points are not interior momentum
    # equations. Preserve the module convention after applying the S*v term.
    if metrics.periodic_x
        Cu[end, :] .= Cu[1, :]
    else
        Cu[1, :] .= 0.0
        Cu[end, :] .= 0.0
    end

    if metrics.periodic_y
        Cv[:, end] .= Cv[:, 1]
    else
        Cv[:, 1] .= 0.0
        Cv[:, end] .= 0.0
    end

    return Cu, Cv
end

end # module MomentumTransport
