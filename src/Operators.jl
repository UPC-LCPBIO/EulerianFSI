"""Finite-difference spatial operators for a two-dimensional Cartesian staggered MAC grid."""
module Operators

using SparseArrays
using LinearAlgebra

using ..BoundaryConditions
using ..MeshGeneration

export AffineOperator,
       ScalarDifferentialOperators,
       VelocityGradientOperators,
       build_scalar_differential_operators,
       build_diffusion_operator,
       build_velocity_gradient_operators,
       build_viscous_operator,
       build_pressure_laplacian,
       scalar_gradient,
       divergence,
       velocity_gradient,
       symmetric_stress_divergence,
       interpolate_center_to_x_faces,
       interpolate_center_to_y_faces,
       interpolate_center_to_corners,
       apply_operator,
       velocity_dirichlet_data

# DATA STRUCTURES

struct AffineOperator
    matrix::SparseMatrixCSC{Float64,Int}
    boundary::Vector{Float64}
end

"""Sparse finite-difference operators mapping between cell centers and the full set of MAC faces."""
struct ScalarDifferentialOperators
    gradient_x::SparseMatrixCSC{Float64,Int}
    gradient_y::SparseMatrixCSC{Float64,Int}
    divergence_x::SparseMatrixCSC{Float64,Int}
    divergence_y::SparseMatrixCSC{Float64,Int}
    gradient_bc_x::Vector{Float64}
    gradient_bc_y::Vector{Float64}
end

"""Compact operators used by the implicit viscous discretization."""
struct VelocityGradientOperators
    du_dx::SparseMatrixCSC{Float64,Int}
    dv_dy::SparseMatrixCSC{Float64,Int}
    du_dy::SparseMatrixCSC{Float64,Int}
    dv_dx::SparseMatrixCSC{Float64,Int}
    du_dx_bc::Vector{Float64}
    dv_dy_bc::Vector{Float64}
end

# INDEXING AND VALIDATION

@inline _center_index(i, j, Nx) = i + (j - 1) * Nx
@inline _xface_index(i, j, Nx) = i + (j - 1) * (Nx + 1)
@inline _yface_index(i, j, Nx) = i + (j - 1) * Nx

function _validate_scalar_field(field, metrics::MeshMetrics, name::AbstractString)
    expected = (metrics.Nx, metrics.Ny)
    size(field) == expected || throw(DimensionMismatch(
        "$name must have size $expected; received $(size(field))."
    ))
    return nothing
end

function _validate_metric_periodicity(metrics::MeshMetrics, bc::BoundaryConditionSetup)
    metrics.periodic_x == is_periodic_x(bc) || throw(ArgumentError(
        "The x-periodicity encoded in MeshMetrics does not match the boundary conditions."
    ))
    metrics.periodic_y == is_periodic_y(bc) || throw(ArgumentError(
        "The y-periodicity encoded in MeshMetrics does not match the boundary conditions."
    ))
    return nothing
end

function _validate_velocity_boundary_periodicity(
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    _validate_metric_periodicity(metrics, bc_U)
    _validate_metric_periodicity(metrics, bc_V)
    return nothing
end

function _require_supported_viscous_boundaries(
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    for (name, bc) in (("U-left", bc_U.left), ("U-right", bc_U.right),
                       ("U-bottom", bc_U.bottom), ("U-top", bc_U.top),
                       ("V-left", bc_V.left), ("V-right", bc_V.right),
                       ("V-bottom", bc_V.bottom), ("V-top", bc_V.top))
        (bc isa Dirichlet || bc isa Periodic) || throw(ArgumentError(
            "$name uses $(typeof(bc)). The current compact viscous operator supports " *
            "Dirichlet or Periodic velocity boundaries. Mechanical Neumann/traction " *
            "conditions should be introduced explicitly when the momentum boundary " *
            "formulation is implemented."
        ))
    end
    return nothing
end

# CENTERED-COEFFICIENT INTERPOLATION

"""Linearly interpolate a cell-centered field to all x-faces of the MAC grid."""
function interpolate_center_to_x_faces(field, metrics::MeshMetrics)
    _validate_scalar_field(field, metrics, "field")
    Nx, Ny = metrics.Nx, metrics.Ny
    face = zeros(Float64, Nx + 1, Ny)

    @inbounds for j in 1:Ny
        if metrics.periodic_x
            value = metrics.wL_c2fx[1] * field[Nx, j] +
                    metrics.wR_c2fx[1] * field[1, j]
            face[1, j] = value
            face[Nx + 1, j] = value
        else
            face[1, j] = field[1, j]
            face[Nx + 1, j] = field[Nx, j]
        end

        for i in 2:Nx
            face[i, j] = metrics.wL_c2fx[i] * field[i - 1, j] +
                         metrics.wR_c2fx[i] * field[i, j]
        end
    end

    return face
end

"""See [`interpolate_center_to_x_faces`](@ref) for the y-direction analogue."""
function interpolate_center_to_y_faces(field, metrics::MeshMetrics)
    _validate_scalar_field(field, metrics, "field")
    Nx, Ny = metrics.Nx, metrics.Ny
    face = zeros(Float64, Nx, Ny + 1)

    @inbounds for i in 1:Nx
        if metrics.periodic_y
            value = metrics.wB_c2fy[1] * field[i, Ny] +
                    metrics.wT_c2fy[1] * field[i, 1]
            face[i, 1] = value
            face[i, Ny + 1] = value
        else
            face[i, 1] = field[i, 1]
            face[i, Ny + 1] = field[i, Ny]
        end

        for j in 2:Ny
            face[i, j] = metrics.wB_c2fy[j] * field[i, j - 1] +
                         metrics.wT_c2fy[j] * field[i, j]
        end
    end

    return face
end

"""Bilinearly interpolate a cell-centered field to all MAC-grid corners."""
function interpolate_center_to_corners(field, metrics::MeshMetrics)
    xface = interpolate_center_to_x_faces(field, metrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    corner = zeros(Float64, Nx + 1, Ny + 1)

    @inbounds for i in 1:(Nx + 1)
        if metrics.periodic_y
            value = metrics.wB_c2fy[1] * xface[i, Ny] +
                    metrics.wT_c2fy[1] * xface[i, 1]
            corner[i, 1] = value
            corner[i, Ny + 1] = value
        else
            corner[i, 1] = xface[i, 1]
            corner[i, Ny + 1] = xface[i, Ny]
        end

        for j in 2:Ny
            corner[i, j] = metrics.wB_c2fy[j] * xface[i, j - 1] +
                           metrics.wT_c2fy[j] * xface[i, j]
        end
    end

    return corner
end

# SCALAR GRADIENT AND DIVERGENCE MATRICES

function build_scalar_differential_operators(
    metrics::MeshMetrics,
    bc::BoundaryConditionSetup,
)
    _validate_metric_periodicity(metrics, bc)

    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    Nfx = (Nx + 1) * Ny
    Nfy = Nx * (Ny + 1)

    Igx, Jgx, Vgx = Int[], Int[], Float64[]
    Igy, Jgy, Vgy = Int[], Int[], Float64[]
    Idx, Jdx, Vdx = Int[], Int[], Float64[]
    Idy, Jdy, Vdy = Int[], Int[], Float64[]

    gx_bc = zeros(Float64, Nfx)
    gy_bc = zeros(Float64, Nfy)

    # X-gradient: cell centers -> all vertical faces.
    @inbounds for j in 1:Ny
        # Left boundary face.
        row = _xface_index(1, j, Nx)
        if metrics.periodic_x
            inv_h = 1.0 / metrics.dx_f[1]
            push!(Igx, row); push!(Jgx, _center_index(Nx, j, Nx)); push!(Vgx, -inv_h)
            push!(Igx, row); push!(Jgx, _center_index(1,  j, Nx)); push!(Vgx,  inv_h)
        elseif bc.left isa Dirichlet
            dist = metrics.dx_f[1] / 2.0
            push!(Igx, row); push!(Jgx, _center_index(1, j, Nx)); push!(Vgx, 1.0 / dist)
            gx_bc[row] = -Float64(boundary_value(bc.left, j)) / dist
        elseif bc.left isa Neumann
            gx_bc[row] = -Float64(boundary_value(bc.left, j))
        elseif bc.left isa MixedBoundary
            if mixed_is_dirichlet(bc.left, j)
                dist = metrics.dx_f[1] / 2.0
                push!(Igx, row); push!(Jgx, _center_index(1, j, Nx)); push!(Vgx, 1.0 / dist)
                gx_bc[row] = -Float64(mixed_dirichlet_value(bc.left, j)) / dist
            else
                gx_bc[row] = -Float64(mixed_neumann_value(bc.left, j))
            end
        elseif bc.left isa AdvectiveOutflow
            gx_bc[row] = 0.0
        else
            error("Unsupported left scalar boundary condition: $(typeof(bc.left)).")
        end

        # Interior faces.
        for i in 2:Nx
            row = _xface_index(i, j, Nx)
            inv_h = 1.0 / metrics.dx_f[i]
            push!(Igx, row); push!(Jgx, _center_index(i - 1, j, Nx)); push!(Vgx, -inv_h)
            push!(Igx, row); push!(Jgx, _center_index(i,     j, Nx)); push!(Vgx,  inv_h)
        end

        # Right boundary face.
        row = _xface_index(Nx + 1, j, Nx)
        if metrics.periodic_x
            source = _xface_index(1, j, Nx)
            # Reinsert the same wrapped finite difference at the duplicate face.
            inv_h = 1.0 / metrics.dx_f[end]
            push!(Igx, row); push!(Jgx, _center_index(Nx, j, Nx)); push!(Vgx, -inv_h)
            push!(Igx, row); push!(Jgx, _center_index(1,  j, Nx)); push!(Vgx,  inv_h)
            gx_bc[row] = gx_bc[source]
        elseif bc.right isa Dirichlet
            dist = metrics.dx_f[end] / 2.0
            push!(Igx, row); push!(Jgx, _center_index(Nx, j, Nx)); push!(Vgx, -1.0 / dist)
            gx_bc[row] = Float64(boundary_value(bc.right, j)) / dist
        elseif bc.right isa Neumann
            gx_bc[row] = Float64(boundary_value(bc.right, j))
        elseif bc.right isa MixedBoundary
            if mixed_is_dirichlet(bc.right, j)
                dist = metrics.dx_f[end] / 2.0
                push!(Igx, row); push!(Jgx, _center_index(Nx, j, Nx)); push!(Vgx, -1.0 / dist)
                gx_bc[row] = Float64(mixed_dirichlet_value(bc.right, j)) / dist
            else
                gx_bc[row] = Float64(mixed_neumann_value(bc.right, j))
            end
        elseif bc.right isa AdvectiveOutflow
            gx_bc[row] = 0.0
        else
            error("Unsupported right scalar boundary condition: $(typeof(bc.right)).")
        end
    end

    # Y-gradient: cell centers -> all horizontal faces.
    @inbounds for i in 1:Nx
        row = _yface_index(i, 1, Nx)
        if metrics.periodic_y
            inv_h = 1.0 / metrics.dy_f[1]
            push!(Igy, row); push!(Jgy, _center_index(i, Ny, Nx)); push!(Vgy, -inv_h)
            push!(Igy, row); push!(Jgy, _center_index(i, 1,  Nx)); push!(Vgy,  inv_h)
        elseif bc.bottom isa Dirichlet
            dist = metrics.dy_f[1] / 2.0
            push!(Igy, row); push!(Jgy, _center_index(i, 1, Nx)); push!(Vgy, 1.0 / dist)
            gy_bc[row] = -Float64(boundary_value(bc.bottom, i)) / dist
        elseif bc.bottom isa Neumann
            gy_bc[row] = -Float64(boundary_value(bc.bottom, i))
        elseif bc.bottom isa MixedBoundary
            if mixed_is_dirichlet(bc.bottom, i)
                dist = metrics.dy_f[1] / 2.0
                push!(Igy, row); push!(Jgy, _center_index(i, 1, Nx)); push!(Vgy, 1.0 / dist)
                gy_bc[row] = -Float64(mixed_dirichlet_value(bc.bottom, i)) / dist
            else
                gy_bc[row] = -Float64(mixed_neumann_value(bc.bottom, i))
            end
        elseif bc.bottom isa AdvectiveOutflow
            gy_bc[row] = 0.0
        else
            error("Unsupported bottom scalar boundary condition: $(typeof(bc.bottom)).")
        end

        for j in 2:Ny
            row = _yface_index(i, j, Nx)
            inv_h = 1.0 / metrics.dy_f[j]
            push!(Igy, row); push!(Jgy, _center_index(i, j - 1, Nx)); push!(Vgy, -inv_h)
            push!(Igy, row); push!(Jgy, _center_index(i, j,     Nx)); push!(Vgy,  inv_h)
        end

        row = _yface_index(i, Ny + 1, Nx)
        if metrics.periodic_y
            inv_h = 1.0 / metrics.dy_f[end]
            push!(Igy, row); push!(Jgy, _center_index(i, Ny, Nx)); push!(Vgy, -inv_h)
            push!(Igy, row); push!(Jgy, _center_index(i, 1,  Nx)); push!(Vgy,  inv_h)
            gy_bc[row] = gy_bc[_yface_index(i, 1, Nx)]
        elseif bc.top isa Dirichlet
            dist = metrics.dy_f[end] / 2.0
            push!(Igy, row); push!(Jgy, _center_index(i, Ny, Nx)); push!(Vgy, -1.0 / dist)
            gy_bc[row] = Float64(boundary_value(bc.top, i)) / dist
        elseif bc.top isa Neumann
            gy_bc[row] = Float64(boundary_value(bc.top, i))
        elseif bc.top isa MixedBoundary
            if mixed_is_dirichlet(bc.top, i)
                dist = metrics.dy_f[end] / 2.0
                push!(Igy, row); push!(Jgy, _center_index(i, Ny, Nx)); push!(Vgy, -1.0 / dist)
                gy_bc[row] = Float64(mixed_dirichlet_value(bc.top, i)) / dist
            else
                gy_bc[row] = Float64(mixed_neumann_value(bc.top, i))
            end
        elseif bc.top isa AdvectiveOutflow
            gy_bc[row] = 0.0
        else
            error("Unsupported top scalar boundary condition: $(typeof(bc.top)).")
        end
    end

    # Divergence: all faces -> cell centers.
    @inbounds for j in 1:Ny, i in 1:Nx
        row = _center_index(i, j, Nx)

        inv_dx = 1.0 / metrics.dx_c[i]
        push!(Idx, row); push!(Jdx, _xface_index(i,     j, Nx)); push!(Vdx, -inv_dx)
        push!(Idx, row); push!(Jdx, _xface_index(i + 1, j, Nx)); push!(Vdx,  inv_dx)

        inv_dy = 1.0 / metrics.dy_c[j]
        push!(Idy, row); push!(Jdy, _yface_index(i, j,     Nx)); push!(Vdy, -inv_dy)
        push!(Idy, row); push!(Jdy, _yface_index(i, j + 1, Nx)); push!(Vdy,  inv_dy)
    end

    return ScalarDifferentialOperators(
        sparse(Igx, Jgx, Vgx, Nfx, N),
        sparse(Igy, Jgy, Vgy, Nfy, N),
        sparse(Idx, Jdx, Vdx, N, Nfx),
        sparse(Idy, Jdy, Vdy, N, Nfy),
        gx_bc,
        gy_bc,
    )
end

# SCALAR DIFFUSION / LAPLACIAN

raw"""Construct the finite-difference operator for \nabla\cdot\left(D\nabla q\right) using a cell-centered diffusivity field."""
function build_diffusion_operator(
    metrics::MeshMetrics,
    diffusivity_center,
    bc::BoundaryConditionSetup,
)
    _validate_scalar_field(diffusivity_center, metrics, "diffusivity_center")
    all(isfinite, diffusivity_center) || throw(ArgumentError("Diffusivity values must be finite."))
    all(diffusivity_center .> 0.0) || throw(ArgumentError("Diffusivity values must be positive."))

    ops = build_scalar_differential_operators(metrics, bc)
    Dx_face = interpolate_center_to_x_faces(diffusivity_center, metrics)
    Dy_face = interpolate_center_to_y_faces(diffusivity_center, metrics)

    Mx = spdiagm(0 => vec(Dx_face))
    My = spdiagm(0 => vec(Dy_face))

    matrix = ops.divergence_x * Mx * ops.gradient_x +
             ops.divergence_y * My * ops.gradient_y

    boundary = ops.divergence_x * (vec(Dx_face) .* ops.gradient_bc_x) +
               ops.divergence_y * (vec(Dy_face) .* ops.gradient_bc_y)

    return AffineOperator(matrix, Vector{Float64}(boundary))
end

"""Construct the unit-coefficient scalar Laplacian used by the pressure/projection solve."""
function build_pressure_laplacian(metrics::MeshMetrics, bc_P::BoundaryConditionSetup)
    coefficient = ones(Float64, metrics.Nx, metrics.Ny)
    return build_diffusion_operator(metrics, coefficient, bc_P)
end

# EXPLICIT SCALAR GRADIENT AND VELOCITY DIVERGENCE

"""Evaluate the gradient of a cell-centered scalar field on all MAC faces."""
function scalar_gradient(field, metrics::MeshMetrics, bc::BoundaryConditionSetup)
    _validate_scalar_field(field, metrics, "field")
    ops = build_scalar_differential_operators(metrics, bc)

    gx = ops.gradient_x * vec(field) + ops.gradient_bc_x
    gy = ops.gradient_y * vec(field) + ops.gradient_bc_y

    return reshape(gx, metrics.Nx + 1, metrics.Ny),
           reshape(gy, metrics.Nx, metrics.Ny + 1)
end

"""Evaluate `∂U/∂x + ∂V/∂y` at cell centers from full staggered velocity arrays."""
function divergence(U, V, metrics::MeshMetrics)
    expected_U = (metrics.Nx + 1, metrics.Ny)
    expected_V = (metrics.Nx, metrics.Ny + 1)
    size(U) == expected_U || throw(DimensionMismatch(
        "U must have size $expected_U; received $(size(U))."
    ))
    size(V) == expected_V || throw(DimensionMismatch(
        "V must have size $expected_V; received $(size(V))."
    ))

    DIV = zeros(Float64, metrics.Nx, metrics.Ny)

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        DIV[i, j] = (U[i + 1, j] - U[i, j]) / metrics.dx_c[i] +
                    (V[i, j + 1] - V[i, j]) / metrics.dy_c[j]
    end

    return DIV
end


# EXPLICIT CELL-CENTERED VELOCITY GRADIENT

"""Evaluate the full two-dimensional velocity gradient at cell centers from the full staggered MAC velocity arrays."""
function velocity_gradient(
    U,
    V,
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    _validate_velocity_boundary_periodicity(metrics, bc_U, bc_V)

    Nx, Ny = metrics.Nx, metrics.Ny
    expected_U = (Nx + 1, Ny)
    expected_V = (Nx, Ny + 1)
    size(U) == expected_U || throw(DimensionMismatch(
        "U must have size $expected_U; received $(size(U))."
    ))
    size(V) == expected_V || throw(DimensionMismatch(
        "V must have size $expected_V; received $(size(V))."
    ))

    du_dx = zeros(Float64, Nx, Ny)
    du_dy = zeros(Float64, Nx, Ny)
    dv_dx = zeros(Float64, Nx, Ny)
    dv_dy = zeros(Float64, Nx, Ny)

    # MAC-normal derivatives are available directly at cell centers.
    @inbounds for j in 1:Ny, i in 1:Nx
        du_dx[i, j] = (U[i + 1, j] - U[i, j]) / metrics.dx_c[i]
        dv_dy[i, j] = (V[i, j + 1] - V[i, j]) / metrics.dy_c[j]
    end

    # Interpolate staggered velocity components to cell centers. These
    # interpolations are exact for affine fields because cell centers lie
    # between the corresponding pair of velocity faces.
    Uc = zeros(Float64, Nx, Ny)
    Vc = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        Uc[i, j] = metrics.wL_f2cx[i] * U[i, j] +
                   metrics.wR_f2cx[i] * U[i + 1, j]
        Vc[i, j] = metrics.wB_f2cy[j] * V[i, j] +
                   metrics.wT_f2cy[j] * V[i, j + 1]
    end

    # du/dy at cell centers.
    @inbounds for i in 1:Nx
        if metrics.periodic_y
            if Ny == 1
                du_dy[i, 1] = 0.0
            else
                for j in 1:Ny
                    jm = j == 1 ? Ny : j - 1
                    jp = j == Ny ? 1 : j + 1
                    hminus = metrics.dy_f[j]
                    hplus = metrics.dy_f[j + 1]
                    du_dy[i, j] = (Uc[i, jp] - Uc[i, jm]) / (hminus + hplus)
                end
            end
        else
            if Ny == 1
                if bc_U.bottom isa Neumann && bc_U.top isa Neumann
                    gb = -Float64(_wall_value_at_x_center(bc_U.bottom, i, metrics))
                    gt =  Float64(_wall_value_at_x_center(bc_U.top, i, metrics))
                    du_dy[i, 1] = 0.5 * (gb + gt)
                elseif bc_U.bottom isa Dirichlet && bc_U.top isa Dirichlet
                    ub = Float64(_wall_value_at_x_center(bc_U.bottom, i, metrics))
                    ut = Float64(_wall_value_at_x_center(bc_U.top, i, metrics))
                    du_dy[i, 1] = (ut - ub) / metrics.Ly
                else
                    throw(ArgumentError(
                        "For Ny=1, du/dy requires matching Dirichlet or Neumann U data on bottom/top."
                    ))
                end
            else
                if bc_U.bottom isa Dirichlet
                    ub = Float64(_wall_value_at_x_center(bc_U.bottom, i, metrics))
                    du_dy[i, 1] = (Uc[i, 1] - ub) / metrics.dy_north_corner[1]
                elseif bc_U.bottom isa Neumann
                    du_dy[i, 1] = -Float64(_wall_value_at_x_center(bc_U.bottom, i, metrics))
                else
                    throw(ArgumentError("Unsupported U-bottom boundary for du/dy."))
                end

                for j in 2:(Ny - 1)
                    du_dy[i, j] = (Uc[i, j + 1] - Uc[i, j - 1]) /
                                  (metrics.dy_f[j] + metrics.dy_f[j + 1])
                end

                if bc_U.top isa Dirichlet
                    ut = Float64(_wall_value_at_x_center(bc_U.top, i, metrics))
                    du_dy[i, Ny] = (ut - Uc[i, Ny]) / metrics.dy_south_corner[end]
                elseif bc_U.top isa Neumann
                    du_dy[i, Ny] = Float64(_wall_value_at_x_center(bc_U.top, i, metrics))
                else
                    throw(ArgumentError("Unsupported U-top boundary for du/dy."))
                end
            end
        end
    end

    # dv/dx at cell centers.
    @inbounds for j in 1:Ny
        if metrics.periodic_x
            if Nx == 1
                dv_dx[1, j] = 0.0
            else
                for i in 1:Nx
                    im = i == 1 ? Nx : i - 1
                    ip = i == Nx ? 1 : i + 1
                    hminus = metrics.dx_f[i]
                    hplus = metrics.dx_f[i + 1]
                    dv_dx[i, j] = (Vc[ip, j] - Vc[im, j]) / (hminus + hplus)
                end
            end
        else
            if Nx == 1
                if bc_V.left isa Neumann && bc_V.right isa Neumann
                    gl = -Float64(_wall_value_at_y_center(bc_V.left, j, metrics))
                    gr =  Float64(_wall_value_at_y_center(bc_V.right, j, metrics))
                    dv_dx[1, j] = 0.5 * (gl + gr)
                elseif bc_V.left isa Dirichlet && bc_V.right isa Dirichlet
                    vl = Float64(_wall_value_at_y_center(bc_V.left, j, metrics))
                    vr = Float64(_wall_value_at_y_center(bc_V.right, j, metrics))
                    dv_dx[1, j] = (vr - vl) / metrics.Lx
                else
                    throw(ArgumentError(
                        "For Nx=1, dv/dx requires matching Dirichlet or Neumann V data on left/right."
                    ))
                end
            else
                if bc_V.left isa Dirichlet
                    vl = Float64(_wall_value_at_y_center(bc_V.left, j, metrics))
                    dv_dx[1, j] = (Vc[1, j] - vl) / metrics.dx_east_corner[1]
                elseif bc_V.left isa Neumann
                    dv_dx[1, j] = -Float64(_wall_value_at_y_center(bc_V.left, j, metrics))
                else
                    throw(ArgumentError("Unsupported V-left boundary for dv/dx."))
                end

                for i in 2:(Nx - 1)
                    dv_dx[i, j] = (Vc[i + 1, j] - Vc[i - 1, j]) /
                                  (metrics.dx_f[i] + metrics.dx_f[i + 1])
                end

                if bc_V.right isa Dirichlet
                    vr = Float64(_wall_value_at_y_center(bc_V.right, j, metrics))
                    dv_dx[Nx, j] = (vr - Vc[Nx, j]) / metrics.dx_west_corner[end]
                elseif bc_V.right isa Neumann
                    dv_dx[Nx, j] = Float64(_wall_value_at_y_center(bc_V.right, j, metrics))
                else
                    throw(ArgumentError("Unsupported V-right boundary for dv/dx."))
                end
            end
        end
    end

    return du_dx, du_dy, dv_dx, dv_dy
end

# Boundary data for a U-component wall running in x, interpolated from x-faces
# to the cell-center x-coordinate.
function _wall_value_at_x_center(
    bc::Union{Dirichlet,Neumann},
    i::Int,
    metrics::MeshMetrics,
)
    value = bc.value
    value isa Number && return value

    left = boundary_value(bc, i)
    right = boundary_value(bc, i + 1)
    return metrics.wL_f2cx[i] * left + metrics.wR_f2cx[i] * right
end

# Boundary data for a V-component wall running in y, interpolated from y-faces
# to the cell-center y-coordinate.
function _wall_value_at_y_center(
    bc::Union{Dirichlet,Neumann},
    j::Int,
    metrics::MeshMetrics,
)
    value = bc.value
    value isa Number && return value

    bottom = boundary_value(bc, j)
    top = boundary_value(bc, j + 1)
    return metrics.wB_f2cy[j] * bottom + metrics.wT_f2cy[j] * top
end

# GENERIC SYMMETRIC-STRESS DIVERGENCE

function symmetric_stress_divergence(
    Txx,
    Tyy,
    Txy,
    metrics::MeshMetrics,
)
    _validate_scalar_field(Txx, metrics, "Txx")
    _validate_scalar_field(Tyy, metrics, "Tyy")
    _validate_scalar_field(Txy, metrics, "Txy")

    Nx, Ny = metrics.Nx, metrics.Ny

    # Txy represented on x-faces at cell-center y locations and on y-faces at
    # cell-center x locations. Physical boundary values use linear
    # extrapolation rather than constant copying.
    Txy_xface = _center_to_x_faces_linear_boundary(Txy, metrics)
    Txy_yface = _center_to_y_faces_linear_boundary(Txy, metrics)

    Fx = zeros(Float64, Nx + 1, Ny)
    Fy = zeros(Float64, Nx, Ny + 1)

    # x-momentum: d(Txx)/dx + d(Txy)/dy on U locations.
    @inbounds for j in 1:Ny
        # Normal derivative d(Txx)/dx on all x-faces.
        if metrics.periodic_x
            wrapped = (Txx[1, j] - Txx[Nx, j]) / metrics.dx_f[1]
            Fx[1, j] = wrapped
            Fx[Nx + 1, j] = wrapped
        else
            left_wall = _linear_extrapolate_left(Txx, j, metrics)
            right_wall = _linear_extrapolate_right(Txx, j, metrics)
            Fx[1, j] = (Txx[1, j] - left_wall) / metrics.dx_east_corner[1]
            Fx[Nx + 1, j] = (right_wall - Txx[Nx, j]) / metrics.dx_west_corner[end]
        end
        for i in 2:Nx
            Fx[i, j] = (Txx[i, j] - Txx[i - 1, j]) / metrics.dx_f[i]
        end
    end

    # Add d(Txy)/dy at each U location.
    @inbounds for i in 1:(Nx + 1)
        if metrics.periodic_y
            if Ny > 1
                for j in 1:Ny
                    jm = j == 1 ? Ny : j - 1
                    jp = j == Ny ? 1 : j + 1
                    Fx[i, j] += (Txy_xface[i, jp] - Txy_xface[i, jm]) /
                                (metrics.dy_f[j] + metrics.dy_f[j + 1])
                end
            end
        elseif Ny == 1
            # With one cell there is no interior information from which to
            # infer a transverse gradient; the neutral extrapolation is zero.
            nothing
        else
            bottom_wall = _linear_extrapolate_bottom_row(Txy_xface, i, metrics)
            top_wall = _linear_extrapolate_top_row(Txy_xface, i, metrics)
            Fx[i, 1] += (Txy_xface[i, 1] - bottom_wall) / metrics.dy_north_corner[1]
            for j in 2:(Ny - 1)
                Fx[i, j] += (Txy_xface[i, j + 1] - Txy_xface[i, j - 1]) /
                            (metrics.dy_f[j] + metrics.dy_f[j + 1])
            end
            Fx[i, Ny] += (top_wall - Txy_xface[i, Ny]) / metrics.dy_south_corner[end]
        end
    end

    # y-momentum: d(Txy)/dx + d(Tyy)/dy on V locations.
    @inbounds for i in 1:Nx
        if metrics.periodic_y
            wrapped = (Tyy[i, 1] - Tyy[i, Ny]) / metrics.dy_f[1]
            Fy[i, 1] = wrapped
            Fy[i, Ny + 1] = wrapped
        else
            bottom_wall = _linear_extrapolate_bottom(Tyy, i, metrics)
            top_wall = _linear_extrapolate_top(Tyy, i, metrics)
            Fy[i, 1] = (Tyy[i, 1] - bottom_wall) / metrics.dy_north_corner[1]
            Fy[i, Ny + 1] = (top_wall - Tyy[i, Ny]) / metrics.dy_south_corner[end]
        end
        for j in 2:Ny
            Fy[i, j] = (Tyy[i, j] - Tyy[i, j - 1]) / metrics.dy_f[j]
        end
    end

    @inbounds for j in 1:(Ny + 1)
        if metrics.periodic_x
            if Nx > 1
                for i in 1:Nx
                    im = i == 1 ? Nx : i - 1
                    ip = i == Nx ? 1 : i + 1
                    Fy[i, j] += (Txy_yface[ip, j] - Txy_yface[im, j]) /
                                (metrics.dx_f[i] + metrics.dx_f[i + 1])
                end
            end
        elseif Nx == 1
            nothing
        else
            left_wall = _linear_extrapolate_left_column(Txy_yface, j, metrics)
            right_wall = _linear_extrapolate_right_column(Txy_yface, j, metrics)
            Fy[1, j] += (Txy_yface[1, j] - left_wall) / metrics.dx_east_corner[1]
            for i in 2:(Nx - 1)
                Fy[i, j] += (Txy_yface[i + 1, j] - Txy_yface[i - 1, j]) /
                            (metrics.dx_f[i] + metrics.dx_f[i + 1])
            end
            Fy[Nx, j] += (right_wall - Txy_yface[Nx, j]) / metrics.dx_west_corner[end]
        end
    end

    if metrics.periodic_x
        Fx[Nx + 1, :] .= Fx[1, :]
    end
    if metrics.periodic_y
        Fy[:, Ny + 1] .= Fy[:, 1]
    end

    return Fx, Fy
end

# Linear extrapolation of a cell-centered field to physical x-boundaries.
@inline function _linear_extrapolate_left(field, j::Int, metrics::MeshMetrics)
    Nx = metrics.Nx
    Nx == 1 && return Float64(field[1, j])
    d = metrics.dx_east_corner[1]
    return Float64(field[1, j]) -
           (Float64(field[2, j]) - Float64(field[1, j])) * d / metrics.dx_f[2]
end

@inline function _linear_extrapolate_right(field, j::Int, metrics::MeshMetrics)
    Nx = metrics.Nx
    Nx == 1 && return Float64(field[Nx, j])
    d = metrics.dx_west_corner[end]
    return Float64(field[Nx, j]) +
           (Float64(field[Nx, j]) - Float64(field[Nx - 1, j])) * d / metrics.dx_f[Nx]
end

@inline function _linear_extrapolate_bottom(field, i::Int, metrics::MeshMetrics)
    Ny = metrics.Ny
    Ny == 1 && return Float64(field[i, 1])
    d = metrics.dy_north_corner[1]
    return Float64(field[i, 1]) -
           (Float64(field[i, 2]) - Float64(field[i, 1])) * d / metrics.dy_f[2]
end

@inline function _linear_extrapolate_top(field, i::Int, metrics::MeshMetrics)
    Ny = metrics.Ny
    Ny == 1 && return Float64(field[i, Ny])
    d = metrics.dy_south_corner[end]
    return Float64(field[i, Ny]) +
           (Float64(field[i, Ny]) - Float64(field[i, Ny - 1])) * d / metrics.dy_f[Ny]
end

function _center_to_x_faces_linear_boundary(field, metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    face = zeros(Float64, Nx + 1, Ny)
    @inbounds for j in 1:Ny
        if metrics.periodic_x
            value = metrics.wL_c2fx[1] * field[Nx, j] +
                    metrics.wR_c2fx[1] * field[1, j]
            face[1, j] = value
            face[Nx + 1, j] = value
        else
            face[1, j] = _linear_extrapolate_left(field, j, metrics)
            face[Nx + 1, j] = _linear_extrapolate_right(field, j, metrics)
        end
        for i in 2:Nx
            face[i, j] = metrics.wL_c2fx[i] * field[i - 1, j] +
                         metrics.wR_c2fx[i] * field[i, j]
        end
    end
    return face
end

function _center_to_y_faces_linear_boundary(field, metrics::MeshMetrics)
    Nx, Ny = metrics.Nx, metrics.Ny
    face = zeros(Float64, Nx, Ny + 1)
    @inbounds for i in 1:Nx
        if metrics.periodic_y
            value = metrics.wB_c2fy[1] * field[i, Ny] +
                    metrics.wT_c2fy[1] * field[i, 1]
            face[i, 1] = value
            face[i, Ny + 1] = value
        else
            face[i, 1] = _linear_extrapolate_bottom(field, i, metrics)
            face[i, Ny + 1] = _linear_extrapolate_top(field, i, metrics)
        end
        for j in 2:Ny
            face[i, j] = metrics.wB_c2fy[j] * field[i, j - 1] +
                         metrics.wT_c2fy[j] * field[i, j]
        end
    end
    return face
end

# Extrapolate a field already located at x-faces along the y direction.
@inline function _linear_extrapolate_bottom_row(field, i::Int, metrics::MeshMetrics)
    metrics.Ny == 1 && return Float64(field[i, 1])
    d = metrics.dy_north_corner[1]
    return Float64(field[i, 1]) -
           (Float64(field[i, 2]) - Float64(field[i, 1])) * d / metrics.dy_f[2]
end

@inline function _linear_extrapolate_top_row(field, i::Int, metrics::MeshMetrics)
    Ny = metrics.Ny
    Ny == 1 && return Float64(field[i, Ny])
    d = metrics.dy_south_corner[end]
    return Float64(field[i, Ny]) +
           (Float64(field[i, Ny]) - Float64(field[i, Ny - 1])) * d / metrics.dy_f[Ny]
end

# Extrapolate a field already located at y-faces along the x direction.
@inline function _linear_extrapolate_left_column(field, j::Int, metrics::MeshMetrics)
    metrics.Nx == 1 && return Float64(field[1, j])
    d = metrics.dx_east_corner[1]
    return Float64(field[1, j]) -
           (Float64(field[2, j]) - Float64(field[1, j])) * d / metrics.dx_f[2]
end

@inline function _linear_extrapolate_right_column(field, j::Int, metrics::MeshMetrics)
    Nx = metrics.Nx
    Nx == 1 && return Float64(field[Nx, j])
    d = metrics.dx_west_corner[end]
    return Float64(field[Nx, j]) +
           (Float64(field[Nx, j]) - Float64(field[Nx - 1, j])) * d / metrics.dx_f[Nx]
end


# COMPACT VELOCITY-GRADIENT OPERATORS

"""Construct the compact finite-difference velocity-gradient operators used by the implicit viscous term."""
function build_velocity_gradient_operators(
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    _validate_velocity_boundary_periodicity(metrics, bc_U, bc_V)
    _require_supported_viscous_boundaries(bc_U, bc_V)

    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny

    Iux, Jux, Vux = Int[], Int[], Float64[]
    Ivy, Jvy, Vvy = Int[], Int[], Float64[]
    Iuy, Juy, Vuy = Int[], Int[], Float64[]
    Ivx, Jvx, Vvx = Int[], Int[], Float64[]

    du_dx_bc = zeros(Float64, N)
    dv_dy_bc = zeros(Float64, N)

    @inbounds for j in 1:Ny, i in 1:Nx
        row = _center_index(i, j, Nx)

        # du/dx at cell centers.
        inv_dx = 1.0 / metrics.dx_c[i]
        push!(Iux, row); push!(Jux, row); push!(Vux, -inv_dx)
        if i < Nx
            push!(Iux, row); push!(Jux, _center_index(i + 1, j, Nx)); push!(Vux, inv_dx)
        elseif metrics.periodic_x
            push!(Iux, row); push!(Jux, _center_index(1, j, Nx)); push!(Vux, inv_dx)
        else
            du_dx_bc[row] = Float64(boundary_value(bc_U.right, j)) * inv_dx
        end

        # dv/dy at cell centers.
        inv_dy = 1.0 / metrics.dy_c[j]
        push!(Ivy, row); push!(Jvy, row); push!(Vvy, -inv_dy)
        if j < Ny
            push!(Ivy, row); push!(Jvy, _center_index(i, j + 1, Nx)); push!(Vvy, inv_dy)
        elseif metrics.periodic_y
            push!(Ivy, row); push!(Jvy, _center_index(i, 1, Nx)); push!(Vvy, inv_dy)
        else
            dv_dy_bc[row] = Float64(boundary_value(bc_V.top, i)) * inv_dy
        end

        # du/dy at compact corners (x_u[i], y_v[j]). Boundary shear terms on
        # bottom/top physical walls are added later as stress-divergence lifting.
        if j > 1
            inv_h = 1.0 / metrics.dy_f[j]
            push!(Iuy, row); push!(Juy, _center_index(i, j - 1, Nx)); push!(Vuy, -inv_h)
            push!(Iuy, row); push!(Juy, row); push!(Vuy, inv_h)
        elseif metrics.periodic_y
            inv_h = 1.0 / metrics.dy_f[1]
            push!(Iuy, row); push!(Juy, _center_index(i, Ny, Nx)); push!(Vuy, -inv_h)
            push!(Iuy, row); push!(Juy, row); push!(Vuy, inv_h)
        end

        # dv/dx at compact corners.
        if i > 1
            inv_h = 1.0 / metrics.dx_f[i]
            push!(Ivx, row); push!(Jvx, _center_index(i - 1, j, Nx)); push!(Vvx, -inv_h)
            push!(Ivx, row); push!(Jvx, row); push!(Vvx, inv_h)
        elseif metrics.periodic_x
            inv_h = 1.0 / metrics.dx_f[1]
            push!(Ivx, row); push!(Jvx, _center_index(Nx, j, Nx)); push!(Vvx, -inv_h)
            push!(Ivx, row); push!(Jvx, row); push!(Vvx, inv_h)
        end
    end

    return VelocityGradientOperators(
        sparse(Iux, Jux, Vux, N, N),
        sparse(Ivy, Jvy, Vvy, N, N),
        sparse(Iuy, Juy, Vuy, N, N),
        sparse(Ivx, Jvx, Vvx, N, N),
        du_dx_bc,
        dv_dy_bc,
    )
end

# VISCOUS STRESS-DIVERGENCE OPERATOR

raw"""Construct the compact affine finite-difference operator for \nabla\cdot\left[2\mu D(\mathbf{v})\right]."""
function build_viscous_operator(
    metrics::MeshMetrics,
    viscosity_center,
    gradients::VelocityGradientOperators,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    _validate_scalar_field(viscosity_center, metrics, "viscosity_center")
    _validate_velocity_boundary_periodicity(metrics, bc_U, bc_V)
    _require_supported_viscous_boundaries(bc_U, bc_V)
    all(isfinite, viscosity_center) || throw(ArgumentError("Viscosity values must be finite."))
    all(viscosity_center .> 0.0) || throw(ArgumentError("Viscosity values must be positive."))

    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny

    mu_center = vec(Float64.(viscosity_center))
    mu_corner_full = interpolate_center_to_corners(viscosity_center, metrics)
    mu_corner = vec(mu_corner_full[1:Nx, 1:Ny])

    # Symmetric rate-of-deformation operator E.
    Z = spzeros(Float64, N, N)
    E = [
        gradients.du_dx              Z;
        0.5 * gradients.du_dy  0.5 * gradients.dv_dx;
        0.5 * gradients.du_dy  0.5 * gradients.dv_dx;
        Z                            gradients.dv_dy
    ]

    E_bc = vcat(
        gradients.du_dx_bc,
        zeros(Float64, N),
        zeros(Float64, N),
        gradients.dv_dy_bc,
    )

    # Constitutive multiplication Tau = 2 mu D(v).
    stress_weights = vcat(
        2.0 .* mu_center,
        2.0 .* mu_corner,
        2.0 .* mu_corner,
        2.0 .* mu_center,
    )
    M = spdiagm(0 => stress_weights)

    # Divergence of the four stress components.
    Ixx, Jxx, Vxx = Int[], Int[], Float64[]
    Ixy, Jxy, Vxy = Int[], Int[], Float64[]
    Iyx, Jyx, Vyx = Int[], Int[], Float64[]
    Iyy, Jyy, Vyy = Int[], Int[], Float64[]

    @inbounds for j in 1:Ny, i in 1:Nx
        row = _center_index(i, j, Nx)

        # d(tau_xx)/dx at compact U locations.
        inv_h = 1.0 / metrics.dx_f[i]
        push!(Ixx, row); push!(Jxx, row); push!(Vxx, inv_h)
        if i > 1
            push!(Ixx, row); push!(Jxx, _center_index(i - 1, j, Nx)); push!(Vxx, -inv_h)
        elseif metrics.periodic_x
            push!(Ixx, row); push!(Jxx, _center_index(Nx, j, Nx)); push!(Vxx, -inv_h)
        end

        # d(tau_xy)/dy at compact U locations.
        inv_h = 1.0 / metrics.dy_c[j]
        push!(Ixy, row); push!(Jxy, row); push!(Vxy, -inv_h)
        if j < Ny
            push!(Ixy, row); push!(Jxy, _center_index(i, j + 1, Nx)); push!(Vxy, inv_h)
        elseif metrics.periodic_y
            push!(Ixy, row); push!(Jxy, _center_index(i, 1, Nx)); push!(Vxy, inv_h)
        end

        # d(tau_yx)/dx at compact V locations.
        inv_h = 1.0 / metrics.dx_c[i]
        push!(Iyx, row); push!(Jyx, row); push!(Vyx, -inv_h)
        if i < Nx
            push!(Iyx, row); push!(Jyx, _center_index(i + 1, j, Nx)); push!(Vyx, inv_h)
        elseif metrics.periodic_x
            push!(Iyx, row); push!(Jyx, _center_index(1, j, Nx)); push!(Vyx, inv_h)
        end

        # d(tau_yy)/dy at compact V locations.
        inv_h = 1.0 / metrics.dy_f[j]
        push!(Iyy, row); push!(Jyy, row); push!(Vyy, inv_h)
        if j > 1
            push!(Iyy, row); push!(Jyy, _center_index(i, j - 1, Nx)); push!(Vyy, -inv_h)
        elseif metrics.periodic_y
            push!(Iyy, row); push!(Jyy, _center_index(i, Ny, Nx)); push!(Vyy, -inv_h)
        end
    end

    Div_xx = sparse(Ixx, Jxx, Vxx, N, N)
    Div_xy = sparse(Ixy, Jxy, Vxy, N, N)
    Div_yx = sparse(Iyx, Jyx, Vyx, N, N)
    Div_yy = sparse(Iyy, Jyy, Vyy, N, N)

    Dstress = [
        Div_xx  Div_xy  Z       Z;
        Z       Z       Div_yx  Div_yy
    ]

    matrix = Dstress * M * E
    boundary = Vector{Float64}(Dstress * (stress_weights .* E_bc))

    # Tangential Dirichlet data contribute shear stress at physical walls.
    # These terms are not represented by the compact corner set and are added
    # directly to the corresponding momentum rows.
    if !metrics.periodic_y
        dist_bottom = metrics.dy_f[1] / 2.0
        dist_top = metrics.dy_f[end] / 2.0

        @inbounds for i in 1:Nx
            if bc_U.bottom isa Dirichlet
                row = _center_index(i, 1, Nx)
                mu = mu_corner_full[i, 1]
                weight = mu / (metrics.dy_c[1] * dist_bottom)
                boundary[row] += weight * Float64(boundary_value(bc_U.bottom, i))
                matrix[row, row] -= weight
            end

            if bc_U.top isa Dirichlet
                row = _center_index(i, Ny, Nx)
                mu = mu_corner_full[i, Ny + 1]
                weight = mu / (metrics.dy_c[Ny] * dist_top)
                boundary[row] += weight * Float64(boundary_value(bc_U.top, i))
                matrix[row, row] -= weight
            end
        end
    end

    if !metrics.periodic_x
        dist_left = metrics.dx_f[1] / 2.0
        dist_right = metrics.dx_f[end] / 2.0

        @inbounds for j in 1:Ny
            if bc_V.left isa Dirichlet
                row = N + _center_index(1, j, Nx)
                mu = mu_corner_full[1, j]
                weight = mu / (metrics.dx_c[1] * dist_left)
                boundary[row] += weight * Float64(boundary_value(bc_V.left, j))
                matrix[row, row] -= weight
            end

            if bc_V.right isa Dirichlet
                row = N + _center_index(Nx, j, Nx)
                mu = mu_corner_full[Nx + 1, j]
                weight = mu / (metrics.dx_c[Nx] * dist_right)
                boundary[row] += weight * Float64(boundary_value(bc_V.right, j))
                matrix[row, row] -= weight
            end
        end
    end

    return AffineOperator(matrix, boundary)
end

function velocity_dirichlet_data(
    metrics::MeshMetrics,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    N = Nx * Ny
    indices = Int[]
    values = Float64[]

    if bc_U.left isa Dirichlet
        for j in 1:Ny
            push!(indices, _center_index(1, j, Nx))
            push!(values, Float64(boundary_value(bc_U.left, j)))
        end
    end

    if bc_V.bottom isa Dirichlet
        for i in 1:Nx
            push!(indices, N + _center_index(i, 1, Nx))
            push!(values, Float64(boundary_value(bc_V.bottom, i)))
        end
    end

    return indices, values
end

# OPERATOR APPLICATION

"""Apply an [`AffineOperator`](@ref): `A*x + boundary`."""
apply_operator(op::AffineOperator, x::AbstractVector) = op.matrix * x + op.boundary

end # module Operators
