"""Optional capillary-force models for the diffuse structural interface."""
module Capillarity

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators

export AbstractCapillarity,
       NoCapillarity,
       DiffuseAreaCapillarity,
       capillary_force

# CAPILLARY MODELS

"""Abstract supertype for capillary-force closures."""
abstract type AbstractCapillarity end

"""Disable capillary forcing explicitly."""
struct NoCapillarity <: AbstractCapillarity end

struct DiffuseAreaCapillarity <: AbstractCapillarity
    coefficient::Float64
    delta_normal::Float64
    smoothing_passes::Int

    function DiffuseAreaCapillarity(
        coefficient::Real;
        delta_normal::Real = 1e-6,
        smoothing_passes::Integer = 0,
    )
        coefficient_f = Float64(coefficient)
        delta_f = Float64(delta_normal)
        passes = Int(smoothing_passes)
        isfinite(coefficient_f) || throw(ArgumentError("coefficient must be finite."))
        coefficient_f >= 0.0 || throw(ArgumentError("coefficient must be nonnegative."))
        isfinite(delta_f) || throw(ArgumentError("delta_normal must be finite."))
        delta_f > 0.0 || throw(ArgumentError("delta_normal must be positive."))
        passes >= 0 || throw(ArgumentError("smoothing_passes must be nonnegative."))
        return new(coefficient_f, delta_f, passes)
    end
end

# VALIDATION AND GEOMETRY

function _validate_phi(phi, metrics::MeshMetrics, bc_phi::BoundaryConditionSetup)
    size(phi) == (metrics.Nx, metrics.Ny) || throw(DimensionMismatch(
        "phi must have size $((metrics.Nx, metrics.Ny)); received $(size(phi))."
    ))
    all(isfinite, phi) || throw(ArgumentError("phi must contain only finite values."))
    metrics.periodic_x == is_periodic_x(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in x."
    ))
    metrics.periodic_y == is_periodic_y(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in y."
    ))
    return nothing
end

function _gradient_at_centers(gx_face, gy_face, metrics::MeshMetrics)
    gx_center = zeros(Float64, metrics.Nx, metrics.Ny)
    gy_center = zeros(Float64, metrics.Nx, metrics.Ny)

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        gx_center[i, j] = metrics.wL_f2cx[i] * gx_face[i, j] +
                          metrics.wR_f2cx[i] * gx_face[i + 1, j]
        gy_center[i, j] = metrics.wB_f2cy[j] * gy_face[i, j] +
                          metrics.wT_f2cy[j] * gy_face[i, j + 1]
    end

    return gx_center, gy_center
end

function _regularized_normal(gx_center, gy_center, delta_normal::Float64)
    nx = similar(gx_center, Float64)
    ny = similar(gy_center, Float64)
    delta2 = delta_normal^2

    @inbounds for index in eachindex(gx_center, gy_center)
        gx = gx_center[index]
        gy = gy_center[index]
        denominator = sqrt(gx * gx + gy * gy + delta2)
        nx[index] = gx / denominator
        ny[index] = gy / denominator
    end

    return nx, ny
end


@inline function _capillary_center_value(phi, i::Int, j::Int, metrics::MeshMetrics)
    ii = metrics.periodic_x ? mod1(i, metrics.Nx) : clamp(i, 1, metrics.Nx)
    jj = metrics.periodic_y ? mod1(j, metrics.Ny) : clamp(j, 1, metrics.Ny)
    return Float64(phi[ii, jj])
end

function _smooth_phase_for_capillarity(phi, metrics::MeshMetrics, passes::Int)
    passes == 0 && return Float64.(phi)

    current = Float64.(phi)
    next = similar(current)
    for _ in 1:passes
        @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
            c  = _capillary_center_value(current, i,   j,   metrics)
            e  = _capillary_center_value(current, i+1, j,   metrics)
            w  = _capillary_center_value(current, i-1, j,   metrics)
            n  = _capillary_center_value(current, i,   j+1, metrics)
            so = _capillary_center_value(current, i,   j-1, metrics)
            ne = _capillary_center_value(current, i+1, j+1, metrics)
            nw = _capillary_center_value(current, i-1, j+1, metrics)
            se = _capillary_center_value(current, i+1, j-1, metrics)
            sw = _capillary_center_value(current, i-1, j-1, metrics)

            next[i, j] = 0.25*c + 0.125*(e+w+n+so) + 0.0625*(ne+nw+se+sw)
        end
        current, next = next, current
    end
    return current
end

# PUBLIC FORCE EVALUATION

"""Return capillary-force components on the full MAC faces."""
function capillary_force(
    ::NoCapillarity,
    phi,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_phi(phi, metrics, bc_phi)
    return zeros(Float64, metrics.Nx + 1, metrics.Ny),
           zeros(Float64, metrics.Nx, metrics.Ny + 1)
end

function capillary_force(
    model::DiffuseAreaCapillarity,
    phi,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_phi(phi, metrics, bc_phi)

    # 1. Capillary-force geometry.  Optionally smooth the phase
    #    field only for normal/curvature reconstruction.  The transported
    #    phase field is never modified, and the final CSF delta factor still
    #    uses the raw phase gradient below.
    phi_geometry = _smooth_phase_for_capillarity(
        phi, metrics, model.smoothing_passes,
    )
    geom_grad_x_face, geom_grad_y_face = scalar_gradient(
        phi_geometry, metrics, bc_phi,
    )

    # Raw gradient used in the final force f_Gamma = gamma*kappa*grad(phi).
    grad_x_face, grad_y_face = scalar_gradient(phi, metrics, bc_phi)

    # 2. Interpolate the geometry gradient to cell centers and construct the
    #    regularized continuum normal there.
    grad_x_center, grad_y_center = _gradient_at_centers(
        geom_grad_x_face,
        geom_grad_y_face,
        metrics,
    )
    normal_x_center, normal_y_center = _regularized_normal(
        grad_x_center,
        grad_y_center,
        model.delta_normal,
    )

    # 3. Place normal components on their corresponding MAC faces. The
    #    interpolation routines use periodic wrapping when required and
    #    constant boundary extrapolation otherwise, avoiding any reuse of phi
    #    Dirichlet data as boundary values for the normal components.
    normal_x_face = interpolate_center_to_x_faces(normal_x_center, metrics)
    normal_y_face = interpolate_center_to_y_faces(normal_y_center, metrics)

    # 4. kappa = -div(n) at cell centers.
    curvature_center = -divergence(normal_x_face, normal_y_face, metrics)

    # 5. Interpolate curvature to momentum locations and form
    #    f_Gamma = coefficient * kappa * grad(phi).
    curvature_x_face = interpolate_center_to_x_faces(curvature_center, metrics)
    curvature_y_face = interpolate_center_to_y_faces(curvature_center, metrics)

    force_x = model.coefficient .* curvature_x_face .* grad_x_face
    force_y = model.coefficient .* curvature_y_face .* grad_y_face

    return force_x, force_y
end

end # module Capillarity
