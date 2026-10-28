"""Numerical interface-flux models for the diffuse structural indicator `phi`."""
module InterfaceRegularization

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators: scalar_gradient,
                   interpolate_center_to_x_faces,
                   interpolate_center_to_y_faces

export AbstractInterfaceFlux,
       NoInterfaceFlux,
       ConservativeDiffuseInterfaceFlux,
       interface_flux

# INTERFACE-FLUX MODELS

"""Abstract supertype for optional numerical diffuse-interface flux models."""
abstract type AbstractInterfaceFlux end

raw"""Disable numerical interface regularization, i.e. \mathbf{j}_\Gamma=\mathbf{0}."""
struct NoInterfaceFlux <: AbstractInterfaceFlux end

struct ConservativeDiffuseInterfaceFlux <: AbstractInterfaceFlux
    inverse_peclet::Float64
    epsilon::Float64
    delta_normal::Float64
end

function ConservativeDiffuseInterfaceFlux(
    inverse_peclet::Real,
    epsilon::Real;
    delta_normal::Real = 1e-6,
)
    inverse_peclet_f = Float64(inverse_peclet)
    epsilon_f = Float64(epsilon)
    delta_normal_f = Float64(delta_normal)

    isfinite(inverse_peclet_f) || throw(ArgumentError(
        "inverse_peclet must be finite."
    ))
    isfinite(epsilon_f) || throw(ArgumentError(
        "epsilon must be finite."
    ))
    isfinite(delta_normal_f) || throw(ArgumentError(
        "delta_normal must be finite."
    ))

    inverse_peclet_f >= 0.0 || throw(ArgumentError(
        "inverse_peclet must be non-negative."
    ))
    epsilon_f > 0.0 || throw(ArgumentError(
        "epsilon must be strictly positive."
    ))
    delta_normal_f > 0.0 || throw(ArgumentError(
        "delta_normal must be strictly positive."
    ))

    return ConservativeDiffuseInterfaceFlux(
        inverse_peclet_f,
        epsilon_f,
        delta_normal_f,
    )
end

# VALIDATION

function _validate_fields(phi, U, V, metrics::MeshMetrics)
    size(phi) == (metrics.Nx, metrics.Ny) || throw(DimensionMismatch(
        "phi must have size $((metrics.Nx, metrics.Ny)); received $(size(phi))."
    ))
    size(U) == (metrics.Nx + 1, metrics.Ny) || throw(DimensionMismatch(
        "U must have size $((metrics.Nx + 1, metrics.Ny)); received $(size(U))."
    ))
    size(V) == (metrics.Nx, metrics.Ny + 1) || throw(DimensionMismatch(
        "V must have size $((metrics.Nx, metrics.Ny + 1)); received $(size(V))."
    ))

    all(isfinite, phi) || throw(ArgumentError("phi contains non-finite values."))
    all(isfinite, U) || throw(ArgumentError("U contains non-finite values."))
    all(isfinite, V) || throw(ArgumentError("V contains non-finite values."))
    return nothing
end

function _validate_periodicity(
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    metrics.periodic_x == is_periodic_x(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in x."
    ))
    metrics.periodic_y == is_periodic_y(bc_phi) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in y."
    ))
    return nothing
end

# DISCRETE GEOMETRY HELPERS

function _cell_centered_phase_gradient(
    phi,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    grad_x_face, grad_y_face = scalar_gradient(phi, metrics, bc_phi)

    Nx, Ny = metrics.Nx, metrics.Ny
    grad_x_center = zeros(Float64, Nx, Ny)
    grad_y_center = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        grad_x_center[i, j] =
            metrics.wL_f2cx[i] * grad_x_face[i, j] +
            metrics.wR_f2cx[i] * grad_x_face[i + 1, j]

        grad_y_center[i, j] =
            metrics.wB_f2cy[j] * grad_y_face[i, j] +
            metrics.wT_f2cy[j] * grad_y_face[i, j + 1]
    end

    return grad_x_center, grad_y_center, grad_x_face, grad_y_face
end

function _sharpening_vector(
    model::ConservativeDiffuseInterfaceFlux,
    phi,
    grad_x_center,
    grad_y_center,
)
    Nx, Ny = size(phi)
    sharpening_x = zeros(Float64, Nx, Ny)
    sharpening_y = zeros(Float64, Nx, Ny)
    delta2 = model.delta_normal^2

    @inbounds for j in 1:Ny, i in 1:Nx
        gx = grad_x_center[i, j]
        gy = grad_y_center[i, j]
        magnitude = sqrt(gx * gx + gy * gy + delta2)
        amplitude = phi[i, j] * (1.0 - phi[i, j])

        sharpening_x[i, j] = amplitude * gx / magnitude
        sharpening_y[i, j] = amplitude * gy / magnitude
    end

    return sharpening_x, sharpening_y
end

function _enforce_interface_flux_boundaries!(
    jx,
    jy,
    metrics::MeshMetrics,
)
    if metrics.periodic_x
        jx[end, :] .= jx[1, :]
    else
        jx[1, :] .= 0.0
        jx[end, :] .= 0.0
    end

    if metrics.periodic_y
        jy[:, end] .= jy[:, 1]
    else
        jy[:, 1] .= 0.0
        jy[:, end] .= 0.0
    end

    return nothing
end

# PUBLIC INTERFACE-FLUX DISPATCH

"""Construct an optional numerical interface flux on the full MAC faces."""
function interface_flux(
    ::NoInterfaceFlux,
    phi,
    U,
    V,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(phi, U, V, metrics)
    _validate_periodicity(metrics, bc_phi)

    return zeros(Float64, metrics.Nx + 1, metrics.Ny),
           zeros(Float64, metrics.Nx, metrics.Ny + 1)
end

function interface_flux(
    model::ConservativeDiffuseInterfaceFlux,
    phi,
    U,
    V,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(phi, U, V, metrics)
    _validate_periodicity(metrics, bc_phi)

    grad_x_center,
    grad_y_center,
    grad_x_face,
    grad_y_face = _cell_centered_phase_gradient(phi, metrics, bc_phi)

    sharpening_x,
    sharpening_y = _sharpening_vector(
        model,
        phi,
        grad_x_center,
        grad_y_center,
    )

    sharpening_x_face = interpolate_center_to_x_faces(sharpening_x, metrics)
    sharpening_y_face = interpolate_center_to_y_faces(sharpening_y, metrics)

    # `epsilon` follows the tanh(d/epsilon) convention used by
    # InterfaceInitialization. The corresponding equilibrium coefficient in
    # phi(1-phi)n - ell*grad(phi) is ell = epsilon/2.
    diffusion_length = 0.5 * model.epsilon

    jx = @. model.inverse_peclet * (
        sharpening_x_face - diffusion_length * grad_x_face
    )
    jy = @. model.inverse_peclet * (
        sharpening_y_face - diffusion_length * grad_y_face
    )

    _enforce_interface_flux_boundaries!(jx, jy, metrics)
    return jx, jy
end

end # module InterfaceRegularization
