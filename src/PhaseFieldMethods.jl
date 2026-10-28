"""Spatial transport operators for the conservative phase-field variable `phi`."""
module PhaseFieldMethods

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators
using ..InterfaceRegularization
using ..InterfaceTransport

export AbstractPhaseFieldMethod,
       LegacySplitPhaseFieldMethod,
       NoPhaseFieldRegularization,
       CurrentCDIPhaseField,
       JainACDIPhaseField,
       PirozzoliUWCDIPhaseField,
       phase_field_rhs,
       uwcdi_directional_rhs,
       uwcdi_strang_step,
       phase_field_label,
       jain_equilibrium_initialization_epsilon

# Types

"""Abstract supertype for complete spatial phase-field transport methods."""
abstract type AbstractPhaseFieldMethod end

"""Compatibility wrapper for the original `interface_flux` API."""
struct LegacySplitPhaseFieldMethod{IF<:AbstractInterfaceFlux} <: AbstractPhaseFieldMethod
    interface_flux::IF
end

"""Upwind material transport with no additional interface regularisation."""
struct NoPhaseFieldRegularization <: AbstractPhaseFieldMethod end

struct CurrentCDIPhaseField <: AbstractPhaseFieldMethod
    flux::ConservativeDiffuseInterfaceFlux
end

function CurrentCDIPhaseField(
    inverse_peclet::Real,
    epsilon::Real;
    delta_normal::Real=1e-6,
)
    return CurrentCDIPhaseField(
        ConservativeDiffuseInterfaceFlux(
            inverse_peclet,
            epsilon;
            delta_normal=delta_normal,
        ),
    )
end

"""Spatial ACDI discretization following Jain (2022)."""
struct JainACDIPhaseField <: AbstractPhaseFieldMethod
    gamma::Float64
    epsilon::Float64
    psi_floor::Float64
    delta_normal::Float64
    dynamic_gamma::Bool
end

function JainACDIPhaseField(
    gamma::Real,
    epsilon::Real;
    psi_floor::Real=1e-100,
    delta_normal::Real=1e-14,
    dynamic_gamma::Bool=true,
)
    gamma_f = Float64(gamma)
    epsilon_f = Float64(epsilon)
    psi_floor_f = Float64(psi_floor)
    delta_normal_f = Float64(delta_normal)

    isfinite(gamma_f) && gamma_f >= 0.0 || throw(ArgumentError(
        "gamma must be finite and non-negative."
    ))
    isfinite(epsilon_f) && epsilon_f > 0.0 || throw(ArgumentError(
        "epsilon must be finite and strictly positive."
    ))
    isfinite(psi_floor_f) && 0.0 < psi_floor_f < 0.5 || throw(ArgumentError(
        "psi_floor must lie strictly between 0 and 0.5."
    ))
    isfinite(delta_normal_f) && delta_normal_f > 0.0 || throw(ArgumentError(
        "delta_normal must be finite and strictly positive."
    ))

    return JainACDIPhaseField(
        gamma_f,
        epsilon_f,
        psi_floor_f,
        delta_normal_f,
        dynamic_gamma,
    )
end

"""Spatial UWCDI implementation following Pirozzoli, Di Giorgio & Rossi (Computers & Fluids 301, 106768, 2025)."""
struct PirozzoliUWCDIPhaseField <: AbstractPhaseFieldMethod
    alpha::Float64
    delta_normal::Float64
    normal_reconstruction::Symbol
end

function PirozzoliUWCDIPhaseField(
    alpha::Real=1.0;
    delta_normal::Real=1e-14,
    normal_reconstruction::Symbol=:standard,
)
    alpha_f = Float64(alpha)
    delta_f = Float64(delta_normal)
    isfinite(alpha_f) && alpha_f >= 1.0 || throw(ArgumentError(
        "alpha must be finite and >= 1 for the published UWCDI stability range."
    ))
    isfinite(delta_f) && delta_f > 0.0 || throw(ArgumentError(
        "delta_normal must be finite and strictly positive."
    ))
    normal_reconstruction in (:standard, :youngs, :smoothed9) || throw(ArgumentError(
        "normal_reconstruction must be :standard, :youngs, or :smoothed9."
    ))
    return PirozzoliUWCDIPhaseField(alpha_f, delta_f, normal_reconstruction)
end

jain_equilibrium_initialization_epsilon(model::JainACDIPhaseField) = 2.0 * model.epsilon

phase_field_label(::NoPhaseFieldRegularization) = "upwind_no_regularization"
phase_field_label(::CurrentCDIPhaseField) = "current_cdi"
phase_field_label(::JainACDIPhaseField) = "jain_acdi"
phase_field_label(::PirozzoliUWCDIPhaseField) = "pirozzoli_uwcdi"
phase_field_label(model::LegacySplitPhaseFieldMethod) =
    "legacy_split_" * string(nameof(typeof(model.interface_flux)))

# Shared validation and boundary helpers

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
    all(isfinite, phi) || throw(ArgumentError("phi contains non-finite values."))
    all(isfinite, U) || throw(ArgumentError("U contains non-finite values."))
    all(isfinite, V) || throw(ArgumentError("V contains non-finite values."))
    return nothing
end

function _validate_periodicity(metrics::MeshMetrics, bc::BoundaryConditionSetup)
    metrics.periodic_x == is_periodic_x(bc) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in x."
    ))
    metrics.periodic_y == is_periodic_y(bc) || throw(ArgumentError(
        "phi boundary conditions do not match MeshMetrics periodicity in y."
    ))
    return nothing
end

function _source_field(value, metrics::MeshMetrics)
    if value isa Number
        v = Float64(value)
        isfinite(v) || throw(ArgumentError("target_divergence must be finite."))
        return fill(v, metrics.Nx, metrics.Ny)
    end
    size(value) == (metrics.Nx, metrics.Ny) || throw(DimensionMismatch(
        "target_divergence has the wrong size."
    ))
    all(isfinite, value) || throw(ArgumentError(
        "target_divergence contains non-finite values."
    ))
    return Float64.(value)
end

@inline function _phase_boundary_value(bc, idx::Int, interior::Float64)
    if bc isa Dirichlet
        return Float64(boundary_value(bc, idx))
    elseif bc isa Neumann
        value = Float64(boundary_value(bc, idx))
        abs(value) <= 100 * eps(Float64) || throw(ArgumentError(
            "Only homogeneous Neumann phase boundaries are supported by the ACDI/UWCDI operators."
        ))
        return interior
    elseif bc isa AdvectiveOutflow
        return interior
    elseif bc isa MixedBoundary
        if mixed_is_dirichlet(bc, idx)
            return Float64(mixed_dirichlet_value(bc, idx))
        end
        value = Float64(mixed_neumann_value(bc, idx))
        abs(value) <= 100 * eps(Float64) || throw(ArgumentError(
            "Only homogeneous-Neumann mixed phase faces are supported here."
        ))
        return interior
    end
    throw(ArgumentError("Unsupported phase boundary condition $(typeof(bc))."))
end

@inline function _x_face_cells(i::Int, Nx::Int, periodic::Bool)
    if i == 1
        return periodic ? (Nx, 1) : (1, 1)
    elseif i == Nx + 1
        return periodic ? (Nx, 1) : (Nx, Nx)
    end
    return i - 1, i
end

@inline function _y_face_cells(j::Int, Ny::Int, periodic::Bool)
    if j == 1
        return periodic ? (Ny, 1) : (1, 1)
    elseif j == Ny + 1
        return periodic ? (Ny, 1) : (Ny, Ny)
    end
    return j - 1, j
end

# Geometry helpers

"""Cell-centered gradient from face gradients, consistent with repository geometry."""
function _center_gradient(field, metrics::MeshMetrics, bc::BoundaryConditionSetup)
    gx_face, gy_face = scalar_gradient(field, metrics, bc)
    gx = zeros(Float64, metrics.Nx, metrics.Ny)
    gy = zeros(Float64, metrics.Nx, metrics.Ny)

    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        gx[i, j] = metrics.wL_f2cx[i] * gx_face[i, j] +
                   metrics.wR_f2cx[i] * gx_face[i + 1, j]
        gy[i, j] = metrics.wB_f2cy[j] * gy_face[i, j] +
                   metrics.wT_f2cy[j] * gy_face[i, j + 1]
    end
    return gx, gy, gx_face, gy_face
end

function _center_normals(field, metrics, bc, delta_normal)
    gx, gy, gx_face, gy_face = _center_gradient(field, metrics, bc)
    nx = similar(gx)
    ny = similar(gy)
    d2 = delta_normal^2
    @inbounds for j in axes(field, 2), i in axes(field, 1)
        mag = sqrt(gx[i, j]^2 + gy[i, j]^2 + d2)
        nx[i, j] = gx[i, j] / mag
        ny[i, j] = gy[i, j] / mag
    end
    return nx, ny, gx_face, gy_face
end

function _face_normal_components_from_centers(nx_c, ny_c, metrics, delta_normal)
    nx_face = interpolate_center_to_x_faces(nx_c, metrics)
    ny_face = interpolate_center_to_y_faces(ny_c, metrics)

    # Renormalise the interpolated vector at each face. The transverse component
    # is interpolated solely for the normalisation; only the component normal to
    # the face enters the flux.
    ny_on_x = interpolate_center_to_x_faces(ny_c, metrics)
    nx_on_y = interpolate_center_to_y_faces(nx_c, metrics)
    d2 = delta_normal^2

    @inbounds for j in 1:metrics.Ny, i in 1:(metrics.Nx + 1)
        mag = sqrt(nx_face[i, j]^2 + ny_on_x[i, j]^2 + d2)
        nx_face[i, j] /= mag
    end
    @inbounds for j in 1:(metrics.Ny + 1), i in 1:metrics.Nx
        mag = sqrt(nx_on_y[i, j]^2 + ny_face[i, j]^2 + d2)
        ny_face[i, j] /= mag
    end

    if metrics.periodic_x
        nx_face[end, :] .= nx_face[1, :]
    end
    if metrics.periodic_y
        ny_face[:, end] .= ny_face[:, 1]
    end
    return nx_face, ny_face
end

function _normal_components_on_faces(field, metrics, bc, delta_normal)
    nx_c, ny_c, _, _ = _center_normals(field, metrics, bc, delta_normal)
    return _face_normal_components_from_centers(nx_c, ny_c, metrics, delta_normal)
end

@inline function _youngs_index(k::Int, n::Int, periodic::Bool)
    periodic && return mod1(k, n)
    return k
end

function _youngs_center_normals(field, metrics, bc, delta_normal)
    _validate_periodicity(metrics, bc)

    # The classical Youngs stencil is defined on a uniform Cartesian mesh.
    # Do not silently reinterpret it on a stretched grid.
    hx = metrics.dx_c[1]
    hy = metrics.dy_c[1]
    tolx = 100 * eps(Float64) * max(1.0, abs(hx))
    toly = 100 * eps(Float64) * max(1.0, abs(hy))
    maximum(abs.(metrics.dx_c .- hx)) <= tolx || throw(ArgumentError(
        "Youngs normal reconstruction currently requires a uniform x grid."
    ))
    maximum(abs.(metrics.dy_c .- hy)) <= toly || throw(ArgumentError(
        "Youngs normal reconstruction currently requires a uniform y grid."
    ))

    # Retain the repository-standard boundary treatment at physical walls.
    # Youngs' 3x3 stencil is applied to every cell with a complete stencil;
    nx, ny, _, _ = _center_normals(field, metrics, bc, delta_normal)
    Nx, Ny = metrics.Nx, metrics.Ny
    d2 = delta_normal^2

    @inbounds for j in 1:Ny, i in 1:Nx
        if (!metrics.periodic_x && (i == 1 || i == Nx)) ||
           (!metrics.periodic_y && (j == 1 || j == Ny))
            continue
        end

        im = _youngs_index(i - 1, Nx, metrics.periodic_x)
        ip = _youngs_index(i + 1, Nx, metrics.periodic_x)
        jm = _youngs_index(j - 1, Ny, metrics.periodic_y)
        jp = _youngs_index(j + 1, Ny, metrics.periodic_y)

        # Youngs' 2-D 3x3 weighted finite-difference gradient, equivalent to
        # averaging the finite-difference gradients evaluated at the four
        # corners of the central cell. The sign follows this solver's existing
        # convention n = +grad(phi)/|grad(phi)|.
        gx = (
            field[ip, jm] + 2.0 * field[ip, j] + field[ip, jp] -
            field[im, jm] - 2.0 * field[im, j] - field[im, jp]
        ) / (8.0 * hx)
        gy = (
            field[im, jp] + 2.0 * field[i, jp] + field[ip, jp] -
            field[im, jm] - 2.0 * field[i, jm] - field[ip, jm]
        ) / (8.0 * hy)

        mag = sqrt(gx^2 + gy^2 + d2)
        nx[i, j] = gx / mag
        ny[i, j] = gy / mag
    end

    return nx, ny
end

function _youngs_normal_components_on_faces(field, metrics, bc, delta_normal)
    nx_c, ny_c = _youngs_center_normals(field, metrics, bc, delta_normal)
    return _face_normal_components_from_centers(nx_c, ny_c, metrics, delta_normal)
end


# Legacy 9-point normal reconstruction

@inline function _smoothed9_index(k::Int, n::Int, periodic::Bool)
    periodic && return mod1(k, n)
    return clamp(k, 1, n)
end

function _validate_smoothed9_boundaries(bc::BoundaryConditionSetup)
    # The legacy lid-driven-cavity implementation used periodic wrapping or
    # zero-normal-gradient ghost values. Restrict this sensitivity option to
    # those cases rather than silently defining a different filter at other
    # physical boundaries.
    for (name, side) in ((:left, bc.left), (:right, bc.right),
                         (:bottom, bc.bottom), (:top, bc.top))
        if side isa Periodic
            continue
        elseif side isa Neumann
            value = Float64(boundary_value(side, 1))
            abs(value) <= 100 * eps(Float64) || throw(ArgumentError(
                "smoothed9 normal reconstruction requires homogeneous Neumann " *
                "or periodic phase boundaries; $name has Neumann value $value."
            ))
        else
            throw(ArgumentError(
                "smoothed9 normal reconstruction requires homogeneous Neumann " *
                "or periodic phase boundaries; $name uses $(typeof(side))."
            ))
        end
    end
    return nothing
end

function _smoothed9_field(field, metrics::MeshMetrics, bc::BoundaryConditionSetup)
    _validate_periodicity(metrics, bc)
    _validate_smoothed9_boundaries(bc)

    Nx, Ny = metrics.Nx, metrics.Ny
    smooth = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        im = _smoothed9_index(i - 1, Nx, metrics.periodic_x)
        ip = _smoothed9_index(i + 1, Nx, metrics.periodic_x)
        jm = _smoothed9_index(j - 1, Ny, metrics.periodic_y)
        jp = _smoothed9_index(j + 1, Ny, metrics.periodic_y)

        smooth[i, j] =
            0.25   * field[i, j] +
            0.125  * (field[ip, j] + field[im, j] + field[i, jp] + field[i, jm]) +
            0.0625 * (field[ip, jp] + field[im, jp] + field[ip, jm] + field[im, jm])
    end

    return smooth
end

function _smoothed9_normal_components_on_faces(field, metrics, bc, delta_normal)
    smooth = _smoothed9_field(field, metrics, bc)
    # This deliberately reuses the repository-standard centered-gradient path
    # after smoothing. On the uniform cavity grid this is the same interior
    return _normal_components_on_faces(smooth, metrics, bc, delta_normal)
end

# Baseline CDI and compatibility path

function phase_field_rhs(
    method::LegacySplitPhaseFieldMethod,
    U,
    V,
    phi,
    target_divergence,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    jx, jy = interface_flux(
        method.interface_flux,
        phi,
        U,
        V,
        metrics,
        bc_phi,
    )
    return interface_explicit_rhs(
        U,
        V,
        phi,
        target_divergence,
        jx,
        jy,
        metrics,
        bc_phi,
    )
end

function phase_field_rhs(
    ::NoPhaseFieldRegularization,
    U,
    V,
    phi,
    target_divergence,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    return phase_field_rhs(
        LegacySplitPhaseFieldMethod(NoInterfaceFlux()),
        U,
        V,
        phi,
        target_divergence,
        metrics,
        bc_phi,
    )
end

function phase_field_rhs(
    method::CurrentCDIPhaseField,
    U,
    V,
    phi,
    target_divergence,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    return phase_field_rhs(
        LegacySplitPhaseFieldMethod(method.flux),
        U,
        V,
        phi,
        target_divergence,
        metrics,
        bc_phi,
    )
end

# ACDI

@inline function _psi_value(phi_value::Real, model::JainACDIPhaseField)
    p = clamp(Float64(phi_value), 0.0, 1.0)
    e = model.psi_floor
    return model.epsilon * log((p + e) / (1.0 - p + e))
end

function _psi_field(phi, model::JainACDIPhaseField)
    psi = similar(Float64.(phi))
    @inbounds for idx in eachindex(phi)
        psi[idx] = _psi_value(phi[idx], model)
    end
    return psi
end

function _psi_boundary_setup(
    phi,
    model::JainACDIPhaseField,
    bc_phi::BoundaryConditionSetup,
)
    # Periodic and homogeneous-Neumann boundaries are used by the current
    # verification and FSI cases. Scalar Dirichlet data are mapped to psi.
    function transform_side(bc, side::Symbol)
        if bc isa Periodic
            return Periodic()
        elseif bc isa Neumann
            value = Float64(boundary_value(bc, 1))
            abs(value) <= 100 * eps(Float64) || throw(ArgumentError(
                "ACDI currently supports only homogeneous Neumann phase gradients."
            ))
            return Neumann(0.0)
        elseif bc isa Dirichlet
            # The boundary container cannot represent a transformed vector-valued
            # Dirichlet condition, so this path requires scalar data.
            value = boundary_value(bc, 1)
            value isa Number || throw(ArgumentError(
                "ACDI psi transformation currently requires scalar Dirichlet phase data."
            ))
            return Dirichlet(_psi_value(value, model))
        end
        throw(ArgumentError(
            "ACDI psi geometry supports Periodic, homogeneous Neumann, or scalar Dirichlet phase boundaries; got $(typeof(bc)) on $side."
        ))
    end

    return BoundaryConditionSetup(
        transform_side(bc_phi.left, :left),
        transform_side(bc_phi.right, :right),
        transform_side(bc_phi.bottom, :bottom),
        transform_side(bc_phi.top, :top),
    )
end

function _acdi_convective_fluxes(U, V, phi, metrics, bc_phi)
    Nx, Ny = metrics.Nx, metrics.Ny
    fx = zeros(Float64, Nx + 1, Ny)
    fy = zeros(Float64, Nx, Ny + 1)

    @inbounds for j in 1:Ny
        if metrics.periodic_x
            phi_face = metrics.wL_c2fx[1] * phi[Nx, j] +
                       metrics.wR_c2fx[1] * phi[1, j]
            fx[1, j] = U[1, j] * phi_face
            fx[end, j] = fx[1, j]
        else
            left_bc = _phase_boundary_value(bc_phi.left, j, Float64(phi[1, j]))
            right_bc = _phase_boundary_value(bc_phi.right, j, Float64(phi[Nx, j]))
            fx[1, j] = U[1, j] * left_bc
            fx[end, j] = U[end, j] * right_bc
        end
        for i in 2:Nx
            phi_face = metrics.wL_c2fx[i] * phi[i - 1, j] +
                       metrics.wR_c2fx[i] * phi[i, j]
            fx[i, j] = U[i, j] * phi_face
        end
    end

    @inbounds for i in 1:Nx
        if metrics.periodic_y
            phi_face = metrics.wB_c2fy[1] * phi[i, Ny] +
                       metrics.wT_c2fy[1] * phi[i, 1]
            fy[i, 1] = V[i, 1] * phi_face
            fy[i, end] = fy[i, 1]
        else
            bottom_bc = _phase_boundary_value(bc_phi.bottom, i, Float64(phi[i, 1]))
            top_bc = _phase_boundary_value(bc_phi.top, i, Float64(phi[i, Ny]))
            fy[i, 1] = V[i, 1] * bottom_bc
            fy[i, end] = V[i, end] * top_bc
        end
        for j in 2:Ny
            phi_face = metrics.wB_c2fy[j] * phi[i, j - 1] +
                       metrics.wT_c2fy[j] * phi[i, j]
            fy[i, j] = V[i, j] * phi_face
        end
    end
    return fx, fy
end

function _acdi_effective_gamma(
    model::JainACDIPhaseField,
    U,
    V,
    metrics::MeshMetrics,
)
    model.dynamic_gamma || return model.gamma

    # Jain uses Gamma = |u|_max. On the MAC grid the two velocity components
    # live on different faces, so evaluate the vector magnitude after the
    # standard face-to-center interpolation and take its global maximum.
    umax = 0.0
    @inbounds for j in 1:metrics.Ny, i in 1:metrics.Nx
        uc = metrics.wL_f2cx[i] * U[i, j] +
             metrics.wR_f2cx[i] * U[i + 1, j]
        vc = metrics.wB_f2cy[j] * V[i, j] +
             metrics.wT_f2cy[j] * V[i, j + 1]
        umax = max(umax, hypot(uc, vc))
    end
    return model.gamma * umax
end

function _acdi_regularization_fluxes(
    model::JainACDIPhaseField,
    U,
    V,
    phi,
    metrics,
    bc_phi,
)
    psi = _psi_field(phi, model)
    bc_psi = _psi_boundary_setup(phi, model, bc_phi)

    # Jain Eq. (21): diffusion remains in phi, whereas both the nonlinear
    # sharpening amplitude and normal are evaluated through psi.
    _, _, grad_phi_x, grad_phi_y = _center_gradient(phi, metrics, bc_phi)
    nx_c, ny_c, _, _ = _center_normals(psi, metrics, bc_psi, model.delta_normal)
    nx_face = interpolate_center_to_x_faces(nx_c, metrics)
    ny_face = interpolate_center_to_y_faces(ny_c, metrics)
    psi_x_face = interpolate_center_to_x_faces(psi, metrics)
    psi_y_face = interpolate_center_to_y_faces(psi, metrics)

    Ax = zeros(Float64, metrics.Nx + 1, metrics.Ny)
    Ay = zeros(Float64, metrics.Nx, metrics.Ny + 1)
    gamma_eff = _acdi_effective_gamma(model, U, V, metrics)

    @inbounds for j in 1:metrics.Ny, i in 1:(metrics.Nx + 1)
        amp = 0.25 * (1.0 - tanh(psi_x_face[i, j] / (2.0 * model.epsilon))^2)
        Ax[i, j] = gamma_eff * (
            model.epsilon * grad_phi_x[i, j] - amp * nx_face[i, j]
        )
    end
    @inbounds for j in 1:(metrics.Ny + 1), i in 1:metrics.Nx
        amp = 0.25 * (1.0 - tanh(psi_y_face[i, j] / (2.0 * model.epsilon))^2)
        Ay[i, j] = gamma_eff * (
            model.epsilon * grad_phi_y[i, j] - amp * ny_face[i, j]
        )
    end

    # The conservative Allen--Cahn regularisation has zero normal flux through
    # non-periodic solid boundaries. Periodic duplicate faces are synchronized.
    if metrics.periodic_x
        Ax[end, :] .= Ax[1, :]
    else
        Ax[1, :] .= 0.0
        Ax[end, :] .= 0.0
    end
    if metrics.periodic_y
        Ay[:, end] .= Ay[:, 1]
    else
        Ay[:, 1] .= 0.0
        Ay[:, end] .= 0.0
    end
    return Ax, Ay
end

function phase_field_rhs(
    model::JainACDIPhaseField,
    U,
    V,
    phi,
    target_divergence,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(U, V, phi, metrics)
    _validate_periodicity(metrics, bc_phi)

    fx, fy = _acdi_convective_fluxes(U, V, phi, metrics, bc_phi)
    Ax, Ay = _acdi_regularization_fluxes(model, U, V, phi, metrics, bc_phi)
    S = _source_field(target_divergence, metrics)

    return -divergence(fx, fy, metrics) .+
           divergence(Ax, Ay, metrics) .+
           Float64.(phi) .* S
end

# UWCDI

function _uwcdi_fluxes(
    model::PirozzoliUWCDIPhaseField,
    U,
    V,
    phi,
    metrics,
    bc_phi,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    nx_face, ny_face = if model.normal_reconstruction === :standard
        _normal_components_on_faces(
            phi,
            metrics,
            bc_phi,
            model.delta_normal,
        )
    elseif model.normal_reconstruction === :youngs
        _youngs_normal_components_on_faces(
            phi,
            metrics,
            bc_phi,
            model.delta_normal,
        )
    elseif model.normal_reconstruction === :smoothed9
        _smoothed9_normal_components_on_faces(
            phi,
            metrics,
            bc_phi,
            model.delta_normal,
        )
    else
        error("Unsupported UWCDI normal reconstruction: $(model.normal_reconstruction)")
    end
    fx = zeros(Float64, Nx + 1, Ny)
    fy = zeros(Float64, Nx, Ny + 1)

    @inbounds for j in 1:Ny, i in 1:(Nx + 1)
        iL, iR = _x_face_cells(i, Nx, metrics.periodic_x)
        u = U[i, j]

        if !metrics.periodic_x && i == 1
            boundary = _phase_boundary_value(bc_phi.left, j, Float64(phi[1, j]))
            phi_up = u >= 0.0 ? boundary : phi[1, j]
        elseif !metrics.periodic_x && i == Nx + 1
            boundary = _phase_boundary_value(bc_phi.right, j, Float64(phi[Nx, j]))
            phi_up = u >= 0.0 ? phi[Nx, j] : boundary
        else
            phi_up = u >= 0.0 ? phi[iL, j] : phi[iR, j]
        end

        # Section 2.3 multidimensional correction. On a uniform grid this
        # reduces to alpha_x = alpha.
        hy_local = metrics.dy_c[j]
        hx_local = metrics.dx_f[i]
        alpha_x = model.alpha * max(1.0, hy_local / hx_local)
        beta_x = abs(u) / alpha_x
        fx[i, j] = phi_up * (
            u + beta_x * nx_face[i, j] * (1.0 - phi_up)
        )
    end

    @inbounds for j in 1:(Ny + 1), i in 1:Nx
        jB, jT = _y_face_cells(j, Ny, metrics.periodic_y)
        v = V[i, j]

        if !metrics.periodic_y && j == 1
            boundary = _phase_boundary_value(bc_phi.bottom, i, Float64(phi[i, 1]))
            phi_up = v >= 0.0 ? boundary : phi[i, 1]
        elseif !metrics.periodic_y && j == Ny + 1
            boundary = _phase_boundary_value(bc_phi.top, i, Float64(phi[i, Ny]))
            phi_up = v >= 0.0 ? phi[i, Ny] : boundary
        else
            phi_up = v >= 0.0 ? phi[i, jB] : phi[i, jT]
        end

        hx_local = metrics.dx_c[i]
        hy_local = metrics.dy_f[j]
        alpha_y = model.alpha * max(1.0, hx_local / hy_local)
        beta_y = abs(v) / alpha_y
        fy[i, j] = phi_up * (
            v + beta_y * ny_face[i, j] * (1.0 - phi_up)
        )
    end

    if metrics.periodic_x
        fx[end, :] .= fx[1, :]
    end
    if metrics.periodic_y
        fy[:, end] .= fy[:, 1]
    end
    return fx, fy
end

function uwcdi_directional_rhs(
    model::PirozzoliUWCDIPhaseField,
    direction::Symbol,
    U,
    V,
    phi,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(U, V, phi, metrics)
    _validate_periodicity(metrics, bc_phi)

    fx, fy = _uwcdi_fluxes(model, U, V, phi, metrics, bc_phi)

    if direction === :x
        zero_y = zeros(Float64, metrics.Nx, metrics.Ny + 1)
        return -divergence(fx, zero_y, metrics)
    elseif direction === :y
        zero_x = zeros(Float64, metrics.Nx + 1, metrics.Ny)
        return -divergence(zero_x, fy, metrics)
    end

    throw(ArgumentError("direction must be :x or :y; received $direction."))
end

@inline function _uwcdi_directional_euler_sweep(
    model::PirozzoliUWCDIPhaseField,
    direction::Symbol,
    U,
    V,
    phi,
    dt::Real,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    dt_f = Float64(dt)
    dt_f >= 0.0 || throw(ArgumentError("Directional sweep dt must be non-negative."))
    dt_f == 0.0 && return copy(Float64.(phi))
    rhs = uwcdi_directional_rhs(model, direction, U, V, phi, metrics, bc_phi)
    return Float64.(phi) .+ dt_f .* rhs
end

function uwcdi_strang_step(
    model::PirozzoliUWCDIPhaseField,
    U,
    V,
    phi,
    dt::Real,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup;
    first_direction::Symbol=:x,
)
    dt_f = Float64(dt)
    isfinite(dt_f) && dt_f >= 0.0 || throw(ArgumentError(
        "dt must be finite and non-negative."
    ))

    if first_direction === :x
        q1 = _uwcdi_directional_euler_sweep(
            model, :x, U, V, phi, 0.5 * dt_f, metrics, bc_phi,
        )
        q2 = _uwcdi_directional_euler_sweep(
            model, :y, U, V, q1, dt_f, metrics, bc_phi,
        )
        return _uwcdi_directional_euler_sweep(
            model, :x, U, V, q2, 0.5 * dt_f, metrics, bc_phi,
        )
    elseif first_direction === :y
        q1 = _uwcdi_directional_euler_sweep(
            model, :y, U, V, phi, 0.5 * dt_f, metrics, bc_phi,
        )
        q2 = _uwcdi_directional_euler_sweep(
            model, :x, U, V, q1, dt_f, metrics, bc_phi,
        )
        return _uwcdi_directional_euler_sweep(
            model, :y, U, V, q2, 0.5 * dt_f, metrics, bc_phi,
        )
    end

    throw(ArgumentError(
        "first_direction must be :x or :y; received $first_direction."
    ))
end

function phase_field_rhs(
    model::PirozzoliUWCDIPhaseField,
    U,
    V,
    phi,
    target_divergence,
    metrics::MeshMetrics,
    bc_phi::BoundaryConditionSetup,
)
    _validate_fields(U, V, phi, metrics)
    _validate_periodicity(metrics, bc_phi)

    fx, fy = _uwcdi_fluxes(model, U, V, phi, metrics, bc_phi)
    S = _source_field(target_divergence, metrics)
    return -divergence(fx, fy, metrics) .+ Float64.(phi) .* S
end

end # module PhaseFieldMethods
