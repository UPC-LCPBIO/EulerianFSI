
module StructureTensor

using ..BoundaryConditions
using ..MeshGeneration
using ..Operators
using ..StructuralConstitutive

export AbstractStructureExtension,
       NoStructureExtension,
       IdentityOutsideStructure,
       GeometricStructureExtension,
       structure_tensor_rhs,
       structure_extension_rhs,
       apply_structure_extension!,
       structure_tensor_diagnostics

# NUMERICAL EXTENSION POLICIES

"""Abstract supertype for numerical extension policies applied to `sigma`."""
abstract type AbstractStructureExtension end

"""Leave the structural tensor unchanged outside the structural region."""
struct NoStructureExtension <: AbstractStructureExtension end

"""Reset `sigma` to the stress-free identity tensor wherever `phi < cutoff`."""
struct IdentityOutsideStructure{T<:Real} <: AbstractStructureExtension
    cutoff::T

    function IdentityOutsideStructure(cutoff::T) where {T<:Real}
        isfinite(cutoff) || throw(ArgumentError("Structure-extension cutoff must be finite."))
        0 <= cutoff <= 1 || throw(ArgumentError(
            "Structure-extension cutoff must lie in [0, 1]."
        ))
        return new{T}(cutoff)
    end
end

struct GeometricStructureExtension <: AbstractStructureExtension
    max_rate::Float64
    exponent::Float64
    safety_cutoff::Float64
end

function GeometricStructureExtension(
    max_rate::Real,
    exponent::Real=4.0;
    safety_cutoff::Real=1e-6,
)
    rate = Float64(max_rate)
    power = Float64(exponent)
    cutoff = Float64(safety_cutoff)

    isfinite(rate) || throw(ArgumentError("Geometric extension rate must be finite."))
    rate >= 0.0 || throw(ArgumentError("Geometric extension rate must be non-negative."))
    isfinite(power) || throw(ArgumentError("Geometric extension exponent must be finite."))
    power > 0.0 || throw(ArgumentError("Geometric extension exponent must be positive."))
    isfinite(cutoff) || throw(ArgumentError("Geometric extension safety cutoff must be finite."))
    0.0 <= cutoff < 1.0 || throw(ArgumentError(
        "Geometric extension safety cutoff must lie in [0, 1)."
    ))

    return GeometricStructureExtension(rate, power, cutoff)
end

# VALIDATION AND FIELD NORMALIZATION

function _validate_center_field(field, metrics::MeshMetrics, name::AbstractString)
    expected = (metrics.Nx, metrics.Ny)
    size(field) == expected || throw(DimensionMismatch(
        "$name must have size $expected; received $(size(field))."
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

function _validate_tensor_boundaries(metrics::MeshMetrics, bc_sigma)
    for name in (:xx, :yy, :xy)
        hasproperty(bc_sigma, name) || throw(ArgumentError(
            "bc_sigma must provide boundary setups named :xx, :yy, and :xy."
        ))
        bc = getproperty(bc_sigma, name)
        bc isa BoundaryConditionSetup || throw(ArgumentError(
            "bc_sigma.$name must be a BoundaryConditionSetup."
        ))
        metrics.periodic_x == is_periodic_x(bc) || throw(ArgumentError(
            "bc_sigma.$name does not match MeshMetrics periodicity in x."
        ))
        metrics.periodic_y == is_periodic_y(bc) || throw(ArgumentError(
            "bc_sigma.$name does not match MeshMetrics periodicity in y."
        ))
    end
    return nothing
end


# FIRST-ORDER UPWIND MATERIAL ADVECTION

@inline function _x_boundary_state(
    field,
    velocity::Real,
    side::Symbol,
    j::Int,
    bc::BoundaryConditionSetup,
)
    Nx = size(field, 1)

    if side === :left
        boundary = bc.left
        interior = field[1, j]
        inflow = velocity > 0.0
    elseif side === :right
        boundary = bc.right
        interior = field[Nx, j]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("x-boundary side must be :left or :right."))
    end

    velocity == 0.0 && return interior
    !inflow && return interior

    if boundary isa Dirichlet
        return Float64(boundary_value(boundary, j))
    elseif boundary isa Neumann
        throw(ArgumentError(
            "A Neumann structural-tensor condition on the $side boundary does not " *
            "define an incoming advected state. Prescribe Dirichlet data for inflow."
        ))
    elseif boundary isa Periodic
        throw(ArgumentError("Periodic x-boundary states must be handled by wrapping."))
    end

    error("Unsupported structural-tensor boundary condition $(typeof(boundary)).")
end

@inline function _y_boundary_state(
    field,
    velocity::Real,
    side::Symbol,
    i::Int,
    bc::BoundaryConditionSetup,
)
    Ny = size(field, 2)

    if side === :bottom
        boundary = bc.bottom
        interior = field[i, 1]
        inflow = velocity > 0.0
    elseif side === :top
        boundary = bc.top
        interior = field[i, Ny]
        inflow = velocity < 0.0
    else
        throw(ArgumentError("y-boundary side must be :bottom or :top."))
    end

    velocity == 0.0 && return interior
    !inflow && return interior

    if boundary isa Dirichlet
        return Float64(boundary_value(boundary, i))
    elseif boundary isa Neumann
        throw(ArgumentError(
            "A Neumann structural-tensor condition on the $side boundary does not " *
            "define an incoming advected state. Prescribe Dirichlet data for inflow."
        ))
    elseif boundary isa Periodic
        throw(ArgumentError("Periodic y-boundary states must be handled by wrapping."))
    end

    error("Unsupported structural-tensor boundary condition $(typeof(boundary)).")
end

function _conservative_upwind_advection(
    field,
    U,
    V,
    metrics::MeshMetrics,
    bc::BoundaryConditionSetup,
)
    Nx, Ny = metrics.Nx, metrics.Ny
    flux_x = zeros(Float64, Nx + 1, Ny)
    flux_y = zeros(Float64, Nx, Ny + 1)

    @inbounds for j in 1:Ny
        if metrics.periodic_x
            u = U[1, j]
            left = field[Nx, j]
            right = field[1, j]
            flux_x[1, j] = u * (u >= 0.0 ? left : right)
        else
            u = U[1, j]
            boundary_state = _x_boundary_state(field, u, :left, j, bc)
            flux_x[1, j] = u * (u >= 0.0 ? boundary_state : field[1, j])
        end

        for i in 2:Nx
            u = U[i, j]
            left = field[i - 1, j]
            right = field[i, j]
            flux_x[i, j] = u * (u >= 0.0 ? left : right)
        end

        if metrics.periodic_x
            flux_x[Nx + 1, j] = flux_x[1, j]
        else
            u = U[Nx + 1, j]
            boundary_state = _x_boundary_state(field, u, :right, j, bc)
            flux_x[Nx + 1, j] = u * (u >= 0.0 ? field[Nx, j] : boundary_state)
        end
    end

    @inbounds for i in 1:Nx
        if metrics.periodic_y
            v = V[i, 1]
            bottom = field[i, Ny]
            top = field[i, 1]
            flux_y[i, 1] = v * (v >= 0.0 ? bottom : top)
        else
            v = V[i, 1]
            boundary_state = _y_boundary_state(field, v, :bottom, i, bc)
            flux_y[i, 1] = v * (v >= 0.0 ? boundary_state : field[i, 1])
        end

        for j in 2:Ny
            v = V[i, j]
            bottom = field[i, j - 1]
            top = field[i, j]
            flux_y[i, j] = v * (v >= 0.0 ? bottom : top)
        end

        if metrics.periodic_y
            flux_y[i, Ny + 1] = flux_y[i, 1]
        else
            v = V[i, Ny + 1]
            boundary_state = _y_boundary_state(field, v, :top, i, bc)
            flux_y[i, Ny + 1] = v * (v >= 0.0 ? field[i, Ny] : boundary_state)
        end
    end

    rhs = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        rhs[i, j] = -(
            (flux_x[i + 1, j] - flux_x[i, j]) / metrics.dx_c[i] +
            (flux_y[i, j + 1] - flux_y[i, j]) / metrics.dy_c[j]
        )
    end

    return rhs
end

# STRUCTURAL-TENSOR RHS

"""Evaluate Eulerian structural-tensor transport."""
function structure_tensor_rhs(
    sigma_xx,
    sigma_yy,
    sigma_xy,
    U,
    V,
    relaxation_model::AbstractTensorRelaxation,
    metrics::MeshMetrics,
    bc_sigma,
    bc_U::BoundaryConditionSetup,
    bc_V::BoundaryConditionSetup;
    phi=nothing,
    structure_extension::AbstractStructureExtension=NoStructureExtension(),
)
    _validate_center_field(sigma_xx, metrics, "sigma_xx")
    _validate_center_field(sigma_yy, metrics, "sigma_yy")
    _validate_center_field(sigma_xy, metrics, "sigma_xy")
    _validate_velocity_fields(U, V, metrics)
    _validate_tensor_boundaries(metrics, bc_sigma)

    adv_xx_cons = _conservative_upwind_advection(
        sigma_xx, U, V, metrics, bc_sigma.xx
    )
    adv_yy_cons = _conservative_upwind_advection(
        sigma_yy, U, V, metrics, bc_sigma.yy
    )
    adv_xy_cons = _conservative_upwind_advection(
        sigma_xy, U, V, metrics, bc_sigma.xy
    )

    div_v = divergence(U, V, metrics)
    adv_xx = @. adv_xx_cons + sigma_xx * div_v
    adv_yy = @. adv_yy_cons + sigma_yy * div_v
    adv_xy = @. adv_xy_cons + sigma_xy * div_v

    du_dx, du_dy, dv_dx, dv_dy = velocity_gradient(U, V, metrics, bc_U, bc_V)

    A11 = du_dx
    A12 = du_dy
    A21 = dv_dx
    A22 = dv_dy

    deformation_xx = @. 2.0 * (A11 * sigma_xx + A12 * sigma_xy)
    deformation_yy = @. 2.0 * (A21 * sigma_xy + A22 * sigma_yy)
    deformation_xy = @. A21 * sigma_xx + A12 * sigma_yy + (A11 + A22) * sigma_xy

    relax_xx, relax_yy, relax_xy = relaxation_rhs(
        relaxation_model, sigma_xx, sigma_yy, sigma_xy
    )

    if phi === nothing
        structure_extension isa NoStructureExtension || throw(ArgumentError(
            "A phase field `phi` is required for a non-trivial structural-tensor extension model."
        ))
        return adv_xx .+ deformation_xx .+ relax_xx,
               adv_yy .+ deformation_yy .+ relax_yy,
               adv_xy .+ deformation_xy .+ relax_xy
    end

    _validate_center_field(phi, metrics, "phi")

    # NoStructureExtension and the legacy hard-reset policy have no continuous
    # extension RHS. Avoid allocating three zero fields in these common cases.
    if structure_extension isa NoStructureExtension ||
       structure_extension isa IdentityOutsideStructure
        return adv_xx .+ deformation_xx .+ relax_xx,
               adv_yy .+ deformation_yy .+ relax_yy,
               adv_xy .+ deformation_xy .+ relax_xy
    end

    ext_xx, ext_yy, ext_xy = structure_extension_rhs(
        structure_extension, sigma_xx, sigma_yy, sigma_xy, phi
    )

    return adv_xx .+ deformation_xx .+ relax_xx .+ ext_xx,
           adv_yy .+ deformation_yy .+ relax_yy .+ ext_yy,
           adv_xy .+ deformation_xy .+ relax_xy .+ ext_xy
end

# NUMERICAL EXTENSION

function structure_extension_rhs(
    ::AbstractStructureExtension,
    sigma_xx,
    sigma_yy,
    sigma_xy,
    phi,
)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy) && size(sigma_xy) == size(phi)) ||
        throw(DimensionMismatch("sigma components and phi must have identical sizes."))
    shape = size(sigma_xx)
    return zeros(Float64, shape), zeros(Float64, shape), zeros(Float64, shape)
end

@inline function _sigma_log_product_2d(a::Float64, d::Float64, b::Float64)
    # For symmetric SPD S, compute S*log(S) without allocating an
    # eigendecomposition. For g(lambda)=lambda*log(lambda), g(S)=alpha*S+beta*I.
    m = 0.5 * (a + d)
    delta = 0.5 * (a - d)
    r = hypot(delta, b)
    lambda_hi = m + r
    lambda_lo = m - r

    if !(isfinite(lambda_lo) && isfinite(lambda_hi) && lambda_lo > 0.0)
        throw(DomainError(
            (a, d, b),
            "GeometricStructureExtension requires an SPD tensor; encountered a non-positive or non-finite eigenvalue.",
        ))
    end

    g_hi = lambda_hi * log(lambda_hi)
    g_lo = lambda_lo * log(lambda_lo)

    scale = max(1.0, abs(m))
    if r <= 1.0e-12 * scale
        # Hermite limit for repeated or nearly repeated eigenvalues.
        alpha = log(m) + 1.0
        beta = -m
    else
        denom = lambda_hi - lambda_lo
        alpha = (g_hi - g_lo) / denom
        beta = (lambda_hi * g_lo - lambda_lo * g_hi) / denom
    end

    return alpha * a + beta,
           alpha * d + beta,
           alpha * b
end

@inline function _geometric_extension_rate(phi::Real, model::GeometricStructureExtension)
    phi01 = clamp(Float64(phi), 0.0, 1.0)
    return model.max_rate * (1.0 - phi01)^model.exponent
end

function structure_extension_rhs(
    model::GeometricStructureExtension,
    sigma_xx,
    sigma_yy,
    sigma_xy,
    phi,
)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy) && size(sigma_xy) == size(phi)) ||
        throw(DimensionMismatch("sigma components and phi must have identical sizes."))

    rhs_xx = zeros(Float64, size(sigma_xx))
    rhs_yy = zeros(Float64, size(sigma_yy))
    rhs_xy = zeros(Float64, size(sigma_xy))

    @inbounds for index in eachindex(phi, sigma_xx, sigma_yy, sigma_xy)
        kappa = _geometric_extension_rate(phi[index], model)
        kappa == 0.0 && continue

        glog_xx, glog_yy, glog_xy = _sigma_log_product_2d(
            Float64(sigma_xx[index]),
            Float64(sigma_yy[index]),
            Float64(sigma_xy[index]),
        )
        rhs_xx[index] = -kappa * glog_xx
        rhs_yy[index] = -kappa * glog_yy
        rhs_xy[index] = -kappa * glog_xy
    end

    return rhs_xx, rhs_yy, rhs_xy
end

function apply_structure_extension!(
    ::NoStructureExtension,
    sigma_xx,
    sigma_yy,
    sigma_xy,
    phi,
)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy) && size(sigma_xy) == size(phi)) ||
        throw(DimensionMismatch("sigma components and phi must have identical sizes."))
    return 0
end

function apply_structure_extension!(
    model::IdentityOutsideStructure,
    sigma_xx,
    sigma_yy,
    sigma_xy,
    phi,
)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy) && size(sigma_xy) == size(phi)) ||
        throw(DimensionMismatch("sigma components and phi must have identical sizes."))

    cutoff = Float64(model.cutoff)
    reset_count = 0

    @inbounds for index in eachindex(phi, sigma_xx, sigma_yy, sigma_xy)
        if phi[index] < cutoff
            sigma_xx[index] = 1.0
            sigma_yy[index] = 1.0
            sigma_xy[index] = 0.0
            reset_count += 1
        end
    end

    return reset_count
end

function apply_structure_extension!(
    model::GeometricStructureExtension,
    sigma_xx,
    sigma_yy,
    sigma_xy,
    phi,
)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy) && size(sigma_xy) == size(phi)) ||
        throw(DimensionMismatch("sigma components and phi must have identical sizes."))

    cutoff = model.safety_cutoff
    cutoff <= 0.0 && return 0

    reset_count = 0
    @inbounds for index in eachindex(phi, sigma_xx, sigma_yy, sigma_xy)
        if phi[index] < cutoff
            sigma_xx[index] = 1.0
            sigma_yy[index] = 1.0
            sigma_xy[index] = 0.0
            reset_count += 1
        end
    end
    return reset_count
end

# SPD DIAGNOSTICS

function structure_tensor_diagnostics(sigma_xx, sigma_yy, sigma_xy)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy)) || throw(DimensionMismatch(
        "sigma_xx, sigma_yy, and sigma_xy must have identical sizes."
    ))

    minimum_determinant = Inf
    maximum_abs_determinant_error = 0.0
    minimum_eigenvalue = Inf
    non_spd_count = 0

    @inbounds for index in eachindex(sigma_xx, sigma_yy, sigma_xy)
        a = Float64(sigma_xx[index])
        d = Float64(sigma_yy[index])
        b = Float64(sigma_xy[index])

        determinant = a * d - b * b
        lambda_min = 0.5 * (a + d - hypot(a - d, 2.0 * b))

        minimum_determinant = min(minimum_determinant, determinant)
        maximum_abs_determinant_error = max(
            maximum_abs_determinant_error,
            abs(determinant - 1.0),
        )
        minimum_eigenvalue = min(minimum_eigenvalue, lambda_min)
        if !(isfinite(lambda_min) && lambda_min > 0.0)
            non_spd_count += 1
        end
    end

    return (
        minimum_determinant = minimum_determinant,
        maximum_abs_determinant_error = maximum_abs_determinant_error,
        minimum_eigenvalue = minimum_eigenvalue,
        non_spd_count = non_spd_count,
    )
end

end # module StructureTensor
