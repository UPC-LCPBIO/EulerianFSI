"""Constitutive closures associated with the Eulerian structural tensor."""
module StructuralConstitutive

export AbstractTensorRelaxation,
       NoRelaxation,
       MaxwellRelaxation,
       AbstractElasticStressModel,
       NoElasticStress,
       LinearInternalTensorStress,
       relaxation_rhs,
       elastic_stress

# RELAXATION MODELS

"""Abstract supertype for structural-tensor relaxation laws."""
abstract type AbstractTensorRelaxation end

"""No constitutive relaxation: `R(sigma) = 0`."""
struct NoRelaxation <: AbstractTensorRelaxation end

"""Maxwell relaxation towards the stress-free state `sigma = I`: For the dimensionless baseline model, `rate = 1/Wi`."""
struct MaxwellRelaxation{T<:Real} <: AbstractTensorRelaxation
    rate::T

    function MaxwellRelaxation(rate::T) where {T<:Real}
        isfinite(rate) || throw(ArgumentError("Maxwell relaxation rate must be finite."))
        rate >= zero(T) || throw(ArgumentError("Maxwell relaxation rate must be non-negative."))
        return new{T}(rate)
    end
end

# ELASTIC-STRESS MODELS

"""Abstract supertype for elastic-stress closures based on the internal tensor."""
abstract type AbstractElasticStressModel end

"""No elastic stress: `tau_e = 0`."""
struct NoElasticStress <: AbstractElasticStressModel end

struct LinearInternalTensorStress{T<:Real} <: AbstractElasticStressModel
    coefficient::T

    function LinearInternalTensorStress(coefficient::T) where {T<:Real}
        isfinite(coefficient) || throw(ArgumentError("Elastic-stress coefficient must be finite."))
        coefficient >= zero(T) || throw(ArgumentError("Elastic-stress coefficient must be non-negative."))
        return new{T}(coefficient)
    end
end

# VALIDATION

function _validate_sigma_components(sigma_xx, sigma_yy, sigma_xy)
    (size(sigma_xx) == size(sigma_yy) && size(sigma_yy) == size(sigma_xy)) || throw(DimensionMismatch(
        "sigma_xx, sigma_yy, and sigma_xy must have identical sizes."
    ))
    return nothing
end

# RELAXATION EVALUATION

function relaxation_rhs(
    ::NoRelaxation,
    sigma_xx,
    sigma_yy,
    sigma_xy,
)
    _validate_sigma_components(sigma_xx, sigma_yy, sigma_xy)
    shape = size(sigma_xx)
    return zeros(Float64, shape), zeros(Float64, shape), zeros(Float64, shape)
end

function relaxation_rhs(
    model::MaxwellRelaxation,
    sigma_xx,
    sigma_yy,
    sigma_xy,
)
    _validate_sigma_components(sigma_xx, sigma_yy, sigma_xy)
    rate = Float64(model.rate)

    rhs_xx = @. -rate * (sigma_xx - 1.0)
    rhs_yy = @. -rate * (sigma_yy - 1.0)
    rhs_xy = @. -rate * sigma_xy
    return rhs_xx, rhs_yy, rhs_xy
end

# ELASTIC-STRESS EVALUATION

function elastic_stress(
    ::NoElasticStress,
    sigma_xx,
    sigma_yy,
    sigma_xy,
)
    _validate_sigma_components(sigma_xx, sigma_yy, sigma_xy)
    shape = size(sigma_xx)
    return zeros(Float64, shape), zeros(Float64, shape), zeros(Float64, shape)
end

function elastic_stress(
    model::LinearInternalTensorStress,
    sigma_xx,
    sigma_yy,
    sigma_xy,
)
    _validate_sigma_components(sigma_xx, sigma_yy, sigma_xy)
    coefficient = Float64(model.coefficient)

    tau_xx = @. coefficient * (sigma_xx - 1.0)
    tau_yy = @. coefficient * (sigma_yy - 1.0)
    tau_xy = @. coefficient * sigma_xy
    return tau_xx, tau_yy, tau_xy
end

end # module StructuralConstitutive
