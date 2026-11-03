"""
    TransportConstitutive

Constitutive closures used by momentum and resource transport.

The numerical operators consume already evaluated coefficient or source fields.
This module therefore separates material modelling from finite-difference
geometry and from time integration.

The viscosity and diffusivity values returned here are the coefficients that
appear directly in the equations being solved. A nondimensional caller may,
for example, include factors such as `1/Re` or `1/Pe` when constructing the
corresponding model.

No clipping of the phase indicator or resource field is performed. If a model
produces a non-positive transport coefficient, the evaluator reports an error
instead of silently modifying the state.
"""
module TransportConstitutive

using ..MeshGeneration

export AbstractViscosityModel,
       NoViscosity,
       ConstantViscosity,
       HarmonicPhaseViscosity,
       viscosity_active,
       viscosity_field!,
       AbstractDiffusivityModel,
       NoDiffusion,
       ConstantDiffusivity,
       HarmonicPhaseDiffusivity,
       diffusion_active,
       diffusivity_field!,
       AbstractResourceConsumption,
       NoResourceConsumption,
       SaturatingResourceConsumption,
       resource_sink!,
       AbstractBodyForce,
       NoBodyForce,
       ConstantBodyForce,
       KolmogorovBodyForce,
       body_force!

# ==============================================================================
# COMMON VALIDATION
# ==============================================================================

function _validate_center_field(field, metrics::MeshMetrics, name::AbstractString)
    expected = (metrics.Nx, metrics.Ny)
    size(field) == expected || throw(DimensionMismatch(
        "$name must have size $expected; received $(size(field))."
    ))
    all(isfinite, field) || throw(ArgumentError("$name contains non-finite values."))
    return nothing
end

function _validate_output_shape(output, metrics::MeshMetrics, name::AbstractString)
    expected = (metrics.Nx, metrics.Ny)
    size(output) == expected || throw(DimensionMismatch(
        "$name must have size $expected; received $(size(output))."
    ))
    return nothing
end

function _validate_positive_field(field, name::AbstractString)
    all(isfinite, field) || throw(ArgumentError("$name contains non-finite values."))
    all(field .> 0.0) || throw(ArgumentError(
        "$name must be strictly positive everywhere."
    ))
    return nothing
end

# ==============================================================================
# VISCOSITY
# ==============================================================================

"""Abstract supertype for effective viscosity closures."""
abstract type AbstractViscosityModel end

"""No viscous stress. The implicit momentum operator is skipped."""
struct NoViscosity <: AbstractViscosityModel end

"""
    ConstantViscosity(value)

Spatially constant positive effective viscosity coefficient.
"""
struct ConstantViscosity{T<:Real} <: AbstractViscosityModel
    value::T

    function ConstantViscosity(value::T) where {T<:Real}
        isfinite(value) || throw(ArgumentError("Viscosity must be finite."))
        value > zero(T) || throw(ArgumentError("Viscosity must be strictly positive."))
        return new{T}(value)
    end
end

raw"""
    HarmonicPhaseViscosity(fluid_value, structure_value)

Harmonic interpolation between two positive phase viscosities,

\mu(\phi)
=
\left[
\frac{1-\phi}{\mu_f}
+
\frac{\phi}{\mu_s}
\right]^{-1}.

For `fluid_value = 1` and `structure_value = ratio_mu`, this reduces to the
viscosity law used in the inherited implementation.
"""
struct HarmonicPhaseViscosity{TF<:Real,TS<:Real} <: AbstractViscosityModel
    fluid_value::TF
    structure_value::TS

    function HarmonicPhaseViscosity(fluid_value::TF, structure_value::TS) where {TF<:Real,TS<:Real}
        isfinite(fluid_value) || throw(ArgumentError("Fluid viscosity must be finite."))
        isfinite(structure_value) || throw(ArgumentError("Structure viscosity must be finite."))
        fluid_value > zero(TF) || throw(ArgumentError("Fluid viscosity must be strictly positive."))
        structure_value > zero(TS) || throw(ArgumentError("Structure viscosity must be strictly positive."))
        return new{TF,TS}(fluid_value, structure_value)
    end
end

viscosity_active(::NoViscosity) = false
viscosity_active(::AbstractViscosityModel) = true

function viscosity_field!(output, ::NoViscosity, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "viscosity output")
    _validate_center_field(phi, metrics, "phi")
    fill!(output, 0.0)
    return output
end

function viscosity_field!(output, model::ConstantViscosity, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "viscosity output")
    _validate_center_field(phi, metrics, "phi")
    fill!(output, Float64(model.value))
    return output
end

function viscosity_field!(output, model::HarmonicPhaseViscosity, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "viscosity output")
    _validate_center_field(phi, metrics, "phi")

    mu_f = Float64(model.fluid_value)
    mu_s = Float64(model.structure_value)

    # The phase fraction is physically meaningful on [0,1].  Conservative
    # interface schemes can produce roundoff/stage-level excursions just
    # outside that interval; with very large property ratios, extrapolating
    # the harmonic law through such excursions can create a non-positive
    # effective viscosity.  Limit only the constitutive evaluation -- the
    # transported phase field itself is left untouched.
    @inbounds for idx in eachindex(output, phi)
        p = clamp(Float64(phi[idx]), 0.0, 1.0)
        output[idx] = 1.0 / ((1.0 - p) / mu_f + p / mu_s)
    end

    _validate_positive_field(output, "evaluated viscosity")
    return output
end

# ==============================================================================
# DIFFUSIVITY
# ==============================================================================

"""Abstract supertype for effective resource-diffusivity closures."""
abstract type AbstractDiffusivityModel end

"""No resource diffusion. The implicit scalar diffusion operator is skipped."""
struct NoDiffusion <: AbstractDiffusivityModel end

"""Spatially constant positive effective diffusivity coefficient."""
struct ConstantDiffusivity{T<:Real} <: AbstractDiffusivityModel
    value::T

    function ConstantDiffusivity(value::T) where {T<:Real}
        isfinite(value) || throw(ArgumentError("Diffusivity must be finite."))
        value > zero(T) || throw(ArgumentError("Diffusivity must be strictly positive."))
        return new{T}(value)
    end
end

raw"""
    HarmonicPhaseDiffusivity(fluid_value, structure_value)

Harmonic interpolation between positive fluid and structural diffusivities,

D(\phi)
=
\left[
\frac{1-\phi}{D_f}
+
\frac{\phi}{D_s}
\right]^{-1}.
"""
struct HarmonicPhaseDiffusivity{TF<:Real,TS<:Real} <: AbstractDiffusivityModel
    fluid_value::TF
    structure_value::TS

    function HarmonicPhaseDiffusivity(fluid_value::TF, structure_value::TS) where {TF<:Real,TS<:Real}
        isfinite(fluid_value) || throw(ArgumentError("Fluid diffusivity must be finite."))
        isfinite(structure_value) || throw(ArgumentError("Structure diffusivity must be finite."))
        fluid_value > zero(TF) || throw(ArgumentError("Fluid diffusivity must be strictly positive."))
        structure_value > zero(TS) || throw(ArgumentError("Structure diffusivity must be strictly positive."))
        return new{TF,TS}(fluid_value, structure_value)
    end
end

diffusion_active(::NoDiffusion) = false
diffusion_active(::AbstractDiffusivityModel) = true

function diffusivity_field!(output, ::NoDiffusion, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "diffusivity output")
    _validate_center_field(phi, metrics, "phi")
    fill!(output, 0.0)
    return output
end

function diffusivity_field!(output, model::ConstantDiffusivity, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "diffusivity output")
    _validate_center_field(phi, metrics, "phi")
    fill!(output, Float64(model.value))
    return output
end

function diffusivity_field!(output, model::HarmonicPhaseDiffusivity, phi, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "diffusivity output")
    _validate_center_field(phi, metrics, "phi")

    D_f = Float64(model.fluid_value)
    D_s = Float64(model.structure_value)
    @. output = 1.0 / ((1.0 - phi) / D_f + phi / D_s)
    _validate_positive_field(output, "evaluated diffusivity")
    return output
end

# ==============================================================================
# RESOURCE CONSUMPTION
# ==============================================================================

"""Abstract supertype for local resource-consumption laws."""
abstract type AbstractResourceConsumption end

"""No resource consumption: `C = 0`."""
struct NoResourceConsumption <: AbstractResourceConsumption end

raw"""
    SaturatingResourceConsumption(rate, half_saturation)

Local sink

\mathcal{C}(\phi,c)
=
\phi\,c_{\max}\,\frac{c}{K_C+c}.

The resource is used exactly as supplied; no clipping or denominator
regularization is introduced here.
"""
struct SaturatingResourceConsumption{TR<:Real,TK<:Real} <: AbstractResourceConsumption
    rate::TR
    half_saturation::TK

    function SaturatingResourceConsumption(rate::TR, half_saturation::TK) where {TR<:Real,TK<:Real}
        isfinite(rate) || throw(ArgumentError("Consumption rate must be finite."))
        isfinite(half_saturation) || throw(ArgumentError("Consumption half-saturation constant must be finite."))
        rate >= zero(TR) || throw(ArgumentError("Consumption rate must be non-negative."))
        half_saturation > zero(TK) || throw(ArgumentError(
            "Consumption half-saturation constant must be strictly positive."
        ))
        return new{TR,TK}(rate, half_saturation)
    end
end

function resource_sink!(output, ::NoResourceConsumption, phi, resource, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "resource sink output")
    _validate_center_field(phi, metrics, "phi")
    _validate_center_field(resource, metrics, "resource")
    fill!(output, 0.0)
    return output
end

function resource_sink!(output, model::SaturatingResourceConsumption, phi, resource, metrics::MeshMetrics)
    _validate_output_shape(output, metrics, "resource sink output")
    _validate_center_field(phi, metrics, "phi")
    _validate_center_field(resource, metrics, "resource")

    rate = Float64(model.rate)
    K = Float64(model.half_saturation)
    @. output = rate * phi * resource / (K + resource)
    all(isfinite, output) || throw(ArgumentError(
        "Evaluated resource sink contains non-finite values."
    ))
    return output
end

# ==============================================================================
# BODY FORCE
# ==============================================================================

"""Abstract supertype for externally prescribed body-force models."""
abstract type AbstractBodyForce end

"""No externally prescribed body force."""
struct NoBodyForce <: AbstractBodyForce end

"""
    ConstantBodyForce(fx, fy)

Spatially and temporally constant force components on the two momentum
equations. The supplied values are the coefficients that appear directly in
the equations being solved.
"""
struct ConstantBodyForce{TX<:Real,TY<:Real} <: AbstractBodyForce
    fx::TX
    fy::TY

    function ConstantBodyForce(fx::TX, fy::TY) where {TX<:Real,TY<:Real}
        isfinite(fx) || throw(ArgumentError("Body-force x-component must be finite."))
        isfinite(fy) || throw(ArgumentError("Body-force y-component must be finite."))
        return new{TX,TY}(fx, fy)
    end
end


"""
    KolmogorovBodyForce(amplitude, wavenumber; phase=0)

Steady streamwise Kolmogorov forcing

    f_x(y) = amplitude * sin(wavenumber*y + phase),
    f_y(y) = 0.

`amplitude` is the coefficient that appears directly in the momentum equation.
"""
struct KolmogorovBodyForce <: AbstractBodyForce
    amplitude::Float64
    wavenumber::Float64
    phase::Float64

    function KolmogorovBodyForce(amplitude::Real, wavenumber::Real; phase::Real=0.0)
        a = Float64(amplitude)
        k = Float64(wavenumber)
        p = Float64(phase)
        isfinite(a) || throw(ArgumentError("Kolmogorov-force amplitude must be finite."))
        isfinite(k) || throw(ArgumentError("Kolmogorov-force wavenumber must be finite."))
        k > 0.0 || throw(ArgumentError("Kolmogorov-force wavenumber must be positive."))
        isfinite(p) || throw(ArgumentError("Kolmogorov-force phase must be finite."))
        return new(a, k, p)
    end
end

function _validate_force_outputs(Fx, Fy, metrics::MeshMetrics)
    expected_x = (metrics.Nx + 1, metrics.Ny)
    expected_y = (metrics.Nx, metrics.Ny + 1)
    size(Fx) == expected_x || throw(DimensionMismatch(
        "Fx must have size $expected_x; received $(size(Fx))."
    ))
    size(Fy) == expected_y || throw(DimensionMismatch(
        "Fy must have size $expected_y; received $(size(Fy))."
    ))
    return nothing
end

function body_force!(Fx, Fy, ::NoBodyForce, t::Real, metrics::MeshMetrics)
    _validate_force_outputs(Fx, Fy, metrics)
    fill!(Fx, 0.0)
    fill!(Fy, 0.0)
    return Fx, Fy
end

function body_force!(Fx, Fy, model::ConstantBodyForce, t::Real, metrics::MeshMetrics)
    _validate_force_outputs(Fx, Fy, metrics)
    isfinite(t) || throw(ArgumentError("Body-force evaluation time must be finite."))
    fill!(Fx, Float64(model.fx))
    fill!(Fy, Float64(model.fy))
    return Fx, Fy
end


function body_force!(Fx, Fy, model::KolmogorovBodyForce, t::Real, metrics::MeshMetrics)
    _validate_force_outputs(Fx, Fy, metrics)
    isfinite(t) || throw(ArgumentError("Body-force evaluation time must be finite."))
    fill!(Fy, 0.0)
    y_center = 0.0
    @inbounds for j in 1:metrics.Ny
        y_center += 0.5 * metrics.dy_c[j]
        force_j = model.amplitude * sin(model.wavenumber * y_center + model.phase)
        @views fill!(Fx[:, j], force_j)
        y_center += 0.5 * metrics.dy_c[j]
    end
    return Fx, Fy
end

end # module TransportConstitutive
