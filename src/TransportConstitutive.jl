"""Viscosity and external forcing closures for the incompressible FSI solver."""
module TransportConstitutive

using ..MeshGeneration

export AbstractViscosityModel,
       NoViscosity,
       ConstantViscosity,
       HarmonicPhaseViscosity,
       viscosity_active,
       viscosity_field!,
       AbstractBodyForce,
       NoBodyForce,
       ConstantBodyForce,
       KolmogorovBodyForce,
       body_force!

# COMMON VALIDATION

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

# VISCOSITY

"""Abstract supertype for effective viscosity closures."""
abstract type AbstractViscosityModel end

"""No viscous stress."""
struct NoViscosity <: AbstractViscosityModel end

"""Spatially constant positive effective viscosity coefficient."""
struct ConstantViscosity{T<:Real} <: AbstractViscosityModel
    value::T

    function ConstantViscosity(value::T) where {T<:Real}
        isfinite(value) || throw(ArgumentError("Viscosity must be finite."))
        value > zero(T) || throw(ArgumentError("Viscosity must be strictly positive."))
        return new{T}(value)
    end
end

"""Harmonic interpolation of phase viscosities."""
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
    @inbounds for idx in eachindex(output, phi)
        p = clamp(Float64(phi[idx]), 0.0, 1.0)
        output[idx] = 1.0 / ((1.0 - p) / mu_f + p / mu_s)
    end

    _validate_positive_field(output, "evaluated viscosity")
    return output
end

# BODY FORCE

"""Abstract supertype for externally prescribed body-force models."""
abstract type AbstractBodyForce end

"""No externally prescribed body force."""
struct NoBodyForce <: AbstractBodyForce end

"""Spatially and temporally constant force components on the two momentum equations."""
struct ConstantBodyForce{TX<:Real,TY<:Real} <: AbstractBodyForce
    fx::TX
    fy::TY

    function ConstantBodyForce(fx::TX, fy::TY) where {TX<:Real,TY<:Real}
        isfinite(fx) || throw(ArgumentError("Body-force x-component must be finite."))
        isfinite(fy) || throw(ArgumentError("Body-force y-component must be finite."))
        return new{TX,TY}(fx, fy)
    end
end


"""Steady streamwise Kolmogorov forcing f_x(y) = amplitude * sin(wavenumber*y + phase), f_y(y) = 0."""
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
