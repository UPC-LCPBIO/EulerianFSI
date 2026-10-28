module BenchmarkUtilities

using Dates
using Printf

export INTERFACE_WIDTH_5_95_FACTOR,
       resolve_interface_epsilon,
       interface_width_5_95,
       interface_cells_5_95,
       conservative_mesh_spacing,
       interface_study_descriptor,
       prepare_study_output_directory,
       scalar_tag,
       tensor_cell_diagnostics_2d

const INTERFACE_WIDTH_5_95_FACTOR = 2.0 * atanh(0.9)

"""Return a conservative representative cell size for diffuse-interface resolution."""
function conservative_mesh_spacing(metrics)
    return max(maximum(metrics.dx_c), maximum(metrics.dy_c))
end

"""Resolve the diffuse-interface scale used by a benchmark."""
function resolve_interface_epsilon(
    fixed_epsilon::Real,
    epsilon_over_h::Real,
    h::Real,
)
    h_value = Float64(h)
    h_value > 0.0 || throw(ArgumentError("Representative mesh spacing must be positive."))

    fixed = Float64(fixed_epsilon)
    ratio = Float64(epsilon_over_h)

    if isfinite(fixed)
        fixed > 0.0 || throw(ArgumentError("Fixed interface epsilon must be positive."))
        epsilon = fixed
        mode = :fixed
    else
        isfinite(ratio) || throw(ArgumentError("epsilon_over_h must be finite in grid-scaled mode."))
        ratio > 0.0 || throw(ArgumentError("epsilon_over_h must be positive in grid-scaled mode."))
        epsilon = ratio * h_value
        mode = :grid_scaled
    end

    actual_ratio = epsilon / h_value
    width = INTERFACE_WIDTH_5_95_FACTOR * epsilon
    cells = width / h_value

    return (
        epsilon = epsilon,
        mode = mode,
        epsilon_over_h = actual_ratio,
        width_5_95 = width,
        cells_5_95 = cells,
    )
end

interface_width_5_95(epsilon::Real) = INTERFACE_WIDTH_5_95_FACTOR * Float64(epsilon)
interface_cells_5_95(epsilon::Real, h::Real) = interface_width_5_95(epsilon) / Float64(h)

@inline function tensor_cell_diagnostics_2d(
    sxx::Real,
    syy::Real,
    sxy::Real,
    phi::Real,
    elastic_modulus::Real,
)
    xx = Float64(sxx)
    yy = Float64(syy)
    xy = Float64(sxy)
    det_sigma = xx * yy - xy * xy
    trace_sigma = xx + yy
    spectral_gap = hypot(xx - yy, 2.0 * xy)
    lambda_min = 0.5 * (trace_sigma - spectral_gap)
    lambda_max = 0.5 * (trace_sigma + spectral_gap)
    stress_frob = abs(Float64(phi) * Float64(elastic_modulus)) * sqrt(
        (xx - 1.0)^2 + (yy - 1.0)^2 + 2.0 * xy^2
    )
    return (
        determinant=det_sigma,
        determinant_error=det_sigma - 1.0,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        effective_stress_frobenius=stress_frob,
    )
end

"""Format a scalar for directory names."""
function scalar_tag(value::Real)
    x = Float64(value)
    isfinite(x) || return "nan"
    s = @sprintf("%.10g", x)
    s = replace(s, "-" => "m", "+" => "", "." => "p")
    return s
end

"""Map the selected interface treatment to the canonical benchmark-study layout."""
function interface_study_descriptor(interface_info)
    if interface_info.mode === :fixed
        return (
            study = "fixed_epsilon",
            parameter = "eps_" * scalar_tag(interface_info.epsilon),
        )
    elseif interface_info.mode === :grid_scaled
        return (
            study = "sharp_interface",
            parameter = "eps_over_h_" * scalar_tag(interface_info.epsilon_over_h),
        )
    elseif interface_info.mode === :sharp
        return (
            study = "sharp_indicator",
            parameter = "no_prescribed_thickness",
        )
    end
    throw(ArgumentError("Unsupported interface mode: $(interface_info.mode)"))
end

"""Create a deterministic study hierarchy and a collision-safe run directory."""
function prepare_study_output_directory(
    output_root::AbstractString,
    interface_info,
    run_label::AbstractString;
    subgroup::Union{Nothing,AbstractString}=nothing,
)
    descriptor = interface_study_descriptor(interface_info)
    study_root = joinpath(output_root, descriptor.study, descriptor.parameter)
    if subgroup !== nothing
        study_root = joinpath(study_root, subgroup)
    end
    mkpath(study_root)

    nominal = joinpath(study_root, run_label)
    if !isdir(nominal) || isempty(readdir(nominal))
        mkpath(nominal)
        return nominal
    end

    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    path = nominal * "_" * stamp
    mkpath(path)
    return path
end


end # module BenchmarkUtilities
