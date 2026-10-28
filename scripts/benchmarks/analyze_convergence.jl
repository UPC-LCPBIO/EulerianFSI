#!/usr/bin/env julia

using Printf
using Statistics

# Summarize cavity mesh-refinement results.

function parse_args(args)
    study_dir = nothing
    for arg in args
        if startswith(arg, "--study-dir=")
            study_dir = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            throw(ArgumentError("Unknown option: $arg"))
        elseif study_dir === nothing
            study_dir = abspath(arg)
        else
            throw(ArgumentError("Unexpected positional argument: $arg"))
        end
    end

    study_dir === nothing && throw(ArgumentError(
        "Provide --study-dir=<path> pointing to a completed refinement study."
    ))
    isdir(study_dir) || throw(ArgumentError("Study directory does not exist: $study_dir"))
    return study_dir
end

function parse_parameters(path::AbstractString)
    params = Dict{String,String}()
    for line in eachline(path)
        occursin(": ", line) || continue
        key, value = split(line, ": "; limit=2)
        params[strip(key)] = strip(value)
    end
    return params
end

function run_info(path::AbstractString)
    params = parse_parameters(path)
    grid = match(r"^(\d+) x (\d+) uniform MAC cells$", params["Grid"])
    domain = match(r"^([0-9.eE+\-]+) x ([0-9.eE+\-]+)$", params["Domain"])
    grid === nothing && error("Invalid grid metadata: $path")
    domain === nothing && error("Invalid domain metadata: $path")
    Nx, Ny = parse.(Int, grid.captures)
    Lx, Ly = parse.(Float64, domain.captures)
    h = max(Lx / Nx, Ly / Ny)

    parts = splitpath(dirname(path))
    index = findfirst(x -> x in ("fixed_epsilon", "sharp_interface", "sharp_indicator"), parts)
    index === nothing && error("Unknown refinement-study directory: $path")
    study = parts[index]
    setting = parts[index + 1]

    manifest = Dict(
        "benchmark" => "sugiyama_lid_driven_cavity",
        "run_label" => "N$(Nx)",
        "study" => study,
        "study_parameter" => setting,
        "Nx" => string(Nx),
        "Ny" => string(Ny),
        "h_interface" => string(h),
        "epsilon" => get(params, "Selected initialization epsilon", "NaN"),
        "epsilon_over_h" => get(params, "Selected initialization epsilon/h", "NaN"),
        "interface_cells_5_95" => get(params, "Selected initialization 5-95 cells", "NaN"),
    )
    return (manifest=manifest, parameters=params)
end

@inline function maybe_float(value)
    value === nothing && return NaN
    parsed = tryparse(Float64, String(value))
    return parsed === nothing ? NaN : parsed
end

function read_numeric_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Empty CSV: $path")
    header = split(strip(lines[1]), ',')
    columns = Dict(name => Float64[] for name in header)

    for line in Iterators.drop(lines, 1)
        stripped = strip(line)
        isempty(stripped) && continue
        values = split(stripped, ',')
        length(values) == length(header) || continue
        parsed = tryparse.(Float64, values)
        any(x -> x === nothing, parsed) && continue
        for (name, value) in zip(header, parsed)
            push!(columns[name], value::Float64)
        end
    end
    return columns
end

function latest_runs_by_label(study_dir::AbstractString)
    paths = String[]
    for (dir, _, files) in walkdir(study_dir)
        "run_parameters.txt" in files && push!(paths, joinpath(dir, "run_parameters.txt"))
    end
    isempty(paths) && error("No completed runs found below: $study_dir")

    selected = Dict{String,Tuple{Float64,String,Dict{String,String},Dict{String,String}}}()
    for path in paths
        info = run_info(path)
        label = info.manifest["run_label"]
        modified = stat(path).mtime
        if !haskey(selected, label) || modified > selected[label][1]
            selected[label] = (modified, dirname(path), info.manifest, info.parameters)
        end
    end

    runs = [(dir=x[2], manifest=x[3], parameters=x[4]) for x in values(selected)]
    sort!(runs; by=r -> maybe_float(get(r.manifest, "h_interface", nothing)), rev=true)
    return runs
end

function sugiyama_qois(run_dir::AbstractString)
    data = read_numeric_csv(joinpath(run_dir, "diagnostics.csv"))
    isempty(data["time"]) && error("No diagnostics rows in $run_dir")
    last_index = length(data["time"])

    return Dict(
        "final_x_centroid" => data["x_centroid"][last_index],
        "final_y_centroid" => data["y_centroid"][last_index],
        "max_abs_phase_integral_error" => maximum(abs, data["relative_phase_integral_error"]),
        "max_divergence_linf" => maximum(data["divergence_linf"]),
        "min_sigma_eigenvalue" => minimum(data["min_sigma_eigenvalue"]),
        "max_abs_sigma_det_error" => maximum(data["max_abs_sigma_det_error"]),
        "max_projection_constraint_linf" => maximum(data["projection_constraint_linf"]),
    )
end

function csv_string(value)
    if value isa AbstractFloat
        return isfinite(value) ? @sprintf("%.16e", value) : "NaN"
    end
    return string(value)
end

const SUGIYAMA_QOIS = [
    "final_x_centroid",
    "final_y_centroid",
    "max_abs_phase_integral_error",
    "max_divergence_linf",
    "min_sigma_eigenvalue",
    "max_abs_sigma_det_error",
    "max_projection_constraint_linf",
]

function write_summary(path, runs)
    common = [
        "run_label", "run_directory", "study", "Nx", "Ny",
        "h_interface", "epsilon", "epsilon_over_h", "interface_cells_5_95",
    ]

    open(path, "w") do io
        println(io, join(vcat(common, SUGIYAMA_QOIS), ","))
        for run in runs
            m = run.manifest
            q = sugiyama_qois(run.dir)
            row = Any[
                get(m, "run_label", ""),
                relpath(run.dir, dirname(path)),
                get(m, "study", ""),
                get(m, "Nx", ""),
                get(m, "Ny", ""),
                maybe_float(get(m, "h_interface", nothing)),
                maybe_float(get(m, "epsilon", nothing)),
                maybe_float(get(m, "epsilon_over_h", nothing)),
                maybe_float(get(m, "interface_cells_5_95", nothing)),
            ]
            append!(row, [get(q, name, NaN) for name in SUGIYAMA_QOIS])
            println(io, join(csv_string.(row), ","))
        end
    end
end

function model_difference_ratio(p, h1, h2, h3)
    denominator = abs(h2^p - h3^p)
    denominator > 0 || return Inf
    return abs(h1^p - h2^p) / denominator
end

function observed_order(q1, q2, q3, h1, h2, h3)
    all(isfinite, (q1, q2, q3, h1, h2, h3)) || return NaN
    h1 > h2 > h3 > 0 || return NaN

    d12 = abs(q1 - q2)
    d23 = abs(q2 - q3)
    min(d12, d23) > 100 * eps(Float64) || return NaN
    target = d12 / d23

    f(p) = model_difference_ratio(p, h1, h2, h3) - target
    p_lo = 1e-4
    p_hi = 12.0
    f_lo = f(p_lo)
    f_hi = f(p_hi)
    isfinite(f_lo) && isfinite(f_hi) || return NaN
    f_lo == 0 && return p_lo
    f_hi == 0 && return p_hi
    signbit(f_lo) == signbit(f_hi) && return NaN

    for _ in 1:100
        p_mid = 0.5 * (p_lo + p_hi)
        f_mid = f(p_mid)
        abs(f_mid) < 1e-10 && return p_mid
        if signbit(f_mid) == signbit(f_lo)
            p_lo = p_mid
            f_lo = f_mid
        else
            p_hi = p_mid
        end
    end
    return 0.5 * (p_lo + p_hi)
end

function write_orders(path, runs)
    study = get(first(runs).manifest, "study", "")
    interpretation = study == "fixed_epsilon" ?
        "spatial_order_fixed_epsilon" : "combined_h_epsilon_order"

    open(path, "w") do io
        println(io, "quantity,run_coarse,run_middle,run_fine,h_coarse,h_middle,h_fine,q_coarse,q_middle,q_fine,observed_order,interpretation")
        length(runs) >= 3 || return
        qois = [sugiyama_qois(run.dir) for run in runs]

        for k in 1:(length(runs) - 2)
            r1, r2, r3 = runs[k], runs[k + 1], runs[k + 2]
            h1 = maybe_float(get(r1.manifest, "h_interface", nothing))
            h2 = maybe_float(get(r2.manifest, "h_interface", nothing))
            h3 = maybe_float(get(r3.manifest, "h_interface", nothing))

            for name in SUGIYAMA_QOIS
                q1 = get(qois[k], name, NaN)
                q2 = get(qois[k + 1], name, NaN)
                q3 = get(qois[k + 2], name, NaN)
                p = observed_order(q1, q2, q3, h1, h2, h3)
                @printf(
                    io,
                    "%s,%s,%s,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s,%s\n",
                    name,
                    get(r1.manifest, "run_label", ""),
                    get(r2.manifest, "run_label", ""),
                    get(r3.manifest, "run_label", ""),
                    h1, h2, h3, q1, q2, q3,
                    isfinite(p) ? @sprintf("%.8f", p) : "NaN",
                    interpretation,
                )
            end
        end
    end
end

function read_trajectory(path)
    data = read_numeric_csv(path)
    return data["time"], data["x_centroid"], data["y_centroid"]
end

function interp_linear(t, values, tq)
    tq <= t[1] && return values[1]
    tq >= t[end] && return values[end]
    k = min(searchsortedlast(t, tq), length(t) - 1)
    theta = (tq - t[k]) / (t[k + 1] - t[k])
    return (1.0 - theta) * values[k] + theta * values[k + 1]
end

function trajectory_rms(run_a, run_b)
    ta, xa, ya = read_trajectory(joinpath(run_a, "centroid_trajectory.csv"))
    tb, xb, yb = read_trajectory(joinpath(run_b, "centroid_trajectory.csv"))
    tmax = min(ta[end], tb[end])
    samples = [t for t in ta if t <= tmax + 1e-12]
    isempty(samples) && return NaN

    squared = Float64[]
    for t in samples
        dx = interp_linear(ta, xa, t) - interp_linear(tb, xb, t)
        dy = interp_linear(ta, ya, t) - interp_linear(tb, yb, t)
        push!(squared, dx^2 + dy^2)
    end
    return sqrt(mean(squared))
end

function write_pairwise_trajectory(path, runs)
    open(path, "w") do io
        println(io, "run_coarse,run_fine,h_coarse,h_fine,rms_centroid_trajectory_difference")
        for k in 1:(length(runs) - 1)
            a, b = runs[k], runs[k + 1]
            h1 = maybe_float(get(a.manifest, "h_interface", nothing))
            h2 = maybe_float(get(b.manifest, "h_interface", nothing))
            rms = trajectory_rms(a.dir, b.dir)
            @printf(
                io,
                "%s,%s,%.16e,%.16e,%.16e\n",
                get(a.manifest, "run_label", ""),
                get(b.manifest, "run_label", ""),
                h1, h2, rms,
            )
        end
    end
end

function main(args)
    study_dir = parse_args(args)
    runs = latest_runs_by_label(study_dir)

    studies = unique(run.manifest["study"] for run in runs)
    length(studies) == 1 || error("Mixed refinement studies in selected directory.")

    # Only resolution-dependent parameters may vary within a refinement study.
    stable = (
        "Domain", "Initial center", "Initial radius", "Lid speed",
        "Density", "Reynolds", "Viscosity coefficient",
        "Neo-Hookean shear modulus G", "phase-field method",
        "Current CDI Pe_Gamma", "ACDI epsilon_Jain/h", "ACDI Gamma (fixed)",
        "UWCDI alpha", "UWCDI normal treatment", "UWCDI initialization",
        "UWCDI initialization epsilon/h (when tanh)",
        "sigma extension rate kappa0", "sigma extension exponent m",
        "sigma far-fluid safety cutoff", "normal regularization delta_n",
        "Final time", "CFL hyperbolic", "CFL interface compression",
        "CFL interface diffusion", "dt_max",
    )
    if only(studies) == "fixed_epsilon"
        stable = (stable..., "Selected initialization epsilon")
    elseif only(studies) == "sharp_interface"
        stable = (stable..., "Selected initialization epsilon/h")
    end
    for key in stable
        values = unique(get(run.parameters, key, "") for run in runs)
        length(values) == 1 || error("Incompatible runs: '$key' differs.")
    end

    summary_path = joinpath(study_dir, "convergence_summary.csv")
    orders_path = joinpath(study_dir, "convergence_orders.csv")
    write_summary(summary_path, runs)
    write_orders(orders_path, runs)

    if length(runs) >= 2
        write_pairwise_trajectory(
            joinpath(study_dir, "centroid_trajectory_pairwise.csv"), runs
        )
    end

    println("Convergence analysis complete")
    println("  runs    : $(length(runs))")
    println("  summary : $summary_path")
    println("  orders  : $orders_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
