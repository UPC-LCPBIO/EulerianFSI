
module InterfaceInitialization

using ..MeshGeneration: MACGrid

export phase_field_from_distance,
       initialize_circle,
       initialize_layer,
       initialize_semicircular_hump,
       initialize_filament,
       initialize_wall_attached_filament,
       initialize_ellipse

# GENERIC DIFFUSE-INTERFACE PROFILE

@inline function _validate_epsilon(epsilon::Real)
    epsilon_f = Float64(epsilon)
    isfinite(epsilon_f) || throw(ArgumentError("epsilon must be finite."))
    epsilon_f > 0.0 || throw(ArgumentError("epsilon must be positive."))
    return epsilon_f
end

@inline function _positive_parameter(value::Real, name::AbstractString)
    value_f = Float64(value)
    isfinite(value_f) || throw(ArgumentError("$name must be finite."))
    value_f > 0.0 || throw(ArgumentError("$name must be positive."))
    return value_f
end

@inline function _finite_parameter(value::Real, name::AbstractString)
    value_f = Float64(value)
    isfinite(value_f) || throw(ArgumentError("$name must be finite."))
    return value_f
end

@inline function _periodic_displacement(
    coordinate::Float64,
    center::Float64,
    length::Float64,
    periodic::Bool,
)
    delta = coordinate - center
    return periodic ? mod(delta + 0.5 * length, length) - 0.5 * length : delta
end

@inline function _profile_from_distance(distance::Float64, epsilon::Float64)
    value = 0.5 * (1.0 - tanh(distance / epsilon))
    return clamp(value, 0.0, 1.0)
end

"""Evaluate a diffuse phase indicator from signed distance."""
function phase_field_from_distance(
    grid::MACGrid,
    signed_distance,
    epsilon::Real,
)
    epsilon_f = _validate_epsilon(epsilon)
    phi = Matrix{Float64}(undef, grid.Nx, grid.Ny)

    for j in 1:grid.Ny, i in 1:grid.Nx
        distance = Float64(signed_distance(grid.x_p[i], grid.y_p[j]))
        isfinite(distance) || throw(DomainError(
            distance,
            "signed_distance must return finite values at every cell center.",
        ))
        phi[i, j] = _profile_from_distance(distance, epsilon_f)
    end

    return phi
end

# CIRCLE

function initialize_circle(
    grid::MACGrid,
    center,
    radius::Real,
    epsilon::Real;
    periodic_x::Bool = false,
    periodic_y::Bool = false,
)
    length(center) == 2 || throw(ArgumentError("center must contain exactly two coordinates."))
    cx = _finite_parameter(center[1], "center x-coordinate")
    cy = _finite_parameter(center[2], "center y-coordinate")
    radius_f = _positive_parameter(radius, "radius")
    epsilon_f = _validate_epsilon(epsilon)

    signed_distance = function (x, y)
        dx = _periodic_displacement(Float64(x), cx, grid.Lx, periodic_x)
        dy = _periodic_displacement(Float64(y), cy, grid.Ly, periodic_y)
        return hypot(dx, dy) - radius_f
    end

    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

# FLAT LAYER

"""Initialize a flat structural layer occupying the region below."""
function initialize_layer(
    grid::MACGrid,
    interface_height::Real,
    epsilon::Real,
)
    height = _finite_parameter(interface_height, "interface_height")
    epsilon_f = _validate_epsilon(epsilon)
    signed_distance = (x, y) -> Float64(y) - height
    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

# LAYER WITH SEMICIRCULAR HUMP

function initialize_semicircular_hump(
    grid::MACGrid,
    base_height::Real,
    radius::Real,
    center_x::Real,
    epsilon::Real;
    periodic_x::Bool = false,
)
    base = _finite_parameter(base_height, "base_height")
    cx = _finite_parameter(center_x, "center_x")
    radius_f = _positive_parameter(radius, "radius")
    epsilon_f = _validate_epsilon(epsilon)

    signed_distance = function (x, y)
        y_f = Float64(y)
        dx = _periodic_displacement(Float64(x), cx, grid.Lx, periodic_x)
        distance_layer = y_f - base
        distance_circle = hypot(dx, y_f - base) - radius_f
        return min(distance_layer, distance_circle)
    end

    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

# VERTICAL FILAMENT / STREAMER

@inline function _signed_distance_to_box(
    x::Float64,
    y::Float64,
    center_x::Float64,
    center_y::Float64,
    half_width::Float64,
    half_height::Float64,
    Lx::Float64,
    periodic_x::Bool,
)
    dx = abs(_periodic_displacement(x, center_x, Lx, periodic_x)) - half_width
    dy = abs(y - center_y) - half_height
    outside_distance = hypot(max(dx, 0.0), max(dy, 0.0))
    inside_distance = min(max(dx, dy), 0.0)
    return outside_distance + inside_distance
end

function initialize_filament(
    grid::MACGrid,
    base_height::Real,
    height::Real,
    cap_radius::Real,
    center_x::Real,
    epsilon::Real;
    periodic_x::Bool = false,
)
    base = _finite_parameter(base_height, "base_height")
    height_f = _positive_parameter(height, "height")
    cap_radius_f = _positive_parameter(cap_radius, "cap_radius")
    cx = _finite_parameter(center_x, "center_x")
    epsilon_f = _validate_epsilon(epsilon)

    height_f >= cap_radius_f || throw(ArgumentError(
        "height must be greater than or equal to cap_radius so that it represents the total filament height."
    ))

    emerged_stem_height = height_f - cap_radius_f
    cap_center_y = base + emerged_stem_height

    # The stem extends one cap radius into the lower layer. This overlap does
    # not alter the union geometry below the layer interface and gives a smooth
    # geometric connection to the cap and substrate.
    stem_top = cap_center_y
    stem_bottom = base - cap_radius_f
    stem_half_height = 0.5 * (stem_top - stem_bottom)
    stem_center_y = 0.5 * (stem_top + stem_bottom)

    signed_distance = function (x, y)
        x_f = Float64(x)
        y_f = Float64(y)
        dx = _periodic_displacement(x_f, cx, grid.Lx, periodic_x)

        distance_layer = y_f - base
        distance_cap = hypot(dx, y_f - cap_center_y) - cap_radius_f
        distance_stem = _signed_distance_to_box(
            x_f,
            y_f,
            cx,
            stem_center_y,
            cap_radius_f,
            stem_half_height,
            grid.Lx,
            periodic_x,
        )

        return min(distance_layer, min(distance_stem, distance_cap))
    end

    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

# WALL-ATTACHED FILAMENT

function initialize_wall_attached_filament(
    grid::MACGrid,
    wall_y::Real,
    height::Real,
    cap_radius::Real,
    center_x::Real,
    epsilon::Real;
    periodic_x::Bool = false,
    embed_depth::Real = cap_radius,
)
    wall = _finite_parameter(wall_y, "wall_y")
    height_f = _positive_parameter(height, "height")
    cap_radius_f = _positive_parameter(cap_radius, "cap_radius")
    cx = _finite_parameter(center_x, "center_x")
    epsilon_f = _validate_epsilon(epsilon)
    embed_depth_f = Float64(embed_depth)

    isfinite(embed_depth_f) || throw(ArgumentError("embed_depth must be finite."))
    embed_depth_f >= 0.0 || throw(ArgumentError("embed_depth must be non-negative."))
    height_f >= cap_radius_f || throw(ArgumentError(
        "height must be greater than or equal to cap_radius so that it represents the total filament height."
    ))

    cap_center_y = wall + height_f - cap_radius_f
    stem_top = cap_center_y
    stem_bottom = wall - embed_depth_f
    stem_half_height = 0.5 * (stem_top - stem_bottom)
    stem_half_height > 0.0 || throw(ArgumentError(
        "The wall-attached stem must have positive height."
    ))
    stem_center_y = 0.5 * (stem_top + stem_bottom)

    signed_distance = function (x, y)
        x_f = Float64(x)
        y_f = Float64(y)
        dx = _periodic_displacement(x_f, cx, grid.Lx, periodic_x)

        distance_cap = hypot(dx, y_f - cap_center_y) - cap_radius_f
        distance_stem = _signed_distance_to_box(
            x_f,
            y_f,
            cx,
            stem_center_y,
            cap_radius_f,
            stem_half_height,
            grid.Lx,
            periodic_x,
        )

        return min(distance_stem, distance_cap)
    end

    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

# ELLIPSE

"""Initialize an elliptical structural region."""
function initialize_ellipse(
    grid::MACGrid,
    center,
    semi_axis_x::Real,
    semi_axis_y::Real,
    epsilon::Real;
    periodic_x::Bool = false,
    periodic_y::Bool = false,
)
    length(center) == 2 || throw(ArgumentError("center must contain exactly two coordinates."))
    cx = _finite_parameter(center[1], "center x-coordinate")
    cy = _finite_parameter(center[2], "center y-coordinate")
    a = _positive_parameter(semi_axis_x, "semi_axis_x")
    b = _positive_parameter(semi_axis_y, "semi_axis_y")
    epsilon_f = _validate_epsilon(epsilon)
    distance_scale = min(a, b)

    signed_distance = function (x, y)
        dx = _periodic_displacement(Float64(x), cx, grid.Lx, periodic_x)
        dy = _periodic_displacement(Float64(y), cy, grid.Ly, periodic_y)
        normalized_radius = hypot(dx / a, dy / b)
        return (normalized_radius - 1.0) * distance_scale
    end

    return phase_field_from_distance(grid, signed_distance, epsilon_f)
end

end # module InterfaceInitialization
