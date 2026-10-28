
module MeshGeneration

export MACGrid, MeshMetrics, generate_grid, build_mesh_metrics

const SUPPORTED_STRETCHING_METHODS = (
    :uniform,
    :tanh_edges,
    :sinh_center,
    :chebyshev_edges,
    :drop_centered,
    :boundary_layer_bottom,
)

# GRID DATA STRUCTURES

struct MACGrid
    Lx::Float64
    Ly::Float64
    Nx::Int
    Ny::Int

    dx_nominal::Float64
    dy_nominal::Float64

    x_p::Vector{Float64}
    y_p::Vector{Float64}
    x_u::Vector{Float64}
    y_u::Vector{Float64}
    x_v::Vector{Float64}
    y_v::Vector{Float64}

    X_p::Matrix{Float64}
    Y_p::Matrix{Float64}
    X_u::Matrix{Float64}
    Y_u::Matrix{Float64}
    X_v::Matrix{Float64}
    Y_v::Matrix{Float64}
end

struct MeshMetrics
    # Fundamental distances
    dx_f::Vector{Float64}  # [Nx+1] center-to-center distance across x-faces
    dx_c::Vector{Float64}  # [Nx]   cell width in x
    dy_f::Vector{Float64}  # [Ny+1] center-to-center distance across y-faces
    dy_c::Vector{Float64}  # [Ny]   cell width in y

    # Center -> face interpolation weights
    wL_c2fx::Vector{Float64}  # [Nx+1] weight of west/left cell center
    wR_c2fx::Vector{Float64}  # [Nx+1] weight of east/right cell center
    wB_c2fy::Vector{Float64}  # [Ny+1] weight of south/bottom cell center
    wT_c2fy::Vector{Float64}  # [Ny+1] weight of north/top cell center

    # Face -> center interpolation weights
    wL_f2cx::Vector{Float64}  # [Nx] weight of left x-face
    wR_f2cx::Vector{Float64}  # [Nx] weight of right x-face
    wB_f2cy::Vector{Float64}  # [Ny] weight of bottom y-face
    wT_f2cy::Vector{Float64}  # [Ny] weight of top y-face

    # Corner-to-center geometry
    dx_west_corner::Vector{Float64}   # [Nx+1]
    dx_east_corner::Vector{Float64}   # [Nx+1]
    dy_south_corner::Vector{Float64}  # [Ny+1]
    dy_north_corner::Vector{Float64}  # [Ny+1]

    periodic_x::Bool
    periodic_y::Bool

    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
end

# LOGICAL-TO-PHYSICAL COORDINATE MAPPING

function _validate_stretching_method(method::Symbol)
    method in SUPPORTED_STRETCHING_METHODS || throw(ArgumentError(
        "Unsupported stretching method: $method. Supported methods are " *
        join(string.(SUPPORTED_STRETCHING_METHODS), ", ")
    ))
    return nothing
end

function _stretch_parameter(method::Symbol, gamma)
    _validate_stretching_method(method)

    if method in (:uniform, :chebyshev_edges)
        return 0.0
    end

    value = gamma === nothing ? (method == :drop_centered ? 0.5 : 2.0) : Float64(gamma)

    if method == :drop_centered
        0.0 <= value < 1.0 || throw(ArgumentError(
            "For :drop_centered, the stretching parameter must satisfy 0 <= gamma < 1."
        ))
    else
        value > 0.0 || throw(ArgumentError(
            "For $method, the stretching parameter gamma must be positive."
        ))
    end

    return value
end

"""Map a logical coordinate `xi in [0,1]` to a physical coordinate in `[0,L]`."""
function map_coordinate(xi::Real, L::Real, method::Symbol, gamma::Real)
    xi_f = Float64(xi)
    L_f = Float64(L)
    gamma_f = Float64(gamma)

    0.0 <= xi_f <= 1.0 || throw(ArgumentError("Logical coordinate xi must lie in [0,1]."))
    L_f > 0.0 || throw(ArgumentError("Physical length L must be positive."))

    if method == :uniform
        return L_f * xi_f
    elseif method == :tanh_edges
        return L_f * 0.5 * (1.0 + tanh(gamma_f * (xi_f - 0.5)) / tanh(gamma_f / 2.0))
    elseif method == :sinh_center
        return L_f * 0.5 * (1.0 + sinh(gamma_f * (xi_f - 0.5)) / sinh(gamma_f / 2.0))
    elseif method == :chebyshev_edges
        return L_f * 0.5 * (1.0 - cos(pi * xi_f))
    elseif method == :drop_centered
        beta = gamma_f
        return L_f * (0.5 + (1.0 - beta) * (xi_f - 0.5) + beta * 16.0 * (xi_f - 0.5)^5)
    elseif method == :boundary_layer_bottom
        return L_f * (1.0 - tanh(gamma_f * (1.0 - xi_f)) / tanh(gamma_f))
    else
        throw(ArgumentError("Unsupported stretching method: $method"))
    end
end

function _meshgrid(x::Vector{Float64}, y::Vector{Float64})
    X = [xi for xi in x, _ in y]
    Y = [yj for _ in x, yj in y]
    return X, Y
end

function _validate_grid_geometry(
    x_p::Vector{Float64},
    y_p::Vector{Float64},
    x_u::Vector{Float64},
    y_v::Vector{Float64},
    Lx::Float64,
    Ly::Float64,
)
    all(diff(x_p) .> 0.0) || throw(ArgumentError("Cell-center x-coordinates must be strictly increasing."))
    all(diff(y_p) .> 0.0) || throw(ArgumentError("Cell-center y-coordinates must be strictly increasing."))
    all(diff(x_u) .> 0.0) || throw(ArgumentError("x-face coordinates must be strictly increasing."))
    all(diff(y_v) .> 0.0) || throw(ArgumentError("y-face coordinates must be strictly increasing."))

    isapprox(x_u[1], 0.0; atol=100eps(Float64), rtol=0.0) ||
        throw(ArgumentError("The first x-face must lie at x=0."))
    isapprox(x_u[end], Lx; atol=100eps(Float64) * max(1.0, Lx), rtol=0.0) ||
        throw(ArgumentError("The last x-face must lie at x=Lx."))
    isapprox(y_v[1], 0.0; atol=100eps(Float64), rtol=0.0) ||
        throw(ArgumentError("The first y-face must lie at y=0."))
    isapprox(y_v[end], Ly; atol=100eps(Float64) * max(1.0, Ly), rtol=0.0) ||
        throw(ArgumentError("The last y-face must lie at y=Ly."))

    for i in eachindex(x_p)
        x_u[i] < x_p[i] < x_u[i+1] || throw(ArgumentError(
            "Each x cell center must lie strictly between its adjacent x-faces."
        ))
    end
    for j in eachindex(y_p)
        y_v[j] < y_p[j] < y_v[j+1] || throw(ArgumentError(
            "Each y cell center must lie strictly between its adjacent y-faces."
        ))
    end

    return nothing
end

# GRID GENERATION

"""generate_grid(Lx, Ly, Nx, Ny; stretch_x=:uniform, stretch_y=:uniform, gamma_x=nothing, gamma_y=nothing)."""
function generate_grid(
    Lx::Real,
    Ly::Real,
    Nx::Integer,
    Ny::Integer;
    stretch_x::Symbol = :uniform,
    stretch_y::Symbol = :uniform,
    gamma_x = nothing,
    gamma_y = nothing,
)
    Lx_f = Float64(Lx)
    Ly_f = Float64(Ly)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)

    Lx_f > 0.0 || throw(ArgumentError("Lx must be positive."))
    Ly_f > 0.0 || throw(ArgumentError("Ly must be positive."))
    Nx_i >= 2 || throw(ArgumentError("Nx must be at least 2."))
    Ny_i >= 2 || throw(ArgumentError("Ny must be at least 2."))

    gamma_x_f = _stretch_parameter(stretch_x, gamma_x)
    gamma_y_f = _stretch_parameter(stretch_y, gamma_y)

    dx_nominal = Lx_f / Nx_i
    dy_nominal = Ly_f / Ny_i

    x_u = [map_coordinate((i - 1.0) / Nx_i, Lx_f, stretch_x, gamma_x_f) for i in 1:(Nx_i + 1)]
    y_v = [map_coordinate((j - 1.0) / Ny_i, Ly_f, stretch_y, gamma_y_f) for j in 1:(Ny_i + 1)]

    # Enforce exact domain endpoints. This avoids tiny roundoff offsets generated
    # by nonlinear mappings at xi=0 and xi=1.
    x_u[1] = 0.0
    x_u[end] = Lx_f
    y_v[1] = 0.0
    y_v[end] = Ly_f

    x_p = [map_coordinate((i - 0.5) / Nx_i, Lx_f, stretch_x, gamma_x_f) for i in 1:Nx_i]
    y_p = [map_coordinate((j - 0.5) / Ny_i, Ly_f, stretch_y, gamma_y_f) for j in 1:Ny_i]

    y_u = copy(y_p)
    x_v = copy(x_p)

    _validate_grid_geometry(x_p, y_p, x_u, y_v, Lx_f, Ly_f)

    X_p, Y_p = _meshgrid(x_p, y_p)
    X_u, Y_u = _meshgrid(x_u, y_u)
    X_v, Y_v = _meshgrid(x_v, y_v)

    return MACGrid(
        Lx_f, Ly_f, Nx_i, Ny_i,
        dx_nominal, dy_nominal,
        x_p, y_p, x_u, y_u, x_v, y_v,
        X_p, Y_p, X_u, Y_u, X_v, Y_v,
    )
end

# ONE-DIMENSIONAL METRIC CONSTRUCTION

function _axis_metrics(
    centers::Vector{Float64},
    faces::Vector{Float64},
    L::Float64,
    periodic::Bool,
)
    N = length(centers)
    length(faces) == N + 1 || throw(DimensionMismatch(
        "A staggered axis with N cell centers must contain N+1 faces."
    ))

    # Cell widths: face-to-face distance, used by center-evaluated divergences.
    h_c = diff(faces)

    # Center-to-center distances across faces, used by face-evaluated gradients.
    h_f = Vector{Float64}(undef, N + 1)
    wrapped_center_distance = centers[1] + L - centers[end]
    h_f[1] = periodic ? wrapped_center_distance : 2.0 * (centers[1] - faces[1])
    for i in 2:N
        h_f[i] = centers[i] - centers[i-1]
    end
    h_f[end] = periodic ? wrapped_center_distance : 2.0 * (faces[end] - centers[end])

    # Center -> face linear interpolation.
    w_minus_c2f = Vector{Float64}(undef, N + 1)
    w_plus_c2f = Vector{Float64}(undef, N + 1)

    if periodic
        d_minus = faces[1] + L - centers[end]
        d_plus = centers[1] - faces[1]
        total = d_minus + d_plus
        w_minus_c2f[1] = d_plus / total
        w_plus_c2f[1] = d_minus / total
    else
        # One-sided constant extrapolation at physical boundaries. Boundary-condition
        # operators may overwrite these values when a prescribed boundary value exists.
        w_minus_c2f[1] = 0.0
        w_plus_c2f[1] = 1.0
    end

    for i in 2:N
        d_minus = faces[i] - centers[i-1]
        d_plus = centers[i] - faces[i]
        total = d_minus + d_plus
        w_minus_c2f[i] = d_plus / total
        w_plus_c2f[i] = d_minus / total
    end

    if periodic
        d_minus = faces[end] - centers[end]
        d_plus = centers[1] + L - faces[end]
        total = d_minus + d_plus
        w_minus_c2f[end] = d_plus / total
        w_plus_c2f[end] = d_minus / total
    else
        w_minus_c2f[end] = 1.0
        w_plus_c2f[end] = 0.0
    end

    # Face -> center linear interpolation.
    w_minus_f2c = Vector{Float64}(undef, N)
    w_plus_f2c = Vector{Float64}(undef, N)
    for i in 1:N
        d_minus = centers[i] - faces[i]
        d_plus = faces[i+1] - centers[i]
        total = d_minus + d_plus
        w_minus_f2c[i] = d_plus / total
        w_plus_f2c[i] = d_minus / total
    end

    # Distances from each corner/face line to the adjacent cell centers.
    d_minus_corner = Vector{Float64}(undef, N + 1)
    d_plus_corner = Vector{Float64}(undef, N + 1)

    d_plus_corner[1] = centers[1] - faces[1]
    d_minus_corner[1] = periodic ? faces[1] + L - centers[end] : d_plus_corner[1]

    for i in 2:N
        d_minus_corner[i] = faces[i] - centers[i-1]
        d_plus_corner[i] = centers[i] - faces[i]
    end

    d_minus_corner[end] = faces[end] - centers[end]
    d_plus_corner[end] = periodic ? centers[1] + L - faces[end] : d_minus_corner[end]

    all(h_c .> 0.0) || throw(ArgumentError("All cell widths must be positive."))
    all(h_f .> 0.0) || throw(ArgumentError("All center-to-center distances must be positive."))

    return (
        h_f = h_f,
        h_c = h_c,
        w_minus_c2f = w_minus_c2f,
        w_plus_c2f = w_plus_c2f,
        w_minus_f2c = w_minus_f2c,
        w_plus_f2c = w_plus_f2c,
        d_minus_corner = d_minus_corner,
        d_plus_corner = d_plus_corner,
    )
end

# MESH METRICS

"""Precompute geometric distances and interpolation weights for `grid`."""
function build_mesh_metrics(
    grid::MACGrid;
    periodic_x::Bool = false,
    periodic_y::Bool = false,
)
    xmetrics = _axis_metrics(grid.x_p, grid.x_u, grid.Lx, periodic_x)
    ymetrics = _axis_metrics(grid.y_p, grid.y_v, grid.Ly, periodic_y)

    return MeshMetrics(
        xmetrics.h_f,
        xmetrics.h_c,
        ymetrics.h_f,
        ymetrics.h_c,
        xmetrics.w_minus_c2f,
        xmetrics.w_plus_c2f,
        ymetrics.w_minus_c2f,
        ymetrics.w_plus_c2f,
        xmetrics.w_minus_f2c,
        xmetrics.w_plus_f2c,
        ymetrics.w_minus_f2c,
        ymetrics.w_plus_f2c,
        xmetrics.d_minus_corner,
        xmetrics.d_plus_corner,
        ymetrics.d_minus_corner,
        ymetrics.d_plus_corner,
        periodic_x,
        periodic_y,
        grid.Nx,
        grid.Ny,
        grid.Lx,
        grid.Ly,
    )
end

end # module MeshGeneration
