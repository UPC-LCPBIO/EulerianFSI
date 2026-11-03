raw"""
    GrowthModels

Constitutive closures for local volumetric growth and growth kinematics.

The module separates two distinct modelling choices:

1. the scalar volumetric source

G = \mathcal{G}(\phi, c, \ldots),

which determines the local rate of natural-volume production; and

2. the growth tensor

\mathbf{K}_g = \mathcal{K}(G, \ldots),
\qquad
\operatorname{tr}(\mathbf{K}_g) = G,

which determines how that volumetric production is distributed directionally.

The module contains no mesh, boundary-condition, pressure, or velocity solve.
In particular, it does not introduce an auxiliary growth potential or an
expansion velocity. The mechanical solver is responsible for enforcing the
kinematic constraint `div(v) = G`.

No clipping of `phi` or the resource field is performed here. Positivity and
boundedness policies belong to the corresponding numerical transport layers.
"""
module GrowthModels

export AbstractGrowthSource,
       NoGrowth,
       ConstantIntrinsicGrowth,
       SaturatingResourceGrowth,
       AbstractGrowthTensorModel,
       IsotropicGrowthTensor,
       DirectionalGrowthTensor,
       growth_source!,
       growth_tensor!,
       growth_diagnostics

# ==============================================================================
# GROWTH-SOURCE MODELS
# ==============================================================================

"""Abstract supertype for scalar volumetric growth-source models."""
abstract type AbstractGrowthSource end

"""No volumetric growth: `G = 0`."""
struct NoGrowth <: AbstractGrowthSource end

raw"""
    ConstantIntrinsicGrowth(rate)

Spatially constant intrinsic volumetric growth rate localized by the diffuse
structural indicator:

G = \phi\,g_0,

where `rate = g_0`.

The stored rate is allowed to be signed. A positive value represents growth;
a negative value may be used deliberately by a model that interprets it as
volumetric loss. The constitutive evaluator does not clamp `phi`.
"""
struct ConstantIntrinsicGrowth{T<:Real} <: AbstractGrowthSource
    rate::T

    function ConstantIntrinsicGrowth(rate::T) where {T<:Real}
        isfinite(rate) || throw(ArgumentError("Intrinsic growth rate must be finite."))
        return new{T}(rate)
    end
end

raw"""
    SaturatingResourceGrowth(rate, half_saturation)

Resource-limited volumetric growth law

G = \phi\,g_{\max}\,\frac{c}{K_G+c},

with `rate = g_max` and `half_saturation = K_G`.

For the dimensionless baseline model, `rate` corresponds to `Da_g`. The
half-saturation constant must be strictly positive. The resource field is used
exactly as supplied; no clipping or denominator regularization is introduced
inside this constitutive law.
"""
struct SaturatingResourceGrowth{TR<:Real,TK<:Real} <: AbstractGrowthSource
    rate::TR
    half_saturation::TK

    function SaturatingResourceGrowth(rate::TR, half_saturation::TK) where {TR<:Real,TK<:Real}
        isfinite(rate) || throw(ArgumentError("Resource-limited growth rate must be finite."))
        isfinite(half_saturation) || throw(ArgumentError("Growth half-saturation constant must be finite."))
        half_saturation > zero(TK) || throw(ArgumentError(
            "Growth half-saturation constant must be strictly positive."
        ))
        return new{TR,TK}(rate, half_saturation)
    end
end

# ==============================================================================
# GROWTH-TENSOR MODELS
# ==============================================================================

"""Abstract supertype for growth-tensor kinematic closures."""
abstract type AbstractGrowthTensorModel end

raw"""
    IsotropicGrowthTensor()

Two-dimensional isotropic growth kinematics:

\mathbf{K}_g = \frac{G}{2}\mathbf{I}.

Therefore

K_{g,xx} = K_{g,yy} = \frac{G}{2},
\qquad
K_{g,xy} = K_{g,yx} = 0,

and `tr(K_g) = G` pointwise.
"""
struct IsotropicGrowthTensor <: AbstractGrowthTensorModel end

raw"""
    DirectionalGrowthTensor(direction_x, direction_y)

Two-dimensional uniaxial growth kinematics along a prescribed material
orientation `t`:

\mathbf{K}_g = G\,\mathbf{t}\otimes\mathbf{t},
\qquad
\|\mathbf{t}\| = 1.

The supplied direction is normalized by the constructor, so the constitutive
compatibility condition is preserved pointwise:

\operatorname{tr}(\mathbf{K}_g)
= G(t_x^2+t_y^2)
= G.

This closure is appropriate when the preferred growth direction is known and
fixed in the Eulerian frame. It is deliberately distinct from a
configuration-dependent director model, which should provide its own local
orientation field.
"""
struct DirectionalGrowthTensor <: AbstractGrowthTensorModel
    direction_x::Float64
    direction_y::Float64

    function DirectionalGrowthTensor(direction_x::Real, direction_y::Real)
        tx = Float64(direction_x)
        ty = Float64(direction_y)
        isfinite(tx) && isfinite(ty) || throw(ArgumentError(
            "Directional-growth components must be finite."
        ))

        magnitude = hypot(tx, ty)
        magnitude > 0.0 || throw(ArgumentError(
            "Directional growth requires a non-zero direction vector."
        ))

        return new(tx / magnitude, ty / magnitude)
    end
end

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

function _validate_output_shape(output, reference, output_name::AbstractString, reference_name::AbstractString)
    expected = size(reference)
    size(output) == expected || throw(DimensionMismatch(
        "$output_name must have size $expected to match $reference_name; received $(size(output))."
    ))
    return nothing
end

function _validate_same_shape(fields::Pair...)
    isempty(fields) && return nothing
    reference_name = first(fields[1])
    reference = last(fields[1])
    expected = size(reference)
    for entry in fields[2:end]
        name = first(entry)
        field = last(entry)
        size(field) == expected || throw(DimensionMismatch(
            "$name must have size $expected to match $reference_name; received $(size(field))."
        ))
    end
    return nothing
end

# ==============================================================================
# GROWTH-SOURCE EVALUATION
# ==============================================================================

"""
    growth_source!(G, model, phi[, resource])

Evaluate the scalar volumetric source in-place.

`G` and `phi` are cell-centred fields with identical sizes. Models that depend
on a transported resource additionally require a cell-centred `resource`
field of the same size.

For time integrators that always carry a resource field, the four-argument
form may also be used with `NoGrowth` and `ConstantIntrinsicGrowth`; the unused
resource is deliberately ignored.
"""
function growth_source!(
    G,
    ::NoGrowth,
    phi,
    resource=nothing,
)
    _validate_output_shape(G, phi, "G", "phi")
    fill!(G, zero(eltype(G)))
    return G
end

function growth_source!(
    G,
    model::ConstantIntrinsicGrowth,
    phi,
    resource=nothing,
)
    _validate_output_shape(G, phi, "G", "phi")
    rate = model.rate
    @. G = rate * phi
    return G
end

function growth_source!(
    G,
    model::SaturatingResourceGrowth,
    phi,
    resource,
)
    _validate_output_shape(G, phi, "G", "phi")
    _validate_output_shape(resource, phi, "resource", "phi")

    rate = model.rate
    half_saturation = model.half_saturation
    @. G = rate * phi * resource / (half_saturation + resource)
    return G
end

# ==============================================================================
# GROWTH-TENSOR EVALUATION
# ==============================================================================

raw"""
    growth_tensor!(Kg_xx, Kg_xy, Kg_yx, Kg_yy, model, G)

Construct the two-dimensional growth tensor in-place from the scalar volumetric
source `G`.

All fields are cell-centred and must have identical sizes. A valid growth
tensor model is responsible for satisfying the constitutive compatibility
condition

\operatorname{tr}(\mathbf{K}_g) = G.
"""
function growth_tensor!(
    Kg_xx,
    Kg_xy,
    Kg_yx,
    Kg_yy,
    ::IsotropicGrowthTensor,
    G,
)
    _validate_same_shape(
        "G" => G,
        "Kg_xx" => Kg_xx,
        "Kg_xy" => Kg_xy,
        "Kg_yx" => Kg_yx,
        "Kg_yy" => Kg_yy,
    )

    @. Kg_xx = 0.5 * G
    @. Kg_yy = 0.5 * G
    fill!(Kg_xy, zero(eltype(Kg_xy)))
    fill!(Kg_yx, zero(eltype(Kg_yx)))
    return Kg_xx, Kg_xy, Kg_yx, Kg_yy
end

function growth_tensor!(
    Kg_xx,
    Kg_xy,
    Kg_yx,
    Kg_yy,
    model::DirectionalGrowthTensor,
    G,
)
    _validate_same_shape(
        "G" => G,
        "Kg_xx" => Kg_xx,
        "Kg_xy" => Kg_xy,
        "Kg_yx" => Kg_yx,
        "Kg_yy" => Kg_yy,
    )

    tx = model.direction_x
    ty = model.direction_y
    txx = tx * tx
    txy = tx * ty
    tyy = ty * ty

    @. Kg_xx = txx * G
    @. Kg_xy = txy * G
    @. Kg_yx = txy * G
    @. Kg_yy = tyy * G
    return Kg_xx, Kg_xy, Kg_yx, Kg_yy
end

# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

"""
    growth_diagnostics(G, Kg_xx, Kg_xy, Kg_yx, Kg_yy)

Return lightweight constitutive diagnostics for the growth fields.

The trace error is evaluated pointwise as

K_{g,xx} + K_{g,yy} - G.

No mesh-weighted integral is computed here because `GrowthModels` is deliberately
independent of mesh geometry.
"""
function growth_diagnostics(G, Kg_xx, Kg_xy, Kg_yx, Kg_yy)
    _validate_same_shape(
        "G" => G,
        "Kg_xx" => Kg_xx,
        "Kg_xy" => Kg_xy,
        "Kg_yx" => Kg_yx,
        "Kg_yy" => Kg_yy,
    )

    trace_error = @. Kg_xx + Kg_yy - G
    return (
        minimum_growth = minimum(G),
        maximum_growth = maximum(G),
        maximum_trace_error = maximum(abs, trace_error),
        maximum_offdiagonal_magnitude = max(maximum(abs, Kg_xy), maximum(abs, Kg_yx)),
    )
end

end # module GrowthModels
