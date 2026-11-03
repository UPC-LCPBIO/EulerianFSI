"""
    EulerianFSI

One-domain Eulerian fluid--structure interaction solver with optional growth,
resource transport, diffuse-interface mechanics, and viscoelastic structural
response.

The package is organized in three layers:

- finite-difference infrastructure (`MeshGeneration`, `BoundaryConditions`,
  `Operators`, and `Projection`);
- transport and constitutive models for momentum, resource, interface, growth,
  and structural mechanics;
- `TimeIntegrationIMEX`, which coordinates the coupled one-velocity model.

The formulation uses a single material velocity and a single volumetric
constraint. Growth enters through a scalar source `G` and a growth tensor `K_g`;
there is no auxiliary growth potential or secondary expansion velocity.
"""
module EulerianFSI

# ==============================================================================
# NUMERICAL INFRASTRUCTURE
# ==============================================================================

include("BoundaryConditions.jl")
include("MeshGeneration.jl")
include("Operators.jl")
include("Projection.jl")

# ==============================================================================
# TRANSPORT AND GEOMETRY
# ==============================================================================

include("MomentumTransport.jl")
include("ResourceTransport.jl")
include("InterfaceInitialization.jl")
include("InterfaceRegularization.jl")
include("InterfaceTransport.jl")
include("PhaseFieldMethods.jl")
include("Capillarity.jl")

# ==============================================================================
# STRUCTURE, GROWTH, AND MATERIAL CLOSURES
# ==============================================================================

include("StructuralConstitutive.jl")
include("StructureTensor.jl")
include("GrowthModels.jl")
include("TransportConstitutive.jl")

# ==============================================================================
# COUPLED TIME INTEGRATION
# ==============================================================================

include("TimeIntegrationIMEX.jl")

using .BoundaryConditions
using .MeshGeneration
using .Operators
using .Projection
using .MomentumTransport
using .ResourceTransport
using .InterfaceInitialization
using .InterfaceRegularization
using .InterfaceTransport
using .PhaseFieldMethods
using .Capillarity
using .StructuralConstitutive
using .StructureTensor
using .GrowthModels
using .TransportConstitutive
using .TimeIntegrationIMEX

# Module namespaces remain public so advanced users can work explicitly within
# each numerical or constitutive layer.
export BoundaryConditions,
       MeshGeneration,
       Operators,
       Projection,
       MomentumTransport,
       ResourceTransport,
       InterfaceInitialization,
       InterfaceRegularization,
       InterfaceTransport,
       PhaseFieldMethods,
       Capillarity,
       StructuralConstitutive,
       StructureTensor,
       GrowthModels,
       TransportConstitutive,
       TimeIntegrationIMEX

# Curated top-level API for ordinary solver construction. Low-level finite-
# difference routines remain available through their module namespaces.
export Periodic,
       Dirichlet,
       Neumann,
       MixedBoundary,
       AdvectiveOutflow,
       BoundaryConditionSetup,
       MACGrid,
       MeshMetrics,
       generate_grid,
       build_mesh_metrics,
       NoGrowth,
       ConstantIntrinsicGrowth,
       SaturatingResourceGrowth,
       IsotropicGrowthTensor,
       DirectionalGrowthTensor,
       NoViscosity,
       ConstantViscosity,
       HarmonicPhaseViscosity,
       NoDiffusion,
       ConstantDiffusivity,
       HarmonicPhaseDiffusivity,
       NoResourceConsumption,
       SaturatingResourceConsumption,
       NoBodyForce,
       ConstantBodyForce,
       KolmogorovBodyForce,
       NoRelaxation,
       MaxwellRelaxation,
       NoElasticStress,
       LinearInternalTensorStress,
       NoCapillarity,
       DiffuseAreaCapillarity,
       NoInterfaceFlux,
       ConservativeDiffuseInterfaceFlux,
       AbstractPhaseFieldMethod,
       LegacySplitPhaseFieldMethod,
       NoPhaseFieldRegularization,
       CurrentCDIPhaseField,
       JainACDIPhaseField,
       PirozzoliUWCDIPhaseField,
       phase_field_rhs,
       uwcdi_directional_rhs,
       uwcdi_strang_step,
       phase_field_label,
       jain_equilibrium_initialization_epsilon,
       NoStructureExtension,
       IdentityOutsideStructure,
       GeometricStructureExtension,
       FSIState,
       FSIBoundaryConditions,
       FSIModels,
       LinearSolverOptions,
       FourStageIMEXFractionalStep,
       IMEXNumerics,
       IMEXContext,
       IMEXWorkspace,
       StepReport,
       build_imex_context,
       allocate_imex_workspace,
       make_velocity_compatible!,
       integrate_step!

end # module EulerianFSI
