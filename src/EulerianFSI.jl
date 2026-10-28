"""Two-dimensional, fully Eulerian fluid–structure interaction on a staggered grid."""
module EulerianFSI

# NUMERICAL INFRASTRUCTURE

include("BoundaryConditions.jl")
include("MeshGeneration.jl")
include("Operators.jl")
include("Projection.jl")

# INTERFACE TRANSPORT AND GEOMETRY

include("MomentumTransport.jl")
include("InterfaceInitialization.jl")
include("InterfaceRegularization.jl")
include("InterfaceTransport.jl")
include("PhaseFieldMethods.jl")
include("Capillarity.jl")

# STRUCTURAL MECHANICS AND MATERIAL CLOSURES

include("StructuralConstitutive.jl")
include("StructureTensor.jl")
include("TransportConstitutive.jl")

# COUPLED TIME INTEGRATION

include("TimeIntegrationIMEX.jl")

using .BoundaryConditions
using .MeshGeneration
using .Operators
using .Projection
using .MomentumTransport
using .InterfaceInitialization
using .InterfaceRegularization
using .InterfaceTransport
using .PhaseFieldMethods
using .Capillarity
using .StructuralConstitutive
using .StructureTensor
using .TransportConstitutive
using .TimeIntegrationIMEX

# Module namespaces remain public so advanced users can work explicitly within
# each numerical or constitutive layer.
export BoundaryConditions,
       MeshGeneration,
       Operators,
       Projection,
       MomentumTransport,
       InterfaceInitialization,
       InterfaceRegularization,
       InterfaceTransport,
       PhaseFieldMethods,
       Capillarity,
       StructuralConstitutive,
       StructureTensor,
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
       NoViscosity,
       ConstantViscosity,
       HarmonicPhaseViscosity,
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
