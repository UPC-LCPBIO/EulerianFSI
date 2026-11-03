using EulerianFSI.GrowthModels
using EulerianFSI.ResourceTransport
using EulerianFSI.TransportConstitutive
using EulerianFSI.Projection

@testset "Growth constitutive closures" begin
    phi = [0.0 0.5; 0.25 1.0]
    resource = fill(0.25, 2, 2)
    G = similar(phi)
    growth_source!(G, NoGrowth(), phi, resource)
    @test all(iszero, G)

    growth_source!(G, ConstantIntrinsicGrowth(0.2), phi, resource)
    @test G ≈ 0.2 .* phi
    growth_source!(G, SaturatingResourceGrowth(0.6, 0.25), phi, resource)
    @test G ≈ 0.3 .* phi
    @test_throws ArgumentError SaturatingResourceGrowth(0.6, 0.0)

    Kxx, Kxy, Kyx, Kyy = (zeros(2, 2) for _ in 1:4)
    growth_tensor!(Kxx, Kxy, Kyx, Kyy, IsotropicGrowthTensor(), G)
    @test Kxx ≈ 0.5 .* G
    @test Kyy ≈ 0.5 .* G
    @test iszero(growth_diagnostics(G, Kxx, Kxy, Kyx, Kyy).maximum_trace_error)

    growth_tensor!(Kxx, Kxy, Kyx, Kyy, DirectionalGrowthTensor(3.0, 4.0), G)
    @test maximum(abs, Kxx .+ Kyy .- G) < 1e-14
    @test Kxy ≈ Kyx
    @test Kxx ≈ (9 / 25) .* G
    @test_throws ArgumentError DirectionalGrowthTensor(0.0, 0.0)
end

@testset "Resource transport and consumption" begin
    grid = generate_grid(1.0, 1.0, 8, 8; stretch_x=:uniform, stretch_y=:uniform)
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=true)
    periodic = BoundaryConditionSetup(Periodic(), Periodic(), Periodic(), Periodic())
    c = [0.5 + 0.1 * sin(2pi * grid.x_p[i]) for i in 1:8, j in 1:8]
    U = fill(0.2, 9, 8)
    V = zeros(8, 9)
    U[end, :] .= U[1, :]

    rhs = resource_advection(U, V, c, metrics, periodic)
    total = sum(rhs[i,j] * metrics.dx_c[i] * metrics.dy_c[j] for i in 1:8, j in 1:8)
    @test abs(total) < 1e-12

    phi = fill(0.5, 8, 8)
    D = zeros(8, 8)
    diffusivity_field!(D, ConstantDiffusivity(0.01), phi, metrics)
    @test all(D .== 0.01)
    diffusivity_field!(D, HarmonicPhaseDiffusivity(0.1, 0.01), phi, metrics)
    @test all(D .> 0.0)
    @test_throws ArgumentError ConstantDiffusivity(0.0)

    sink = similar(phi)
    resource_sink!(sink, SaturatingResourceConsumption(0.4, 0.5), phi, c, metrics)
    @test all(sink .>= 0.0)
    @test resource_explicit_rhs(U, V, c, 0.0, metrics, periodic) ≈ rhs
end

@testset "Legacy no-growth public constructors" begin
    U, V = zeros(9, 8), zeros(8, 9)
    phi = zeros(8, 8)
    state = FSIState(U, V, phi, ones(8, 8), ones(8, 8), zeros(8, 8))
    @test size(state.resource) == size(phi)
    @test all(iszero, state.resource)

    bc = BoundaryConditionSetup(Periodic(), Periodic(), Periodic(), Periodic())
    boundaries = FSIBoundaryConditions(bc, bc, bc, bc, (xx=bc, yy=bc, xy=bc))
    @test boundaries.resource === bc
    @test FSIModels().growth_source isa NoGrowth
    @test FSIModels().diffusivity isa NoDiffusion
end

@testset "Growth divergence constraint and domain compatibility" begin
    # A positive net growth source cannot be imposed in a closed/periodic domain.
    grid = generate_grid(1.0, 1.0, 8, 8; stretch_x=:uniform, stretch_y=:uniform)
    periodic_metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=true)
    periodic = BoundaryConditionSetup(Periodic(), Periodic(), Periodic(), Periodic())
    closed_system = build_projection_system(periodic_metrics, periodic; use_amg=false)
    @test_throws ArgumentError project!(
        zeros(9, 8), zeros(8, 9), fill(0.1, 8, 8),
        closed_system, periodic_metrics,
    )

    # Dirichlet correction boundaries permit a compensating normal flux.
    open_metrics = build_mesh_metrics(grid; periodic_x=false, periodic_y=false)
    bc_q = BoundaryConditionSetup(Dirichlet(0.0), Dirichlet(0.0), Dirichlet(0.0), Dirichlet(0.0))
    open_system = build_projection_system(open_metrics, bc_q; use_amg=false)
    U, V = zeros(9, 8), zeros(8, 9)
    source = fill(0.05, 8, 8)
    result = project!(U, V, source, open_system, open_metrics;
                      reltol=1e-12, maxiter=2000)
    @test result.constraint_linf < 1e-7
    @test maximum(abs, EulerianFSI.Operators.divergence(U, V, open_metrics) .- source) < 1e-7
end

@testset "Coupled IMEX growth smoke test (compatible outflow boundary)" begin
    # Nonzero net growth requires a net outward flux. The compact IMEX velocity
    # predictor only supports periodic U-right and Dirichlet/periodic V-top.
    # Use x-periodicity, a fixed impermeable upper wall, and an open lower edge:
    # q = 0 at the bottom allows a correction-induced outflow, whereas dq/dn = 0
    # at the top preserves the imposed zero normal velocity there.
    grid = generate_grid(1.0, 1.0, 8, 8; stretch_x=:uniform, stretch_y=:uniform)
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=false)
    px = Periodic()
    bc_neumann = BoundaryConditionSetup(px, px, Neumann(0.0), Neumann(0.0))
    bc_u = bc_neumann
    bc_v = BoundaryConditionSetup(px, px, Neumann(0.0), Dirichlet(0.0))
    bc_q = BoundaryConditionSetup(px, px, Dirichlet(0.0), Neumann(0.0))
    bc_c = BoundaryConditionSetup(px, px, Dirichlet(1.0), Dirichlet(1.0))
    bcs = FSIBoundaryConditions(
        bc_u, bc_v, bc_q, bc_neumann, bc_c,
        (xx=bc_neumann, yy=bc_neumann, xy=bc_neumann),
    )
    state = FSIState(
        zeros(9, 8), zeros(8, 9), fill(0.5, 8, 8), fill(1.0, 8, 8),
        ones(8, 8), ones(8, 8), zeros(8, 8),
    )
    models = FSIModels(
        growth_source=SaturatingResourceGrowth(0.02, 0.5),
        growth_tensor=IsotropicGrowthTensor(),
        diffusivity=ConstantDiffusivity(0.01),
        resource_consumption=SaturatingResourceConsumption(0.002, 0.5),
    )
    context = build_imex_context(metrics, bcs, models; use_projection_amg=false)
    workspace = allocate_imex_workspace(metrics)
    initial_projection = make_velocity_compatible!(state, context, workspace)
    @test initial_projection.constraint_linf < 1e-7
    @test maximum(abs, state.U) > 0.0
    # Diagnostic: the initialization projection solves for a correction of
    # order G, while subsequent IMEX predictors may need corrections orders
    # of magnitude smaller. Do not warm-start the first stage projection
    # with the initialization potential in this isolated smoke test.
    fill!(workspace.correction, 0.0)
    report = integrate_step!(state, 0.0, 1e-4, context, workspace)
    @test report.maximum_projection_constraint_linf < 1e-6
    @test maximum(abs, state.U[end, :] .- state.U[1, :]) < 1e-12
    @test maximum(abs, state.V[:, end]) < 1e-12
    @test report.maximum_growth_trace_error < 1e-12
    @test report.minimum_resource > 0.0
    @test all(isfinite, state.phi)
    @test all(isfinite, state.sigma_xx)
end

@testset "Original FTaC no-growth integration path" begin
    grid = generate_grid(1.0, 1.0, 8, 8; stretch_x=:uniform, stretch_y=:uniform)
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=true)
    bc = BoundaryConditionSetup(Periodic(), Periodic(), Periodic(), Periodic())
    bcs = FSIBoundaryConditions(bc, bc, bc, bc, (xx=bc, yy=bc, xy=bc))
    state = FSIState(
        zeros(9, 8), zeros(8, 9), fill(0.5, 8, 8),
        ones(8, 8), ones(8, 8), zeros(8, 8),
    )
    context = build_imex_context(metrics, bcs, FSIModels(); use_projection_amg=false)
    workspace = allocate_imex_workspace(metrics)
    make_velocity_compatible!(state, context, workspace)
    report = integrate_step!(state, 0.0, 1e-4, context, workspace)
    @test report.maximum_growth_trace_error == 0.0
    @test all(iszero, state.resource)
    @test maximum(abs, state.U) < 1e-9
    @test maximum(abs, state.V) < 1e-9
    @test maximum(abs, state.phi .- 0.5) < 1e-9
end
