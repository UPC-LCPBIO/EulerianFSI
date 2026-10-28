using Test
using LinearAlgebra
using EulerianFSI
using EulerianFSI.StructureTensor
using EulerianFSI.StructuralConstitutive

@testset "Geometric structural-tensor extension" begin
    model = GeometricStructureExtension(20.0, 4.0; safety_cutoff=1e-6)

    @test model.max_rate == 20.0
    @test model.exponent == 4.0
    @test model.safety_cutoff == 1e-6
    @test_throws ArgumentError GeometricStructureExtension(-1.0)
    @test_throws ArgumentError GeometricStructureExtension(1.0, 0.0)
    @test_throws ArgumentError GeometricStructureExtension(1.0, 4.0; safety_cutoff=1.0)

    # The neutral state is a fixed point for every phase value.
    sigma_xx = ones(2, 2)
    sigma_yy = ones(2, 2)
    sigma_xy = zeros(2, 2)
    phi = [1.0 0.5; 0.2 0.0]
    rx, ry, rxy = structure_extension_rhs(model, sigma_xx, sigma_yy, sigma_xy, phi)
    @test rx == zeros(2, 2)
    @test ry == zeros(2, 2)
    @test rxy == zeros(2, 2)

    # For an SPD unimodular state, the geometric extension preserves
    # d/dt log(det(sigma)) = tr(inv(sigma) R_ext) = 0.
    sigma_xx .= 2.0
    sigma_yy .= 0.5
    sigma_xy .= 0.0
    phi .= 0.2
    rx, ry, rxy = structure_extension_rhs(model, sigma_xx, sigma_yy, sigma_xy, phi)

    S = [2.0 0.0; 0.0 0.5]
    R = [rx[1] rxy[1]; rxy[1] ry[1]]
    @test abs(tr(inv(S) * R)) <= 1e-12

    # A rotated SPD state checks the off-diagonal spectral evaluation.
    theta = 0.37
    Q = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    eigenvalues = [3.0, 1.0 / 3.0]
    Srot = Q * Diagonal(eigenvalues) * Q'
    expected_sigma_log_sigma =
        Q * Diagonal(eigenvalues .* log.(eigenvalues)) * Q'
    phi_rot = fill(0.3, 1, 1)
    rx_rot, ry_rot, rxy_rot = structure_extension_rhs(
        model,
        fill(Srot[1, 1], 1, 1),
        fill(Srot[2, 2], 1, 1),
        fill(Srot[1, 2], 1, 1),
        phi_rot,
    )
    kappa_rot = model.max_rate * (1.0 - phi_rot[1])^model.exponent
    @test rx_rot[1] ≈ -kappa_rot * expected_sigma_log_sigma[1, 1] atol=1e-12
    @test ry_rot[1] ≈ -kappa_rot * expected_sigma_log_sigma[2, 2] atol=1e-12
    @test rxy_rot[1] ≈ -kappa_rot * expected_sigma_log_sigma[1, 2] atol=1e-12

    # The extension contribution is dissipative for the elastic free energy.
    energy_gradient = 0.5 .* ([1.0 0.0; 0.0 1.0] - inv(S))
    @test tr(Matrix(energy_gradient)' * R) <= 1e-12

    # The far-fluid reset is only a safety mechanism and acts below the
    # configured tiny phase threshold.
    sigma_xx .= 2.0
    sigma_yy .= 0.5
    sigma_xy .= 0.1
    phi .= 1.0
    phi[1] = 0.5e-6
    count = apply_structure_extension!(model, sigma_xx, sigma_yy, sigma_xy, phi)
    @test count == 1
    @test sigma_xx[1] == 1.0
    @test sigma_yy[1] == 1.0
    @test sigma_xy[1] == 0.0
    @test sigma_xx[2] == 2.0
end

@testset "Physical Maxwell relaxation remains independent" begin
    maxwell = MaxwellRelaxation(0.5)
    sigma_xx = fill(2.0, 1, 1)
    sigma_yy = fill(0.5, 1, 1)
    sigma_xy = fill(0.1, 1, 1)
    rx, ry, rxy = relaxation_rhs(maxwell, sigma_xx, sigma_yy, sigma_xy)
    @test rx[1] == -0.5
    @test ry[1] == 0.25
    @test rxy[1] == -0.05
end


@testset "Kolmogorov body force" begin
    grid = generate_grid(
        6.0, 2.0, 24, 8;
        stretch_x=:uniform,
        stretch_y=:uniform,
    )
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=false)

    n = 4
    ky = 2pi * n / 2.0
    amplitude = ky^2 / 100.0
    model = KolmogorovBodyForce(amplitude, ky)

    Fx = zeros(metrics.Nx + 1, metrics.Ny)
    Fy = zeros(metrics.Nx, metrics.Ny + 1)
    EulerianFSI.TransportConstitutive.body_force!(
        Fx, Fy, model, 0.0, metrics
    )

    @test all(Fy .== 0.0)
    @test all(isfinite, Fx)

    y = 0.0
    for j in 1:metrics.Ny
        y += 0.5 * metrics.dy_c[j]
        expected = amplitude * sin(ky * y)
        @test maximum(abs.(Fx[:, j] .- expected)) <= 1e-12
        y += 0.5 * metrics.dy_c[j]
    end
end

include(joinpath(@__DIR__, "..", "scripts", "benchmarks", "BenchmarkUtilities.jl"))
using .BenchmarkUtilities

@testset "Benchmark interface-study utilities" begin
    fixed = resolve_interface_epsilon(0.03125, 2.0, 1 / 64)
    @test fixed.mode === :fixed
    @test fixed.epsilon == 0.03125
    @test fixed.epsilon_over_h ≈ 2.0
    @test fixed.cells_5_95 ≈ INTERFACE_WIDTH_5_95_FACTOR * 2.0
    @test interface_study_descriptor(fixed).study == "fixed_epsilon"
    @test interface_study_descriptor(fixed).parameter == "eps_0p03125"

    joint = resolve_interface_epsilon(NaN, 2.0, 1 / 128)
    @test joint.mode === :grid_scaled
    @test joint.epsilon ≈ 2 / 128
    @test joint.epsilon_over_h ≈ 2.0
    @test interface_study_descriptor(joint).study == "sharp_interface"
    @test interface_study_descriptor(joint).parameter == "eps_over_h_2"

    tensor = tensor_cell_diagnostics_2d(2.0, 0.5, 0.0, 0.2, 10.0)
    @test tensor.determinant ≈ 1.0
    @test tensor.determinant_error ≈ 0.0
    @test tensor.lambda_min ≈ 0.5
    @test tensor.lambda_max ≈ 2.0
    @test tensor.effective_stress_frobenius > 0.0
end


include(joinpath(@__DIR__, "..", "scripts", "benchmarks", "analyze_convergence.jl"))

@testset "Convergence-order postprocessor" begin
    # q(h) = q_exact + 3 h^2 should recover second order on non-uniform h.
    h1, h2, h3 = 1 / 32, 1 / 48, 1 / 64
    q1 = 2.0 + 3.0 * h1^2
    q2 = 2.0 + 3.0 * h2^2
    q3 = 2.0 + 3.0 * h3^2
    @test observed_order(q1, q2, q3, h1, h2, h3) ≈ 2.0 atol=1e-7
end

@testset "Interchangeable phase-field methods" begin
    grid = generate_grid(1.0, 1.0, 16, 16; stretch_x=:uniform, stretch_y=:uniform)
    metrics = build_mesh_metrics(grid; periodic_x=true, periodic_y=true)
    bc = BoundaryConditionSetup(Periodic(), Periodic(), Periodic(), Periodic())

    phi = fill(0.37, 16, 16)
    U = fill(0.2, 17, 16)
    V = fill(-0.1, 16, 17)
    U[end, :] .= U[1, :]
    V[:, end] .= V[:, 1]

    current = CurrentCDIPhaseField(4.0, 2 / 16; delta_normal=1e-6)
    acdi = JainACDIPhaseField(0.2, 0.51 / 16)
    uwcdi = PirozzoliUWCDIPhaseField(1.0)
    uwcdi_youngs = PirozzoliUWCDIPhaseField(1.0; normal_reconstruction=:youngs)
    uwcdi_smoothed9 = PirozzoliUWCDIPhaseField(1.0; normal_reconstruction=:smoothed9)

    @test phase_field_label(current) == "current_cdi"
    @test phase_field_label(acdi) == "jain_acdi"
    @test phase_field_label(uwcdi) == "pirozzoli_uwcdi"
    @test jain_equilibrium_initialization_epsilon(acdi) ≈ 1.02 / 16

    # A spatially constant phase field under constant divergence-free velocity
    # must remain constant for all three complete phase-field operators.
    for method in (current, acdi, uwcdi, uwcdi_youngs, uwcdi_smoothed9)
        rhs = phase_field_rhs(method, U, V, phi, 0.0, metrics, bc)
        @test maximum(abs, rhs) <= 1e-11
    end

    # Conservative periodic fluxes must integrate to zero at the semi-discrete
    # level for a nontrivial smooth phase field.
    phi_wave = [
        0.5 + 0.2 * sin(2pi * grid.x_p[i]) * cos(2pi * grid.y_p[j])
        for i in 1:16, j in 1:16
    ]
    for method in (current, acdi, uwcdi, uwcdi_youngs, uwcdi_smoothed9)
        rhs = phase_field_rhs(method, U, V, phi_wave, 0.0, metrics, bc)
        integral = 0.0
        @inbounds for j in 1:16, i in 1:16
            integral += rhs[i, j] * metrics.dx_c[i] * metrics.dy_c[j]
        end
        @test abs(integral) <= 1e-11
    end

    # Dynamic ACDI mobility vanishes with a zero velocity field.
    U0 = zeros(17, 16)
    V0 = zeros(16, 17)
    acdi_dynamic = JainACDIPhaseField(1.0, 0.51 / 16; dynamic_gamma=true)
    rhs_zero_velocity = phase_field_rhs(acdi_dynamic, U0, V0, phi_wave, 0.0, metrics, bc)
    @test maximum(abs, rhs_zero_velocity) <= 1e-12

    # The published UWCDI directional update remains conservative on a
    # periodic grid.
    mass0 = sum(
        phi_wave[i, j] * metrics.dx_c[i] * metrics.dy_c[j]
        for i in 1:16, j in 1:16
    )
    phi_split = uwcdi_strang_step(uwcdi, U, V, phi_wave, 0.01, metrics, bc)
    mass1 = sum(
        phi_split[i, j] * metrics.dx_c[i] * metrics.dy_c[j]
        for i in 1:16, j in 1:16
    )
    @test mass1 ≈ mass0 atol=1e-12

    @test_throws ArgumentError PirozzoliUWCDIPhaseField(0.9)
    @test_throws ArgumentError PirozzoliUWCDIPhaseField(1.0; normal_reconstruction=:invalid)
    @test_throws ArgumentError JainACDIPhaseField(-1.0, 0.01)
end
