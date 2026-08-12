import numpy as np
import pytest
from mqed.Dyadic_GF.GF_NLayer import LayerSpec, NLayerGreenFunction
from mqed.Dyadic_GF.GF_Sommerfeld import Greens_function_analytical
from mqed.utils.dgf_data import load_gf_h5, save_gf_h5
from mqed.Dyadic_GF.dcim import (
    fit_exponentials,
    integrate_complex_images,
    integrate_complex_images_range,
)
from mqed.Dyadic_GF.sommerfeld_singularities import (
    BranchCutConfig,
    PoleResidue,
    PoleSearchConfig,
    SommerfeldPole,
    find_poles_by_winding,
    residue_vector_by_contour,
    residue_vector_by_limit,
)
from mqed.utils.SI_unit import c


def test_nlayer_reflection_reduces_to_bare_interfaces():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=4.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=1e7,
    )
    q = 1.0e6

    assert np.isclose(
        solver.reflection_coefficient(q, "down", "s"),
        solver._fresnel(1, 0, q, "s"),
    )
    assert np.isclose(
        solver.reflection_coefficient(q, "up", "p"),
        solver._fresnel(1, 2, q, "p"),
    )


def test_nlayer_no_contrast_has_zero_scattering_kernels():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=1e7,
    )

    kernels = solver.bessel_free_kernels(1.0e6, 40e-9, 40e-9)

    assert np.allclose(kernels, np.zeros(7, dtype=complex))


def test_top_exterior_source_reduces_to_two_layer_sommerfeld_kernels():
    energy_eV = 2.0
    omega = energy_eV * 1.602176634e-19 / 1.054571817e-34
    metal_epsilon = -10.0 + 0.5j
    q = 1.5e7
    z_observer = 30e-9
    z_source = 45e-9
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=metal_epsilon, thickness_m=None),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=2.0e7,
    )
    sommerfeld = Greens_function_analytical(
        metal_epsi=metal_epsilon,
        omega=omega,
        eps_0=1.0,
        qmax=2.0e7,
    )

    actual = solver.bessel_free_kernels(q, z_observer, z_source)
    beta = sommerfeld._kz(0, q)
    phase = np.exp(1j * beta * (z_observer + z_source))
    reflection_s = sommerfeld._rs(q)
    reflection_p = sommerfeld._rp(q)
    expected = np.array(
        [
            reflection_s * q * phase / (2 * beta),
            reflection_s * q * phase / (2 * beta),
            reflection_p * q * beta * phase / (2 * solver.k0**2),
            reflection_p * q * beta * phase / (2 * solver.k0**2),
            1j * reflection_p * q**2 * phase / solver.k0**2,
            1j * reflection_p * q**2 * phase / solver.k0**2,
            reflection_p * q**3 * phase / (beta * solver.k0**2),
        ],
        dtype=complex,
    )

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_top_exterior_source_runs_for_finite_silver_film_on_substrate():
    energy_eV = 2.0
    omega = energy_eV * 1.602176634e-19 / 1.054571817e-34
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None, name="bottom_air"),
            LayerSpec(epsilon=15.8 + 0.24j, thickness_m=5e-6, name="substrate"),
            LayerSpec(epsilon=-17.0 + 2.2j, thickness_m=100e-9, name="silver"),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None, name="top_air"),
        ],
        source_layer=3,
        omega=omega,
        qmax_factor=4.0,
        epsabs=1e-6,
        epsrel=1e-6,
        limit=80,
        integration_method="direct",
    )

    tensor = solver.calculate_total_Green_function(12e-9, 0.0, 300e-9, 300e-9)

    assert tensor.shape == (3, 3)
    assert np.all(np.isfinite(tensor))


def test_top_exterior_source_rejects_negative_interface_heights():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
    )

    with pytest.raises(ValueError, match="finite non-negative heights"):
        solver.bessel_free_kernels(1.0e6, -1e-9, 20e-9)


@pytest.mark.parametrize("position", [np.nan, np.inf, -np.inf])
def test_top_exterior_source_rejects_nonfinite_interface_heights(position):
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
    )

    with pytest.raises(ValueError, match="finite non-negative heights"):
        solver.bessel_free_kernels(1.0e6, position, 20e-9)


def test_top_exterior_airy_denominator_is_regular_at_zero_reflection():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
    )

    assert solver.reflection_coefficient(1.0e6, "down", "p") == 0.0
    assert np.isfinite(solver.airy_denominator(1.0e6, "p"))
    assert solver.airy_denominator(1.0e6, "p") != 0.0


def test_top_exterior_airy_denominator_vanishes_at_two_layer_spp():
    metal_epsilon = -10.0 + 0.5j
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=metal_epsilon, thickness_m=None),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=2.0e7,
    )
    expected_q = solver.k0 * np.lib.scimath.sqrt(
        metal_epsilon / (metal_epsilon + 1.0)
    )

    assert abs(solver.airy_denominator(expected_q, "p")) < 1e-8


def test_nlayer_off_center_evanescent_kernels_remain_finite():
    omega = 2.0 * 1.602176634e-19 / 1.054571817e-34
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=3.0 + 0.0j, thickness_m=20e-9),
            LayerSpec(epsilon=4.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=None,
    )

    q = 10.0 * solver.k0
    beta = solver._kz(solver.source_layer, q)
    original_scaled = np.asarray(solver.amplitude_coefficients(q, 3e-9, 15e-9)) * np.exp(
        1j * beta * solver.source_thickness_m
    )
    stable_scaled = np.asarray(solver._scaled_amplitude_coefficients(q, 3e-9, 15e-9))

    assert np.allclose(stable_scaled, original_scaled, rtol=1e-12, atol=1e-12)

    kernels = solver.bessel_free_kernels(5.0e10, 1e-9, 1e-9)
    assert np.all(np.isfinite(kernels))


def test_dcim_single_exponential_matches_laplace_bessel_transform():
    q_values = np.linspace(0.0, 1.0e8, 80)
    coefficient = 2.0 - 0.25j
    exponent = 3.0e-8 + 0.5e-8j
    samples = coefficient * np.exp(-exponent * q_values)

    fit = fit_exponentials(q_values, samples, image_count=1)
    expected = coefficient / np.sqrt(exponent**2 + (25e-9) ** 2)
    actual = integrate_complex_images(fit, rho=25e-9, order=0)

    assert np.isclose(actual, expected, rtol=1e-5)


def test_shifted_dcim_range_matches_single_exponential_tail():
    q_values = np.linspace(4.0e7, 1.2e8, 80)
    coefficient = 1.4 + 0.3j
    exponent = 2.5e-8 + 0.2e-8j
    samples = coefficient * np.exp(-exponent * (q_values - q_values[0]))

    fit = fit_exponentials(q_values, samples, image_count=1, q_origin=q_values[0])
    actual = integrate_complex_images_range(
        fit,
        rho=30e-9,
        order=0,
        lower=q_values[0],
        upper=np.inf,
    )
    shifted_coefficient = coefficient * np.exp(exponent * q_values[0])
    full_integral = shifted_coefficient / np.sqrt(exponent**2 + (30e-9) ** 2)
    lower_fit = fit_exponentials(
        np.linspace(0.0, q_values[0], 80),
        shifted_coefficient * np.exp(-exponent * np.linspace(0.0, q_values[0], 80)),
        image_count=1,
    )
    lower_integral = integrate_complex_images_range(
        lower_fit,
        rho=30e-9,
        order=0,
        lower=0.0,
        upper=q_values[0],
    )

    assert np.isclose(actual, full_integral - lower_integral, rtol=1e-5)


def test_total_green_function_runs_for_small_five_layer_stack():
    energy_eV = 2.0
    omega = energy_eV * 1.602176634e-19 / 1.054571817e-34
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=-10.0 + 0.5j, thickness_m=20e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=60e-9),
            LayerSpec(epsilon=-10.0 + 0.5j, thickness_m=20e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=2,
        omega=omega,
        qmax=2.0e7,
        epsabs=1e-6,
        epsrel=1e-6,
        limit=80,
    )

    tensor = solver.calculate_total_Green_function(20e-9, 0.0, 30e-9, 30e-9)

    assert tensor.shape == (3, 3)
    assert np.all(np.isfinite(tensor))
    assert solver.k0 == omega / c


def test_compute_one_energy_rejects_nonfinite_green_tensor(monkeypatch):
    from mqed.Dyadic_GF import main_nlayer

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    class NonfiniteCalculator:
        def __init__(self, **kwargs):
            pass

        def calculate_total_Green_function(self, **kwargs):
            return np.full((3, 3), np.nan + 0.0j)

        def vacuum_component(self, **kwargs):
            return np.eye(3, dtype=complex)

    monkeypatch.setattr(
        main_nlayer,
        "build_layers",
        lambda stack_cfg, materials_cfg, omega: [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=20e-9, name="emitter"),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
    )
    monkeypatch.setattr(main_nlayer, "NLayerGreenFunction", NonfiniteCalculator)

    with pytest.raises(FloatingPointError, match="energy 2.000000 eV.*Rx index 0"):
        main_nlayer._compute_one_energy(
            idx=0,
            energy_eV=2.0,
            lambda_m=600e-9,
            rx_values_m=np.array([0.0]),
            z_observer=10e-9,
            z_source=10e-9,
            stack_cfg=AttrDict({"source_layer": 1}),
            materials_cfg=AttrDict({}),
            integ_cfg=None,
        )


def test_hybrid_dcim_fallback_runs_and_records_report():
    energy_eV = 2.0
    omega = energy_eV * 1.602176634e-19 / 1.054571817e-34
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=-10.0 + 0.5j, thickness_m=20e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=60e-9),
            LayerSpec(epsilon=-10.0 + 0.5j, thickness_m=20e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=2,
        omega=omega,
        qmax=1.2e7,
        epsabs=1e-5,
        epsrel=1e-5,
        limit=50,
        integration_method="hybrid_dcim",
        hybrid_direct_q_stop=4.0e6,
        hybrid_tail_q_stop=1.2e7,
        hybrid_sample_count=40,
        hybrid_image_count=6,
        hybrid_validation_rtol=1e3,
    )

    integrals = solver.compute_integrals(20e-9, 30e-9, 30e-9)

    assert integrals.shape == (7,)
    assert np.all(np.isfinite(integrals))
    assert solver.last_hybrid_dcim_report is not None


def test_winding_pole_search_finds_artificial_simple_root():
    target = 2.0 - 0.25j
    config = PoleSearchConfig(
        real_min=0.0,
        real_max=3.0,
        imag_min=-1.0,
        imag_max=0.25,
        contour_points_per_side=12,
        max_depth=5,
        min_box_size=1e-4,
        residual_tol=1e-10,
        dedup_tol=1e-3,
    )

    poles = find_poles_by_winding(lambda q: q - target, "p", config)

    assert len(poles) == 1
    assert poles[0].polarization == "p"
    assert np.isclose(poles[0].q, target)
    assert poles[0].residual < 1e-10


def test_pole_residue_helpers_recover_artificial_vector_residue():
    pole = find_poles_by_winding(
        lambda q: q - (1.0 - 0.2j),
        "s",
        PoleSearchConfig(
            real_min=0.0,
            real_max=2.0,
            imag_min=-0.6,
            imag_max=0.1,
            contour_points_per_side=12,
            max_depth=5,
            min_box_size=1e-4,
            residual_tol=1e-10,
        ),
    )[0]
    expected = np.array([1.5 - 0.25j, -0.75 + 0.5j], dtype=complex)

    def kernel(q):
        return expected / (q - pole.q) + np.array([2.0, -1.0j], dtype=complex)

    contour = residue_vector_by_contour(kernel, pole, radius=0.05, points=64)
    limit = residue_vector_by_limit(kernel, pole, step=0.05)

    assert np.allclose(contour.residues, expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(limit.residues, expected, rtol=1e-12, atol=1e-12)


def test_singularity_aware_direct_quadrature_records_report_for_no_contrast_stack():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=2.0e7,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=40,
        integration_method="singularity_aware",
        pole_search_max_depth=3,
    )

    integrals = solver.compute_integrals(20e-9, 40e-9, 40e-9)

    assert integrals.shape == (7,)
    assert np.allclose(integrals, np.zeros(7, dtype=complex), atol=1e-18)
    assert solver.last_singularity_report is not None
    assert solver.last_singularity_report["method"] == "singularity_aware"
    assert solver.last_singularity_report["pole_count"] == 0
    assert solver.singularity_breakpoints(include_poles=False)


def test_componentwise_quadrature_has_zero_scattering_for_no_contrast_stack():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax_factor=20.0,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=40,
        integration_method="componentwise",
        pole_search_max_depth=2,
    )

    tensor = solver.scatter_component(20e-9, 0.0, 40e-9, 40e-9)

    assert tensor.shape == (3, 3)
    assert np.allclose(tensor, np.zeros((3, 3), dtype=complex), atol=1e-18)
    assert solver.last_componentwise_report is not None
    assert solver.last_componentwise_report["method"] == "componentwise"


def test_compute_integrals_rejects_componentwise_api_ambiguity():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax_factor=20.0,
        integration_method="componentwise",
    )

    with pytest.raises(ValueError, match="componentwise.*scattering tensor.*seven scalar"):
        solver.compute_integrals(20e-9, 40e-9, 40e-9)


def test_scattering_components_from_integrals_split_te_tm_and_sum():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
    )
    integrals = np.arange(1, 8, dtype=float).astype(complex) + 0.5j

    scattering, scattering_te, scattering_tm = solver.scattering_components_from_integrals(
        20e-9,
        10e-9,
        integrals,
    )

    prefactor = 1j / (4 * np.pi)
    assert np.allclose(scattering_te, prefactor * solver.scattering_s_component(20e-9, 10e-9, integrals))
    assert np.allclose(scattering_tm, prefactor * solver.scattering_p_component(20e-9, 10e-9, integrals))
    assert np.allclose(scattering, scattering_te + scattering_tm)
    assert np.allclose(scattering, solver.scatter_component_from_integrals(20e-9, 10e-9, integrals))


def test_green_function_components_reuse_one_integral_call(monkeypatch):
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
    )
    calls = []

    def fake_compute_integrals(rho, z_observer, z_source):
        calls.append((rho, z_observer, z_source))
        return np.arange(1, 8, dtype=float).astype(complex)

    monkeypatch.setattr(solver, "compute_integrals", fake_compute_integrals)

    total, vacuum, scattering_te, scattering_tm = solver.calculate_Green_function_components(
        3e-9,
        4e-9,
        40e-9,
        40e-9,
    )

    assert len(calls) == 1
    assert np.isclose(calls[0][0], 5e-9)
    assert np.allclose(total, vacuum + scattering_te + scattering_tm)


def test_fixed_grid_components_reuse_one_batch_integral_call(monkeypatch):
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=1.0e7,
        integration_method="fixed_grid",
    )
    calls = []

    def fake_batch(rhos, z_observer, z_source):
        calls.append((np.array(rhos), z_observer, z_source))
        return np.tile(np.arange(1, 8, dtype=float).astype(complex), (len(rhos), 1))

    monkeypatch.setattr(solver, "compute_integrals_fixed_grid_for_rhos", fake_batch)

    total, vacuum, scattering_te, scattering_tm = solver.calculate_Green_function_components_for_x_values(
        np.array([0.0, 3e-9]),
        4e-9,
        40e-9,
        40e-9,
    )

    assert len(calls) == 1
    assert total.shape == (2, 3, 3)
    assert np.allclose(total, vacuum + scattering_te + scattering_tm)


def test_componentwise_rho_zero_smoke_for_no_contrast_stack():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax_factor=20.0,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=40,
        integration_method="componentwise",
        pole_search_max_depth=2,
    )

    tensor = solver.scatter_component(0.0, 0.0, 40e-9, 40e-9)

    assert tensor.shape == (3, 3)
    assert np.all(np.isfinite(tensor))
    assert np.allclose(tensor, np.zeros((3, 3), dtype=complex), atol=1e-18)
    assert solver.last_componentwise_report is not None


@pytest.mark.parametrize("qmax_factor", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_qmax_factor_rejects_nonpositive_and_nonfinite_values(qmax_factor):
    with pytest.raises(ValueError, match="finite positive multiplier"):
        NLayerGreenFunction(
            layers=[
                LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
                LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
                LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            ],
            source_layer=1,
            omega=2.0e15,
            qmax=1.0,
            qmax_factor=qmax_factor,
        )


def test_qmax_factor_scales_with_frequency_and_takes_precedence():
    layers = [
        LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
        LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
    ]
    solver = NLayerGreenFunction(
        layers=layers,
        source_layer=1,
        omega=2.0e15,
        qmax=1.0,
        qmax_factor=25.0,
    )
    qmax_only_solver = NLayerGreenFunction(
        layers=layers,
        source_layer=1,
        omega=2.0e15,
        qmax=1.0,
        qmax_factor=None,
    )

    assert np.isclose(solver.qmax, 25.0 * abs(solver.k0))
    assert qmax_only_solver.qmax == 1.0


def test_componentwise_matches_singularity_aware_for_finite_stack():
    common = {
        "layers": [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=4.0 + 0.1j, thickness_m=None),
        ],
        "source_layer": 1,
        "omega": 2.0e15,
        "qmax_factor": 8.0,
        "epsabs": 1e-7,
        "epsrel": 1e-7,
        "limit": 80,
        "pole_search_max_depth": 2,
    }
    componentwise = NLayerGreenFunction(**common, integration_method="componentwise")
    singularity_aware = NLayerGreenFunction(**common, integration_method="singularity_aware")

    componentwise_tensor = componentwise.scatter_component(20e-9, 0.0, 40e-9, 40e-9)
    reference_tensor = singularity_aware.scatter_component(20e-9, 0.0, 40e-9, 40e-9)

    assert np.allclose(componentwise_tensor, reference_tensor, rtol=1e-5, atol=1e-2)


def test_branch_cut_samples_are_finite_for_no_contrast_stack():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=2.0e7,
        dcim_sample_count=6,
    )
    t_values = np.linspace(-0.2 * solver.k0, 0.2 * solver.k0, 5)

    q_values, samples = solver.branch_cut_samples(40e-9, 40e-9, t_values=t_values)
    values = solver.branch_cut_integrals(
        20e-9,
        40e-9,
        40e-9,
        config=BranchCutConfig.from_k0(
            solver.k0,
            branch_layer=1,
            t_limit_factor=0.2,
            side_offset_factor=1e-6,
            epsabs=1e-7,
            epsrel=1e-7,
            limit=40,
        ),
    )

    assert q_values.shape == (5,)
    assert samples.shape == (5, 7)
    assert values.shape == (7,)
    assert np.all(np.isfinite(q_values))
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(values))
    assert np.allclose(samples, np.zeros((5, 7), dtype=complex), atol=1e-18)
    assert np.allclose(values, np.zeros(7, dtype=complex), atol=1e-18)


def test_find_poles_rejects_unknown_polarization():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=2.0e7,
    )

    with pytest.raises(ValueError, match="polarizations"):
        solver.find_poles(polarizations=("x",))


def test_main_nlayer_passes_singularity_aware_options(monkeypatch):
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")
    captured = {}

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc


    class FakeCalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def calculate_total_Green_function(self, x, y, z_observer, z_source):
            return np.eye(3, dtype=complex) * (1.0 + x + y + z_observer + z_source)

        def vacuum_component(self, x, y, z_observer, z_source):
            return np.eye(3, dtype=complex)

    monkeypatch.setattr(
        main_nlayer,
        "build_layers",
        lambda stack_cfg, materials_cfg, omega: [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
    )
    monkeypatch.setattr(main_nlayer, "NLayerGreenFunction", FakeCalculator)
    stack_cfg = AttrDict({"source_layer": 1})
    integ_cfg = AttrDict(
        {
            "method": "singularity_aware",
            "qmax": 2.0e7,
            "qmax_factor": 30.0,
            "epsabs": 1e-6,
            "epsrel": 1e-6,
            "limit": 50,
            "split_propagating": True,
            "dcim_q_start_factor": 0.25,
            "dcim_q_stop_factor": 12.0,
            "hybrid_direct_q_stop_factor": 6.0,
            "hybrid_tail_q_stop_factor": 20.0,
            "pole_search_real_min_factor": 0.1,
            "pole_search_real_max_factor": 8.0,
            "pole_search_imag_min_factor": -3.0,
            "pole_search_imag_max_factor": 1e-4,
            "pole_search_max_depth": 7,
            "pole_search_contour_points": 16,
            "pole_search_residual_tol": 1e-7,
            "pole_search_branch_guard_factor": 2e-4,
            "pole_residue_radius_factor": 3e-4,
            "pole_residue_points": 80,
            "branch_cut_t_limit_factor": 5.0,
            "branch_cut_side_offset_factor": 2e-6,
            "branch_cut_layers": "source",
            "branch_cut_sample_count": 17,
            "branch_cut_image_count": 4,
            "branch_cut_validation_rtol": 0.25,
            "branch_cut_validate": True,
            "branch_cut_fallback_to_singularity_aware": False,
            "branch_cut_include_poles": False,
            "branch_cut_prefactor": "1+0j",
            "branch_cut_jump_sign": -1.0,
            "branch_cut_use_hankel": False,
            "pole_subtraction_validate": True,
            "pole_subtraction_validation_rtol": 0.1,
            "pole_subtraction_validation_atol": 1e-11,
            "pole_subtraction_fallback_to_singularity_aware": False,
            "pole_subtraction_include_poles": True,
            "pole_subtraction_residue_method": "limit",
        }
    )

    _, total, vacuum = main_nlayer._compute_one_energy(
        idx=0,
        energy_eV=2.0,
        lambda_m=600e-9,
        rx_values_m=np.array([10e-9]),
        z_observer=40e-9,
        z_source=40e-9,
        stack_cfg=stack_cfg,
        materials_cfg=AttrDict({}),
        integ_cfg=integ_cfg,
    )

    assert total.shape == (1, 3, 3)
    assert vacuum.shape == (1, 3, 3)
    assert captured["integration_method"] == "singularity_aware"
    assert captured["qmax_factor"] == 30.0
    assert captured["dcim_q_start_factor"] == 0.25
    assert captured["dcim_q_stop_factor"] == 12.0
    assert captured["hybrid_direct_q_stop_factor"] == 6.0
    assert captured["hybrid_tail_q_stop_factor"] == 20.0
    assert captured["pole_search_real_min_factor"] == 0.1
    assert captured["pole_search_real_max_factor"] == 8.0
    assert captured["pole_search_imag_min_factor"] == -3.0
    assert captured["pole_search_imag_max_factor"] == 1e-4
    assert captured["pole_search_max_depth"] == 7
    assert captured["pole_search_contour_points"] == 16
    assert captured["pole_search_residual_tol"] == 1e-7
    assert captured["pole_search_branch_guard_factor"] == 2e-4
    assert captured["pole_residue_radius_factor"] == 3e-4
    assert captured["pole_residue_points"] == 80
    assert captured["branch_cut_t_limit_factor"] == 5.0
    assert captured["branch_cut_side_offset_factor"] == 2e-6
    assert captured["branch_cut_layers"] == "source"
    assert captured["branch_cut_sample_count"] == 17
    assert captured["branch_cut_image_count"] == 4
    assert captured["branch_cut_validation_rtol"] == 0.25
    assert captured["branch_cut_validate"] is True
    assert captured["branch_cut_fallback_to_singularity_aware"] is False
    assert captured["branch_cut_include_poles"] is False
    assert captured["branch_cut_prefactor"] == 1.0 + 0.0j
    assert captured["branch_cut_jump_sign"] == -1.0
    assert captured["branch_cut_use_hankel"] is False
    assert captured["pole_subtraction_validate"] is True
    assert captured["pole_subtraction_validation_rtol"] == 0.1
    assert captured["pole_subtraction_validation_atol"] == 1e-11
    assert captured["pole_subtraction_fallback_to_singularity_aware"] is False
    assert captured["pole_subtraction_include_poles"] is True
    assert captured["pole_subtraction_residue_method"] == "limit"


def test_compute_one_energy_passes_componentwise_method(monkeypatch):
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")
    captured = {}

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    class FakeCalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def calculate_total_Green_function(self, x, y, z_observer, z_source):
            assert captured["integration_method"] == "componentwise"
            return np.eye(3, dtype=complex) * (1.0 + x + y + z_observer + z_source)

        def vacuum_component(self, x, y, z_observer, z_source):
            return np.eye(3, dtype=complex)

    monkeypatch.setattr(
        main_nlayer,
        "build_layers",
        lambda stack_cfg, materials_cfg, omega: [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
    )
    monkeypatch.setattr(main_nlayer, "NLayerGreenFunction", FakeCalculator)
    integ_cfg = AttrDict(
        {
            "method": "componentwise",
            "qmax": 2.0e7,
            "qmax_factor": 30.0,
            "epsabs": 1e-6,
            "epsrel": 1e-6,
            "limit": 50,
            "split_propagating": False,
        }
    )

    _, total, vacuum = main_nlayer._compute_one_energy(
        idx=0,
        energy_eV=2.0,
        lambda_m=600e-9,
        rx_values_m=np.array([0.0, 10e-9]),
        z_observer=40e-9,
        z_source=40e-9,
        stack_cfg=AttrDict({"source_layer": 1}),
        materials_cfg=AttrDict({}),
        integ_cfg=integ_cfg,
    )

    assert total.shape == (2, 3, 3)
    assert vacuum.shape == (2, 3, 3)
    assert captured["integration_method"] == "componentwise"
    assert captured["qmax_factor"] == 30.0


def test_compute_one_energy_opt_in_returns_structure_and_polarization_arrays(monkeypatch):
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    class FakeCalculator:
        def __init__(self, **kwargs):
            pass

        def calculate_Green_function_components(self, x, y, z_observer, z_source):
            vacuum = np.eye(3, dtype=complex)
            scattering_te = np.eye(3, dtype=complex) * (2.0 + x)
            scattering_tm = np.eye(3, dtype=complex) * (3.0 + y + z_observer + z_source)
            return vacuum + scattering_te + scattering_tm, vacuum, scattering_te, scattering_tm

    monkeypatch.setattr(
        main_nlayer,
        "build_layers",
        lambda stack_cfg, materials_cfg, omega: [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
    )
    monkeypatch.setattr(main_nlayer, "NLayerGreenFunction", FakeCalculator)

    result = main_nlayer._compute_one_energy(
        idx=0,
        energy_eV=2.0,
        lambda_m=600e-9,
        rx_values_m=np.array([0.0, 10e-9]),
        z_observer=40e-9,
        z_source=40e-9,
        stack_cfg=AttrDict({"source_layer": 1}),
        materials_cfg=AttrDict({}),
        integ_cfg=AttrDict({"method": "direct", "qmax": 2.0e7, "epsabs": 1e-6, "epsrel": 1e-6, "limit": 50, "split_propagating": False}),
        save_polarization_components=True,
    )

    _, total, vacuum, structure, scattering_te, scattering_tm = result
    assert total.shape == (2, 3, 3)
    assert np.allclose(structure, scattering_te + scattering_tm)
    assert np.allclose(total, vacuum + structure)


def test_compute_one_energy_normalizes_fixed_grid_method(monkeypatch):
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    calls = {"batch": 0}

    class FakeCalculator:
        def __init__(self, **kwargs):
            assert kwargs["integration_method"] == "fixed_grid"

        def calculate_total_Green_functions_for_x_values(self, x_values, **kwargs):
            calls["batch"] += 1
            return np.zeros((len(x_values), 3, 3), dtype=complex)

        def vacuum_component(self, **kwargs):
            return np.zeros((3, 3), dtype=complex)

    monkeypatch.setattr(
        main_nlayer,
        "build_layers",
        lambda stack_cfg, materials_cfg, omega: [
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
    )
    monkeypatch.setattr(main_nlayer, "NLayerGreenFunction", FakeCalculator)

    main_nlayer._compute_one_energy(
        idx=0,
        energy_eV=2.0,
        lambda_m=600e-9,
        rx_values_m=np.array([0.0, 10e-9]),
        z_observer=40e-9,
        z_source=40e-9,
        stack_cfg=AttrDict({"source_layer": 1}),
        materials_cfg=AttrDict({}),
        integ_cfg=AttrDict(
            {
                "method": " Fixed_Grid ",
                "qmax": 2.0e7,
                "epsabs": 1e-6,
                "epsrel": 1e-6,
                "limit": 50,
                "split_propagating": False,
            }
        ),
    )

    assert calls["batch"] == 1


def test_compute_one_energy_rejects_componentwise_polarization_output():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    with pytest.raises(ValueError, match="save_polarization_components.*componentwise"):
        main_nlayer._compute_one_energy(
            idx=0,
            energy_eV=2.0,
            lambda_m=600e-9,
            rx_values_m=np.array([0.0]),
            z_observer=40e-9,
            z_source=40e-9,
            stack_cfg=AttrDict({"source_layer": 1}),
            materials_cfg=AttrDict({}),
            integ_cfg=AttrDict({"method": "componentwise", "qmax": 2.0e7, "epsabs": 1e-6, "epsrel": 1e-6, "limit": 50, "split_propagating": False}),
            save_polarization_components=True,
        )


def test_mpi_task_slices_split_rx_when_energies_cannot_fill_ranks():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    tasks = main_nlayer._mpi_task_slices(
        n_energy=1,
        n_rx=7,
        size=3,
        integration_method="componentwise",
    )

    assert [energy_index for energy_index, _ in tasks] == [0, 0, 0]
    assert [indices.tolist() for _, indices in tasks] == [[0, 1, 2], [3, 4], [5, 6]]
    assert sorted(index for _, indices in tasks for index in indices) == list(range(7))


def test_mpi_task_slices_keep_fixed_grid_batched_by_energy():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    tasks = main_nlayer._mpi_task_slices(
        n_energy=1,
        n_rx=7,
        size=4,
        integration_method="fixed_grid",
    )

    assert len(tasks) == 1
    assert tasks[0][0] == 0
    assert tasks[0][1].tolist() == list(range(7))


def test_mpi_task_slices_keep_energy_rows_when_energies_fill_ranks():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    tasks = main_nlayer._mpi_task_slices(
        n_energy=4,
        n_rx=5,
        size=3,
        integration_method="componentwise",
    )

    assert [energy_index for energy_index, _ in tasks] == [0, 1, 2, 3]
    assert all(indices.tolist() == list(range(5)) for _, indices in tasks)


def test_mpi_task_slices_split_multiple_scarce_energies():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    tasks = main_nlayer._mpi_task_slices(
        n_energy=2,
        n_rx=3,
        size=5,
        integration_method="componentwise",
    )

    assert len(tasks) == 6
    for energy_index in range(2):
        assigned = [indices.item() for task_energy, indices in tasks if task_energy == energy_index]
        assert assigned == [0, 1, 2]


def test_assemble_mpi_results_restores_rx_order_with_empty_rank():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    def tensors(indices, offset):
        return np.asarray([np.eye(3, dtype=complex) * (offset + index) for index in indices])

    gathered = [
        [(0, np.array([0, 3]), tensors([0, 3], 10), tensors([0, 3], 20))],
        [(0, np.array([1, 4]), tensors([1, 4], 10), tensors([1, 4], 20))],
        [(0, np.array([2]), tensors([2], 10), tensors([2], 20))],
        [],
    ]

    total, vacuum = main_nlayer._assemble_mpi_results(
        gathered,
        n_energy=1,
        n_rx=5,
        save_components=False,
    )

    assert total.shape == (1, 5, 3, 3)
    for rx_index in range(5):
        assert np.allclose(total[0, rx_index], np.eye(3) * (10 + rx_index))
        assert np.allclose(vacuum[0, rx_index], np.eye(3) * (20 + rx_index))


def test_assemble_mpi_results_preserves_polarization_components():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")
    vacuum = np.ones((2, 3, 3), dtype=complex)
    scattering_te = np.ones_like(vacuum) * 2.0
    scattering_tm = np.ones_like(vacuum) * 3.0
    structure = scattering_te + scattering_tm
    total = vacuum + structure
    gathered = [[(0, np.array([0, 1]), total, vacuum, structure, scattering_te, scattering_tm)]]

    assembled = main_nlayer._assemble_mpi_results(
        gathered,
        n_energy=1,
        n_rx=2,
        save_components=True,
    )

    result_total, result_vacuum, result_structure, result_te, result_tm = assembled
    assert np.allclose(result_structure, result_te + result_tm)
    assert np.allclose(result_total, result_vacuum + result_structure)


def test_assemble_mpi_results_rejects_missing_or_duplicate_work():
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")
    tensor = np.zeros((1, 3, 3), dtype=complex)
    gathered = [[
        (0, np.array([0]), tensor, tensor),
        (0, np.array([0]), tensor, tensor),
    ]]

    with pytest.raises(RuntimeError, match="missing=.*duplicate="):
        main_nlayer._assemble_mpi_results(
            gathered,
            n_energy=1,
            n_rx=2,
            save_components=False,
        )


@pytest.mark.parametrize(
    "result, message",
    [
        ((0, np.array([-1]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))), r"in \[0, 1\)"),
        ((0, np.array([1]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))), r"in \[0, 1\)"),
        ((0, np.array([0.5]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))), "integers"),
        ((0, np.array([[0]]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))), "one-dimensional"),
        ((0, np.array([], dtype=int), np.zeros((0, 3, 3)), np.zeros((0, 3, 3))), "non-empty"),
        ((0, np.array([0, 0]), np.zeros((2, 3, 3)), np.zeros((2, 3, 3))), "unique"),
        ((-1, np.array([0]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))), "energy index"),
        ((0, np.array([0]), np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), None), "4 fields"),
    ],
)
def test_assemble_mpi_results_rejects_invalid_worker_contracts(result, message):
    main_nlayer = pytest.importorskip("mqed.Dyadic_GF.main_nlayer")

    with pytest.raises(ValueError, match=message):
        main_nlayer._assemble_mpi_results(
            [[result]],
            n_energy=1,
            n_rx=1,
            save_components=False,
        )


def test_save_gf_h5_round_trips_optional_polarization_datasets(tmp_path):
    path = tmp_path / "polarized_gf.h5"
    total = np.ones((1, 2, 3, 3), dtype=complex) * 4.0
    vacuum = np.ones_like(total)
    scattering_te = np.ones_like(total) * 1.0
    scattering_tm = np.ones_like(total) * 2.0
    structure = scattering_te + scattering_tm

    save_gf_h5(
        str(path),
        total,
        vacuum,
        np.array([2.0]),
        np.array([0.0, 2.0]),
        40e-9,
        40e-9,
        Gstructure=structure,
        G_scattering_te=scattering_te,
        G_scattering_tm=scattering_tm,
        attrs={"source": "test"},
    )
    loaded = load_gf_h5(str(path))

    assert np.allclose(loaded["G_structure"], structure)
    assert np.allclose(loaded["G_scattering_te"], scattering_te)
    assert np.allclose(loaded["G_scattering_tm"], scattering_tm)


def test_save_gf_h5_default_schema_omits_optional_polarization_datasets(tmp_path):
    path = tmp_path / "default_gf.h5"
    total = np.ones((1, 1, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)

    save_gf_h5(str(path), total, vacuum, np.array([2.0]), np.array([0.0]), 0.0, 0.0)
    loaded = load_gf_h5(str(path))

    assert "G_structure" not in loaded
    assert "G_scattering_te" not in loaded
    assert "G_scattering_tm" not in loaded


def test_dcim_family_routes_rho_zero_to_singularity_aware(monkeypatch):
    dcim_methods = [
        "dcim",
        "hybrid_dcim",
        "branch_cut_dcim",
        "pole_aware_hybrid_dcim",
    ]
    for method in dcim_methods:
        solver = NLayerGreenFunction(
            layers=[
                LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
                LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
                LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            ],
            source_layer=1,
            omega=2.0e15,
            qmax=8.0e6,
            epsabs=1e-6,
            epsrel=1e-6,
            limit=40,
            integration_method=method,
            dcim_q_stop_factor=4.0,
            hybrid_direct_q_stop_factor=2.0,
            hybrid_tail_q_stop_factor=4.0,
            branch_cut_layers="source",
            branch_cut_sample_count=9,
            branch_cut_image_count=3,
            branch_cut_include_poles=False,
            pole_subtraction_include_poles=False,
            hybrid_sample_count=9,
            hybrid_image_count=3,
            pole_search_max_depth=2,
            pole_search_contour_points=8,
        )
        sentinel = np.arange(7, dtype=float).astype(complex) + 1j
        calls = []

        def fake_singularity_aware(rho, z_observer, z_source):
            calls.append((rho, z_observer, z_source))
            return sentinel

        monkeypatch.setattr(solver, "compute_integrals_singularity_aware", fake_singularity_aware)

        values = solver.compute_integrals(0.0, 40e-9, 40e-9)

        assert calls == [(0.0, 40e-9, 40e-9)]
        assert np.array_equal(values, sentinel)


def test_dcim_q_factor_settings_resolve_against_k0():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=9.0e8,
        dcim_q_start=123.0,
        dcim_q_stop=456.0,
        dcim_q_start_factor=0.5,
        dcim_q_stop_factor=10.0,
        hybrid_direct_q_stop=789.0,
        hybrid_tail_q_stop=987.0,
        hybrid_direct_q_stop_factor=6.0,
        hybrid_tail_q_stop_factor=20.0,
    )

    assert np.isclose(solver._resolve_q_value(solver.dcim_q_start, solver.dcim_q_start_factor), 0.5 * abs(solver.k0))
    assert np.isclose(solver._resolve_q_value(solver.dcim_q_stop, solver.dcim_q_stop_factor), 10.0 * abs(solver.k0))
    assert np.isclose(solver._hybrid_default_direct_q_stop(), 6.0 * abs(solver.k0))
    assert np.isclose(solver._hybrid_default_tail_q_stop(6.0 * abs(solver.k0)), 20.0 * abs(solver.k0))


def test_branch_cut_dcim_zero_stack_accepts_zero_approximation():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=2.0e7,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=40,
        integration_method="branch_cut_dcim",
        branch_cut_layers="source",
        branch_cut_sample_count=9,
        branch_cut_image_count=3,
        branch_cut_t_limit_factor=0.2,
        branch_cut_include_poles=False,
        branch_cut_use_hankel=False,
        branch_cut_validate=True,
        pole_search_max_depth=2,
    )

    integrals = solver.compute_integrals(20e-9, 40e-9, 40e-9)

    assert integrals.shape == (7,)
    assert np.all(np.isfinite(integrals))
    assert np.allclose(integrals, np.zeros(7, dtype=complex), atol=1e-18)
    assert solver.last_branch_cut_dcim_report is not None
    assert solver.last_branch_cut_dcim_report["method"] == "branch_cut_dcim"
    assert solver.last_branch_cut_dcim_report["accepted"] is True
    assert solver.last_branch_cut_dcim_report["branch_layers"] == [1]
    assert solver.last_branch_cut_dcim_report["pole_count"] == 0


def test_branch_cut_dcim_falls_back_when_validation_fails():
    omega = 2.0e15
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=omega,
        qmax=8.0e6,
        epsabs=1e-6,
        epsrel=1e-6,
        limit=40,
        integration_method="branch_cut_dcim",
        branch_cut_layers="source",
        branch_cut_sample_count=9,
        branch_cut_image_count=3,
        branch_cut_t_limit_factor=0.15,
        branch_cut_prefactor=0.0 + 0.0j,
        branch_cut_include_poles=False,
        branch_cut_use_hankel=False,
        branch_cut_validation_rtol=0.0,
        branch_cut_fallback_to_singularity_aware=True,
        pole_search_real_max_factor=1.0,
        pole_search_max_depth=2,
        pole_search_contour_points=8,
    )

    integrals = solver.compute_integrals(20e-9, 40e-9, 40e-9)
    report = solver.last_branch_cut_dcim_report

    assert report is not None
    assert report["accepted"] is False
    assert report["reference"] is not None
    assert np.all(np.isfinite(integrals))
    assert np.allclose(integrals, report["reference"])
    assert report["max_relative_error"] > 0.0


def test_branch_cut_dcim_rho_zero_routes_to_singularity_aware():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=8.0e6,
        epsabs=1e-6,
        epsrel=1e-6,
        limit=40,
        integration_method="branch_cut_dcim",
        branch_cut_layers="source",
        branch_cut_sample_count=9,
        branch_cut_image_count=3,
        branch_cut_t_limit_factor=0.15,
        branch_cut_include_poles=False,
        branch_cut_use_hankel=True,
        branch_cut_fallback_to_singularity_aware=True,
        pole_search_real_max_factor=1.0,
        pole_search_max_depth=2,
        pole_search_contour_points=8,
    )

    integrals = solver.compute_integrals(0.0, 40e-9, 40e-9)
    report = solver.last_singularity_report

    assert report is not None
    assert report["method"] == "singularity_aware"
    assert solver.last_branch_cut_dcim_report is None
    assert np.all(np.isfinite(integrals))


def test_pole_subtracted_direct_zero_stack_accepts_zero_result():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=8.0e6,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=40,
        integration_method="pole_subtracted_direct",
        pole_subtraction_include_poles=True,
        pole_subtraction_validate=True,
        pole_search_real_max_factor=1.0,
        pole_search_max_depth=2,
        pole_search_contour_points=8,
    )

    values = solver.compute_integrals(20e-9, 40e-9, 40e-9)
    report = solver.last_pole_subtraction_report

    assert values.shape == (7,)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, np.zeros(7, dtype=complex), atol=1e-18)
    assert report is not None
    assert report["method"] == "pole_subtracted_direct"
    assert report["accepted"] is True
    assert report["returned"] == "approximation"
    assert report["pole_count"] == 0


def test_pole_subtracted_direct_subtracts_and_adds_back_artificial_pole(monkeypatch):
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=1.0 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=3.0e6,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=50,
        integration_method="pole_subtracted_direct",
        pole_subtraction_validate=True,
        pole_subtraction_validation_rtol=1e-6,
        pole_subtraction_validation_atol=1e-10,
        pole_subtraction_fallback_to_singularity_aware=False,
    )
    pole = SommerfeldPole(q=1.2e6 - 1.5e5j, polarization="p", residual=0.0)
    residues = np.array([1.0, -0.4j, 0.6 + 0.2j, 0.1, -0.2, 0.3j, -0.5], dtype=complex)
    pole_residue = PoleResidue(pole=pole, residues=residues, contour_radius=1.0)
    smooth_scale = 9.0e5
    smooth_weights = np.array([0.2, -0.1j, 0.05, 0.03j, 0.04, -0.02j, 0.01], dtype=complex)

    def artificial_kernel(q, z_observer, z_source):
        del z_observer, z_source
        return smooth_weights * np.exp(-q / smooth_scale) + residues / (q - pole.q)

    monkeypatch.setattr(solver, "bessel_free_kernels", artificial_kernel)
    monkeypatch.setattr(solver, "find_poles", lambda *args, **kwargs: [pole])
    monkeypatch.setattr(
        solver,
        "pole_residues",
        lambda *args, **kwargs: [pole_residue],
    )

    direct = solver.direct_integrals_over_range(25e-9, 40e-9, 40e-9, 0.0, solver.qmax)
    values = solver.compute_integrals(25e-9, 40e-9, 40e-9)
    report = solver.last_pole_subtraction_report

    assert report is not None
    assert report["pole_count"] == 1
    assert report["accepted"] is True
    assert report["returned"] == "approximation"
    assert np.allclose(values, direct, rtol=1e-6, atol=1e-10)
    assert np.linalg.norm(report["pole_values"]) > 0.0
    assert np.linalg.norm(report["smooth_values"]) > 0.0


def test_pole_aware_hybrid_dcim_reports_and_falls_back_for_zero_stack():
    solver = NLayerGreenFunction(
        layers=[
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=80e-9),
            LayerSpec(epsilon=2.25 + 0.0j, thickness_m=None),
        ],
        source_layer=1,
        omega=2.0e15,
        qmax=8.0e6,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=40,
        integration_method="pole_aware_hybrid_dcim",
        hybrid_direct_q_stop=2.0e6,
        hybrid_tail_q_stop=8.0e6,
        hybrid_sample_count=9,
        hybrid_image_count=3,
        hybrid_validation_rtol=1e-2,
        pole_subtraction_include_poles=True,
        pole_subtraction_validate=True,
        pole_search_real_max_factor=1.0,
        pole_search_max_depth=2,
        pole_search_contour_points=8,
    )

    values = solver.compute_integrals(20e-9, 40e-9, 40e-9)
    report = solver.last_pole_aware_hybrid_report

    assert values.shape == (7,)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, np.zeros(7, dtype=complex), atol=1e-18)
    assert report is not None
    assert report["method"] == "pole_aware_hybrid_dcim"
    assert report["accepted"] is True
    assert report["returned"] == "approximation"
    assert report["pole_count"] == 0
