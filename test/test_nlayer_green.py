import numpy as np

from mqed.Dyadic_GF.GF_NLayer import LayerSpec, NLayerGreenFunction
from mqed.Dyadic_GF.dcim import (
    fit_exponentials,
    integrate_complex_images,
    integrate_complex_images_range,
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
