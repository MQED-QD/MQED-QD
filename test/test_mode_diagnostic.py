import numpy as np
import pytest
from omegaconf import OmegaConf

from mqed.Dyadic_GF.GF_NLayer import LayerSpec, NLayerGreenFunction
from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.mode_diagnostic import compute_mode_diagnostic, main as diagnostic_main
from mqed.plotting.plot_mode_diagnostic import (
    _source_light_line,
    main as plot_main,
    plot_mode_diagnostic,
)
from mqed.utils.SI_unit import c, eV_to_J, hbar


@pytest.fixture
def interface_config():
    return OmegaConf.create({
        "materials": {
            "metal": {"source_type": "Drude", "drude_config": {
                "eps_inf": 1.0, "omega_p_eV": 8.0, "gamma_eV": 0.1,
            }},
            "air": {"source_type": "constant", "constant_value": 1.0},
        },
        "stack": {"source_layer": 1, "layers": [
            {"name": "metal", "material": "metal"},
            {"name": "air", "material": "air"},
        ]},
        "simulation": {"energy_eV": {"min": 1.8, "max": 2.2, "points": 2}},
    })


@pytest.fixture
def spacer_config(interface_config):
    cfg = OmegaConf.create(OmegaConf.to_container(interface_config, resolve=True))
    cfg.stack.source_layer = 1
    cfg.stack.layers.insert(1, {
        "name": "spacer", "thickness_nm": 80,
        "material_config": {"source_type": "constant", "constant_value": 2.25},
    })
    return cfg


def test_interface_axes_units_dispersion_and_exterior_denominator(interface_config):
    energies = np.array([1.8, 2.2])
    momenta = np.array([0.4, 1.3, 1.8])
    data = compute_mode_diagnostic(interface_config, energies, momenta)
    metal = DataProvider(interface_config.materials.metal)

    np.testing.assert_array_equal(data["energy_eV"], energies)
    np.testing.assert_array_equal(data["q_over_k0"], momenta)
    np.testing.assert_allclose(data["k0_m_inv"], energies * eV_to_J / (hbar * c))
    np.testing.assert_allclose(data["q_m_inv"], data["k0_m_inv"][:, None] * momenta)
    assert data["q_m_inv"].shape == (2, 3)
    assert data["reflection"].shape == (2, 3, 2, 2)
    assert data["characteristic"].shape == (2, 3, 2)
    assert data["directions"].tolist() == ["down", "up"]
    assert data["polarizations"].tolist() == ["s", "p"]
    assert data["source_layer"].item() == 1
    assert data["source_layer_name"].item() == "air"
    assert data["characteristic_kind"].item() == "top_exterior_reflection_denominator"
    np.testing.assert_array_equal(data["source_epsilon"], [1.0, 1.0])
    assert data["reflection_valid"][:, :, 0, :].all()
    assert not data["reflection_valid"][:, :, 1, :].any()
    assert np.isnan(data["reflection"][:, :, 1, :]).all()
    assert data["characteristic_valid"].all()

    for index, energy in enumerate(energies):
        omega = energy * eV_to_J / hbar
        metal_epsilon = metal.get_epsilon(omega)
        np.testing.assert_allclose(metal_epsilon, 1 - 64 / (energy**2 + 0.1j * energy),
                                   rtol=1e-8)
        solver = NLayerGreenFunction(
            layers=[LayerSpec(metal_epsilon, None), LayerSpec(1.0, None)],
            source_layer=1, omega=omega,
        )
        for pol_index, pol in enumerate(("s", "p")):
            for q_index, q in enumerate(data["q_m_inv"][index]):
                np.testing.assert_allclose(
                    data["reflection"][index, q_index, 0, pol_index],
                    solver.reflection_coefficient(q, "down", pol), rtol=1e-12,
                )
                np.testing.assert_allclose(
                    data["characteristic"][index, q_index, pol_index],
                    solver._fresnel_terms(1, 0, q, pol)[1], rtol=1e-12,
                )
    assert not np.isclose(data["source_epsilon"][0].real, metal_epsilon.real)
    assert not np.isclose(data["reflection"][0, 1, 0, 0], data["reflection"][0, 1, 0, 1])


def test_finite_spacer_both_mirrors_airy_and_source_override(spacer_config):
    energies = np.array([1.8, 2.2])
    momenta = np.array([0.5, 1.3, 1.8])
    data = compute_mode_diagnostic(spacer_config, energies, momenta)
    metal = DataProvider(spacer_config.materials.metal)
    assert data["source_layer_name"].item() == "spacer"
    assert data["characteristic_kind"].item() == "finite_source_airy_denominator"
    np.testing.assert_array_equal(data["source_epsilon"], [2.25, 2.25])
    assert data["reflection_valid"].all()
    assert data["characteristic_valid"].all()

    for energy_index, energy in enumerate(energies):
        metal_epsilon = metal.get_epsilon(energy * eV_to_J / hbar)
        solver = NLayerGreenFunction(
            layers=[LayerSpec(metal_epsilon, None), LayerSpec(2.25, 80e-9),
                    LayerSpec(1.0, None)],
            source_layer=1, omega=energy * eV_to_J / hbar,
        )
        for q_index, q in enumerate(data["q_m_inv"][energy_index]):
            for pol_index, pol in enumerate(("s", "p")):
                for direction_index, direction in enumerate(("down", "up")):
                    np.testing.assert_allclose(
                        data["reflection"][energy_index, q_index, direction_index, pol_index],
                        solver.reflection_coefficient(q, direction, pol), rtol=1e-12,
                    )
                down, up = data["reflection"][energy_index, q_index, :, pol_index]
                expected = 1 - down * up * np.exp(2j * solver._kz(1, q) * 80e-9)
                np.testing.assert_allclose(
                    data["characteristic"][energy_index, q_index, pol_index],
                    expected, rtol=1e-12,
                )

    exterior = compute_mode_diagnostic(spacer_config, [2.0], [1.3], source_layer=2)
    assert exterior["source_layer_name"].item() == "air"
    assert exterior["characteristic_kind"].item() == "top_exterior_reflection_denominator"
    assert exterior["reflection_valid"][0, 0, 0].all()
    assert not exterior["reflection_valid"][0, 0, 1].any()


def test_lossy_drude_air_tm_pole_matches_analytic_root(interface_config):
    energy = 2.0
    omega = energy * eV_to_J / hbar
    epsilon = DataProvider(interface_config.materials.metal).get_epsilon(omega)
    expected = np.sqrt(epsilon / (epsilon + 1))
    solver = NLayerGreenFunction(
        [LayerSpec(epsilon, None), LayerSpec(1.0, None)], source_layer=1, omega=omega,
    )
    assert expected.imag > 0
    assert all(solver._kz(layer, expected * solver.k0).imag > 0 for layer in (0, 1))

    data = compute_mode_diagnostic(interface_config, [energy], [1.01, 1.08],
                                   pole_energy_eV=energy)
    assert data["pole_polarization"].tolist() == ["p"]
    np.testing.assert_allclose(data["pole_q_over_k0"], [expected], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(data["pole_q_m_inv"], expected * solver.k0, rtol=1e-6)
    assert data["pole_residual"][0] < 1e-6
    assert abs(solver._pole_characteristic(data["pole_q_m_inv"][0], "p")) < 1e-6


def test_pole_search_outside_real_window_returns_no_candidates(interface_config):
    data = compute_mode_diagnostic(interface_config, [2.0], [1.2, 1.4], pole_energy_eV=2.0)
    assert data["pole_q_m_inv"].size == 0
    assert data["pole_q_over_k0"].size == 0
    assert data["pole_polarization"].size == 0
    assert data["pole_residual"].size == 0


@pytest.mark.parametrize(("energies", "momenta", "message"), [
    ([], [1.3], "energy_eV"),
    ([0.0], [1.3], "energy_eV"),
    ([np.inf], [1.3], "energy_eV"),
    ([2.0, 1.8], [1.3], "energy_eV"),
    ([2.0, 2.0], [1.3], "energy_eV"),
    ([[2.0]], [1.3], "energy_eV"),
    ([2.0], [], "q_over_k0"),
    ([2.0], [-0.1], "q_over_k0"),
    ([2.0], [np.nan], "q_over_k0"),
    ([2.0], [1.5, 1.3], "q_over_k0"),
    ([2.0], [1.3, 1.3], "q_over_k0"),
    ([2.0], [[1.3]], "q_over_k0"),
])
def test_invalid_grids(interface_config, energies, momenta, message):
    with pytest.raises(ValueError, match=message):
        compute_mode_diagnostic(interface_config, energies, momenta)


@pytest.mark.parametrize("source", [-1, 2])
def test_invalid_source_layer(interface_config, source):
    with pytest.raises(ValueError, match="source_layer"):
        compute_mode_diagnostic(interface_config, [2.0], [1.3], source_layer=source)


@pytest.mark.parametrize("energy", [0, -1, np.nan, np.inf])
def test_invalid_pole_energy(interface_config, energy):
    with pytest.raises(ValueError, match="pole_energy_eV"):
        compute_mode_diagnostic(interface_config, [2.0], [1.3], pole_energy_eV=energy)


def test_single_point_grid_and_pole_interval(interface_config, tmp_path):
    data = compute_mode_diagnostic(interface_config, [2.0], [1.3])
    assert data["reflection"].shape == (1, 1, 2, 2)
    assert data["characteristic"].shape == (1, 1, 2)
    source_line = _source_light_line(data, data["energy_eV"])
    np.testing.assert_allclose(source_line, [1.0])
    input_path = tmp_path / "one_point.npz"
    output_path = tmp_path / "one_point.png"
    np.savez_compressed(input_path, **data)
    plot_mode_diagnostic(input_path, output_path)
    assert output_path.is_file() and output_path.stat().st_size > 0
    with pytest.raises(ValueError, match="nonzero q_over_k0 interval"):
        compute_mode_diagnostic(interface_config, [2.0], [1.3], pole_energy_eV=2.0)


@pytest.mark.parametrize(("energy", "momentum"), [
    ([2.0, 2.0], [1.3, 1.5]),
    ([2.0, 2.2], [1.3, 1.3]),
])
def test_plot_rejects_duplicate_grid_coordinates(interface_config, tmp_path, energy, momentum):
    data = compute_mode_diagnostic(interface_config, [2.0, 2.2], [1.3, 1.5])
    data["energy_eV"] = np.asarray(energy)
    data["q_over_k0"] = np.asarray(momentum)
    input_path = tmp_path / "duplicate.npz"
    output_path = tmp_path / "duplicate.png"
    np.savez_compressed(input_path, **data)
    with pytest.raises(ValueError, match="energy_eV and q_over_k0"):
        plot_mode_diagnostic(input_path, output_path)
    assert not output_path.exists()


def test_source_light_line_only_for_real_positive_permittivity():
    energy = np.array([1.8, 2.0, 2.2])
    np.testing.assert_allclose(
        _source_light_line({"source_epsilon": np.array([2.25, -1.0, 1.0 + 0.1j])}, energy),
        [1.5, np.nan, np.nan], equal_nan=True,
    )
    assert _source_light_line({"source_epsilon": np.array([-1.0, 1.0j, np.nan])}, energy) is None
    with pytest.raises(ValueError, match="source_epsilon"):
        _source_light_line({"source_epsilon": [1.0, 2.0]}, energy)


def test_cli_yaml_npz_and_plot_artifacts(interface_config, tmp_path, capsys):
    config_path = tmp_path / "stack.yaml"
    npz_path = tmp_path / "diagnostic.npz"
    image_path = tmp_path / "diagnostic.png"
    OmegaConf.save(interface_config, config_path)
    diagnostic_main(["--config", str(config_path), "--output", str(npz_path),
                     "--q-min", "1.2", "--q-max", "1.8", "--q-points", "3"])
    assert npz_path.is_file() and npz_path.stat().st_size > 0
    assert "Saved" in capsys.readouterr().out
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["reflection"].shape == (2, 3, 2, 2)
        np.testing.assert_allclose(data["energy_eV"], [1.8, 2.2])
        np.testing.assert_allclose(data["q_over_k0"], [1.2, 1.5, 1.8])
        np.testing.assert_allclose(data["source_epsilon"], [1.0, 1.0])
    plot_main([str(npz_path), "--output", str(image_path)])
    assert image_path.is_file() and image_path.stat().st_size > 0
    assert "Saved" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Direction"):
        plot_mode_diagnostic(npz_path, tmp_path / "invalid.png", direction="sideways")
    assert not (tmp_path / "invalid.png").exists()


@pytest.mark.parametrize("arguments", [
    ["--energy-min", "0"], ["--energy-points", "0"],
    ["--q-min", "-1"], ["--q-points", "0"],
    ["--q-min", "2", "--q-max", "1"],
])
def test_cli_rejects_invalid_ranges(interface_config, tmp_path, arguments):
    config_path = tmp_path / "stack.yaml"
    output_path = tmp_path / "invalid.npz"
    OmegaConf.save(interface_config, config_path)
    with pytest.raises(SystemExit, match="2"):
        diagnostic_main(["--config", str(config_path), "--output", str(output_path),
                         *arguments])
    assert not output_path.exists()


@pytest.mark.parametrize("arguments", [
    ["--energy-min", "2", "--energy-max", "2", "--energy-points", "2"],
    ["--q-min", "1.3", "--q-max", "1.3", "--q-points", "2"],
])
def test_cli_rejects_duplicate_grid_coordinates(interface_config, tmp_path, arguments):
    config_path = tmp_path / "stack.yaml"
    output_path = tmp_path / "duplicate.npz"
    OmegaConf.save(interface_config, config_path)
    with pytest.raises(ValueError, match="strictly ascending order"):
        diagnostic_main(["--config", str(config_path), "--output", str(output_path),
                         *arguments])
    assert not output_path.exists()
