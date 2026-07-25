from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from mqed.analysis.emission_spectrum import (
    _read_green_component,
    compute_emission_spectrum,
    project_pair_green,
    project_separation_green_to_pair,
    resolve_emitter_orientations,
    run_from_config,
    self_energy_from_projected_green,
)
from mqed.plotting.plot_emission_spectrum import (
    _plot_curves,
    _plot_map,
)


def _write_pair_gf(path: Path, emitter_orientations=None) -> None:
    energy_eV = np.array([1.0, 1.1, 1.2])
    G_total = np.zeros((3, 2, 2, 3, 3), dtype=complex)
    G_vac = np.zeros_like(G_total)
    G_structure = np.zeros_like(G_total)
    for m in range(3):
        G_structure[m, 0, 0, 2, 2] = 1.0e7j * (m + 1)
        G_structure[m, 1, 1, 2, 2] = 1.1e7j * (m + 1)
        G_structure[m, 0, 1, 2, 2] = 2.0e6j * (m + 1)
        G_structure[m, 1, 0, 2, 2] = 2.0e6j * (m + 1)
    G_total[:] = G_vac + G_structure
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset("green_function_total", data=G_total)
        h5.create_dataset("green_function_vacuum", data=G_vac)
        h5.create_dataset("green_function_structure", data=G_structure)
        h5.create_dataset("energy_eV", data=energy_eV)
        h5.create_dataset("emitter_positions_nm", data=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        if emitter_orientations is not None:
            h5.create_dataset("emitter_orientations", data=np.asarray(emitter_orientations, dtype=float))
        position_fixed = h5.create_group("position_fixed")
        position_fixed.attrs["zD_meters"] = 0.0
        position_fixed.attrs["zA_meters"] = 0.0


def _write_separation_gf(path: Path, include_structure: bool = True, include_channels: bool = True) -> None:
    energy_eV = np.array([1.0, 1.1, 1.2])
    rx_nm = np.array([0.0, 2.0, 4.0])
    shape = (energy_eV.size, rx_nm.size, 3, 3)
    vacuum = np.zeros(shape, dtype=complex)
    structure = np.zeros(shape, dtype=complex)
    scattering_te = np.zeros(shape, dtype=complex)
    scattering_tm = np.zeros(shape, dtype=complex)
    for m in range(energy_eV.size):
        for k in range(rx_nm.size):
            structure[m, k, 2, 2] = (10 + m + k) + 1j * (1 + k)
            vacuum[m, k, 2, 2] = (100 + k) + 1j * (50 + k)
            scattering_te[m, k, 2, 2] = 1 + k + 0.1j
            scattering_tm[m, k, 2, 2] = 2 + k + 0.2j
    total = structure + vacuum
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "separation"
        h5.create_dataset("green_function_total", data=total)
        h5.create_dataset("green_function_vacuum", data=vacuum)
        if include_structure:
            h5.create_dataset("green_function_structure", data=structure)
        if include_channels:
            h5.create_dataset("green_function_scattering_te", data=scattering_te)
            h5.create_dataset("green_function_scattering_tm", data=scattering_tm)
        h5.create_dataset("energy_eV", data=energy_eV)
        h5.create_dataset("Rx_nm", data=rx_nm)
        position_fixed = h5.create_group("position_fixed")
        position_fixed.attrs["zD_meters"] = 0.0
        position_fixed.attrs["zA_meters"] = 0.0


def test_resolve_emitter_orientations_accepts_explicit_vectors():
    cfg = OmegaConf.create({"orientations": {"emitter_orientations": [[0, 0, 2], [0, 3, 0]]}})

    orientations = resolve_emitter_orientations(cfg, 2)

    assert np.allclose(orientations, [[0, 0, 1], [0, 1, 0]])


def test_resolve_emitter_orientations_uses_stored_when_config_omits_orientation():
    stored = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])

    orientations = resolve_emitter_orientations({}, 2, stored_orientations=stored)

    assert np.allclose(orientations, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_resolve_emitter_orientations_explicit_config_overrides_stored():
    stored = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cfg = {"emitter_orientations": [[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]]}

    orientations = resolve_emitter_orientations(cfg, 2, stored_orientations=stored)

    assert np.allclose(orientations, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def test_resolve_emitter_orientations_angle_config_overrides_stored():
    stored = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    orientations = resolve_emitter_orientations({"orientations": {"theta_deg": 0.0}}, 2, stored_orientations=stored)

    assert np.allclose(orientations, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def test_resolve_emitter_orientations_rejects_nonfinite_stored_vectors():
    with pytest.raises(ValueError, match="orientations must be finite"):
        resolve_emitter_orientations(
            {},
            2,
            stored_orientations=np.array([[0.0, 1.0, 0.0], [np.nan, 0.0, 0.0]]),
        )


def test_resolve_emitter_orientations_normalizes_large_finite_vectors():
    orientations = resolve_emitter_orientations(
        {"emitter_orientations": [[1.0e308, 1.0e308, 0.0]]},
        1,
    )

    assert np.all(np.isfinite(orientations))
    assert np.allclose(orientations, [[np.sqrt(0.5), np.sqrt(0.5), 0.0]])


def test_project_pair_green_uses_left_observer_and_right_source_orientations():
    G_pair = np.zeros((1, 2, 2, 3, 3), dtype=complex)
    G_pair[0, 0, 1, 0, 2] = 5.0 + 1.0j
    orientations = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    projected = project_pair_green(G_pair, orientations)

    assert projected.shape == (1, 2, 2)
    assert projected[0, 0, 1] == 5.0 + 1.0j


def test_varguet_effective_component_uses_scattered_diagonal_and_real_vacuum_off_diagonal(
    tmp_path,
):
    path = tmp_path / "varguet_component.h5"
    structure = np.full((1, 2, 2, 3, 3), 2.0 + 3.0j, dtype=complex)
    vacuum = np.full_like(structure, 5.0 + 7.0j)
    total = structure + vacuum
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset("green_function_total", data=total)
        h5.create_dataset("green_function_vacuum", data=vacuum)
        h5.create_dataset("green_function_structure", data=structure)
        h5.create_dataset("energy_eV", data=[2.85])
        h5.create_dataset("emitter_positions_nm", data=np.zeros((2, 3)))

    green_data = _read_green_component(path, "varguet_effective")

    assert np.all(green_data["G"][:, 0, 0] == structure[:, 0, 0])
    assert np.all(green_data["G"][:, 1, 1] == structure[:, 1, 1])
    assert np.all(green_data["G"][:, 0, 1] == structure[:, 0, 1] + 5.0)
    assert np.all(green_data["G"][:, 1, 0] == structure[:, 1, 0] + 5.0)


def test_varguet_effective_component_falls_back_to_total_minus_vacuum(tmp_path):
    path = tmp_path / "varguet_component_fallback.h5"
    structure = np.full((1, 2, 2, 3, 3), 2.0 + 3.0j, dtype=complex)
    vacuum = np.full_like(structure, 5.0 + 7.0j)
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset("green_function_total", data=structure + vacuum)
        h5.create_dataset("green_function_vacuum", data=vacuum)
        h5.create_dataset("energy_eV", data=[2.85])
        h5.create_dataset("emitter_positions_nm", data=np.zeros((2, 3)))

    green_data = _read_green_component(path, "varguet_effective")

    assert np.all(green_data["G"][:, 0, 0] == structure[:, 0, 0])
    assert np.all(green_data["G"][:, 0, 1] == structure[:, 0, 1] + 5.0)


def test_varguet_effective_fallback_rejects_broadcastable_total_shape(tmp_path):
    path = tmp_path / "varguet_component_broadcast_fallback.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset(
            "green_function_total",
            data=np.zeros((1, 1, 1, 3, 3), dtype=complex),
        )
        h5.create_dataset(
            "green_function_vacuum",
            data=np.zeros((1, 2, 2, 3, 3), dtype=complex),
        )
        h5.create_dataset("energy_eV", data=[2.85])
        h5.create_dataset("emitter_positions_nm", data=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="matching shapes"):
        _read_green_component(path, "varguet_effective")


@pytest.mark.parametrize(
    "dataset_name,dataset_shape,error_match",
    [
        ("green_function_vacuum", (1, 2, 1, 3, 3), "matching shapes"),
        ("green_function_structure", (1, 2, 2, 2, 2), r"shape \(M,N,N,3,3\)"),
    ],
)
def test_varguet_effective_component_rejects_malformed_pair_tensors(
    tmp_path,
    dataset_name,
    dataset_shape,
    error_match,
):
    path = tmp_path / f"malformed_{dataset_name}.h5"
    shapes = {
        "green_function_total": (1, 2, 2, 3, 3),
        "green_function_vacuum": (1, 2, 2, 3, 3),
        "green_function_structure": (1, 2, 2, 3, 3),
    }
    shapes[dataset_name] = dataset_shape
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        for name, shape in shapes.items():
            h5.create_dataset(name, data=np.zeros(shape, dtype=complex))
        h5.create_dataset("energy_eV", data=[2.85])
        h5.create_dataset("emitter_positions_nm", data=np.zeros((2, 3)))

    with pytest.raises(ValueError, match=error_match):
        _read_green_component(path, "varguet_effective")


def test_varguet_effective_component_rejects_inconsistent_pair_metadata(tmp_path):
    path = tmp_path / "malformed_metadata.h5"
    tensor = np.zeros((2, 2, 2, 3, 3), dtype=complex)
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset("green_function_total", data=tensor)
        h5.create_dataset("green_function_vacuum", data=tensor)
        h5.create_dataset("green_function_structure", data=tensor)
        h5.create_dataset("energy_eV", data=[2.85])
        h5.create_dataset("emitter_positions_nm", data=np.zeros((3, 3)))

    with pytest.raises(ValueError, match="energy_eV length"):
        _read_green_component(path, "varguet_effective")


def test_project_separation_green_to_pair_maps_chain_separations():
    G_sep = np.zeros((1, 3, 3, 3), dtype=complex)
    G_sep[0, 0, 2, 2] = 1.0
    G_sep[0, 1, 2, 2] = 2.0
    G_sep[0, 2, 2, 2] = 3.0
    orientations = np.tile([0.0, 0.0, 1.0], (3, 1))

    projected = project_separation_green_to_pair(G_sep, np.array([0.0, 2.0, 4.0]), 3, 2.0, orientations)

    assert np.allclose(projected[0], [[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]])


def test_separation_varguet_effective_uses_structure_self_and_real_vacuum_off_diagonal(tmp_path):
    path = tmp_path / "sep_varguet.h5"
    _write_separation_gf(path)

    green_data = _read_green_component(path, "varguet_effective")

    with h5py.File(path, "r") as h5:
        structure = h5["green_function_structure"][:]
        vacuum = h5["green_function_vacuum"][:]
    assert np.allclose(green_data["G"][:, 0], structure[:, 0])
    assert np.allclose(green_data["G"][:, 1], structure[:, 1] + np.real(vacuum[:, 1]))
    assert green_data["green_convention"] == "varguet_effective"


def test_separation_varguet_effective_falls_back_to_total_minus_vacuum(tmp_path):
    path = tmp_path / "sep_varguet_fallback.h5"
    _write_separation_gf(path, include_structure=False)

    green_data = _read_green_component(path, "varguet_effective")

    with h5py.File(path, "r") as h5:
        structure = h5["green_function_total"][:] - h5["green_function_vacuum"][:]
    assert np.allclose(green_data["G"][:, 0], structure[:, 0])


def test_separation_renormalized_total_uses_structure_self_and_complex_total_off_diagonal(tmp_path):
    path = tmp_path / "sep_renormalized.h5"
    _write_separation_gf(path)

    green_data = _read_green_component(path, "renormalized_total")

    with h5py.File(path, "r") as h5:
        structure = h5["green_function_structure"][:]
        total = h5["green_function_total"][:]
    assert np.allclose(green_data["G"][:, 0], structure[:, 0])
    assert np.allclose(green_data["G"][:, 1], total[:, 1])


@pytest.mark.parametrize("channel,dataset", [("te", "green_function_scattering_te"), ("tm", "green_function_scattering_tm")])
def test_te_tm_channel_selection_uses_stored_scattering_only(tmp_path, channel, dataset):
    path = tmp_path / f"sep_{channel}.h5"
    _write_separation_gf(path)

    green_data = _read_green_component(path, "varguet_effective", channel)

    with h5py.File(path, "r") as h5:
        expected = h5[dataset][:]
    assert np.allclose(green_data["G"], expected)
    assert green_data["green_channel"] == channel
    assert green_data["green_convention"] == f"{channel}_scattering_only"


def test_te_tm_channel_selection_requires_dataset(tmp_path):
    path = tmp_path / "sep_missing_channel.h5"
    _write_separation_gf(path, include_channels=False)

    with pytest.raises(KeyError, match="green_function_scattering_te"):
        _read_green_component(path, "structure", "te")


def test_te_tm_channel_selection_validates_separation_metadata(tmp_path):
    path = tmp_path / "sep_malformed_channel.h5"
    _write_separation_gf(path)
    with h5py.File(path, "a") as h5:
        del h5["green_function_scattering_te"]
        h5.create_dataset(
            "green_function_scattering_te",
            data=np.zeros((2, 3, 3, 3), dtype=complex),
        )

    with pytest.raises(ValueError, match="matching shapes"):
        _read_green_component(path, "structure", "te")


def test_emission_spectrum_peaks_at_transition_without_self_energy():
    energy_eV = np.array([1.0, 1.1, 1.2])
    self_energy = np.zeros((3, 2, 2), dtype=complex)

    spectrum = compute_emission_spectrum(self_energy, energy_eV, np.array([1.1]), gamma0_eV=0.05)

    assert spectrum.shape == (1, 3)
    assert int(np.argmax(spectrum[0])) == 1
    assert np.all(spectrum >= 0.0)


def test_self_energy_from_projected_green_returns_energy_units():
    projected_G = np.zeros((2, 1, 1), dtype=complex)
    projected_G[:, 0, 0] = [1.0e7j, 2.0e7j]

    self_energy = self_energy_from_projected_green(
        projected_G,
        np.array([1.0, 2.0]),
        mu_debye=3.8,
        shift_method="real_green",
    )

    assert self_energy.shape == (2, 1, 1)
    assert np.all(np.imag(self_energy[:, 0, 0]) > 0.0)


def test_run_from_config_writes_emission_hdf5(tmp_path):
    gf_path = tmp_path / "gf_pair.h5"
    _write_pair_gf(gf_path)
    cfg = OmegaConf.create({
        "input_file": str(gf_path),
        "output_prefix": "emission_test",
        "green_component": "structure",
        "shift_method": "real_green",
        "mu_debye": 3.8,
        "gamma0_eV": 0.05,
        "transition_energy_eV": [1.1],
        "normalize": True,
        "orientations": {"theta_deg": 0.0, "phi_deg": 0.0},
    })

    output_path = run_from_config(cfg, tmp_path, tmp_path)

    with h5py.File(output_path, "r") as h5:
        assert h5["emission_spectrum"].shape == (1, 3)
        assert h5["projected_G"].shape == (3, 2, 2)
        assert h5["self_energy_eV"].shape == (3, 2, 2)
        assert h5["emitter_orientations"].shape == (2, 3)
        assert h5.attrs["gf_layout"] == "pair"
        assert h5.attrs["green_component"] == "structure"
        assert np.isclose(np.max(h5["emission_spectrum"][:]), 1.0)


def test_run_from_config_uses_stored_hdf5_orientations_when_config_omits_them(tmp_path):
    gf_path = tmp_path / "gf_pair_stored_orientations.h5"
    _write_pair_gf(gf_path, emitter_orientations=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    cfg = OmegaConf.create({
        "input_file": str(gf_path),
        "output_prefix": "emission_stored",
        "green_component": "structure",
        "transition_energy_eV": [1.1],
    })

    output_path = run_from_config(cfg, tmp_path, tmp_path)

    with h5py.File(output_path, "r") as h5:
        assert np.allclose(h5["emitter_orientations"][:], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        assert np.linalg.norm(h5["projected_G"][:]) > 0.0


def test_run_from_config_explicit_orientations_override_stored_hdf5(tmp_path):
    gf_path = tmp_path / "gf_pair_stored_overridden.h5"
    _write_pair_gf(gf_path, emitter_orientations=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cfg = OmegaConf.create({
        "input_file": str(gf_path),
        "output_prefix": "emission_override",
        "green_component": "structure",
        "transition_energy_eV": [1.1],
        "emitter_orientations": [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
    })

    output_path = run_from_config(cfg, tmp_path, tmp_path)

    with h5py.File(output_path, "r") as h5:
        assert np.allclose(h5["emitter_orientations"][:], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        assert np.linalg.norm(h5["projected_G"][:]) > 0.0


def test_run_from_config_requires_explicit_chain_parameters_for_separation(tmp_path):
    gf_path = tmp_path / "gf_sep.h5"
    _write_separation_gf(gf_path)
    cfg = OmegaConf.create({
        "input_file": str(gf_path),
        "green_component": "varguet_effective",
        "transition_energy_eV": [1.1],
    })

    with pytest.raises(ValueError, match="requires explicit n_emitters.*d_nm"):
        run_from_config(cfg, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "overrides,error_match",
    [
        ({"n_emitters": 0, "d_nm": 2.0}, "n_emitters must be a positive integer"),
        ({"n_emitters": 2.5, "d_nm": 2.0}, "n_emitters must be a positive integer"),
        ({"n_emitters": np.inf, "d_nm": 2.0}, "n_emitters must be a positive integer"),
        ({"n_emitters": 2, "d_nm": 0.0}, "d_nm must be finite and positive"),
        (
            {"n_emitters": 2, "d_nm": 2.0, "rx_tolerance_nm": -1.0},
            "rx_tolerance_nm must be finite and non-negative",
        ),
    ],
)
def test_run_from_config_rejects_invalid_chain_parameters(tmp_path, overrides, error_match):
    gf_path = tmp_path / "gf_sep_invalid_chain.h5"
    _write_separation_gf(gf_path)
    cfg = {
        "input_file": str(gf_path),
        "green_component": "varguet_effective",
        "transition_energy_eV": [1.1],
        **overrides,
    }

    with pytest.raises(ValueError, match=error_match):
        run_from_config(cfg, tmp_path, tmp_path)


def test_run_from_config_tiny_separation_chain_writes_provenance_and_finite_spectrum(tmp_path):
    gf_path = tmp_path / "gf_sep_chain.h5"
    _write_separation_gf(gf_path)
    cfg = OmegaConf.create({
        "input_file": str(gf_path),
        "output_prefix": "sep_chain",
        "green_component": "varguet_effective",
        "green_channel": "full",
        "n_emitters": 3,
        "d_nm": 2.0,
        "transition_energy_eV": [1.1],
        "mu_debye": 3.8,
        "gamma0_eV": 0.05,
        "orientations": {"theta_deg": 0.0, "phi_deg": 0.0},
    })

    output_path = run_from_config(cfg, tmp_path, tmp_path)

    with h5py.File(output_path, "r") as h5:
        assert h5["projected_G"].shape == (3, 3, 3)
        assert h5["emission_spectrum"].shape == (1, 3)
        assert np.all(np.isfinite(h5["emission_spectrum"][:]))
        assert h5.attrs["green_component"] == "varguet_effective"
        assert h5.attrs["green_channel"] == "full"
        assert h5.attrs["green_convention"] == "varguet_effective"


def test_run_from_config_honors_exact_output_filename(tmp_path):
    gf_path = tmp_path / "gf_pair_exact_output.h5"
    _write_pair_gf(gf_path)
    cfg = {
        "input_file": str(gf_path),
        "output_filename": "exact_emission.h5",
        "green_component": "structure",
        "transition_energy_eV": [1.1],
        "orientations": {"theta_deg": 0.0, "phi_deg": 0.0},
    }

    output_path = run_from_config(cfg, tmp_path, tmp_path)

    assert output_path == tmp_path / "exact_emission.h5"
    assert output_path.exists()


def test_plot_emission_map_and_curves_create_figures():
    spectrum = np.array([[1.0, 2.0, 1.5], [0.5, 1.0, 0.7]])
    emission_energy = np.array([1.0, 1.1, 1.2])
    transition_energy = np.array([1.05, 1.15])
    map_cfg = OmegaConf.create({"plot_settings": {"figsize": [4, 3], "title": "Map"}})
    curves_cfg = OmegaConf.create({
        "plot_settings": {
            "figsize": [4, 3],
            "transition_values_eV": [1.15],
            "label_template": "E0 = {omega0:.2f} eV",
            "xlabel": "Emission energy (eV)",
        }
    })

    map_fig = _plot_map(spectrum, emission_energy, transition_energy, map_cfg)
    curves_fig = _plot_curves(spectrum, emission_energy, transition_energy, curves_cfg)

    map_ax = map_fig.axes[0]
    image = map_ax.images[0]
    assert map_ax.get_title() == "Map"
    assert map_ax.get_xlabel() == r"Transition energy $\omega_0$ (eV)"
    assert map_ax.get_ylabel() == r"Emission energy $\omega$ (eV)"
    assert image.get_array().shape == (emission_energy.size, transition_energy.size)
    assert np.allclose(image.get_array(), spectrum.T)
    assert np.allclose(
        image.get_extent(),
        [
            transition_energy[0],
            transition_energy[-1],
            emission_energy[0],
            emission_energy[-1],
        ],
    )

    curve_ax = curves_fig.axes[0]
    assert len(curve_ax.lines) == 1
    assert curve_ax.get_xlabel() == "Emission energy (eV)"
    assert curve_ax.lines[0].get_label() == "E0 = 1.15 eV"
    assert np.allclose(curve_ax.lines[0].get_xdata(), emission_energy)
    assert np.allclose(curve_ax.lines[0].get_ydata(), spectrum[1])


def test_plot_emission_spectrum_hydra_entry_saves_png(tmp_path):
    cfg = OmegaConf.create({
        "font": {},
        "plot_settings": {
            "plot_type": "curves",
            "transition_indices": [0],
            "filename": "emission.png",
            "save_plot": True,
            "dpi": 80,
        },
    })
    fig = _plot_curves(np.array([[1.0, 2.0, 1.0]]), np.array([1.0, 1.1, 1.2]), np.array([1.1]), cfg)
    out = tmp_path / "emission.png"
    fig.savefig(out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_emission_spectrum_configs_parse():
    for relpath in [
        "configs/analysis/emission_spectrum.yaml",
        "configs/analysis/emission_spectrum_example.yaml",
        "configs/plots/plt_emission_spectrum.yaml",
    ]:
        with open(relpath, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        assert loaded is not None


def test_emission_spectrum_configs_follow_stored_arbitrary_count_orientations():
    stored = np.column_stack([
        -np.sin(2.0 * np.pi * np.arange(15) / 15),
        np.cos(2.0 * np.pi * np.arange(15) / 15),
        np.zeros(15),
    ])

    for relpath in [
        "configs/analysis/emission_spectrum.yaml",
        "configs/analysis/emission_spectrum_example.yaml",
    ]:
        config = OmegaConf.load(relpath)
        assert "orientations" not in config
        assert str(config.input_file).endswith(
            "mie_silver_sphere_ring_Emin_2.60_Emax_3.20_121pts_emitters_15pts.hdf5"
        )
        assert np.allclose(resolve_emitter_orientations(config, 15, stored), stored)
