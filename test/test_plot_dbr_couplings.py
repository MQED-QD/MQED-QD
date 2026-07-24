from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf
from PIL import Image

from mqed.analysis.plot_dbr_couplings import (
    compute_dbr_couplings,
    run_from_config,
    select_energy_index,
    select_rx_indices,
)
from mqed.utils.SI_unit import D2CMM, c, eps0, eV_to_J, hbar


def _write_separation_gf(path: Path, with_structure: bool = False) -> None:
    energy_eV = np.array([1.0, 1.5, 2.0])
    Rx_nm = np.array([0.0, 5.0, 10.0])
    G_vac = np.zeros((3, 3, 3, 3), dtype=complex)
    G_total = np.zeros_like(G_vac)
    G_structure = np.zeros_like(G_vac)
    for energy_index in range(3):
        for rx_index in range(3):
            G_vac[energy_index, rx_index, 0, 0] = 0.25 * (rx_index + 1)
            G_total[energy_index, rx_index, 0, 0] = (
                (energy_index + 1) * (rx_index + 1)
                + 1j * (energy_index + 2) * (rx_index + 1)
            )
            G_structure[energy_index, rx_index, 0, 0] = (
                -0.5 * (rx_index + 1) + 0.75j * (rx_index + 1)
            )
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "separation"
        h5.create_dataset("green_function_total", data=G_total)
        h5.create_dataset("green_function_vacuum", data=G_vac)
        if with_structure:
            h5.create_dataset("green_function_structure", data=G_structure)
        h5.create_dataset("energy_eV", data=energy_eV)
        h5.create_dataset("Rx_nm", data=Rx_nm)
        position_fixed = h5.create_group("position_fixed")
        position_fixed.attrs["zD_meters"] = 0.0
        position_fixed.attrs["zA_meters"] = 0.0


def _base_cfg(input_path: Path) -> OmegaConf:
    return OmegaConf.create({
        "input_file": str(input_path),
        "output_prefix": "coupling_test",
        "green_component": "structure",
        "energy_selection": {"index": 1, "nearest": True, "tolerance_eV": 1e-12},
        "mu_D_debye": 2.0,
        "mu_A_debye": 3.0,
        "orientations": {
            "donor": {"theta_deg": 90.0, "phi_deg": 0.0},
            "acceptor": {"theta_deg": 90.0, "phi_deg": 0.0},
        },
        "plot": {
            "absolute": True,
            "filename": "couplings.png",
            "dpi": 80,
            "figsize": [4, 4],
            "grid": True,
        },
    })


def test_compute_dbr_couplings_matches_ddi_formula_sign_factor_two_and_gamma_conversion():
    energy_eV = 1.7
    G_slice = np.zeros((2, 3, 3), dtype=complex)
    G_slice[:, 0, 2] = np.array([4.0 + 5.0j, -2.0 - 3.0j])
    p_acceptor = np.array([1.0, 0.0, 0.0])
    p_donor = np.array([0.0, 0.0, 1.0])
    mu_D_debye = 2.0
    mu_A_debye = 3.0

    result = compute_dbr_couplings(
        G_slice,
        np.array([0.0, 4.0]),
        energy_eV,
        p_donor,
        p_acceptor,
        mu_D_debye,
        mu_A_debye,
    )

    omega = energy_eV * eV_to_J / hbar
    prefactor = (omega ** 2 / (eps0 * c ** 2)) * mu_D_debye * D2CMM * mu_A_debye * D2CMM
    projected = np.array([4.0 + 5.0j, -2.0 - 3.0j])
    expected_v = -(prefactor * np.real(projected)) / eV_to_J
    expected_hbar_gamma = +(2.0 * prefactor * np.imag(projected)) / eV_to_J

    assert np.allclose(result["projected_G"], projected)
    assert np.allclose(result["V_eV"], expected_v)
    assert np.allclose(result["hbarGamma_eV"], expected_hbar_gamma)
    assert np.allclose(result["Gamma_s_inv"], expected_hbar_gamma * eV_to_J / hbar)
    assert result["V_eV"][0] < 0.0
    assert result["hbarGamma_eV"][1] < 0.0
    assert np.allclose(result["abs_V_eV"], np.abs(expected_v))


def test_select_energy_index_by_index_and_value_with_provenance():
    energy_eV = np.array([1.0, 1.5, 2.0])

    by_index = select_energy_index(energy_eV, {"index": 2})
    by_value = select_energy_index(energy_eV, {"value_eV": 1.6, "nearest": True})

    assert by_index["selected_energy_index"] == 2
    assert by_index["selected_energy_eV"] == 2.0
    assert by_index["energy_selection_mode"] == "index"
    assert by_value["selected_energy_index"] == 1
    assert by_value["selected_energy_eV"] == 1.5
    assert by_value["requested_energy_eV"] == 1.6
    assert by_value["energy_selection_mode"] == "value_nearest"
    assert np.isclose(by_value["energy_selection_delta_eV"], 0.1)


def test_select_energy_index_rejects_out_of_bounds_and_nonexact_value():
    with pytest.raises(IndexError, match="out of bounds"):
        select_energy_index(np.array([1.0, 2.0]), {"index": 2})
    with pytest.raises(ValueError, match="not on the grid"):
        select_energy_index(
            np.array([1.0, 2.0]),
            {"value_eV": 1.25, "nearest": False, "tolerance_eV": 1e-9},
        )

    with pytest.raises(ValueError, match="finite"):
        select_energy_index(np.array([1.0, np.nan]), {"index": 0})


def test_select_rx_indices_preserves_disjoint_value_order_and_duplicates():
    selection = select_rx_indices(
        np.array([1.0, 2.0, 50.0, 500.0, 700.0, 1000.0]),
        {
            "values_nm": [50.0, 500.0, 1.0, 1000.0, 50.0],
            "nearest": False,
            "tolerance_nm": 1e-9,
        },
    )

    assert selection["rx_selection_mode"] == "values_exact"
    assert np.array_equal(selection["selected_Rx_indices"], [2, 3, 0, 5, 2])
    assert np.array_equal(selection["selected_Rx_nm"], [50.0, 500.0, 1.0, 1000.0, 50.0])
    assert np.array_equal(selection["requested_Rx_nm"], [50.0, 500.0, 1.0, 1000.0, 50.0])
    assert np.allclose(selection["rx_selection_delta_nm"], 0.0)


def test_select_rx_indices_supports_indices_and_nearest_values():
    Rx_nm = np.array([1.0, 10.0, 50.0, 500.0, 1000.0])

    by_indices = select_rx_indices(Rx_nm, {"indices": [4, 0, 3]})
    by_nearest = select_rx_indices(
        Rx_nm,
        {"values_nm": [48.0, 510.0], "nearest": True},
    )

    assert by_indices["rx_selection_mode"] == "indices"
    assert np.array_equal(by_indices["selected_Rx_indices"], [4, 0, 3])
    assert np.array_equal(by_indices["selected_Rx_nm"], [1000.0, 1.0, 500.0])
    assert by_nearest["rx_selection_mode"] == "values_nearest"
    assert np.array_equal(by_nearest["selected_Rx_indices"], [2, 3])
    assert np.array_equal(by_nearest["selected_Rx_nm"], [50.0, 500.0])
    assert np.allclose(by_nearest["rx_selection_delta_nm"], [2.0, 10.0])


def test_select_rx_indices_rejects_invalid_requests():
    Rx_nm = np.array([1.0, 10.0, 50.0])

    with pytest.raises(ValueError, match="either indices or values_nm"):
        select_rx_indices(Rx_nm, {"indices": [0], "values_nm": [1.0]})
    with pytest.raises(IndexError, match="out of bounds"):
        select_rx_indices(Rx_nm, {"indices": [-1]})
    with pytest.raises(ValueError, match="finite values"):
        select_rx_indices(Rx_nm, {"values_nm": [np.nan]})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        select_rx_indices(Rx_nm, {"values_nm": [1.0], "tolerance_nm": -1.0})
    with pytest.raises(ValueError, match=r"nearest is Rx_nm\[1\]=10 nm"):
        select_rx_indices(
            Rx_nm,
            {"values_nm": [11.0], "nearest": False, "tolerance_nm": 1e-9},
        )


def test_run_from_config_writes_hdf5_csv_png_and_provenance(tmp_path):
    gf_path = tmp_path / "gf.h5"
    _write_separation_gf(gf_path, with_structure=False)
    cfg = _base_cfg(gf_path)
    cfg.energy_selection.index = None
    cfg.energy_selection.value_eV = 1.49
    cfg.energy_selection.nearest = True

    h5_path = run_from_config(cfg, tmp_path, tmp_path)
    csv_path = h5_path.with_suffix(".csv")
    png_path = tmp_path / "couplings.png"

    assert h5_path.exists()
    assert csv_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0
    csv = np.genfromtxt(csv_path, delimiter=",", names=True)
    csv_text = csv_path.read_text(encoding="utf-8")
    with Image.open(png_path) as png:
        assert png.size[0] > 0
        assert png.size[1] > 0
        assert png.info["Title"] == "MQED-QD DBR physical couplings"
        assert png.info["Software"] == "MQED-QD"
        assert "selected_energy_eV=1.5" in png.info["Description"]
        assert "green_component=structure" in png.info["Description"]
        assert f"input_path={gf_path}" in png.info["Description"]
    assert "# selected_energy_eV: 1.5" in csv_text
    assert "# green_component: structure" in csv_text
    assert "# input_path:" in csv_text
    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["input_path"] == str(gf_path)
        assert h5.attrs["green_component"] == "structure"
        assert h5.attrs["gf_layout"] == "separation"
        assert h5.attrs["selected_energy_index"] == 1
        assert np.isclose(h5.attrs["requested_energy_eV"], 1.49)
        assert h5.attrs["energy_selection_mode"] == "value_nearest"
        assert h5["p_donor"].shape == (3,)
        assert h5["p_acceptor"].shape == (3,)
        assert h5["Rx_nm"].shape == (3,)
        assert h5["projected_G_real"].shape == (3,)
        assert np.allclose(h5["abs_V_eV"][:], np.abs(h5["V_eV"][:]))
        assert np.allclose(h5["abs_hbarGamma_eV"][:], np.abs(h5["hbarGamma_eV"][:]))
        assert "total-vacuum" not in h5.attrs["formula_V_eV"]
        assert np.allclose(csv["V_eV"], h5["V_eV"][:])
        assert np.allclose(csv["hbarGamma_eV"], h5["hbarGamma_eV"][:])


def test_run_from_config_selects_disjoint_rx_values_and_records_provenance(tmp_path):
    gf_path = tmp_path / "gf.h5"
    _write_separation_gf(gf_path, with_structure=True)
    cfg = _base_cfg(gf_path)
    cfg.rx_selection = {
        "indices": None,
        "values_nm": [10.0, 0.0, 5.0],
        "nearest": False,
        "tolerance_nm": 1e-9,
    }

    h5_path = run_from_config(cfg, tmp_path, tmp_path)
    csv_path = h5_path.with_suffix(".csv")

    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["rx_selection_mode"] == "values_exact"
        assert h5.attrs["rx_selection_nearest"] == 0
        assert np.array_equal(h5["selected_Rx_indices"][:], [2, 0, 1])
        assert np.array_equal(h5["requested_Rx_nm"][:], [10.0, 0.0, 5.0])
        assert np.array_equal(h5["Rx_nm"][:], [10.0, 0.0, 5.0])
        assert np.allclose(h5["projected_G_real"][:], [-1.5, -0.5, -1.0])
        assert np.allclose(h5["rx_selection_delta_nm"][:], 0.0)

    csv = np.genfromtxt(csv_path, delimiter=",", names=True)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert np.array_equal(csv["Rx_nm"], [10.0, 0.0, 5.0])
    assert "# selected_Rx_indices: 2 0 1" in csv_text
    assert "# requested_Rx_nm: 10 0 5" in csv_text


def test_run_from_config_uses_stored_structure_component_when_present(tmp_path):
    gf_path = tmp_path / "gf_structure.h5"
    _write_separation_gf(gf_path, with_structure=True)

    h5_path = run_from_config(_base_cfg(gf_path), tmp_path, tmp_path)

    with h5py.File(h5_path, "r") as h5:
        assert np.allclose(h5["projected_G_real"][:], [-0.5, -1.0, -1.5])
        assert np.allclose(h5["projected_G_imag"][:], [0.75, 1.5, 2.25])


def test_run_from_config_creates_nested_output_directories(tmp_path):
    gf_path = tmp_path / "gf.h5"
    _write_separation_gf(gf_path)
    cfg = _base_cfg(gf_path)
    cfg.output_prefix = "data/nested/coupling_test"
    cfg.plot.filename = "plots/nested/couplings.png"

    h5_path = run_from_config(cfg, tmp_path, tmp_path)

    assert h5_path == tmp_path / "data/nested/coupling_test_E_1.5eV.h5"
    assert h5_path.exists()
    assert h5_path.with_suffix(".csv").exists()
    assert (tmp_path / "plots/nested/couplings.png").exists()


def test_run_from_config_rejects_pair_layout(tmp_path):
    gf_path = tmp_path / "gf_pair.h5"
    with h5py.File(gf_path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"
        h5.create_dataset("green_function_total", data=np.zeros((1, 1, 1, 3, 3), dtype=complex))
        h5.create_dataset("green_function_vacuum", data=np.zeros((1, 1, 1, 3, 3), dtype=complex))
        h5.create_dataset("energy_eV", data=np.array([1.0]))
        h5.create_dataset("emitter_positions_nm", data=np.zeros((1, 3)))
        position_fixed = h5.create_group("position_fixed")
        position_fixed.attrs["zD_meters"] = 0.0
        position_fixed.attrs["zA_meters"] = 0.0

    with pytest.raises(ValueError, match="supports only gf_layout='separation'"):
        run_from_config(_base_cfg(gf_path), tmp_path, tmp_path)


def test_compute_dbr_couplings_rejects_nonfinite_selected_green_data():
    G_slice = np.zeros((1, 3, 3), dtype=complex)
    G_slice[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        compute_dbr_couplings(
            G_slice,
            np.array([0.0]),
            1.0,
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Rx_nm": np.array([np.inf])}, "Rx_nm"),
        ({"energy_eV": np.nan}, "energy_eV"),
        ({"p_donor": np.array([np.nan, 0.0, 0.0])}, "orientation"),
        ({"mu_D_debye": np.inf}, "magnitudes"),
    ],
)
def test_compute_dbr_couplings_rejects_nonfinite_physical_inputs(overrides, message):
    inputs = {
        "G_slice": np.zeros((1, 3, 3), dtype=complex),
        "Rx_nm": np.array([0.0]),
        "energy_eV": 1.0,
        "p_donor": np.array([1.0, 0.0, 0.0]),
        "p_acceptor": np.array([1.0, 0.0, 0.0]),
        "mu_D_debye": 1.0,
        "mu_A_debye": 1.0,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        compute_dbr_couplings(**inputs)


def test_plot_dbr_couplings_yaml_parses_and_entry_point_exists():
    with open("configs/analysis/plot_dbr_couplings.yaml", "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    with open("pyproject.toml", "r", encoding="utf-8") as handle:
        pyproject_text = handle.read()

    assert loaded["green_component"] == "total"
    assert loaded["rx_selection"]["values_nm"] is None
    assert loaded["rx_selection"]["nearest"] is False
    assert loaded["plot"]["absolute"] is True
    assert "mqed_plot_dbr_couplings" in pyproject_text
