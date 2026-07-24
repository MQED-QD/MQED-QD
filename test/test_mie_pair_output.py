import h5py
import numpy as np
import pytest
import yaml
from pathlib import Path
from omegaconf import OmegaConf

from mqed.Dyadic_GF.GF_Mie import c, MieGreenFunction
from mqed.Dyadic_GF.main_mie import (
    MaterialResolver,
    observer_positions_m,
    output_path_from_config,
    resolve_emitter_geometry_m,
    run_from_config,
    spectral_grid,
)
from mqed.utils.dgf_data import load_gf_h5, save_gf_pair_h5
from mqed.utils.emitter_geometry import equatorial_ring_nearest_neighbor_chord_nm


def _projected_pair(G_pair, orientations, observer_index, source_index):
    return np.einsum(
        "a,ab,b->",
        orientations[observer_index],
        G_pair[observer_index, source_index],
        orientations[source_index],
    )


def test_mie_spectral_grid_accepts_energy_ev_without_wavelength_input():
    energy_eV, wavelength_m, wavelength_nm = spectral_grid(
        {"spectral_param": "eV", "energy_eV": {"min": 1.8, "max": 2.0, "points": 3}}
    )

    assert np.allclose(energy_eV, [1.8, 1.9, 2.0])
    assert np.all(wavelength_m > 0.0)
    assert np.allclose(wavelength_nm, wavelength_m * 1e9)


def test_mie_horizontal_rx_scan_uses_source_yz_by_default():
    observers = observer_positions_m(
        {
            "source_position_nm": [5.0, 3.0, 10.0],
            "position": {"Rx_nm": {"min": 0.0, "max": 120.0, "points": 3}},
        }
    )

    assert np.allclose(observers * 1e9, [[5.0, 3.0, 10.0], [65.0, 3.0, 10.0], [125.0, 3.0, 10.0]])


def test_mie_horizontal_rx_scan_accepts_explicit_observer_scan_key():
    observers = observer_positions_m(
        {
            "source_position_nm": [0.0, 0.0, 0.0],
            "observer_scan_nm": {"Rx_nm": [0.0, 2.0, 20.0], "y_nm": 1.0, "zA_nm": 4.0},
        }
    )

    assert np.allclose(observers * 1e9, [[0.0, 1.0, 4.0], [2.0, 1.0, 4.0], [20.0, 1.0, 4.0]])


def test_mie_pair_output_loads_with_shared_gf_loader(tmp_path):
    config = {
        "simulation": {
            "spectral_param": "wavelength_nm",
            "wavelength_nm": 600.0,
            "nmax": 1,
            "geometry": {"boundary": "sphere", "radius_nm": 1.0},
            "emitter_positions_nm": [[0.0, 0.0, 5.0], [0.0, 0.0, 8.0]],
            "strict_regions": True,
        },
        "materials": {"regions": [{"n": 1.0}, {"n": 1.0}]},
        "parallel": {"backend": "sequential"},
        "output": {"layout": "pair", "directory": str(tmp_path), "prefix": "mie_pair_test"},
    }
    config_path = tmp_path / "mie_pair.yaml"
    output_path = tmp_path / "mie_pair.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_from_config(config_path, output_path)
    data = load_gf_h5(str(output_path))

    assert data["gf_layout"] == "pair"
    assert data["G_total"].shape == (1, 2, 2, 3, 3)
    assert data["G_vac"].shape == (1, 2, 2, 3, 3)
    assert data["emitter_positions_nm"].shape == (2, 3)


def test_mie_pair_output_saves_generated_ring_orientations(tmp_path):
    config = {
        "simulation": {
            "spectral_param": "wavelength_nm",
            "wavelength_nm": 600.0,
            "nmax": 1,
            "geometry": {"boundary": "sphere", "radius_nm": 1.0},
            "emitter_ring": {
                "emitter_count": 2,
                "sphere_radius_nm": 1.0,
                "emitter_surface_gap_nm": 4.0,
                "orientation": "orthoradial",
            },
            "strict_regions": True,
        },
        "materials": {"regions": [{"n": 1.0}, {"n": 1.0}]},
        "parallel": {"backend": "sequential"},
        "output": {"layout": "pair", "directory": str(tmp_path), "prefix": "mie_ring_test"},
    }
    config_path = tmp_path / "mie_ring.yaml"
    output_path = tmp_path / "mie_ring.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_from_config(config_path, output_path)
    data = load_gf_h5(str(output_path))

    assert data["G_total"].shape == (1, 2, 2, 3, 3)
    assert np.allclose(data["emitter_positions_nm"], [[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    assert np.allclose(data["emitter_orientations"], [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], atol=1e-12)


def test_mie_pair_hdf5_orientations_are_optional_and_backward_compatible(tmp_path):
    h5_path = tmp_path / "pair_with_orientations.h5"
    old_h5_path = tmp_path / "pair_without_orientations.h5"
    G = np.zeros((1, 2, 2, 3, 3), dtype=complex)
    positions_nm = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    orientations = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    save_gf_pair_h5(str(h5_path), G, G, np.array([1.0]), positions_nm, 5e-9, 5e-9, emitter_orientations=orientations)
    save_gf_pair_h5(str(old_h5_path), G, G, np.array([1.0]), positions_nm, 5e-9, 5e-9)

    assert np.allclose(load_gf_h5(str(h5_path))["emitter_orientations"], orientations)
    assert "emitter_orientations" not in load_gf_h5(str(old_h5_path))


def test_mie_pair_hdf5_validates_and_normalizes_geometry_before_writing(tmp_path):
    h5_path = tmp_path / "invalid_pair.h5"
    G = np.zeros((1, 1, 1, 3, 3), dtype=complex)

    with pytest.raises(ValueError, match="positions_nm must be finite"):
        save_gf_pair_h5(
            str(h5_path), G, G, np.array([1.0]), np.array([[np.nan, 0.0, 0.0]]), 0.0, 0.0
        )
    assert not h5_path.exists()

    with pytest.raises(ValueError, match="orientations must be finite"):
        save_gf_pair_h5(
            str(h5_path),
            G,
            G,
            np.array([1.0]),
            np.array([[0.0, 0.0, 0.0]]),
            0.0,
            0.0,
            emitter_orientations=np.array([[np.inf, 0.0, 0.0]]),
        )
    assert not h5_path.exists()

    save_gf_pair_h5(
        str(h5_path),
        G,
        G,
        np.array([1.0]),
        np.array([[0.0, 0.0, 0.0]]),
        0.0,
        0.0,
        emitter_orientations=np.array([[0.0, 3.0, 0.0]]),
    )
    assert np.allclose(load_gf_h5(str(h5_path))["emitter_orientations"], [[0.0, 1.0, 0.0]])


def test_mie_pair_hdf5_rejects_inconsistent_pair_shapes_before_writing(tmp_path):
    h5_path = tmp_path / "inconsistent_pair.h5"
    positions_nm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid_G = np.zeros((1, 2, 2, 3, 3), dtype=complex)

    with pytest.raises(ValueError, match="Gtot must have shape"):
        save_gf_pair_h5(
            str(h5_path),
            np.zeros((1, 1, 1, 3, 3), dtype=complex),
            valid_G,
            np.array([1.0]),
            positions_nm,
            0.0,
            0.0,
        )
    assert not h5_path.exists()

    with pytest.raises(ValueError, match="Gstructure must have shape"):
        save_gf_pair_h5(
            str(h5_path),
            valid_G,
            valid_G,
            np.array([1.0]),
            positions_nm,
            0.0,
            0.0,
            Gstructure=np.zeros((1, 1, 1, 3, 3), dtype=complex),
        )
    assert not h5_path.exists()

    with pytest.raises(ValueError, match="wavelength_m must have shape"):
        save_gf_pair_h5(
            str(h5_path),
            valid_G,
            valid_G,
            np.array([1.0]),
            positions_nm,
            0.0,
            0.0,
            wavelength_m=np.array([1.0, 2.0]),
        )
    assert not h5_path.exists()

    with pytest.raises(ValueError, match="observer_region must have shape"):
        save_gf_pair_h5(
            str(h5_path),
            valid_G,
            valid_G,
            np.array([1.0]),
            positions_nm,
            0.0,
            0.0,
            observer_region=np.zeros((1, 2), dtype=int),
        )
    assert not h5_path.exists()


def test_mie_pair_hdf5_preserves_legacy_positional_attrs_argument(tmp_path):
    h5_path = tmp_path / "legacy_attrs.h5"
    G = np.zeros((1, 1, 1, 3, 3), dtype=complex)
    positions_nm = np.array([[0.0, 0.0, 0.0]])

    save_gf_pair_h5(
        str(h5_path),
        G,
        G,
        np.array([1.0]),
        positions_nm,
        0.0,
        0.0,
        None,
        None,
        None,
        {"legacy_attribute": "preserved"},
    )

    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["legacy_attribute"] == "preserved"


def test_mie_resolver_preserves_explicit_lists_and_normalizes_orientations():
    positions_m, orientations = resolve_emitter_geometry_m({
        "emitter_positions_nm": [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        "emitter_orientations": [[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
    })

    assert np.allclose(positions_m * 1e9, [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    assert np.allclose(orientations, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    _, large_orientations = resolve_emitter_geometry_m({
        "emitter_positions_nm": [[0.0, 0.0, 1.0]],
        "emitter_orientations": [[1.0e308, 1.0e308, 0.0]],
    })
    assert np.allclose(large_orientations, [[np.sqrt(0.5), np.sqrt(0.5), 0.0]])


def test_mie_resolver_rejects_empty_and_nonfinite_explicit_positions():
    with pytest.raises(ValueError, match="at least one"):
        resolve_emitter_geometry_m({"emitter_positions_nm": []})

    with pytest.raises(ValueError, match="positions must be finite"):
        resolve_emitter_geometry_m({"emitter_positions_nm": [[np.nan, 0.0, 0.0]]})


def test_mie_resolver_rejects_ring_with_explicit_positions_orientations():
    simulation = {
        "emitter_ring": {"emitter_count": 2, "emitter_radius_nm": 10.0},
        "emitter_positions_nm": [[0.0, 0.0, 0.0]],
    }

    with pytest.raises(ValueError, match="either simulation.emitter_ring"):
        resolve_emitter_geometry_m(simulation)

    with pytest.raises(ValueError, match="either simulation.emitter_ring"):
        resolve_emitter_geometry_m({
            "emitter_ring": {"emitter_count": 2, "emitter_radius_nm": 10.0},
            "emitter_orientations": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        })


def test_coreshell_core_pair_has_vacuum_and_structure_terms():
    calculator = MieGreenFunction(
        refractive_indices=[1.0, 2.0 + 0.1j, 1.0],
        radii_m=[160e-9, 60e-9],
        omega=2.0 * np.pi * c / 600e-9,
        nmax=3,
        geometry="coreshell",
        strict_regions=True,
    )
    source = np.array([0.0, 0.0, 10e-9])
    observer = np.array([20e-9, 0.0, 0.0])

    result = calculator.calculate_components(observer, source)
    infinite_cavity = MieGreenFunction(
        refractive_indices=[2.0 + 0.1j, 1.0],
        radii_m=[60e-9],
        omega=2.0 * np.pi * c / 600e-9,
        nmax=3,
        geometry="simplecavity",
        strict_regions=True,
    ).calculate_components(observer, source)

    assert result.source_region == 2
    assert result.observer_region == 2
    assert np.linalg.norm(result.vacuum) > 0.0
    assert np.linalg.norm(result.structure) > 0.0
    assert np.linalg.norm(result.total) > 0.0
    relative_difference = np.linalg.norm(result.structure - infinite_cavity.structure) / np.linalg.norm(
        infinite_cavity.structure
    )
    assert relative_difference > 1e-4


def test_mie_pair_output_supports_finite_shell_cavity_core_emitters(tmp_path):
    config = {
        "simulation": {
            "spectral_param": "wavelength_nm",
            "wavelength_nm": 600.0,
            "nmax": 3,
            "geometry": {"boundary": "coreshell", "radii_nm": [160.0, 60.0]},
            "emitter_positions_nm": [[0.0, 0.0, 10.0], [20.0, 0.0, 0.0]],
            "strict_regions": True,
        },
        "materials": {
            "regions": [
                {"n": 1.0},
                {"epsilon": {"real": 4.0, "imag": 0.4}},
                {"n": 1.0},
            ]
        },
        "parallel": {"backend": "sequential"},
        "output": {"layout": "pair", "directory": str(tmp_path), "prefix": "mie_shell_test"},
    }
    config_path = tmp_path / "mie_shell.yaml"
    output_path = tmp_path / "mie_shell.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_from_config(config_path, output_path)
    data = load_gf_h5(str(output_path))

    assert data["gf_layout"] == "pair"
    assert data["G_total"].shape == (1, 2, 2, 3, 3)
    assert np.linalg.norm(data["G_vac"][0, 0, 1]) > 0.0
    assert np.linalg.norm(data["G_total"][0, 0, 1]) > 0.0


def test_mie_scan_output_supports_energy_ev_and_horizontal_rx_points(tmp_path):
    config = {
        "simulation": {
            "spectral_param": "energy_eV",
            "energy_eV": [1.8, 2.0],
            "nmax": 2,
            "geometry": {"boundary": "coreshell", "radii_nm": [160.0, 60.0]},
            "source_position_nm": [0.0, 0.0, 10.0],
            "position": {"Rx_nm": [0.0, 20.0]},
            "source_orientation": [0.0, 0.0, 1.0],
            "observer_orientation": [0.0, 0.0, 1.0],
            "strict_regions": True,
        },
        "materials": {
            "regions": [
                {"n": 1.0},
                {"epsilon": {"real": 4.0, "imag": 0.4}},
                {"n": 1.0},
            ]
        },
        "parallel": {"backend": "sequential"},
        "output": {"layout": "scan", "directory": str(tmp_path), "prefix": "mie_scan_test"},
    }
    config_path = tmp_path / "mie_scan.yaml"
    output_path = tmp_path / "mie_scan.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_from_config(config_path, output_path)

    with h5py.File(output_path, "r") as h5:
        assert h5.attrs["gf_layout"] == "scan"
        assert h5["green_function_total"].shape == (2, 2, 3, 3)
        assert h5["G_total"].shape == (2, 2, 3, 3)
        assert h5["G_structure"].shape == (2, 2, 3, 3)
        assert np.allclose(h5["energy_eV"][:], [1.8, 2.0])
        assert np.allclose(
            h5["observer_positions_m"][:] * 1e9,
            [[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]],
        )
        assert np.linalg.norm(h5["G_total"][0, 1]) > 0.0

    data = load_gf_h5(str(output_path))
    assert data["gf_layout"] == "scan"
    assert data["G_structure"].shape == (2, 2, 3, 3)
    assert np.allclose(data["observer_positions_nm"], [[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]])


def test_mie_output_path_uses_yaml_prefix_and_hdf5_parameter_suffix(tmp_path):
    config = {
        "simulation": {
            "spectral_param": "energy_eV",
            "energy_eV": [1.8, 2.0],
            "nmax": 1,
            "geometry": {"boundary": "sphere", "radius_nm": 1.0},
            "source_position_nm": [0.0, 0.0, 5.0],
            "position": {"Rx_nm": [0.0, 20.0]},
            "strict_regions": True,
        },
        "materials": {"regions": [{"n": 1.0}, {"n": 1.0}]},
        "parallel": {"backend": "sequential"},
        "output": {"layout": "scan", "directory": str(tmp_path), "prefix": "mie_yaml_prefix"},
    }
    config_path = tmp_path / "mie_filename.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_path = run_from_config(config_path)

    assert output_path.name == "mie_yaml_prefix_Emin_1.80_Emax_2.00_2pts_Rx_20nm_2pts.hdf5"
    assert output_path.exists()
    assert load_gf_h5(str(output_path))["gf_layout"] == "scan"


def test_mie_scan_mpi_backend_runs_single_rank_when_mpi4py_available(tmp_path):
    pytest.importorskip("mpi4py")
    config = {
        "simulation": {
            "spectral_param": "energy_eV",
            "energy_eV": 1.8,
            "nmax": 1,
            "geometry": {"boundary": "sphere", "radius_nm": 1.0},
            "source_position_nm": [0.0, 0.0, 5.0],
            "position": {"Rx_nm": [0.0, 1.0]},
            "strict_regions": True,
        },
        "materials": {"regions": [{"n": 1.0}, {"n": 1.0}]},
        "parallel": {"backend": "mpi", "mpi_auto_launch": False},
        "output": {"layout": "scan", "directory": str(tmp_path), "prefix": "mie_mpi_scan_test"},
    }
    config_path = tmp_path / "mie_mpi_scan.yaml"
    output_path = tmp_path / "mie_mpi_scan.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_from_config(config_path, output_path)

    with h5py.File(output_path, "r") as h5:
        assert h5["G_total"].shape == (1, 2, 3, 3)


def test_gf_sphere_example_config_defines_equatorial_orthoradial_ring():
    config_path = Path("configs/Dyadic_GF/GF_sphere_example.yaml")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    simulation = config["simulation"]

    positions_m, orientations = resolve_emitter_geometry_m(simulation)
    positions_nm = positions_m * 1e9
    radii_nm = np.linalg.norm(positions_nm, axis=1)

    assert simulation["geometry"] == {"boundary": "sphere", "radius_nm": 8.0}
    assert config["output"]["layout"] == "pair"
    assert simulation["emitter_ring"]["emitter_count"] == 15
    assert simulation["emitter_ring"]["emitter_surface_gap_nm"] == 2.0
    assert positions_nm.shape == (15, 3)
    assert orientations.shape == (15, 3)
    assert np.allclose(radii_nm, 10.0)
    assert np.allclose(positions_nm[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(orientations, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", positions_nm, orientations), 0.0, atol=1e-12)
    assert np.isclose(equatorial_ring_nearest_neighbor_chord_nm(15, 10.0), 4.158, atol=5e-4)
    assert simulation["sphere_example"]["dipole_moment_debye"] == 24.0


def test_gf_sphere_example_direct_output_matches_emission_input_default():
    config_path = Path("configs/Dyadic_GF/GF_sphere_example.yaml").resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    output_path = output_path_from_config(config, config_path)

    emission_config = OmegaConf.load("configs/analysis/emission_spectrum_example.yaml")
    expected_input = Path(str(emission_config.input_file).replace("${oc.env:MQED_ROOT,./outputs}", "outputs"))

    assert output_path.resolve() == (Path.cwd() / expected_input).resolve()


def test_gf_sphere_example_drude_silver_is_absorbing_near_lsp_band():
    config = yaml.safe_load(Path("configs/Dyadic_GF/GF_sphere_example.yaml").read_text(encoding="utf-8"))
    resolver = MaterialResolver(Path("configs/Dyadic_GF"))
    silver = config["materials"]["regions"][1]
    refractive_index = resolver.refractive_index(silver, energy_eV=2.95, wavelength_nm=420.0, omega=0.0)

    assert np.imag(refractive_index) > 0.0
    assert np.imag(refractive_index**2) > 0.0


def test_gf_sphere_example_pair_couplings_are_circulant_by_rotation_symmetry():
    config = OmegaConf.to_container(
        OmegaConf.load(Path("configs/Dyadic_GF/GF_sphere_example.yaml")),
        resolve=True,
    )
    positions_m, orientations = resolve_emitter_geometry_m(config["simulation"])
    energy_eV = 2.95
    omega = energy_eV * 1.602176634e-19 / 1.054571817e-34
    calculator = MieGreenFunction(
        refractive_indices=[1.0, 0.3 + 3.0j],
        radii_m=[8e-9],
        omega=omega,
        nmax=4,
        geometry="sphere",
        strict_regions=True,
    )

    n_emitters = positions_m.shape[0]
    G_pair = np.empty((n_emitters, n_emitters, 3, 3), dtype=complex)
    for observer_index, observer in enumerate(positions_m):
        for source_index, source in enumerate(positions_m):
            G_pair[observer_index, source_index] = calculator.calculate_total_Green_function(
                observer, source
            )

    first_row_projected = np.array(
        [_projected_pair(G_pair, orientations, 0, source_index) for source_index in range(n_emitters)]
    )
    for observer_index in range(n_emitters):
        projected_row = np.array(
            [
                _projected_pair(G_pair, orientations, observer_index, source_index)
                for source_index in range(n_emitters)
            ]
        )
        assert np.allclose(projected_row, np.roll(first_row_projected, observer_index), rtol=1e-5, atol=1e-2)

    assert np.linalg.norm(G_pair[0, 0]) > 0.0
    assert np.linalg.norm(G_pair[0, 1]) > 0.0
