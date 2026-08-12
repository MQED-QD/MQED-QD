import h5py
import numpy as np
import pytest

from mqed.analysis.spectral_density import (
    _save_spectral_density_h5,
    compute_spectral_density_pair,
    compute_spectral_density_ring_circulant,
    compute_spectral_density_scan,
    compute_spectral_density_separation,
)


def test_pair_spectral_density_preserves_close_0_2_20_nm_curves():
    energy_eV = np.array([1.0, 1.5, 2.0])
    g_imag = np.zeros((3, 3, 3, 3, 3), dtype=float)
    shared_zz = np.array([1.0, 1.2, 1.4])
    g_imag[:, 0, 0, 2, 2] = shared_zz
    g_imag[:, 0, 1, 2, 2] = 1.001 * shared_zz
    g_imag[:, 0, 2, 2, 2] = 0.999 * shared_zz
    orientations = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))

    j_eV = compute_spectral_density_pair(g_imag, energy_eV, orientations, mu_debye=3.8)

    assert j_eV.shape == (3, 3, 3)
    assert np.allclose(j_eV[0, 1], j_eV[0, 0], rtol=1.1e-3)
    assert np.allclose(j_eV[0, 2], j_eV[0, 0], rtol=1.1e-3)


def test_ring_circulant_spectral_density_matches_projected_pair():
    energy_eV = np.array([1.0, 1.5])
    projected_row = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    g_imag = np.zeros((2, 3, 3, 3, 3), dtype=float)
    offsets = (np.arange(3)[None, :] - np.arange(3)[:, None]) % 3
    g_imag[:, :, :, 2, 2] = projected_row[:, offsets]
    orientations = np.tile([0.0, 0.0, 1.0], (3, 1))

    expected = compute_spectral_density_pair(g_imag, energy_eV, orientations, mu_debye=2.0)
    actual = compute_spectral_density_ring_circulant(projected_row, energy_eV, mu_debye=2.0)

    assert np.allclose(actual, expected)


def test_scan_spectral_density_preserves_fixed_source_observer_curves():
    energy_eV = np.array([1.0, 1.5, 2.0])
    g_imag = np.zeros((3, 3, 3, 3), dtype=float)
    shared_xx = np.array([1.0, 1.2, 1.4])
    g_imag[:, 0, 0, 0] = shared_xx
    g_imag[:, 1, 0, 0] = 1.001 * shared_xx
    g_imag[:, 2, 0, 0] = 0.999 * shared_xx
    orientation = np.array([1.0, 0.0, 0.0])

    j_eV = compute_spectral_density_scan(
        g_imag,
        energy_eV,
        orientation,
        orientation,
        mu_source_debye=1.0,
        mu_observer_debye=1.0,
    )

    assert j_eV.shape == (3, 3)
    assert np.allclose(j_eV[1], j_eV[0], rtol=1.1e-3)
    assert np.allclose(j_eV[2], j_eV[0], rtol=1.1e-3)


def test_separation_spectral_density_rejects_nonfinite_green_data():
    g_imag = np.zeros((2, 1, 3, 3), dtype=float)
    g_imag[1, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="G_imag contains 1 non-finite value"):
        compute_spectral_density_separation(
            g_imag,
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )


def test_save_spectral_density_h5_preserves_scan_position_metadata(tmp_path):
    output_path = tmp_path / "scan_spectral_density.h5"
    observer_positions_nm = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])

    _save_spectral_density_h5(
        output_path,
        {
            "J_eV": np.ones((3, 2)),
            "energy_eV": np.array([1.0, 2.0]),
            "gf_layout": "scan",
            "observer_positions_nm": observer_positions_nm,
            "source_position_nm": np.array([0.0, 0.0, 0.0]),
            "observer_distances_nm": np.array([0.0, 2.0, 20.0]),
        },
    )

    with h5py.File(output_path, "r") as h5:
        assert h5.attrs["gf_layout"] == "scan"
        assert h5["J_eV"].shape == (3, 2)
        assert np.allclose(h5["observer_positions_nm"][:], observer_positions_nm)
        assert np.allclose(h5["observer_distances_nm"][:], [0.0, 2.0, 20.0])
