import numpy as np

from mqed.analysis.spectral_density import compute_spectral_density_pair


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
