import numpy as np
import pytest

from mqed.utils.emitter_geometry import (
    equatorial_ring_nearest_neighbor_chord_nm,
    equatorial_ring_positions_orientations_nm,
    generate_equatorial_ring_from_config,
    normalize_orientation_vectors,
    resolve_equatorial_ring_radius_nm,
)


def test_equatorial_ring_n1_geometry_and_spacing():
    positions_nm, orientations = equatorial_ring_positions_orientations_nm(1, 10.0)

    assert np.allclose(positions_nm, [[10.0, 0.0, 0.0]])
    assert np.allclose(orientations, [[0.0, 1.0, 0.0]])
    assert equatorial_ring_nearest_neighbor_chord_nm(1, 10.0) == 0.0


def test_equatorial_ring_n8_matches_static_backwards_equivalent_points():
    positions_nm, orientations = equatorial_ring_positions_orientations_nm(8, 10.0)
    root_half = np.sqrt(0.5)

    expected_positions = np.array([
        [10.0, 0.0, 0.0],
        [10.0 * root_half, 10.0 * root_half, 0.0],
        [0.0, 10.0, 0.0],
        [-10.0 * root_half, 10.0 * root_half, 0.0],
        [-10.0, 0.0, 0.0],
        [-10.0 * root_half, -10.0 * root_half, 0.0],
        [0.0, -10.0, 0.0],
        [10.0 * root_half, -10.0 * root_half, 0.0],
    ])
    expected_orientations = np.array([
        [0.0, 1.0, 0.0],
        [-root_half, root_half, 0.0],
        [-1.0, 0.0, 0.0],
        [-root_half, -root_half, 0.0],
        [0.0, -1.0, 0.0],
        [root_half, -root_half, 0.0],
        [1.0, 0.0, 0.0],
        [root_half, root_half, 0.0],
    ])

    assert np.allclose(positions_nm, expected_positions, atol=1e-12)
    assert np.allclose(orientations, expected_orientations, atol=1e-12)


@pytest.mark.parametrize("count, expected_spacing", [(15, 4.158), (50, 1.256)])
def test_equatorial_ring_invariants_and_paper_spacings(count, expected_spacing):
    positions_nm, orientations = equatorial_ring_positions_orientations_nm(count, 10.0)

    assert positions_nm.shape == (count, 3)
    assert orientations.shape == (count, 3)
    assert np.allclose(np.linalg.norm(positions_nm[:, :2], axis=1), 10.0)
    assert np.allclose(positions_nm[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(orientations, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", positions_nm, orientations), 0.0, atol=1e-12)
    assert np.isclose(equatorial_ring_nearest_neighbor_chord_nm(count, 10.0), expected_spacing, atol=5e-4)


def test_equatorial_ring_phase_offset_and_radial_mode():
    positions_nm, orientations = equatorial_ring_positions_orientations_nm(
        4,
        2.0,
        z_nm=1.5,
        phase_offset_deg=90.0,
        orientation="radial",
    )

    assert np.allclose(positions_nm[0], [0.0, 2.0, 1.5], atol=1e-12)
    assert np.allclose(orientations[0], [0.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "config",
    [
        {"emitter_count": 0, "emitter_radius_nm": 10.0},
        {"emitter_count": 1, "emitter_radius_nm": 0.0},
        {"emitter_count": 1, "emitter_radius_nm": float("nan")},
        {"emitter_count": 1, "sphere_radius_nm": 8.0, "emitter_surface_gap_nm": -1.0},
        {"emitter_count": 1, "orientation": "tangential"},
    ],
)
def test_equatorial_ring_rejects_invalid_inputs(config):
    with pytest.raises(ValueError):
        generate_equatorial_ring_from_config(config)


def test_equatorial_ring_radius_rejects_inconsistent_explicit_and_gap_radius():
    with pytest.raises(ValueError):
        resolve_equatorial_ring_radius_nm({
            "sphere_radius_nm": 8.0,
            "emitter_surface_gap_nm": 2.0,
            "emitter_radius_nm": 11.0,
        })


def test_equatorial_ring_radius_derives_from_sphere_gap():
    assert resolve_equatorial_ring_radius_nm({
        "sphere_radius_nm": 8.0,
        "emitter_surface_gap_nm": 2.0,
    }) == 10.0


def test_orientation_normalization_is_stable_for_large_finite_vectors():
    orientations = normalize_orientation_vectors(
        np.array([[1.0e308, 1.0e308, 0.0], [0.0, -1.0e308, 1.0e308]]),
        expected_count=2,
    )

    assert np.all(np.isfinite(orientations))
    assert np.allclose(np.linalg.norm(orientations, axis=1), 1.0)
    assert np.allclose(orientations[0], [np.sqrt(0.5), np.sqrt(0.5), 0.0])
