import numpy as np
from omegaconf import OmegaConf

from mqed.Dyadic_GF.main import build_grid


def test_build_grid_supports_piecewise_ranges_without_duplicate_boundaries():
    config = OmegaConf.create({
        "segments": [
            {"min": 0.0, "max": 3.0, "points": 4},
            {"min": 3.0, "max": 4.5, "points": 4},
            {"min": 4.5, "max": 6.0, "points": 4},
        ]
    })

    grid = build_grid(config)

    assert np.isclose(grid[0], 0.0)
    assert np.isclose(grid[-1], 6.0)
    assert len(grid) == 12
    assert np.all(np.diff(grid) > 0.0)
    assert np.count_nonzero(np.isclose(grid, 3.0)) == 1
    assert np.count_nonzero(np.isclose(grid, 4.5)) == 1


def test_build_grid_supports_legacy_piecewise_list_of_dicts():
    config = OmegaConf.create([
        {"min": 0.0, "max": 1.0, "points": 3},
        {"min": 1.0, "max": 2.0, "points": 3},
    ])

    grid = build_grid(config)

    assert np.all(np.diff(grid) > 0.0)
    assert np.count_nonzero(np.isclose(grid, 1.0)) == 1


def test_build_grid_keeps_explicit_numeric_lists_unchanged():
    config = OmegaConf.create([1.0, 1.5, 2.0])

    grid = build_grid(config)

    assert np.allclose(grid, [1.0, 1.5, 2.0])
