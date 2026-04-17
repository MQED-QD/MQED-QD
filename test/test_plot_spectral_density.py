import numpy as np
import pytest
from omegaconf import OmegaConf

from mqed.plotting.plot_spectral_density import (
    _format_scaled_label,
    _normalize_curve_scales,
    _normalize_pair_indices,
    _normalize_separation_indices,
    _plot_pair_layout,
    _plot_separation_layout,
)


def test_normalize_separation_indices_supports_multiple_values():
    assert _normalize_separation_indices([0, 3]) == [0, 3]
    assert _normalize_separation_indices("[0, 3]") == [0, 3]
    assert _normalize_separation_indices(3) == [3]


def test_normalize_pair_indices_supports_single_and_multiple_pairs():
    assert _normalize_pair_indices([0, 3]) == [[0, 3]]
    assert _normalize_pair_indices("[[0, 0], [0, 3]]") == [[0, 0], [0, 3]]


def test_normalize_curve_scales_supports_single_and_per_curve_values():
    assert _normalize_curve_scales([1.0, 100.0], 2, "plot_settings.separation_multipliers") == [
        1.0,
        100.0,
    ]
    assert _normalize_curve_scales(1000.0, 2, "plot_settings.separation_multipliers") == [
        1000.0,
        1000.0,
    ]


def test_normalize_curve_styles_supports_broadcast_and_lists():
    from mqed.plotting.plot_spectral_density import _normalize_curve_styles

    assert _normalize_curve_styles("tab:red", 2, "plot_settings.separation_colors") == [
        "tab:red",
        "tab:red",
    ]
    assert _normalize_curve_styles(["tab:blue", "tab:orange"], 2, "plot_settings.separation_colors") == [
        "tab:blue",
        "tab:orange",
    ]


def test_format_scaled_label_appends_multiplier_only_when_needed():
    assert _format_scaled_label("Rx = 120.0 nm", 1.0) == "Rx = 120.0 nm"
    assert _format_scaled_label("Rx = 120.0 nm", 1000.0) == "Rx = 120.0 nm ×1000"
    assert _format_scaled_label(r"$J_{\alpha=0,\beta=2}(\omega)$", 1000.0) == (
        r"$J_{\alpha=0,\beta=2}(\omega)\,\times\,1000$"
    )


def test_plot_layout_rejects_nonpositive_log_multiplier():
    cfg = OmegaConf.create({
        "plot_settings": {
            "separation_indices": [0],
            "separation_multipliers": [0.0],
            "yscale": "log",
        }
    })

    with pytest.raises(ValueError, match="separation_multipliers"):
        _plot_separation_layout(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), np.array([0.0]), cfg)


def test_plot_separation_layout_plots_multiple_curves():
    cfg = OmegaConf.create({
        "plot_settings": {
            "separation_indices": [0, 3],
            "separation_multipliers": [1.0, 100.0],
            "separation_colors": ["tab:blue", "tab:red"],
            "separation_linestyles": ["-", "--"],
            "figsize": [4, 3],
        }
    })
    energy_eV = np.array([1.0, 2.0, 3.0])
    rx_nm = np.array([0.0, 1.0, 2.0, 3.0])
    j_eV = np.array([
        [1.0, 1.5, 2.0],
        [1.2, 1.7, 2.2],
        [1.4, 1.9, 2.4],
        [1.6, 2.1, 2.6],
    ])

    fig = _plot_separation_layout(j_eV, energy_eV, rx_nm, cfg)

    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].lines[0].get_label() == "Rx = 0.0 nm"
    assert fig.axes[0].lines[1].get_label() == "Rx = 3.0 nm ×100"
    assert np.allclose(fig.axes[0].lines[1].get_ydata(), 100.0 * j_eV[3, :])
    assert fig.axes[0].lines[0].get_color() == "tab:blue"
    assert fig.axes[0].lines[1].get_linestyle() == "--"


def test_plot_pair_layout_plots_multiple_curves():
    cfg = OmegaConf.create({
        "plot_settings": {
            "pair_indices": [[0, 0], [0, 2]],
            "pair_multipliers": [1.0, 1000.0],
            "pair_colors": ["black", "tab:green"],
            "pair_linestyles": ["-", ":"],
            "figsize": [4, 3],
        }
    })
    energy_eV = np.array([1.0, 2.0, 3.0])
    j_eV = np.arange(27, dtype=float).reshape(3, 3, 3)

    fig = _plot_pair_layout(j_eV, energy_eV, cfg)

    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].lines[0].get_label() == r"$J_{\alpha=0,\beta=0}(\omega)$"
    assert fig.axes[0].lines[1].get_label() == r"$J_{\alpha=0,\beta=2}(\omega)\,\times\,1000$"
    assert np.allclose(fig.axes[0].lines[1].get_ydata(), 1000.0 * j_eV[0, 2, :])
    assert fig.axes[0].lines[0].get_color() == "black"
    assert fig.axes[0].lines[1].get_linestyle() == ":"
