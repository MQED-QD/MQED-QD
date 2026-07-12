import numpy as np
import pytest
from omegaconf import OmegaConf

from mqed.plotting.plot_spectral_density import (
    _format_scaled_label,
    _iter_input_curves,
    _normalize_curve_scales,
    _normalize_pair_indices,
    _normalize_separation_indices,
    _plot_dataset_on_axes,
    _plot_pair_layout,
    _plot_scan_layout,
    _plot_separation_layout,
    _resolve_pair_indices,
    _resolve_scan_indices,
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


def test_resolve_pair_indices_from_physical_separations():
    cfg = OmegaConf.create({
        "pair_separation_values_nm": [0.0, 2.0, 20.0],
        "pair_reference_index": 0,
        "pair_separation_tolerance_nm": 1e-9,
    })
    emitter_positions_nm = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])

    pair_indices, distances_nm = _resolve_pair_indices(cfg, emitter_positions_nm, 3)

    assert pair_indices == [[0, 0], [0, 1], [0, 2]]
    assert distances_nm == [0.0, 2.0, 20.0]


def test_plot_pair_layout_selects_curves_by_physical_separation():
    cfg = OmegaConf.create({
        "plot_settings": {
            "pair_separation_values_nm": [0.0, 2.0, 20.0],
            "pair_reference_index": 0,
            "pair_separation_tolerance_nm": 1e-9,
            "pair_label_template": "R = {distance_nm:.0f} nm",
            "figsize": [4, 3],
        }
    })
    energy_eV = np.array([1.0, 2.0])
    j_eV = np.zeros((3, 3, 2), dtype=float)
    j_eV[0, 0, :] = [1.0, 1.1]
    j_eV[0, 1, :] = [2.0, 2.2]
    j_eV[0, 2, :] = [20.0, 22.0]
    emitter_positions_nm = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])

    fig = _plot_pair_layout(j_eV, energy_eV, cfg, emitter_positions_nm)

    assert len(fig.axes[0].lines) == 3
    assert [line.get_label() for line in fig.axes[0].lines] == [
        "R = 0 nm",
        "R = 2 nm",
        "R = 20 nm",
    ]
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), j_eV[0, 0, :])
    assert np.allclose(fig.axes[0].lines[1].get_ydata(), j_eV[0, 1, :])
    assert np.allclose(fig.axes[0].lines[2].get_ydata(), j_eV[0, 2, :])


def test_resolve_scan_indices_from_physical_distances():
    cfg = OmegaConf.create({
        "scan_distance_values_nm": [0.0, 2.0, 20.0],
        "scan_distance_tolerance_nm": 1e-9,
    })

    indices, distances_nm = _resolve_scan_indices(cfg, np.array([0.0, 2.0, 20.0]), 3)

    assert indices == [0, 1, 2]
    assert distances_nm == [0.0, 2.0, 20.0]


def test_plot_scan_layout_selects_curves_by_physical_distance():
    cfg = OmegaConf.create({
        "plot_settings": {
            "scan_distance_values_nm": [0.0, 2.0, 20.0],
            "scan_distance_tolerance_nm": 1e-9,
            "scan_label_template": "R = {distance_nm:.0f} nm",
            "figsize": [4, 3],
        }
    })
    energy_eV = np.array([1.0, 2.0])
    j_eV = np.array([
        [1.0, 1.1],
        [2.0, 2.2],
        [20.0, 22.0],
    ])

    fig = _plot_scan_layout(j_eV, energy_eV, cfg, np.array([0.0, 2.0, 20.0]))

    assert len(fig.axes[0].lines) == 3
    assert [line.get_label() for line in fig.axes[0].lines] == [
        "R = 0 nm",
        "R = 2 nm",
        "R = 20 nm",
    ]
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), j_eV[0, :])
    assert np.allclose(fig.axes[0].lines[1].get_ydata(), j_eV[1, :])
    assert np.allclose(fig.axes[0].lines[2].get_ydata(), j_eV[2, :])


def test_iter_input_curves_preserves_single_file_default():
    cfg = OmegaConf.create({"input_file": "single.h5", "curves": []})

    assert _iter_input_curves(cfg) == []


def test_plot_dataset_on_axes_overlays_multiple_input_labels():
    cfg = OmegaConf.create({
        "plot_settings": {
            "separation_indices": [0],
            "separation_multipliers": [1.0],
            "label_template": "Rx = {Rx:.0f} nm",
            "figsize": [4, 3],
        }
    })
    energy_eV = np.array([1.0, 2.0])
    data_a = {
        "gf_layout": "separation",
        "J_eV": np.array([[1.0, 2.0]]),
        "energy_eV": energy_eV,
        "Rx_nm": np.array([0.0]),
    }
    data_b = {
        "gf_layout": "separation",
        "J_eV": np.array([[3.0, 4.0]]),
        "energy_eV": energy_eV,
        "Rx_nm": np.array([0.0]),
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_dataset_on_axes(data_a, cfg, ax, curve_cfg=OmegaConf.create({"label": "direct"}))
    _plot_dataset_on_axes(data_b, cfg, ax, curve_cfg=OmegaConf.create({"label": "dcim"}))

    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "direct: Rx = 0 nm"
    assert ax.lines[1].get_label() == "dcim: Rx = 0 nm"
    assert np.allclose(ax.lines[1].get_ydata(), [3.0, 4.0])
    plt.close(fig)


def test_multi_file_separation_styles_override_file_defaults():
    cfg = OmegaConf.create({
        "plot_settings": {
            "separation_indices": [0, 1],
            "separation_multipliers": [1.0, 1.0],
            "separation_colors": ["black", "gray"],
            "separation_linestyles": [":", "-."],
            "label_template": "Rx = {Rx:.0f} nm",
            "figsize": [4, 3],
        }
    })
    data = {
        "gf_layout": "separation",
        "J_eV": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "energy_eV": np.array([1.0, 2.0]),
        "Rx_nm": np.array([0.0, 3.0]),
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_dataset_on_axes(
        data,
        cfg,
        ax,
        curve_cfg=OmegaConf.create({
            "label": "direct",
            "color": "tab:blue",
            "linestyle": "-",
            "separation_styles": [
                {"linestyle": "-", "marker": "o"},
                {"linestyle": "--", "marker": "s", "color": "tab:red"},
            ],
        }),
    )

    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "direct: Rx = 0 nm"
    assert ax.lines[0].get_color() == "tab:blue"
    assert ax.lines[0].get_linestyle() == "-"
    assert ax.lines[0].get_marker() == "o"
    assert ax.lines[1].get_label() == "direct: Rx = 3 nm"
    assert ax.lines[1].get_color() == "tab:red"
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.lines[1].get_marker() == "s"
    plt.close(fig)


def test_multi_file_pair_styles_support_compact_lists():
    cfg = OmegaConf.create({
        "plot_settings": {
            "pair_indices": [[0, 0], [0, 1]],
            "pair_multipliers": [1.0, 1.0],
            "pair_colors": ["black", "gray"],
            "pair_linestyles": [":", "-."],
            "figsize": [4, 3],
        }
    })
    data = {
        "gf_layout": "pair",
        "J_eV": np.arange(8, dtype=float).reshape(2, 2, 2),
        "energy_eV": np.array([1.0, 2.0]),
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_dataset_on_axes(
        data,
        cfg,
        ax,
        curve_cfg=OmegaConf.create({
            "label": "hybrid",
            "color": "tab:green",
            "pair_linestyles": ["-", "--"],
            "pair_markers": ["o", "^"],
        }),
    )

    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() == "tab:green"
    assert ax.lines[0].get_linestyle() == "-"
    assert ax.lines[0].get_marker() == "o"
    assert ax.lines[1].get_color() == "tab:green"
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.lines[1].get_marker() == "^"
    plt.close(fig)


def test_plot_dataset_on_axes_accepts_scan_layout():
    cfg = OmegaConf.create({
        "plot_settings": {
            "scan_distance_values_nm": [2.0],
            "scan_label_template": "R = {distance_nm:.0f} nm",
            "figsize": [4, 3],
        }
    })
    data = {
        "gf_layout": "scan",
        "J_eV": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "energy_eV": np.array([1.0, 2.0]),
        "observer_distances_nm": np.array([0.0, 2.0]),
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_dataset_on_axes(data, cfg, ax)

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "R = 2 nm"
    assert np.allclose(ax.lines[0].get_ydata(), [3.0, 4.0])
    plt.close(fig)
