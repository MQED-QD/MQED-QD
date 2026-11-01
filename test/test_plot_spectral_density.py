import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from omegaconf import OmegaConf

from mqed.plotting import plot_spectral_density as spectral_plot
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


def _write_separation_h5(path, energy_eV, rx_nm, j_eV):
    with h5py.File(path, "w") as handle:
        handle.attrs["gf_layout"] = "separation"
        handle.create_dataset("energy_eV", data=energy_eV)
        handle.create_dataset("Rx_nm", data=rx_nm)
        handle.create_dataset("J_eV", data=j_eV)


def _run_plot_config(cfg, output_dir, monkeypatch):
    monkeypatch.setattr(
        spectral_plot.HydraConfig,
        "get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir))),
    )
    monkeypatch.setattr(spectral_plot, "setup_loggers_hydra_aware", lambda: None)
    figures = []
    original_close = plt.close

    def capture_and_close(fig):
        figures.append(fig)
        original_close(fig)

    monkeypatch.setattr(spectral_plot.plt, "close", capture_and_close)
    with plt.rc_context():
        spectral_plot.plot_spectral_density.__wrapped__(cfg)
    return figures[0]


def test_entrypoint_overlays_real_files_by_physical_distance(tmp_path, monkeypatch):
    first = tmp_path / "direct.h5"
    second = tmp_path / "aware.h5"
    _write_separation_h5(
        first, [1.0, 2.0], [0.0, 12.0, 120.0], [[1, 2], [3, 4], [5, 6]],
    )
    _write_separation_h5(
        second, [1.5, 2.5, 3.5], [120.0, 0.0, 12.0],
        [[50, 60, 70], [10, 20, 30], [30, 40, 50]],
    )
    cfg = OmegaConf.create({
        "input_file": str(tmp_path / "unused.h5"),
        "curves": [
            {"path": str(first), "label": "Direct", "color": "tab:blue", "linestyle": "-"},
            {"path": str(second), "label": "Aware", "color": "tab:orange", "linestyle": "--"},
        ],
        "plot_settings": {
            "separation_indices": [0],
            "separation_values_nm": [12.0],
            "label_template": "Rx = {Rx:.0f} nm",
            "figsize": [4, 3],
            "filename": "overlay.png",
            "dpi": 72,
        },
    })

    fig = _run_plot_config(cfg, tmp_path, monkeypatch)

    # 12 nm is index 1 in the first file and index 2 in the second: using
    # index 1 for both would silently compare different physical separations.
    lines = fig.axes[0].lines
    assert len(lines) == 2
    assert [line.get_label() for line in lines] == [
        "Direct: Rx = 12 nm", "Aware: Rx = 12 nm",
    ]
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        "Direct: Rx = 12 nm", "Aware: Rx = 12 nm",
    ]
    np.testing.assert_array_equal(lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_array_equal(lines[1].get_xdata(), [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(lines[0].get_ydata(), [3, 4])
    np.testing.assert_array_equal(lines[1].get_ydata(), [30, 40, 50])
    assert [line.get_color() for line in lines] == ["tab:blue", "tab:orange"]
    assert [line.get_linestyle() for line in lines] == ["-", "--"]
    assert (tmp_path / "overlay.png").stat().st_size > 0


def test_entrypoint_empty_curves_uses_single_file_and_indices(tmp_path, monkeypatch):
    source = tmp_path / "single.h5"
    _write_separation_h5(
        source, [1.0, 2.0], [0.0, 12.0, 120.0], [[1, 2], [3, 4], [5, 6]],
    )
    cfg = OmegaConf.create({
        "input_file": str(source),
        "curves": [],
        "plot_settings": {
            "separation_indices": [2],
            "separation_values_nm": [],
            "figsize": [4, 3],
            "filename": "single.png",
        },
    })

    fig = _run_plot_config(cfg, tmp_path, monkeypatch)

    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "Rx = 120.0 nm"
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [5, 6])
    assert (tmp_path / "single.png").stat().st_size > 0


def test_entrypoint_missing_curve_file_fails_instead_of_using_input_file(tmp_path, monkeypatch):
    source = tmp_path / "single.h5"
    _write_separation_h5(source, [1.0], [0.0], [[1.0]])
    cfg = OmegaConf.create({
        "input_file": str(source),
        "curves": [{"path": str(tmp_path / "missing.h5"), "label": "Missing"}],
        "plot_settings": {"figsize": [4, 3]},
    })

    before = set(plt.get_fignums())
    with pytest.raises(FileNotFoundError, match="missing.h5"):
        _run_plot_config(cfg, tmp_path, monkeypatch)
    for figure_number in set(plt.get_fignums()) - before:
        plt.close(figure_number)


def test_hydra_cli_single_file_override_saves_plot(tmp_path):
    source = tmp_path / "cli.h5"
    _write_separation_h5(
        source, [1.0, 2.0], [0.0, 12.0, 120.0], [[1, 2], [3, 4], [5, 6]],
    )
    output_dir = tmp_path / "hydra_output"
    result = subprocess.run(
        [
            sys.executable, "-m", "mqed.plotting.plot_spectral_density",
            "--config-name=plt_spec_dens_direct_sg", "curves=[]",
            "plot_settings.separation_values_nm=[]", "plot_settings.separation_indices=[2]",
            "plot_settings.separation_multipliers=[1.0]", f"input_file={source}",
            "plot_settings.filename=cli.png", f"hydra.run.dir={output_dir}",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "cli.png").stat().st_size > 0


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


def test_plot_dataset_routes_ring_circulant_output_through_pair_layout():
    cfg = OmegaConf.create({
        "plot_settings": {
            "pair_indices": [[0, 2]],
            "pair_multipliers": [1.0],
            "figsize": [4, 3],
        }
    })
    data = {
        "gf_layout": "ring_circulant",
        "J_eV": np.arange(18, dtype=float).reshape(3, 3, 2),
        "energy_eV": np.array([1.0, 2.0]),
        "emitter_positions_nm": np.zeros((3, 3)),
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _plot_dataset_on_axes(data, cfg, ax)

    assert len(ax.lines) == 1
    assert np.allclose(ax.lines[0].get_ydata(), data["J_eV"][0, 2, :])
    plt.close(fig)


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
