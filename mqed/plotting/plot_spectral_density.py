"""
Plot Spectral Density
=====================

Plots the generalized spectral density :math:`J_{\\alpha\\beta}(\\omega)` as a
function of energy (eV) for user-selected emitter pairs or separations.

Reads spectral density data from the HDF5 file produced by
:mod:`mqed.analysis.spectral_density`.

Usage::

    python -m mqed.plotting.plot_spectral_density

Configuration via ``configs/plots/plt_spec_dens_direct_sg.yaml``.
"""

from ast import literal_eval
from typing import Any

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.SI_unit import eV_to_J, hbar
from mqed.utils.file_utils import _resolve_input_path
from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------


def _load_spectral_density_h5(filepath: str) -> dict:
    """Load spectral density data from HDF5.

    Returns:
        Dictionary with keys: J_eV, energy_eV, gf_layout, and layout-specific
        position metadata.
    """
    data = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
        for key in f.attrs:
            data[key] = f.attrs[key]
    return data


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def _apply_font_config(cfg):
    """Apply font configuration from config, following plot_pr.py conventions."""
    font_cfg = cfg.get("font", {})
    plt.rcParams.update({
        "font.family": font_cfg.get("family", "Arial"),
        "axes.labelsize": font_cfg.get("labelsize", 18),
        "xtick.labelsize": font_cfg.get("ticksize", 16),
        "ytick.labelsize": font_cfg.get("ticksize", 16),
        "legend.fontsize": font_cfg.get("legendsize", 14),
        "axes.titlesize": font_cfg.get("titlesize", 18),
        "axes.labelweight": font_cfg.get("labelweight", "bold"),
        "axes.titleweight": font_cfg.get("titleweight", "bold"),
    })


def _parse_index_selection(raw_selection: Any, default_value: Any, selection_name: str) -> Any:
    """Normalize Hydra index selection values into plain Python objects."""
    if raw_selection is None:
        return default_value

    if isinstance(raw_selection, str):
        stripped = raw_selection.strip()
        if not stripped:
            return default_value
        try:
            return literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid {selection_name}: {raw_selection!r}. "
                "Use Python-style list syntax such as [0, 3] or [[0, 0], [0, 3]]."
            ) from exc

    if hasattr(raw_selection, "__iter__") and not isinstance(raw_selection, (bytes, bytearray)):
        return list(raw_selection)

    return raw_selection


def _normalize_separation_indices(raw_selection: Any) -> list[int]:
    """Return a validated list of separation indices."""
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[0],
        selection_name="plot_settings.separation_indices",
    )

    if isinstance(normalized, (int, float)):
        return [int(normalized)]

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.separation_indices must be an integer or a list of integers."
        )

    if not normalized:
        return [0]

    return [int(idx) for idx in normalized]


def _normalize_separation_values_nm(raw_selection: Any) -> list[float]:
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[],
        selection_name="plot_settings.separation_values_nm",
    )

    if isinstance(normalized, (int, float)):
        return [float(normalized)]

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.separation_values_nm must be a number or a list of numbers."
        )

    return [float(value) for value in normalized]


def _resolve_separation_indices(ps, Rx_nm) -> list[int]:
    values_nm = _normalize_separation_values_nm(ps.get("separation_values_nm", []))
    if not values_nm:
        return _normalize_separation_indices(ps.get("separation_indices", [0]))

    tolerance_nm = float(ps.get("separation_value_tolerance_nm", 1e-9))
    rx_array = np.asarray(Rx_nm, dtype=float)
    indices = []
    for value_nm in values_nm:
        matches = np.where(np.isclose(rx_array, value_nm, rtol=0.0, atol=tolerance_nm))[0]
        if matches.size == 0:
            nearest_idx = int(np.argmin(np.abs(rx_array - value_nm)))
            nearest_value = float(rx_array[nearest_idx])
            logger.warning(
                "Separation value {:.6g} nm not found within {:.3g} nm; nearest is "
                "index {} at {:.6g} nm, skipping.",
                value_nm,
                tolerance_nm,
                nearest_idx,
                nearest_value,
            )
            continue
        indices.append(int(matches[0]))
    return indices


def _is_nested_index_collection(value: Any) -> bool:
    """Return True when a selection entry is itself a collection of indices."""
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray))


def _normalize_pair_indices(raw_selection: Any) -> list[list[int]]:
    """Return a validated list of [alpha, beta] pair indices."""
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[[0, 0]],
        selection_name="plot_settings.pair_indices",
    )

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.pair_indices must be a pair like [0, 3] or a list of pairs."
        )

    if not normalized:
        return [[0, 0]]

    if len(normalized) == 2 and all(not _is_nested_index_collection(item) for item in normalized):
        first = normalized[0]
        second = normalized[1]
        return [[int(first), int(second)]]

    pair_indices = []
    for pair in normalized:
        if not _is_nested_index_collection(pair) or len(pair) != 2:
            raise ValueError(
                "Each pair in plot_settings.pair_indices must have exactly two entries."
            )
        pair_indices.append([int(pair[0]), int(pair[1])])

    return pair_indices


def _normalize_pair_separation_values_nm(raw_selection: Any) -> list[float]:
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[],
        selection_name="plot_settings.pair_separation_values_nm",
    )

    if isinstance(normalized, (int, float)):
        return [float(normalized)]

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.pair_separation_values_nm must be a number or a list of numbers."
        )

    return [float(value) for value in normalized]


def _resolve_pair_indices(ps, emitter_positions_nm, n_emitters: int) -> tuple[list[list[int]], list[float | None]]:
    values_nm = _normalize_pair_separation_values_nm(ps.get("pair_separation_values_nm", []))
    if not values_nm:
        pair_indices = _normalize_pair_indices(ps.get("pair_indices", [[0, 0]]))
        return pair_indices, [None] * len(pair_indices)

    if emitter_positions_nm is None:
        raise ValueError(
            "plot_settings.pair_separation_values_nm requires emitter_positions_nm in the "
            "spectral-density HDF5 file."
        )

    positions = np.asarray(emitter_positions_nm, dtype=float)
    if positions.shape != (n_emitters, 3):
        raise ValueError(
            f"emitter_positions_nm shape {positions.shape} does not match J shape with N={n_emitters}."
        )

    reference_index = int(ps.get("pair_reference_index", 0))
    if reference_index < 0 or reference_index >= n_emitters:
        raise ValueError(f"plot_settings.pair_reference_index {reference_index} is out of range for N={n_emitters}.")

    tolerance_nm = float(ps.get("pair_separation_tolerance_nm", 1e-6))
    distances_nm = np.linalg.norm(positions - positions[reference_index], axis=1)
    pairs: list[list[int]] = []
    resolved_values: list[float | None] = []
    for value_nm in values_nm:
        matches = np.where(np.isclose(distances_nm, value_nm, rtol=0.0, atol=tolerance_nm))[0]
        if matches.size == 0:
            nearest_idx = int(np.argmin(np.abs(distances_nm - value_nm)))
            logger.warning(
                "Pair separation {:.6g} nm not found within {:.3g} nm from emitter {}; "
                "nearest is emitter {} at {:.6g} nm, skipping.",
                value_nm,
                tolerance_nm,
                reference_index,
                nearest_idx,
                float(distances_nm[nearest_idx]),
            )
            continue
        beta = int(matches[0])
        pairs.append([reference_index, beta])
        resolved_values.append(float(distances_nm[beta]))

    return pairs, resolved_values


def _resolve_scan_indices(ps, observer_distances_nm, n_observers: int) -> tuple[list[int], list[float | None]]:
    values_nm = _normalize_separation_values_nm(ps.get("scan_distance_values_nm", []))
    if not values_nm:
        values_nm = _normalize_separation_values_nm(ps.get("separation_values_nm", []))
    if not values_nm:
        indices = _normalize_separation_indices(ps.get("scan_indices", ps.get("separation_indices", [0])))
        return indices, [None] * len(indices)

    if observer_distances_nm is None:
        raise ValueError(
            "plot_settings.scan_distance_values_nm requires observer_distances_nm in the "
            "spectral-density HDF5 file."
        )

    distances = np.asarray(observer_distances_nm, dtype=float)
    if distances.shape != (n_observers,):
        raise ValueError(
            f"observer_distances_nm shape {distances.shape} does not match J shape with P={n_observers}."
        )

    tolerance_nm = float(ps.get("scan_distance_tolerance_nm", ps.get("separation_value_tolerance_nm", 1e-9)))
    indices: list[int] = []
    resolved_values: list[float | None] = []
    for value_nm in values_nm:
        matches = np.where(np.isclose(distances, value_nm, rtol=0.0, atol=tolerance_nm))[0]
        if matches.size == 0:
            nearest_idx = int(np.argmin(np.abs(distances - value_nm)))
            logger.warning(
                "Scan distance {:.6g} nm not found within {:.3g} nm; nearest is "
                "observer {} at {:.6g} nm, skipping.",
                value_nm,
                tolerance_nm,
                nearest_idx,
                float(distances[nearest_idx]),
            )
            continue
        idx = int(matches[0])
        indices.append(idx)
        resolved_values.append(float(distances[idx]))

    return indices, resolved_values


def _normalize_curve_scales(raw_scales: Any, count: int, setting_name: str) -> list[float]:
    """Return one multiplicative scale factor per plotted curve."""
    normalized = _parse_index_selection(
        raw_scales,
        default_value=[1.0] * count,
        selection_name=setting_name,
    )

    if isinstance(normalized, (int, float)):
        return [float(normalized)] * count

    if not isinstance(normalized, list):
        raise ValueError(f"{setting_name} must be a number or a list of numbers.")

    if not normalized:
        return [1.0] * count

    if len(normalized) == 1 and count > 1:
        return [float(normalized[0])] * count

    if len(normalized) != count:
        raise ValueError(
            f"{setting_name} must contain {count} value(s) to match the selected curves."
        )

    return [float(scale) for scale in normalized]


def _normalize_curve_styles(raw_styles: Any, count: int, setting_name: str) -> list[Any]:
    """Return one optional style entry per plotted curve."""
    if raw_styles is None:
        return [None] * count

    if isinstance(raw_styles, str):
        return [raw_styles] * count

    if not isinstance(raw_styles, list):
        raw_styles = list(raw_styles)

    if not raw_styles:
        return [None] * count

    if len(raw_styles) == 1 and count > 1:
        return [raw_styles[0]] * count

    if len(raw_styles) != count:
        raise ValueError(
            f"{setting_name} must contain {count} value(s) to match the selected curves."
        )

    return list(raw_styles)


def _resolve_curve_multipliers(ps, primary_key: str, legacy_key: str, count: int) -> list[float]:
    """Load curve multipliers with a backwards-compatible fallback key."""
    raw_multipliers = ps.get(primary_key, None)
    if raw_multipliers is None:
        raw_multipliers = ps.get(legacy_key, None)

    return _normalize_curve_scales(raw_multipliers, count=count, setting_name=f"plot_settings.{primary_key}")


def _format_scaled_label(base_label: str, scale_factor: float) -> str:
    """Append a multiplier annotation to the legend label when needed."""
    if scale_factor == 1.0:
        return base_label

    if base_label.startswith("$") and base_label.endswith("$"):
        return f"{base_label[:-1]}\\,\\times\\,{scale_factor:g}$"

    return f"{base_label} ×{scale_factor:g}"


def _validate_curve_multiplier(scale_factor: float, yscale: str, setting_name: str) -> None:
    """Reject invalid multipliers for the active y-axis scale."""
    if yscale == "log" and scale_factor <= 0:
        raise ValueError(f"{setting_name} values must be positive when plot_settings.yscale is 'log'.")


def _resolve_curve_styles(ps, prefix: str, count: int) -> tuple[list[Any], list[Any]]:
    """Load optional per-curve colors and linestyles for one plot layout."""
    colors = _normalize_curve_styles(
        ps.get(f"{prefix}_colors", None),
        count=count,
        setting_name=f"plot_settings.{prefix}_colors",
    )
    linestyles = _normalize_curve_styles(
        ps.get(f"{prefix}_linestyles", None),
        count=count,
        setting_name=f"plot_settings.{prefix}_linestyles",
    )
    return colors, linestyles


def _convert_spectral_density_for_plot(J_eV, unit: str):
    normalized_unit = str(unit).strip().lower()
    if normalized_unit in {"ev", "electronvolt", "electronvolts"}:
        return J_eV, "eV"
    if normalized_unit in {"si", "s^-1", "1/s", "per_s", "per_second", "rad/s"}:
        return J_eV * eV_to_J / hbar, r"s$^{-1}$"

    raise ValueError(
        "plot_settings.spectral_density_unit must be 'eV' or 's^-1' "
        f"(got {unit!r})."
    )


def _resolve_ylabel(ps, default_label: str) -> str:
    custom_label = ps.get("ylabel", None)
    if custom_label is None:
        return default_label
    return custom_label


def _apply_y_sci_formatting(ax, ps) -> None:
    y_sci = ps.get("y_sci", None)
    if not y_sci or not y_sci.get("enabled", False):
        return

    if y_sci.get("style", "sci") == "sci":
        scilimits = y_sci.get("scilimits", [-2, 2])
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(int(scilimits[0]), int(scilimits[1])),
            useMathText=bool(y_sci.get("use_math_text", True)),
        )
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(int(y_sci.get("offset_text_size", 16)))
        offset_text.set_fontweight(str(y_sci.get("offset_text_weight", "normal")))
        return

    ax.ticklabel_format(axis="y", style="plain")


def _curve_label_prefix(curve_cfg: Any) -> str | None:
    if curve_cfg is None:
        return None
    label = curve_cfg.get("label", None)
    if label is None:
        return None
    return str(label)


def _with_curve_label(label: str, curve_cfg: Any) -> str:
    prefix = _curve_label_prefix(curve_cfg)
    if prefix is None:
        return label
    return f"{prefix}: {label}"


def _curve_style(curve_cfg: Any, key: str, fallback: Any = None) -> Any:
    if curve_cfg is None:
        return fallback
    return curve_cfg.get(key, fallback)


def _style_list_key(style_key: str) -> str:
    """Return the plural config key used for per-selected-curve style lists."""
    if style_key == "linestyle":
        return "linestyles"
    if style_key == "lw":
        return "lws"
    if style_key == "alpha":
        return "alphas"
    if style_key == "marker":
        return "markers"
    return f"{style_key}s"


def _style_entry_value(entry: Any, style_key: str) -> Any:
    """Read one style value from a per-selection style entry."""
    if entry is None:
        return None
    if isinstance(entry, str):
        if style_key in {"color", "linestyle", "marker"}:
            return entry
        return None
    if not hasattr(entry, "get"):
        return None
    if style_key == "lw":
        return entry.get("lw", entry.get("linewidth", None))
    if style_key == "linestyle":
        return entry.get("linestyle", entry.get("style", None))
    return entry.get(style_key, None)


def _curve_sequence_style(curve_cfg: Any, prefix: str, selection_position: int, style_key: str) -> Any:
    """Return a per-selected separation/pair style from a curve config, if present.

    Supported forms under each ``curves`` entry are either explicit style dicts::

        separation_styles:
          - {linestyle: "-", marker: "o"}
          - {linestyle: "--", marker: "s"}

    or compact per-property lists/scalars::

        separation_linestyles: ["-", "--"]
        separation_markers: ["o", "s"]

    The list order follows ``plot_settings.separation_indices`` or
    ``plot_settings.pair_indices`` after physical-value resolution.
    """
    if curve_cfg is None:
        return None

    style_entries = curve_cfg.get(f"{prefix}_styles", None)
    if style_entries is not None:
        entries = list(style_entries)
        if len(entries) == 1:
            value = _style_entry_value(entries[0], style_key)
        elif selection_position < len(entries):
            value = _style_entry_value(entries[selection_position], style_key)
        else:
            raise ValueError(
                f"curves[].{prefix}_styles must contain at least "
                f"{selection_position + 1} entry(ies) for the selected {prefix} curves."
            )
        if value is not None:
            return value

    raw_values = curve_cfg.get(f"{prefix}_{_style_list_key(style_key)}", None)
    if raw_values is None and style_key == "lw":
        raw_values = curve_cfg.get(f"{prefix}_linewidths", None)
    if raw_values is None:
        return None

    if isinstance(raw_values, str):
        return raw_values

    if not isinstance(raw_values, list):
        raw_values = list(raw_values)

    if not raw_values:
        return None
    if len(raw_values) == 1:
        return raw_values[0]
    if selection_position >= len(raw_values):
        raise ValueError(
            f"curves[].{prefix}_{_style_list_key(style_key)} must contain at least "
            f"{selection_position + 1} value(s) for the selected {prefix} curves."
        )
    return raw_values[selection_position]


def _curve_style_for_selection(
    curve_cfg: Any,
    prefix: str,
    selection_position: int,
    style_key: str,
    fallback: Any = None,
) -> Any:
    """Resolve style precedence for one selected separation or pair.

    Precedence is per-selected style on the input-file curve, then file-level
    curve defaults, then the global plot-settings fallback for that selected
    separation or pair.
    """
    selected_style = _curve_sequence_style(curve_cfg, prefix, selection_position, style_key)
    if selected_style is not None:
        return selected_style

    if style_key == "linestyle":
        return _curve_style(curve_cfg, "linestyle", _curve_style(curve_cfg, "style", fallback))
    if style_key == "lw":
        return _curve_style(curve_cfg, "lw", _curve_style(curve_cfg, "linewidth", fallback))
    return _curve_style(curve_cfg, style_key, fallback)


def _plot_separation_layout(J_eV, energy_eV, Rx_nm, cfg, ax=None, curve_cfg=None):
    """Plot J(ω) for separation-indexed data.

    Produces one curve per selected separation Rx.
    """
    ps = cfg.plot_settings
    J_plot, unit_label = _convert_spectral_density_for_plot(
        J_eV,
        ps.get("spectral_density_unit", "eV"),
    )

    # Select which separations to plot
    sep_indices = _resolve_separation_indices(ps, Rx_nm)
    yscale = ps.get("yscale", "linear")
    scale_factors = _resolve_curve_multipliers(
        ps,
        primary_key="separation_multipliers",
        legacy_key="separation_scale_factors",
        count=len(sep_indices),
    )
    colors, linestyles = _resolve_curve_styles(ps, prefix="separation", count=len(sep_indices))

    if ax is None:
        fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
    else:
        fig = ax.figure

    for selection_position, (idx, scale_factor, color, linestyle) in enumerate(zip(
        sep_indices,
        scale_factors,
        colors,
        linestyles,
    )):
        if idx >= len(Rx_nm):
            logger.warning(f"Separation index {idx} out of range "
                           f"(max {len(Rx_nm) - 1}), skipping.")
            continue

        _validate_curve_multiplier(
            scale_factor,
            yscale=yscale,
            setting_name="plot_settings.separation_multipliers",
        )

        label = ps.get("label_template", "Rx = {Rx:.1f} nm").format(Rx=Rx_nm[idx])
        label = _with_curve_label(label, curve_cfg)
        label = _format_scaled_label(label, scale_factor)
        ax.plot(
            energy_eV,
            scale_factor * J_plot[idx, :],
            lw=_curve_style_for_selection(curve_cfg, "separation", selection_position, "lw", ps.get("lw", 1.5)),
            label=label,
            color=_curve_style_for_selection(curve_cfg, "separation", selection_position, "color", color),
            linestyle=_curve_style_for_selection(
                curve_cfg,
                "separation",
                selection_position,
                "linestyle",
                linestyle,
            ),
            marker=_curve_style_for_selection(curve_cfg, "separation", selection_position, "marker", None),
            alpha=_curve_style_for_selection(curve_cfg, "separation", selection_position, "alpha", None),
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(_resolve_ylabel(ps, rf"$J(\omega)$ ({unit_label})"))

    title_template = ps.get("title", r"Spectral Density $J(\omega)$")
    ax.set_title(title_template)

    if yscale == "log":
        ax.set_yscale("log")
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    x_range = ps.get("x_range_eV", None)
    if x_range is not None:
        ax.set_xlim(x_range)

    y_range = ps.get("y_range", None)
    if y_range is not None:
        ax.set_ylim(y_range)

    _apply_y_sci_formatting(ax, ps)

    if ps.get("grid", True):
        ax.grid(True, alpha=0.3)

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid separation indices were plotted.")
    fig.tight_layout()
    return fig


def _plot_pair_layout(J_eV, energy_eV, cfg, emitter_positions_nm=None, ax=None, curve_cfg=None):
    """Plot J_αβ(ω) for pair-indexed data.

    Produces one curve per selected (α, β) pair.
    """
    ps = cfg.plot_settings
    J_plot, unit_label = _convert_spectral_density_for_plot(
        J_eV,
        ps.get("spectral_density_unit", "eV"),
    )

    pair_indices, pair_distances_nm = _resolve_pair_indices(ps, emitter_positions_nm, J_eV.shape[0])
    yscale = ps.get("yscale", "linear")
    scale_factors = _resolve_curve_multipliers(
        ps,
        primary_key="pair_multipliers",
        legacy_key="pair_scale_factors",
        count=len(pair_indices),
    )
    colors, linestyles = _resolve_curve_styles(ps, prefix="pair", count=len(pair_indices))

    if ax is None:
        fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
    else:
        fig = ax.figure

    N = J_eV.shape[0]
    for selection_position, (pair, distance_nm, scale_factor, color, linestyle) in enumerate(zip(
        pair_indices,
        pair_distances_nm,
        scale_factors,
        colors,
        linestyles,
    )):
        alpha, beta = int(pair[0]), int(pair[1])
        if alpha >= N or beta >= N:
            logger.warning(f"Pair ({alpha}, {beta}) out of range "
                           f"(N={N}), skipping.")
            continue

        _validate_curve_multiplier(
            scale_factor,
            yscale=yscale,
            setting_name="plot_settings.pair_multipliers",
        )

        label_template = ps.get(
            "pair_label_template",
            ps.get(
                "label_template",
                r"$J_{{\alpha={a},\beta={b}}}(\omega)$",
            ),
        )
        try:
            label = label_template.format(a=alpha, b=beta, distance_nm=distance_nm, Rx=distance_nm)
        except (KeyError, TypeError, ValueError):
            label = (
                r"$J_{{\alpha={a},\beta={b}}}(\omega)$"
            ).format(a=alpha, b=beta)
        if distance_nm is not None and "distance_nm" not in label_template and "Rx" not in label_template:
            label = f"{label} ({distance_nm:g} nm)"
        label = _with_curve_label(label, curve_cfg)
        label = _format_scaled_label(label, scale_factor)
        ax.plot(
            energy_eV,
            scale_factor * J_plot[alpha, beta, :],
            lw=_curve_style_for_selection(curve_cfg, "pair", selection_position, "lw", ps.get("lw", 1.5)),
            label=label,
            color=_curve_style_for_selection(curve_cfg, "pair", selection_position, "color", color),
            linestyle=_curve_style_for_selection(
                curve_cfg,
                "pair",
                selection_position,
                "linestyle",
                linestyle,
            ),
            marker=_curve_style_for_selection(curve_cfg, "pair", selection_position, "marker", None),
            alpha=_curve_style_for_selection(curve_cfg, "pair", selection_position, "alpha", None),
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(_resolve_ylabel(ps, rf"$J_{{\alpha\beta}}(\omega)$ ({unit_label})"))

    title_template = ps.get(
        "title", r"Spectral Density $J_{\alpha\beta}(\omega)$"
    )
    ax.set_title(title_template)

    if yscale == "log":
        ax.set_yscale("log")
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    x_range = ps.get("x_range_eV", None)
    if x_range is not None:
        ax.set_xlim(x_range)

    y_range = ps.get("y_range", None)
    if y_range is not None:
        ax.set_ylim(y_range)

    _apply_y_sci_formatting(ax, ps)

    if ps.get("grid", True):
        ax.grid(True, alpha=0.3)

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid pair indices were plotted.")
    fig.tight_layout()
    return fig


def _plot_scan_layout(J_eV, energy_eV, cfg, observer_distances_nm=None, ax=None, curve_cfg=None):
    """Plot J(ω) for fixed-source scan data."""
    ps = cfg.plot_settings
    J_plot, unit_label = _convert_spectral_density_for_plot(
        J_eV,
        ps.get("spectral_density_unit", "eV"),
    )

    scan_indices, scan_distances_nm = _resolve_scan_indices(
        ps,
        observer_distances_nm,
        J_eV.shape[0],
    )
    yscale = ps.get("yscale", "linear")
    scale_factors = _resolve_curve_multipliers(
        ps,
        primary_key="scan_multipliers",
        legacy_key="separation_scale_factors",
        count=len(scan_indices),
    )
    colors, linestyles = _resolve_curve_styles(ps, prefix="scan", count=len(scan_indices))

    if ax is None:
        fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
    else:
        fig = ax.figure

    for selection_position, (idx, distance_nm, scale_factor, color, linestyle) in enumerate(zip(
        scan_indices,
        scan_distances_nm,
        scale_factors,
        colors,
        linestyles,
    )):
        if idx >= J_eV.shape[0]:
            logger.warning(f"Scan index {idx} out of range (max {J_eV.shape[0] - 1}), skipping.")
            continue

        _validate_curve_multiplier(
            scale_factor,
            yscale=yscale,
            setting_name="plot_settings.scan_multipliers",
        )

        if distance_nm is None and observer_distances_nm is not None:
            distance_nm = float(np.asarray(observer_distances_nm, dtype=float)[idx])
        label_template = ps.get("scan_label_template", ps.get("label_template", "R = {distance_nm:.1f} nm"))
        try:
            label = label_template.format(index=idx, distance_nm=distance_nm, Rx=distance_nm)
        except (KeyError, TypeError, ValueError):
            label = f"Observer {idx}"
            if distance_nm is not None:
                label = f"{label} ({distance_nm:g} nm)"
        label = _with_curve_label(label, curve_cfg)
        label = _format_scaled_label(label, scale_factor)
        ax.plot(
            energy_eV,
            scale_factor * J_plot[idx, :],
            lw=_curve_style_for_selection(curve_cfg, "scan", selection_position, "lw", ps.get("lw", 1.5)),
            label=label,
            color=_curve_style_for_selection(curve_cfg, "scan", selection_position, "color", color),
            linestyle=_curve_style_for_selection(
                curve_cfg,
                "scan",
                selection_position,
                "linestyle",
                linestyle,
            ),
            marker=_curve_style_for_selection(curve_cfg, "scan", selection_position, "marker", None),
            alpha=_curve_style_for_selection(curve_cfg, "scan", selection_position, "alpha", None),
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(_resolve_ylabel(ps, rf"$J(\omega)$ ({unit_label})"))
    ax.set_title(ps.get("title", r"Spectral Density $J(\omega)$"))

    if yscale == "log":
        ax.set_yscale("log")
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    x_range = ps.get("x_range_eV", None)
    if x_range is not None:
        ax.set_xlim(x_range)
    y_range = ps.get("y_range", None)
    if y_range is not None:
        ax.set_ylim(y_range)

    _apply_y_sci_formatting(ax, ps)

    if ps.get("grid", True):
        ax.grid(True, alpha=0.3)

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid scan indices were plotted.")
    fig.tight_layout()
    return fig


def _resolve_single_input_path(cfg) -> Path:
    input_path = Path(cfg.input_file)
    if not input_path.is_absolute():
        input_path = Path(hydra.utils.get_original_cwd()) / input_path
    return input_path


def _plot_dataset_on_axes(data: dict, cfg, ax, curve_cfg=None) -> None:
    J_eV = data["J_eV"]
    energy_eV = data["energy_eV"]
    gf_layout = data["gf_layout"]
    logger.info(f"GF layout: {gf_layout}, J shape: {J_eV.shape}")

    if gf_layout == "separation":
        _plot_separation_layout(J_eV, energy_eV, data["Rx_nm"], cfg, ax=ax, curve_cfg=curve_cfg)
        return
    if gf_layout in {"pair", "ring_circulant"}:
        _plot_pair_layout(
            J_eV,
            energy_eV,
            cfg,
            data.get("emitter_positions_nm", None),
            ax=ax,
            curve_cfg=curve_cfg,
        )
        return
    if gf_layout == "scan":
        _plot_scan_layout(
            J_eV,
            energy_eV,
            cfg,
            data.get("observer_distances_nm", None),
            ax=ax,
            curve_cfg=curve_cfg,
        )
        return
    raise ValueError(f"Unknown GF layout: {gf_layout}")


def _iter_input_curves(cfg) -> list[Any]:
    curves = cfg.get("curves", None)
    if curves:
        return list(curves)
    return []


# ---------------------------------------------------------------------------
#  Hydra CLI entry point
# ---------------------------------------------------------------------------


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("plots", __file__)

@hydra.main(
    config_path=HYDRA_CONFIG_PATH,
    config_name="plt_spec_dens_direct_sg",
    version_base=None,
)
def plot_spectral_density(cfg=None) -> None:
    """Plot spectral density from pre-computed HDF5 data.

    This is the Hydra CLI entry point.  Configuration is loaded from
    ``configs/plots/plt_spec_dens_direct_sg.yaml`` by default.
    """
    if cfg is None:
        raise ValueError("Hydra did not provide a plotting configuration.")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("Plotting spectral density")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Apply font config ---
    _apply_font_config(cfg)

    # --- Plot ---
    ps = cfg.plot_settings
    input_curves = _iter_input_curves(cfg)
    if input_curves:
        fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
        for curve_cfg in input_curves:
            input_path = _resolve_input_path(curve_cfg)
            logger.info(f"Loading spectral density from: {input_path}")
            data = _load_spectral_density_h5(str(input_path))
            _plot_dataset_on_axes(data, cfg, ax, curve_cfg=curve_cfg)
    else:
        input_path = _resolve_single_input_path(cfg)
        logger.info(f"Loading spectral density from: {input_path}")
        data = _load_spectral_density_h5(str(input_path))
        fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
        _plot_dataset_on_axes(data, cfg, ax)

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid spectral-density curves were plotted.")
    fig.tight_layout()

    # --- Save ---
    filename = ps.get("filename", "spectral_density.png")
    dpi = ps.get("dpi", 300)
    plot_filepath = output_dir / filename

    if ps.get("save_plot", True):
        fig.savefig(plot_filepath, dpi=dpi, bbox_inches="tight")
        logger.success(f"Saved plot to: {plot_filepath}")
    else:
        logger.info("save_plot=False; plot not saved.")

    plt.close(fig)


if __name__ == "__main__":
    plot_spectral_density()
