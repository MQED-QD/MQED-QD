"""Plot physical DBR couplings from separation-indexed Green tensors.

The optional ``rx_selection`` config selects arbitrary, disjoint separations
while preserving the requested order. For example, a single plot can combine
near-field and far-field samples::

    rx_selection:
      values_nm: [1, 2, 5, 10, 20, 50, 500, 600, 700, 1000]
      nearest: false
      tolerance_nm: 1.0e-9

Use ``values_nm: null`` (the default) to retain the complete input ``Rx_nm``
grid. Set ``nearest: true`` only when requested values need not lie exactly on
the stored grid; requested values, selected values, source indices, and deltas
are then recorded in the output provenance.
"""
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple

import h5py
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.SI_unit import D2CMM, c, eps0, eV_to_J, hbar
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.orientation import resolve_angle_deg, spherical_to_cartesian_dipole


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(val) for val in value]
    if hasattr(value, "items"):
        return {key: _to_plain(val) for key, val in value.items()}
    return value


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _require_separation_layout(gf_layout: Any) -> None:
    layout = str(gf_layout)
    if layout != "separation":
        raise ValueError(
            "plot_dbr_couplings supports only gf_layout='separation'; "
            f"got gf_layout={layout!r}."
        )


def _select_green_component(gf_data: Dict[str, Any], component: str) -> np.ndarray:
    component_key = str(component).strip().lower()
    if component_key == "total":
        return np.asarray(gf_data["G_total"])
    if component_key == "vacuum":
        return np.asarray(gf_data["G_vac"])
    if component_key in {"structure", "scattered"}:
        if "G_structure" in gf_data:
            return np.asarray(gf_data["G_structure"])
        return np.asarray(gf_data["G_total"]) - np.asarray(gf_data["G_vac"])
    raise ValueError("green_component must be 'total', 'vacuum', 'structure', or 'scattered'.")


def select_energy_index(energy_eV: np.ndarray, selection_cfg: Any) -> Dict[str, Any]:
    """Select one energy grid point by index or requested value.

    Args:
        energy_eV: One-dimensional available energy grid in eV.
        selection_cfg: Mapping with either ``index`` or ``value_eV``. When
            selecting by value, ``nearest`` controls whether the nearest grid
            point may be used instead of requiring an exact/tolerance match.

    Returns:
        Provenance dictionary with selected index/value and request metadata.
    """
    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("energy_eV must be a non-empty 1D array.")
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy_eV must contain only finite values.")

    cfg = _to_plain(selection_cfg or {})
    index_raw = _get(cfg, "index", None)
    value_raw = _get(cfg, "value_eV", None)
    if index_raw is not None and value_raw is not None:
        raise ValueError("Select energy by either index or value_eV, not both.")

    if index_raw is not None:
        index = int(index_raw)
        if index < 0 or index >= energy.size:
            raise IndexError(
                f"energy_selection.index={index} is out of bounds for {energy.size} energies."
            )
        return {
            "selected_energy_index": index,
            "selected_energy_eV": float(energy[index]),
            "requested_energy_eV": None,
            "energy_selection_mode": "index",
            "energy_selection_nearest": False,
            "energy_selection_delta_eV": 0.0,
        }

    if value_raw is None:
        index = 0
        return {
            "selected_energy_index": index,
            "selected_energy_eV": float(energy[index]),
            "requested_energy_eV": None,
            "energy_selection_mode": "index_default",
            "energy_selection_nearest": False,
            "energy_selection_delta_eV": 0.0,
        }

    requested = float(value_raw)
    nearest = bool(_get(cfg, "nearest", True))
    tolerance = float(_get(cfg, "tolerance_eV", 1e-12))
    deltas = np.abs(energy - requested)
    index = int(np.argmin(deltas))
    delta = float(deltas[index])
    if (not nearest) and delta > tolerance:
        raise ValueError(
            f"Requested energy {requested:g} eV is not on the grid within "
            f"tolerance {tolerance:g} eV; nearest is {energy[index]:g} eV."
        )
    return {
        "selected_energy_index": index,
        "selected_energy_eV": float(energy[index]),
        "requested_energy_eV": requested,
        "energy_selection_mode": "value_nearest" if nearest else "value_exact",
        "energy_selection_nearest": nearest,
        "energy_selection_delta_eV": delta,
    }


def select_rx_indices(Rx_nm: np.ndarray, selection_cfg: Any) -> Dict[str, Any]:
    """Select ordered separation-grid rows by indices or physical values.

    Configure ``rx_selection.values_nm`` with any disjoint near-/far-field
    values, for example ``[1, 2, 5, 10, 50, 500, 700, 1000]``. Exact matching
    within ``tolerance_nm`` is the default. With ``nearest: true``, each
    requested value maps to its nearest stored grid point. Alternatively,
    ``rx_selection.indices`` selects source rows directly. Requested order and
    duplicates are preserved. Null or empty selections retain the full grid.

    Args:
        Rx_nm: One-dimensional available separation grid in nm.
        selection_cfg: Mapping containing either ``indices`` or ``values_nm``,
            plus optional ``nearest`` and ``tolerance_nm`` settings.

    Returns:
        Provenance dictionary containing ordered selected indices/values,
        requested values, matching deltas, and selection settings.
    """
    rx = np.asarray(Rx_nm, dtype=float)
    if rx.ndim != 1 or rx.size == 0:
        raise ValueError("Rx_nm must be a non-empty 1D array.")
    if not np.all(np.isfinite(rx)):
        raise ValueError("Rx_nm must contain only finite values.")

    cfg = _to_plain(selection_cfg or {})
    indices_raw = _get(cfg, "indices", None)
    values_raw = _get(cfg, "values_nm", None)

    indices = np.asarray([] if indices_raw is None else indices_raw)
    values = np.asarray([] if values_raw is None else values_raw, dtype=float)
    if indices.ndim == 0:
        indices = indices.reshape(1)
    if values.ndim == 0:
        values = values.reshape(1)
    if indices.ndim != 1:
        raise ValueError("rx_selection.indices must be an integer or a 1D list of integers.")
    if values.ndim != 1:
        raise ValueError("rx_selection.values_nm must be a number or a 1D list of numbers.")
    if indices.size and values.size:
        raise ValueError("Select Rx by either indices or values_nm, not both.")

    tolerance = float(_get(cfg, "tolerance_nm", 1e-9))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rx_selection.tolerance_nm must be finite and nonnegative.")
    nearest = bool(_get(cfg, "nearest", False))

    if indices.size:
        try:
            indices_float = indices.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("rx_selection.indices must contain only integers.") from exc
        if not np.all(np.isfinite(indices_float)) or not np.all(indices_float == np.floor(indices_float)):
            raise ValueError("rx_selection.indices must contain only finite integers.")
        selected_indices = indices_float.astype(int)
        invalid = selected_indices[(selected_indices < 0) | (selected_indices >= rx.size)]
        if invalid.size:
            raise IndexError(
                f"rx_selection index {int(invalid[0])} is out of bounds for {rx.size} separations."
            )
        selected_values = rx[selected_indices]
        return {
            "selected_Rx_indices": selected_indices,
            "selected_Rx_nm": selected_values,
            "requested_Rx_nm": None,
            "rx_selection_delta_nm": np.zeros(selected_indices.size, dtype=float),
            "rx_selection_mode": "indices",
            "rx_selection_nearest": False,
            "rx_selection_tolerance_nm": tolerance,
        }

    if not values.size:
        selected_indices = np.arange(rx.size, dtype=int)
        return {
            "selected_Rx_indices": selected_indices,
            "selected_Rx_nm": rx.copy(),
            "requested_Rx_nm": None,
            "rx_selection_delta_nm": np.zeros(rx.size, dtype=float),
            "rx_selection_mode": "all",
            "rx_selection_nearest": False,
            "rx_selection_tolerance_nm": tolerance,
        }

    if not np.all(np.isfinite(values)):
        raise ValueError("rx_selection.values_nm must contain only finite values.")

    selected_indices_list = []
    deltas = []
    for requested in values:
        differences = np.abs(rx - requested)
        nearest_index = int(np.argmin(differences))
        if nearest:
            selected_index = nearest_index
        else:
            matches = np.flatnonzero(np.isclose(rx, requested, rtol=0.0, atol=tolerance))
            if not matches.size:
                raise ValueError(
                    f"Requested Rx {requested:g} nm is not on the grid within tolerance "
                    f"{tolerance:g} nm; nearest is Rx_nm[{nearest_index}]={rx[nearest_index]:g} nm "
                    f"(delta {differences[nearest_index]:g} nm)."
                )
            selected_index = int(matches[0])
        selected_indices_list.append(selected_index)
        deltas.append(float(abs(rx[selected_index] - requested)))

    selected_indices = np.asarray(selected_indices_list, dtype=int)
    return {
        "selected_Rx_indices": selected_indices,
        "selected_Rx_nm": rx[selected_indices],
        "requested_Rx_nm": values,
        "rx_selection_delta_nm": np.asarray(deltas, dtype=float),
        "rx_selection_mode": "values_nearest" if nearest else "values_exact",
        "rx_selection_nearest": nearest,
        "rx_selection_tolerance_nm": tolerance,
    }


def resolve_dipole_orientations(cfg: Any) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Resolve donor and acceptor dipole unit vectors from theta/phi config."""
    plain = _to_plain(cfg)
    orientations = _get(plain, "orientations", {})
    donor = _get(orientations, "donor", {})
    acceptor = _get(orientations, "acceptor", {})

    theta_donor = resolve_angle_deg(_get(donor, "theta_deg", _get(plain, "theta_D_deg", 90.0)))
    phi_donor = resolve_angle_deg(_get(donor, "phi_deg", _get(plain, "phi_D_deg", 0.0)))
    theta_acceptor = resolve_angle_deg(
        _get(acceptor, "theta_deg", _get(plain, "theta_A_deg", theta_donor))
    )
    phi_acceptor = resolve_angle_deg(_get(acceptor, "phi_deg", _get(plain, "phi_A_deg", phi_donor)))

    p_donor = np.asarray(spherical_to_cartesian_dipole(theta_donor, phi_donor), dtype=float)
    p_acceptor = np.asarray(spherical_to_cartesian_dipole(theta_acceptor, phi_acceptor), dtype=float)
    return p_donor, p_acceptor, theta_donor, phi_donor, theta_acceptor, phi_acceptor


def compute_dbr_couplings(
    G_slice: np.ndarray,
    Rx_nm: np.ndarray,
    energy_eV: float,
    p_donor: np.ndarray,
    p_acceptor: np.ndarray,
    mu_D_debye: float,
    mu_A_debye: float,
) -> Dict[str, np.ndarray]:
    r"""Compute physical DBR couplings versus separation.

    Args:
        G_slice: Selected Green tensor slice, shape ``(K, 3, 3)``.
        Rx_nm: Separation grid in nm, shape ``(K,)``.
        energy_eV: Selected transition energy in eV.
        p_donor: Donor dipole unit vector, shape ``(3,)``.
        p_acceptor: Acceptor dipole unit vector, shape ``(3,)``.
        mu_D_debye: Donor dipole magnitude in Debye.
        mu_A_debye: Acceptor dipole magnitude in Debye.

    Returns:
        Dictionary containing projected Green data and signed/absolute
        ``V_eV``, ``hbarGamma_eV``, and ``Gamma_s_inv`` arrays.
    """
    G = np.asarray(G_slice)
    rx = np.asarray(Rx_nm, dtype=float)
    if G.ndim != 3 or G.shape[-2:] != (3, 3):
        raise ValueError(f"G_slice must have shape (K, 3, 3), got {G.shape}.")
    if rx.shape != (G.shape[0],):
        raise ValueError(f"Rx_nm must have shape ({G.shape[0]},), got {rx.shape}.")
    if not np.all(np.isfinite(rx)):
        raise ValueError("Rx_nm must contain only finite values.")
    if not np.all(np.isfinite(G)):
        invalid_count = int(np.size(G) - np.count_nonzero(np.isfinite(G)))
        raise ValueError(
            f"Selected Green tensor contains {invalid_count} non-finite values; "
            "repair the input before DBR coupling analysis."
        )

    p_d = np.asarray(p_donor, dtype=float).reshape(3)
    p_a = np.asarray(p_acceptor, dtype=float).reshape(3)
    if not np.isfinite(energy_eV):
        raise ValueError("energy_eV must be finite.")
    if not np.all(np.isfinite(p_d)) or not np.all(np.isfinite(p_a)):
        raise ValueError("Dipole orientation vectors must contain only finite values.")
    if not np.isfinite(mu_D_debye) or not np.isfinite(mu_A_debye):
        raise ValueError("Dipole magnitudes must be finite.")
    projected_G = np.einsum("a,kab,b->k", p_a, G, p_d)
    omega = float(energy_eV) * eV_to_J / hbar
    mu_D = float(mu_D_debye) * D2CMM
    mu_A = float(mu_A_debye) * D2CMM
    prefactor = (omega ** 2 / (eps0 * c ** 2)) * mu_D * mu_A

    V_eV = -(prefactor * np.real(projected_G)) / eV_to_J
    hbarGamma_eV = +(2.0 * prefactor * np.imag(projected_G)) / eV_to_J
    Gamma_s_inv = hbarGamma_eV * eV_to_J / hbar
    return {
        "Rx_nm": rx,
        "projected_G": projected_G,
        "projected_G_real": np.real(projected_G),
        "projected_G_imag": np.imag(projected_G),
        "V_eV": V_eV,
        "hbarGamma_eV": hbarGamma_eV,
        "Gamma_s_inv": Gamma_s_inv,
        "abs_V_eV": np.abs(V_eV),
        "abs_hbarGamma_eV": np.abs(hbarGamma_eV),
        "abs_Gamma_s_inv": np.abs(Gamma_s_inv),
    }


def _save_h5(filepath: Path, data: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as h5:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                h5.create_dataset(key, data=value)
            elif isinstance(value, str):
                h5.attrs[key] = value
            elif value is not None:
                h5.attrs[key] = value


def _write_csv_metadata(handle: TextIO, provenance: Dict[str, Any]) -> None:
    for key, value in provenance.items():
        if isinstance(value, np.ndarray):
            formatted = " ".join(f"{item:.17g}" for item in value.ravel())
        elif value is None:
            formatted = "null"
        else:
            formatted = str(value).replace("\n", " ")
        handle.write(f"# {key}: {formatted}\n")


def _save_csv(
    filepath: Path,
    data: Dict[str, np.ndarray],
    provenance: Dict[str, Any],
) -> None:
    columns = [
        "Rx_nm",
        "projected_G_real",
        "projected_G_imag",
        "V_eV",
        "hbarGamma_eV",
        "Gamma_s_inv",
        "abs_V_eV",
        "abs_hbarGamma_eV",
        "abs_Gamma_s_inv",
    ]
    rows = np.column_stack([np.asarray(data[column]) for column in columns])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        _write_csv_metadata(handle, provenance)
        np.savetxt(handle, rows, delimiter=",")


def _plot_couplings(
    results: Dict[str, np.ndarray],
    cfg: Any,
    filepath: Path,
    provenance: Dict[str, Any],
) -> None:
    plot_cfg = _to_plain(_get(cfg, "plot", _get(cfg, "plot_settings", {})))
    use_absolute = bool(_get(plot_cfg, "absolute", _get(cfg, "absolute", False)))
    v_key = "abs_V_eV" if use_absolute else "V_eV"
    gamma_key = "abs_hbarGamma_eV" if use_absolute else "hbarGamma_eV"

    figsize = _get(plot_cfg, "figsize", [6.5, 6.0])
    dpi = int(_get(plot_cfg, "dpi", 150))
    style = _get(plot_cfg, "style", {}) or {}
    color_v = _get(style, "V_color", "tab:blue")
    color_gamma = _get(style, "gamma_color", "tab:orange")
    linestyle = _get(style, "linestyle", "-")
    marker = _get(style, "marker", None)
    linewidth = float(_get(style, "linewidth", 1.8))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(
        results["Rx_nm"], results[v_key], color=color_v, linestyle=linestyle,
        marker=marker, linewidth=linewidth, label=_get(plot_cfg, "V_label", "V")
    )
    axes[1].plot(
        results["Rx_nm"], results[gamma_key], color=color_gamma, linestyle=linestyle,
        marker=marker, linewidth=linewidth, label=_get(plot_cfg, "gamma_label", "hbarGamma")
    )

    axes[0].set_ylabel(_get(plot_cfg, "V_ylabel", r"$V_{ij}$ (eV)"))
    axes[1].set_ylabel(_get(plot_cfg, "gamma_ylabel", r"$\hbar\Gamma_{ij}$ (eV)"))
    axes[1].set_xlabel(_get(plot_cfg, "xlabel", r"$R_x$ (nm)"))
    title = _get(plot_cfg, "title", None)
    if title:
        axes[0].set_title(title)

    xscale = _get(plot_cfg, "xscale", "linear")
    yscale = _get(plot_cfg, "yscale", "linear")
    for ax in axes:
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if bool(_get(plot_cfg, "grid", True)):
            ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    xlim = _get(plot_cfg, "xlim", _get(plot_cfg, "x_range_nm", None))
    if xlim is not None:
        axes[1].set_xlim(float(xlim[0]), float(xlim[1]))
    v_ylim = _get(plot_cfg, "V_ylim", None)
    gamma_ylim = _get(plot_cfg, "gamma_ylim", None)
    if v_ylim is not None:
        axes[0].set_ylim(float(v_ylim[0]), float(v_ylim[1]))
    if gamma_ylim is not None:
        axes[1].set_ylim(float(gamma_ylim[0]), float(gamma_ylim[1]))

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    png_metadata = {
        "Title": "MQED-QD DBR physical couplings",
        "Description": "; ".join(
            f"{key}={value}"
            for key, value in provenance.items()
            if not isinstance(value, np.ndarray)
        ),
        "Software": "MQED-QD",
    }
    fig.savefig(filepath, dpi=dpi, metadata=png_metadata)
    plt.close(fig)


def run_from_config(cfg: Any, output_dir: Path, original_cwd: Optional[Path] = None) -> Path:
    """Run DBR coupling analysis from a Hydra/OmegaConf-compatible config.

    Args:
        cfg: Configuration mapping or OmegaConf object.
        output_dir: Directory where HDF5, CSV, and PNG files are written.
        original_cwd: Base directory for resolving relative input paths.

    Returns:
        Path to the written HDF5 result file.
    """
    plain_cfg = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    original_cwd = original_cwd or Path.cwd()
    output_dir = Path(output_dir)

    input_path = Path(str(plain_cfg["input_file"]))
    if not input_path.is_absolute():
        input_path = original_cwd / input_path

    gf_data = load_gf_h5(str(input_path))
    _require_separation_layout(gf_data["gf_layout"])
    energy_eV = np.asarray(gf_data["energy_eV"], dtype=float)
    Rx_nm = np.asarray(gf_data["Rx_nm"], dtype=float)
    green_component = str(_get(plain_cfg, "green_component", "total")).strip().lower()
    G = _select_green_component(gf_data, green_component)

    if G.ndim != 4 or G.shape[-2:] != (3, 3):
        raise ValueError(f"Separation Green tensor must have shape (M, K, 3, 3), got {G.shape}.")
    if G.shape[0] != energy_eV.size:
        raise ValueError("Green tensor energy axis does not match energy_eV.")
    if G.shape[1] != Rx_nm.size:
        raise ValueError("Green tensor separation axis does not match Rx_nm.")

    selection = select_energy_index(energy_eV, _get(plain_cfg, "energy_selection", {}))
    selected_index = int(selection["selected_energy_index"])
    selected_G_full = G[selected_index]
    rx_selection = select_rx_indices(Rx_nm, _get(plain_cfg, "rx_selection", {}))
    selected_rx_indices = np.asarray(rx_selection["selected_Rx_indices"], dtype=int)
    selected_Rx_nm = Rx_nm[selected_rx_indices]
    selected_G = selected_G_full[selected_rx_indices]

    p_donor, p_acceptor, theta_d, phi_d, theta_a, phi_a = resolve_dipole_orientations(plain_cfg)
    mu_D_debye = float(_get(plain_cfg, "mu_D_debye", _get(plain_cfg, "mu_debye", 1.0)))
    mu_A_debye = float(_get(plain_cfg, "mu_A_debye", _get(plain_cfg, "mu_debye", mu_D_debye)))
    results = compute_dbr_couplings(
        selected_G,
        selected_Rx_nm,
        float(selection["selected_energy_eV"]),
        p_donor,
        p_acceptor,
        mu_D_debye,
        mu_A_debye,
    )

    output_prefix = str(_get(plain_cfg, "output_prefix", "dbr_couplings"))
    stem = f"{output_prefix}_E_{selection['selected_energy_eV']:.6g}eV"
    h5_path = output_dir / f"{stem}.h5"
    csv_path = output_dir / f"{stem}.csv"
    plot_cfg = _to_plain(_get(plain_cfg, "plot", _get(plain_cfg, "plot_settings", {})))
    png_name = str(_get(plot_cfg, "filename", f"{stem}.png"))
    png_path = output_dir / png_name

    provenance = {
        "input_path": str(input_path),
        "green_component": green_component,
        "gf_layout": str(gf_data["gf_layout"]),
        "energy_selection_mode": str(selection["energy_selection_mode"]),
        "energy_selection_nearest": int(bool(selection["energy_selection_nearest"])),
        "energy_selection_delta_eV": float(selection["energy_selection_delta_eV"]),
        "selected_energy_index": selected_index,
        "selected_energy_eV": float(selection["selected_energy_eV"]),
        "requested_energy_eV": selection["requested_energy_eV"],
        "rx_selection_mode": str(rx_selection["rx_selection_mode"]),
        "rx_selection_nearest": int(bool(rx_selection["rx_selection_nearest"])),
        "rx_selection_tolerance_nm": float(rx_selection["rx_selection_tolerance_nm"]),
        "requested_Rx_nm": rx_selection["requested_Rx_nm"],
        "selected_Rx_indices": selected_rx_indices,
        "selected_Rx_nm": np.asarray(rx_selection["selected_Rx_nm"], dtype=float),
        "rx_selection_delta_nm": np.asarray(rx_selection["rx_selection_delta_nm"], dtype=float),
        "requested_Rx_nm_csv": (
            "null" if rx_selection["requested_Rx_nm"] is None else
            ",".join(f"{value:.17g}" for value in rx_selection["requested_Rx_nm"])
        ),
        "selected_Rx_indices_csv": ",".join(str(index) for index in selected_rx_indices),
        "selected_Rx_nm_csv": ",".join(
            f"{value:.17g}" for value in rx_selection["selected_Rx_nm"]
        ),
        "rx_selection_delta_nm_csv": ",".join(
            f"{value:.17g}" for value in rx_selection["rx_selection_delta_nm"]
        ),
        "p_donor": p_donor,
        "p_acceptor": p_acceptor,
        "mu_D_debye": mu_D_debye,
        "mu_A_debye": mu_A_debye,
        "mu_D_Cm": mu_D_debye * D2CMM,
        "mu_A_Cm": mu_A_debye * D2CMM,
        "theta_D_deg": theta_d,
        "phi_D_deg": phi_d,
        "theta_A_deg": theta_a,
        "phi_A_deg": phi_a,
        "omega_rad_s": float(selection["selected_energy_eV"]) * eV_to_J / hbar,
        "formula_V_eV": "-(omega**2/(eps0*c**2))*mu_D*mu_A*Re(projected_G)/eV_to_J",
        "formula_hbarGamma_eV": "+2*(omega**2/(eps0*c**2))*mu_D*mu_A*Im(projected_G)/eV_to_J",
        "formula_Gamma_s_inv": "hbarGamma_eV*eV_to_J/hbar",
    }
    h5_data = dict(results)
    h5_data.update(provenance)
    _save_h5(h5_path, h5_data)
    _save_csv(csv_path, results, provenance)
    _plot_couplings(results, plain_cfg, png_path, provenance)
    logger.success(f"Saved DBR coupling analysis to: {h5_path}")
    return h5_path


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("analysis", __file__)


@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="plot_dbr_couplings", version_base=None)
def main(cfg: Any) -> None:
    """Hydra CLI entry point for DBR coupling plotting."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()
    logger.info("Plotting physical DBR couplings from projected Green tensors")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    run_from_config(cfg, output_dir, Path(hydra.utils.get_original_cwd()))


if __name__ == "__main__":
    main()
