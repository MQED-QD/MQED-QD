from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.SI_unit import D2CMM, c, eps0, eV_to_J, hbar
from mqed.utils.emitter_geometry import normalize_orientation_vectors
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


def _normalize_vectors(vectors: np.ndarray, expected_count: int) -> np.ndarray:
    return normalize_orientation_vectors(vectors, expected_count, allow_single_vector=True)


def resolve_emitter_orientations(
    config: Any,
    n_emitters: int,
    stored_orientations: np.ndarray | None = None,
) -> np.ndarray:
    cfg = _to_plain(config)
    orientations_cfg = _get(cfg, "orientations", {})
    explicit = _get(cfg, "emitter_orientations", None)
    if explicit is None:
        explicit = _get(orientations_cfg, "emitter_orientations", None)
    if explicit is None:
        explicit = _get(orientations_cfg, "emitters", None)
    if explicit is None:
        explicit = _get(orientations_cfg, "U_list", None)
    if explicit is not None:
        return _normalize_vectors(np.asarray(explicit, dtype=float), n_emitters)

    has_angle_config = any(
        _get(orientations_cfg, key, None) is not None or _get(cfg, key, None) is not None
        for key in ("theta_deg", "phi_deg")
    )
    if not has_angle_config and stored_orientations is not None:
        return _normalize_vectors(np.asarray(stored_orientations, dtype=float), n_emitters)

    theta_raw = _get(orientations_cfg, "theta_deg", _get(cfg, "theta_deg", 90.0))
    phi_raw = _get(orientations_cfg, "phi_deg", _get(cfg, "phi_deg", 0.0))

    def resolve_many(raw_value):
        if isinstance(raw_value, (list, tuple)):
            return [resolve_angle_deg(value) for value in raw_value]
        return resolve_angle_deg(raw_value)

    theta = resolve_many(theta_raw)
    phi = resolve_many(phi_raw)
    vectors = spherical_to_cartesian_dipole(theta, phi)
    return _normalize_vectors(vectors, n_emitters)


def project_pair_green(G_pair: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    G_pair = np.asarray(G_pair)
    orientations = np.asarray(orientations, dtype=float)
    if G_pair.ndim != 5 or G_pair.shape[-2:] != (3, 3):
        raise ValueError(f"Pair Green tensor must have shape (M,N,N,3,3), got {G_pair.shape}.")
    if orientations.shape != (G_pair.shape[1], 3):
        raise ValueError(
            f"Orientations shape {orientations.shape} does not match pair Green tensor N={G_pair.shape[1]}."
        )
    return np.einsum("ia,mijab,jb->mij", orientations, G_pair, orientations)


DEFAULT_MAX_CIRCULANT_EXPANSION_BYTES = 2 * 1024**3


def circulant_row_to_pair(
    projected_row: np.ndarray,
    *,
    max_allocation_bytes: int = DEFAULT_MAX_CIRCULANT_EXPANSION_BYTES,
) -> np.ndarray:
    """Expand projected circulant rows to scalar pair matrices."""
    row = np.asarray(projected_row, dtype=complex)
    if row.ndim != 2 or row.shape[1] == 0:
        raise ValueError(f"Circulant Green row must have shape (M,N), got {row.shape}.")
    if max_allocation_bytes <= 0:
        raise ValueError("max_allocation_bytes must be positive.")
    energy_count, emitter_count = row.shape
    output_bytes = energy_count * emitter_count**2 * row.dtype.itemsize
    offset_bytes = emitter_count**2 * np.dtype(np.intp).itemsize
    required_bytes = output_bytes + offset_bytes
    if required_bytes > max_allocation_bytes:
        raise ValueError(
            "Expanding the circulant Green row requires approximately "
            f"{required_bytes / 1024**2:.1f} MiB, exceeding the configured "
            f"{max_allocation_bytes / 1024**2:.1f} MiB limit."
        )
    indices = np.arange(row.shape[1])
    offsets = (indices[np.newaxis, :] - indices[:, np.newaxis]) % row.shape[1]
    return row[:, offsets]


def project_separation_green_to_pair(
    G_separation: np.ndarray,
    Rx_nm: np.ndarray,
    n_emitters: int,
    d_nm: float,
    orientations: np.ndarray,
    tolerance_nm: float = 1e-6,
) -> np.ndarray:
    G_separation = np.asarray(G_separation)
    Rx_nm = np.asarray(Rx_nm, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    if G_separation.ndim != 4 or G_separation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Separation Green tensor must have shape (M,K,3,3), got {G_separation.shape}."
        )
    if orientations.shape != (n_emitters, 3):
        raise ValueError(f"Orientations must have shape ({n_emitters}, 3), got {orientations.shape}.")

    projected = np.zeros((G_separation.shape[0], n_emitters, n_emitters), dtype=complex)
    for alpha in range(n_emitters):
        for beta in range(n_emitters):
            separation_nm = abs(alpha - beta) * d_nm
            matches = np.where(np.isclose(Rx_nm, separation_nm, rtol=0.0, atol=tolerance_nm))[0]
            if matches.size == 0:
                nearest = int(np.argmin(np.abs(Rx_nm - separation_nm)))
                raise ValueError(
                    f"No Rx_nm entry for emitter pair ({alpha},{beta}) separation "
                    f"{separation_nm:g} nm within {tolerance_nm:g} nm; nearest is "
                    f"Rx_nm[{nearest}]={Rx_nm[nearest]:g} nm."
                )
            tensor = G_separation[:, int(matches[0]), :, :]
            projected[:, alpha, beta] = np.einsum(
                "a,mab,b->m", orientations[alpha], tensor, orientations[beta]
            )
    return projected


def _separation_tensor_to_pair(
    G_separation: np.ndarray,
    Rx_nm: np.ndarray,
    n_emitters: int,
    d_nm: float,
    tolerance_nm: float = 1e-6,
) -> np.ndarray:
    G_separation = np.asarray(G_separation)
    Rx_nm = np.asarray(Rx_nm, dtype=float)
    if G_separation.ndim != 4 or G_separation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Separation Green tensor must have shape (M,K,3,3), got {G_separation.shape}."
        )
    pair = np.zeros((G_separation.shape[0], n_emitters, n_emitters, 3, 3), dtype=complex)
    for alpha in range(n_emitters):
        for beta in range(n_emitters):
            separation_nm = abs(alpha - beta) * d_nm
            matches = np.where(np.isclose(Rx_nm, separation_nm, rtol=0.0, atol=tolerance_nm))[0]
            if matches.size == 0:
                nearest = int(np.argmin(np.abs(Rx_nm - separation_nm)))
                raise ValueError(
                    f"No Rx_nm entry for emitter pair ({alpha},{beta}) separation "
                    f"{separation_nm:g} nm within {tolerance_nm:g} nm; nearest is "
                    f"Rx_nm[{nearest}]={Rx_nm[nearest]:g} nm."
                )
            pair[:, alpha, beta] = G_separation[:, int(matches[0])]
    return pair


def _project_tensor_pair(G_pair: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    return np.einsum("ia,mijab,jb->mij", orientations, G_pair, orientations)


def self_energy_from_projected_green(
    projected_G: np.ndarray,
    energy_eV: np.ndarray,
    mu_debye: float,
    shift_method: str = "real_green",
) -> np.ndarray:
    projected_G = np.asarray(projected_G, dtype=complex)
    energy_eV = np.asarray(energy_eV, dtype=float)
    if projected_G.ndim != 3:
        raise ValueError(f"projected_G must have shape (M,N,N), got {projected_G.shape}.")
    if projected_G.shape[0] != energy_eV.size:
        raise ValueError("projected_G energy axis does not match energy_eV.")

    mu_si = float(mu_debye) * D2CMM
    omega = energy_eV * eV_to_J / hbar
    prefactor = mu_si * mu_si * omega**2 / (hbar * eps0 * c**2)
    method = str(shift_method).strip().lower()

    if method in {"real_green", "green", "direct"}:
        self_energy_rad_s = prefactor[:, np.newaxis, np.newaxis] * projected_G
    elif method in {"principal_value", "pv", "kramers_kronig"}:
        imag_part = np.imag(projected_G)
        imaginary_rad_s = 1j * prefactor[:, np.newaxis, np.newaxis] * imag_part
        integrand = (omega[:, np.newaxis, np.newaxis] ** 2 / c**2) * imag_part
        real_rad_s = np.zeros_like(imag_part, dtype=float)
        for idx, omega_value in enumerate(omega):
            mask = np.ones_like(omega, dtype=bool)
            mask[idx] = False
            if np.any(mask):
                denominator = omega[mask, np.newaxis, np.newaxis] - omega_value
                real_rad_s[idx] = (mu_si * mu_si / (hbar * np.pi * eps0)) * np.trapz(
                    integrand[mask] / denominator,
                    omega[mask],
                    axis=0,
                )
        self_energy_rad_s = real_rad_s + imaginary_rad_s
    else:
        raise ValueError("shift_method must be 'real_green' or 'principal_value'.")

    return self_energy_rad_s * hbar / eV_to_J


def compute_emission_spectrum(
    self_energy_eV: np.ndarray,
    emission_energy_eV: np.ndarray,
    transition_energy_eV: np.ndarray,
    gamma0_eV: float,
    bright_weights: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    self_energy_eV = np.asarray(self_energy_eV, dtype=complex)
    emission_energy_eV = np.asarray(emission_energy_eV, dtype=float)
    transition_energy_eV = np.atleast_1d(np.asarray(transition_energy_eV, dtype=float))
    if self_energy_eV.ndim != 3 or self_energy_eV.shape[1] != self_energy_eV.shape[2]:
        raise ValueError(f"self_energy_eV must have shape (M,N,N), got {self_energy_eV.shape}.")
    if self_energy_eV.shape[0] != emission_energy_eV.size:
        raise ValueError("self_energy_eV energy axis does not match emission_energy_eV.")

    n_emitters = self_energy_eV.shape[1]
    if bright_weights is None:
        bright = np.ones(n_emitters, dtype=complex) / np.sqrt(n_emitters)
    else:
        bright = np.asarray(bright_weights, dtype=complex)
        if bright.shape != (n_emitters,):
            raise ValueError(f"bright_weights must have shape ({n_emitters},), got {bright.shape}.")
        norm = np.linalg.norm(bright)
        if norm == 0.0:
            raise ValueError("bright_weights must not be the zero vector.")
        bright = bright / norm

    identity = np.eye(n_emitters, dtype=complex)
    spectra = np.empty((transition_energy_eV.size, emission_energy_eV.size), dtype=float)
    gamma0_eV = float(gamma0_eV)

    for transition_index, transition_energy in enumerate(transition_energy_eV):
        for energy_index, emission_energy in enumerate(emission_energy_eV):
            matrix = (
                (emission_energy - transition_energy + 0.5j * gamma0_eV) * identity
                + self_energy_eV[energy_index]
            )
            response = np.linalg.solve(matrix, bright)
            amplitude = np.vdot(bright, response)
            spectra[transition_index, energy_index] = gamma0_eV / (2.0 * np.pi) * abs(amplitude) ** 2

    if normalize:
        max_value = float(np.max(spectra)) if spectra.size else 0.0
        if max_value > 0.0:
            spectra = spectra / max_value
    return spectra


def _validate_separation_green_data(
    primary: np.ndarray,
    vacuum: np.ndarray,
    energy_dataset: h5py.Dataset,
    rx_dataset: h5py.Dataset,
    primary_name: str,
) -> None:
    if primary.ndim != 4 or primary.shape[-2:] != (3, 3):
        raise ValueError(
            "Separation Green tensors must have shape (M,K,3,3); "
            f"got {primary_name} shape {primary.shape}."
        )
    if vacuum.shape != primary.shape:
        raise ValueError(
            "Separation effective Green tensors require matching shapes; "
            f"got {primary_name} {primary.shape} and vacuum {vacuum.shape}."
        )
    if energy_dataset.shape != (primary.shape[0],):
        raise ValueError(
            f"energy_eV length must match M={primary.shape[0]}; got shape {energy_dataset.shape}."
        )
    if rx_dataset.shape != (primary.shape[1],):
        raise ValueError(f"Rx_nm length must match K={primary.shape[1]}; got shape {rx_dataset.shape}.")


def _read_structure_with_fallback(h5: h5py.File, layout: str) -> np.ndarray:
    if "green_function_structure" in h5:
        return h5["green_function_structure"][:]
    total = h5["green_function_total"][:]
    vacuum = h5["green_function_vacuum"][:]
    if total.shape != vacuum.shape:
        raise ValueError(
            "Structure fallback requires matching shapes for total/vacuum; "
            f"got total {total.shape} and vacuum {vacuum.shape}."
        )
    logger.warning("green_function_structure absent for {} layout; using total-vacuum.", layout)
    return total - vacuum


def _apply_effective_pair_convention(
    component_key: str,
    structure: np.ndarray,
    vacuum: np.ndarray,
) -> np.ndarray:
    if component_key == "renormalized_total":
        G = structure + vacuum
    else:
        G = structure + np.real(vacuum)
    diagonal = np.arange(G.shape[1])
    G[:, diagonal, diagonal] = structure[:, diagonal, diagonal]
    return G


def _apply_effective_separation_convention(
    component_key: str,
    structure: np.ndarray,
    vacuum: np.ndarray,
    rx_nm: np.ndarray,
) -> np.ndarray:
    G = np.array(structure, copy=True)
    rx_nm = np.asarray(rx_nm, dtype=float)
    nonzero = ~np.isclose(rx_nm, 0.0, rtol=0.0, atol=1e-9)
    if component_key == "renormalized_total":
        G[:, nonzero] = structure[:, nonzero] + vacuum[:, nonzero]
    else:
        G[:, nonzero] = structure[:, nonzero] + np.real(vacuum[:, nonzero])
    return G


def _apply_effective_ring_convention(
    component_key: str,
    structure: np.ndarray,
    vacuum: np.ndarray,
) -> np.ndarray:
    G = structure + (vacuum if component_key == "renormalized_total" else np.real(vacuum))
    G[:, 0] = structure[:, 0]
    return G


def _read_green_component(input_path: Path, component: str, channel: str = "full") -> dict[str, Any]:
    component_key = str(component).strip().lower()
    channel_key = str(channel).strip().lower()
    if channel_key not in {"full", "te", "tm"}:
        raise ValueError("green_channel must be 'full', 'te', or 'tm'.")
    dataset_by_component = {
        "total": "green_function_total",
        "vacuum": "green_function_vacuum",
        "structure": "green_function_structure",
        "scattered": "green_function_structure",
    }
    dataset_name = dataset_by_component.get(component_key)
    is_effective = component_key in {"varguet_effective", "renormalized_total"}
    if dataset_name is None and not is_effective:
        raise ValueError(
            "green_component must be 'total', 'vacuum', 'structure', 'scattered', "
            "'varguet_effective', or 'renormalized_total'."
        )
    if channel_key in {"te", "tm"} and component_key not in {
        "structure",
        "scattered",
        "varguet_effective",
        "renormalized_total",
    }:
        raise ValueError("green_channel='te' or 'tm' is only defined for structure/effective components.")

    with h5py.File(input_path, "r") as h5:
        layout = h5.attrs.get("gf_layout", "separation")
        if isinstance(layout, bytes):
            layout = layout.decode()
        convention = component_key
        if channel_key in {"te", "tm"}:
            channel_dataset = f"green_function_scattering_{channel_key}"
            if channel_dataset not in h5:
                raise KeyError(f"Missing dataset {channel_dataset!r} in {input_path}.")
            G = h5[channel_dataset][:]
            vacuum = h5["green_function_vacuum"][:]
            if layout == "pair":
                _validate_pair_green_data(
                    G,
                    vacuum,
                    h5["energy_eV"],
                    h5["emitter_positions_nm"],
                    primary_name=channel_dataset,
                )
            elif layout == "separation":
                _validate_separation_green_data(
                    G,
                    vacuum,
                    h5["energy_eV"],
                    h5["Rx_nm"],
                    primary_name=channel_dataset,
                )
            elif layout == "ring_circulant":
                _validate_ring_green_data(
                    G,
                    vacuum,
                    h5["energy_eV"],
                    h5["emitter_positions_nm"],
                    h5.get("emitter_orientations"),
                    h5.attrs.get("green_representation", ""),
                    channel_dataset,
                )
            else:
                raise ValueError(f"green_channel='{channel_key}' is not supported for {layout} layout.")
            convention = f"{channel_key}_scattering_only"
        elif is_effective:
            vacuum = h5["green_function_vacuum"][:]
            structure = _read_structure_with_fallback(h5, layout)
            if layout == "pair":
                _validate_pair_green_data(
                    structure,
                    vacuum,
                    h5["energy_eV"],
                    h5["emitter_positions_nm"],
                    primary_name="structure",
                )
                G = _apply_effective_pair_convention(component_key, structure, vacuum)
            elif layout == "separation":
                _validate_separation_green_data(
                    structure,
                    vacuum,
                    h5["energy_eV"],
                    h5["Rx_nm"],
                    primary_name="structure",
                )
                G = _apply_effective_separation_convention(
                    component_key,
                    structure,
                    vacuum,
                    h5["Rx_nm"][:].astype(float),
                )
            elif layout == "ring_circulant":
                _validate_ring_green_data(
                    structure,
                    vacuum,
                    h5["energy_eV"],
                    h5["emitter_positions_nm"],
                    h5.get("emitter_orientations"),
                    h5.attrs.get("green_representation", ""),
                    "structure",
                )
                G = _apply_effective_ring_convention(component_key, structure, vacuum)
            else:
                raise ValueError(f"green_component='{component_key}' is not supported for {layout} layout.")
        elif dataset_name not in h5:
            if component_key in {"structure", "scattered"}:
                logger.warning(
                    "Requested green_component='{}' but {} is absent; using total-vacuum.",
                    component,
                    dataset_name,
                )
                G = h5["green_function_total"][:] - h5["green_function_vacuum"][:]
            else:
                raise KeyError(f"Missing dataset {dataset_name!r} in {input_path}.")
        else:
            G = h5[dataset_name][:]
        if layout == "ring_circulant" and not is_effective and channel_key not in {"te", "tm"}:
            _validate_ring_green_data(
                G,
                h5["green_function_vacuum"][:],
                h5["energy_eV"],
                h5["emitter_positions_nm"],
                h5.get("emitter_orientations"),
                h5.attrs.get("green_representation", ""),
                dataset_name or "structure",
            )
        data: dict[str, Any] = {
            "G": G,
            "energy_eV": h5["energy_eV"][:].astype(float),
            "gf_layout": layout,
            "green_component": component_key,
            "green_channel": channel_key,
            "green_convention": convention,
        }
        if layout in {"pair", "ring_circulant"}:
            data["emitter_positions_nm"] = h5["emitter_positions_nm"][:].astype(float)
            if "emitter_orientations" in h5:
                data["emitter_orientations"] = h5["emitter_orientations"][:].astype(float)
        else:
            data["Rx_nm"] = h5["Rx_nm"][:].astype(float)
    return data


def _validate_pair_green_data(
    primary: np.ndarray,
    vacuum: np.ndarray,
    energy_dataset: h5py.Dataset,
    positions_dataset: h5py.Dataset,
    primary_name: str,
) -> None:
    if primary.ndim != 5 or primary.shape[-2:] != (3, 3):
        raise ValueError(
            "Pair Green tensors must have shape (M,N,N,3,3); "
            f"got {primary_name} shape {primary.shape}."
        )
    if primary.shape[1] != primary.shape[2]:
        raise ValueError(f"Pair Green emitter axes must be square; got {primary.shape}.")
    if vacuum.shape != primary.shape:
        raise ValueError(
            "Varguet effective Green tensors require matching shapes; "
            f"got {primary_name} {primary.shape} and vacuum {vacuum.shape}."
        )
    energy_shape = energy_dataset.shape
    if energy_shape != (primary.shape[0],):
        raise ValueError(
            f"energy_eV length must match M={primary.shape[0]}; got shape {energy_shape}."
        )
    positions_shape = positions_dataset.shape
    if positions_shape != (primary.shape[1], 3):
        raise ValueError(
            f"emitter_positions_nm must have shape ({primary.shape[1]}, 3); "
            f"got {positions_shape}."
        )


def _validate_ring_green_data(
    primary: np.ndarray,
    vacuum: np.ndarray,
    energy_dataset: h5py.Dataset,
    positions_dataset: h5py.Dataset,
    orientations_dataset: h5py.Dataset | None,
    representation: str | bytes,
    primary_name: str,
) -> None:
    if isinstance(representation, bytes):
        representation = representation.decode()
    if representation != "dipole_projected_scalar_circulant_row":
        raise ValueError(
            "ring_circulant files must declare the dipole-projected scalar representation."
        )
    if primary.ndim != 2 or primary.shape[1] == 0:
        raise ValueError(
            f"Circulant Green rows must have shape (M,N); got {primary_name} shape {primary.shape}."
        )
    if vacuum.shape != primary.shape:
        raise ValueError(
            f"Circulant effective Green rows require matching shapes; got {primary.shape} and {vacuum.shape}."
        )
    if energy_dataset.shape != (primary.shape[0],):
        raise ValueError(f"energy_eV length must match M={primary.shape[0]}.")
    if positions_dataset.shape != (primary.shape[1], 3):
        raise ValueError(f"emitter_positions_nm must have shape ({primary.shape[1]}, 3).")
    if orientations_dataset is None or orientations_dataset.shape != positions_dataset.shape:
        raise ValueError(
            f"emitter_orientations must have shape ({primary.shape[1]}, 3) for ring_circulant data."
        )
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(vacuum)):
        raise ValueError("Circulant Green rows must contain only finite values.")


def _transition_grid(cfg, energy_eV: np.ndarray) -> np.ndarray:
    raw = cfg.get("transition_energy_eV", None)
    if raw is None:
        raw = cfg.get("omega0_eV", None)
    if raw is None:
        transition_cfg = cfg.get("transition_energy_grid_eV", None)
        if transition_cfg is None:
            return np.asarray(energy_eV, dtype=float)
        return np.linspace(
            float(transition_cfg["min"]),
            float(transition_cfg["max"]),
            int(transition_cfg["points"]),
        )
    if isinstance(raw, (int, float)):
        return np.array([float(raw)])
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lower() in {"same", "energy", "emission"}:
            return np.asarray(energy_eV, dtype=float)
        return np.array([float(stripped)])
    return np.asarray(list(raw), dtype=float)


def _bright_weights(cfg, n_emitters: int) -> np.ndarray | None:
    raw = cfg.get("bright_weights", None)
    if raw is None:
        return None
    weights = np.asarray(raw, dtype=complex)
    if weights.shape != (n_emitters,):
        raise ValueError(f"bright_weights must have shape ({n_emitters},), got {weights.shape}.")
    return weights


def _separation_chain_parameters(cfg: Any) -> tuple[int, float, float]:
    n_raw = cfg.get("n_emitters", cfg.get("N_mol", None))
    d_raw = cfg.get("d_nm", None)
    if n_raw is None or d_raw is None:
        raise ValueError(
            "Separation-layout emission analysis requires explicit n_emitters (or N_mol) "
            "and d_nm."
        )
    if isinstance(n_raw, bool):
        raise ValueError("n_emitters must be a positive integer, not a boolean.")
    n_value = float(n_raw)
    if not np.isfinite(n_value) or not n_value.is_integer() or n_value <= 0.0:
        raise ValueError("n_emitters must be a positive integer.")
    n_emitters = int(n_value)
    d_nm = float(d_raw)
    tolerance_nm = float(cfg.get("rx_tolerance_nm", 1e-6))
    if not np.isfinite(d_nm) or d_nm <= 0.0:
        raise ValueError("d_nm must be finite and positive.")
    if not np.isfinite(tolerance_nm) or tolerance_nm < 0.0:
        raise ValueError("rx_tolerance_nm must be finite and non-negative.")
    return n_emitters, d_nm, tolerance_nm


def _save_emission_h5(filepath: Path, data: dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as h5:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                h5.create_dataset(key, data=value)
            elif isinstance(value, str):
                h5.attrs[key] = value
            elif value is not None:
                h5.attrs[key] = value


def run_from_config(cfg: Any, output_dir: Path, original_cwd: Path | None = None) -> Path:
    cfg = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    original_cwd = original_cwd or Path.cwd()
    input_path = Path(str(cfg["input_file"]))
    if not input_path.is_absolute():
        input_path = original_cwd / input_path

    green_component = str(cfg.get("green_component", "total"))
    green_channel = str(cfg.get("green_channel", "full"))
    green_data = _read_green_component(input_path, green_component, green_channel)
    energy_eV = green_data["energy_eV"]
    gf_layout = green_data["gf_layout"]
    G = green_data["G"]

    if gf_layout == "pair":
        n_emitters = G.shape[1]
        orientations = resolve_emitter_orientations(
            cfg,
            n_emitters,
            stored_orientations=green_data.get("emitter_orientations"),
        )
        projected_G = project_pair_green(G, orientations)
    elif gf_layout == "ring_circulant":
        n_emitters = G.shape[1]
        stored_orientations = green_data.get("emitter_orientations")
        if stored_orientations is None:
            raise ValueError("ring_circulant data must store emitter_orientations provenance.")
        orientations = _normalize_vectors(stored_orientations, n_emitters)
        projected_G = circulant_row_to_pair(G)
    elif gf_layout == "separation":
        n_emitters, d_nm, tolerance_nm = _separation_chain_parameters(cfg)
        orientations = resolve_emitter_orientations(cfg, n_emitters)
        projected_G = _project_tensor_pair(
            _separation_tensor_to_pair(
                G,
                green_data["Rx_nm"],
                n_emitters,
                d_nm,
                tolerance_nm=tolerance_nm,
            ),
            orientations,
        )
    else:
        raise ValueError(f"Unknown GF layout: {gf_layout}")

    mu_debye = float(cfg.get("mu_debye", cfg.get("dipole_moment_debye", 1.0)))
    gamma0_eV = float(cfg.get("gamma0_eV", 0.05))
    transition_energy_eV = _transition_grid(cfg, energy_eV)
    shift_method = str(cfg.get("shift_method", "real_green"))
    normalize = bool(cfg.get("normalize", False))
    if not np.all(np.isfinite(projected_G)):
        invalid = np.argwhere(~np.isfinite(projected_G))[0]
        raise FloatingPointError(f"Selected/projected Green tensor contains non-finite data at index {tuple(invalid)}.")
    self_energy_eV = self_energy_from_projected_green(
        projected_G,
        energy_eV,
        mu_debye=mu_debye,
        shift_method=shift_method,
    )
    emission_spectrum = compute_emission_spectrum(
        self_energy_eV,
        energy_eV,
        transition_energy_eV,
        gamma0_eV=gamma0_eV,
        bright_weights=_bright_weights(cfg, self_energy_eV.shape[1]),
        normalize=normalize,
    )

    output_filename = cfg.get("output_filename", None)
    if output_filename is None:
        output_prefix = str(cfg.get("output_prefix", "emission_spectrum"))
        output_filename = (
            f"{output_prefix}_Emin_{energy_eV[0]:.3f}_Emax_{energy_eV[-1]:.3f}_"
            f"{len(energy_eV)}pts.h5"
        )
    output_file = output_dir / str(output_filename)
    result = {
        "emission_spectrum": emission_spectrum,
        "emission_energy_eV": energy_eV,
        "transition_energy_eV": transition_energy_eV,
        "projected_G": projected_G,
        "self_energy_eV": self_energy_eV,
        "emitter_orientations": orientations,
        "gf_layout": str(gf_layout),
        "green_component": green_component,
        "green_channel": green_data["green_channel"],
        "green_convention": green_data["green_convention"],
        "shift_method": shift_method,
        "mu_debye": mu_debye,
        "gamma0_eV": gamma0_eV,
        "normalized": int(normalize),
    }
    if "emitter_positions_nm" in green_data:
        result["emitter_positions_nm"] = green_data["emitter_positions_nm"]
    if "Rx_nm" in green_data:
        result["Rx_nm"] = green_data["Rx_nm"]

    _save_emission_h5(output_file, result)
    logger.success(f"Saved emission spectrum to: {output_file}")
    return output_file


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("analysis", __file__)


@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="emission_spectrum", version_base=None)
def compute_and_save_emission_spectrum(cfg) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()
    logger.info("Computing frequency-domain emission spectrum")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    run_from_config(cfg, output_dir, Path(hydra.utils.get_original_cwd()))


if __name__ == "__main__":
    compute_and_save_emission_spectrum()
