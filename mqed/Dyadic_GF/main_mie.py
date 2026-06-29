"""Run generalized Mie dyadic-Green simulations and save results to HDF5.

This driver mirrors the execution pattern of the project's planar ``main.py``:

1. Build a spectral grid from a YAML/JSON configuration.
2. Resolve the dielectric function in each spherical region at every frequency.
3. Instantiate ``MieGreenFunction`` independently at each energy.
4. Loop over observer positions and assemble total/vacuum/structure Green tensors.
5. Save HDF5 output for post-processing.

Minimal run
-----------
Create an example configuration first::

    python main_mie.py --write-example-config mie_example.yaml

Then run::

    python main_mie.py --config mie_example.yaml

The output HDF5 stores complex arrays directly.  Dataset shapes are:

``G_total``
    ``(n_energy, n_observer, 3, 3)`` Cartesian dyadic Green tensor.
``G_vacuum``
    Homogeneous/direct part included only when source and observer are in the
    same implemented source region.
``G_structure``
    Scattered or transmitted Mie contribution.
``projected_G``
    Optional ``e_A · G · e_D`` if both orientations are supplied.
``purcell``
    Optional source-position Purcell factor if ``simulation.compute_purcell`` is
    true.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import warnings

import h5py
import hydra
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.io import loadmat
from tqdm import tqdm

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("main_mie")  # type: ignore

try:  # installed as a sibling file or as mqed.Dyadic_GF.GF_Mie
    from GF_Mie import MieGreenFunction, c, eV_to_J, hbar
except Exception:  # pragma: no cover
    from mqed.Dyadic_GF.GF_Mie import MieGreenFunction, c, eV_to_J, hbar  # type: ignore

from mqed.utils.dgf_data import save_gf_pair_h5
from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware


# -----------------------------------------------------------------------------
#  Generic config/grid helpers
# -----------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key from a config mapping."""

    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def build_grid(config: Any) -> np.ndarray:
    """Build a 1-D numerical grid from flexible config input.

    Accepted formats:
        * scalar: ``2.0`` -> ``[2.0]``
        * list: ``[1.0, 2.0, 3.0]``
        * linspace dict: ``{min: 1, max: 3, points: 5}``
        * legacy linspace dict: ``{start: 0, stop: 100, points: 51}``
        * piecewise dict: ``{segments: [{min: ..., max: ..., points: ...}, ...]}``
        * values dict: ``{values: [...]}``
    """

    if _is_number(config):
        return np.array([float(config)], dtype=float)
    if isinstance(config, (list, tuple)):
        if config and all(isinstance(item, Mapping) for item in config):
            return _build_piecewise_grid(config)
        return np.asarray(config, dtype=float).reshape(-1)
    if isinstance(config, Mapping):
        if "values" in config:
            return build_grid(config["values"])
        if "segments" in config:
            return _build_piecewise_grid(config["segments"])
        if {"min", "max", "points"} <= set(config):
            return np.linspace(float(config["min"]), float(config["max"]), int(config["points"]))
        if {"start", "stop", "points"} <= set(config):
            return np.linspace(float(config["start"]), float(config["stop"]), int(config["points"]))
    raise TypeError(f"Unsupported grid config: {config!r}")


def _build_piecewise_grid(segments: Sequence[Mapping[str, Any]]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for i, segment in enumerate(segments):
        if not {"min", "max", "points"} <= set(segment):
            raise ValueError("Each grid segment must define min, max, and points.")
        endpoint = i == len(segments) - 1
        arrays.append(
            np.linspace(
                float(segment["min"]),
                float(segment["max"]),
                int(segment["points"]),
                endpoint=endpoint,
            )
        )
    if not arrays:
        return np.array([], dtype=float)
    grid = np.concatenate(arrays)
    if grid.size <= 1:
        return grid
    keep = np.ones(grid.size, dtype=bool)
    keep[1:] = ~np.isclose(grid[1:], grid[:-1])
    return grid[keep]


def spectral_grid(simulation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(energy_eV, wavelength_m, wavelength_nm)`` sorted by energy."""

    if "spectral_param" in simulation:
        kind = str(simulation["spectral_param"]).lower()
    elif "energy_eV" in simulation:
        kind = "energy_eV"
    else:
        kind = "wavelength_nm"

    if kind in {"energy_eV", "energy_ev", "energy", "ev", "eV".lower()}:
        energy_eV = build_grid(simulation["energy_eV"])
    elif kind in {"wavelength_nm", "wavelength", "lambda_nm", "lambda"}:
        wavelength_nm = build_grid(simulation["wavelength_nm"])
        energy_eV = 2.0 * np.pi * hbar * c / (wavelength_nm * 1e-9 * eV_to_J)
    else:
        raise ValueError("simulation.spectral_param must be 'energy_eV'/'eV' or 'wavelength_nm'.")

    sort_idx = np.argsort(energy_eV)
    energy_eV = energy_eV[sort_idx]
    wavelength_m = 2.0 * np.pi * hbar * c / (energy_eV * eV_to_J)
    wavelength_nm = wavelength_m * 1e9
    # Deduplicate neighboring energy points while preserving companions.
    if energy_eV.size > 1:
        keep = np.ones(energy_eV.size, dtype=bool)
        keep[1:] = ~np.isclose(energy_eV[1:], energy_eV[:-1])
        energy_eV = energy_eV[keep]
        wavelength_m = wavelength_m[keep]
        wavelength_nm = wavelength_nm[keep]
    return energy_eV, wavelength_m, wavelength_nm


def _position_array_m(value: Sequence[float], unit: str) -> np.ndarray:
    factor = 1e-9 if unit == "nm" else 1.0
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Position must be a 3-vector; got shape {arr.shape}.")
    return arr * factor


def source_position_m(simulation: Mapping[str, Any]) -> np.ndarray:
    """Read the source position from modern or legacy config keys."""

    if "source_position_m" in simulation:
        return _position_array_m(simulation["source_position_m"], "m")
    if "source_position_nm" in simulation:
        return _position_array_m(simulation["source_position_nm"], "nm")
    if "position" in simulation:
        pos = simulation["position"]
        if "source_nm" in pos:
            return _position_array_m(pos["source_nm"], "nm")
        if "source_m" in pos:
            return _position_array_m(pos["source_m"], "m")
        zD_nm = _get(pos, "zD_nm", default=None)
        zD_m = _get(pos, "zD", "zD_m", default=None)
        if zD_nm is not None:
            return np.array([0.0, 0.0, float(zD_nm) * 1e-9])
        if zD_m is not None:
            return np.array([0.0, 0.0, float(zD_m)])
    raise ValueError(
        "Define simulation.source_position_nm/m or legacy simulation.position.zD_nm/zD."
    )


def observer_positions_m(simulation: Mapping[str, Any]) -> np.ndarray:
    """Read observer positions from arbitrary-point, Cartesian-grid, or legacy Rx config."""

    if "observer_positions_m" in simulation:
        arr = np.asarray(simulation["observer_positions_m"], dtype=float)
        return arr.reshape(-1, 3)
    if "observer_positions_nm" in simulation:
        arr = np.asarray(simulation["observer_positions_nm"], dtype=float)
        return arr.reshape(-1, 3) * 1e-9
    if "observer_position_nm" in simulation:
        return _position_array_m(simulation["observer_position_nm"], "nm").reshape(1, 3)
    if "observer_position_m" in simulation:
        return _position_array_m(simulation["observer_position_m"], "m").reshape(1, 3)

    if "observer_grid_nm" in simulation:
        grid = simulation["observer_grid_nm"]
        x = build_grid(grid.get("x", grid.get("x_nm", 0.0)))
        y = build_grid(grid.get("y", grid.get("y_nm", 0.0)))
        z = build_grid(grid.get("z", grid.get("z_nm", 0.0)))
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) * 1e-9

    if "observer_scan_nm" in simulation:
        return horizontal_observer_scan_m(simulation, simulation["observer_scan_nm"])
    if "observer_rx_nm" in simulation:
        return horizontal_observer_scan_m(simulation, {"Rx_nm": simulation["observer_rx_nm"]})

    # Legacy planar-driver-like syntax: position.Rx_nm with fixed y and zA.
    if "position" in simulation:
        pos = simulation["position"]
        if "observer_nm" in pos:
            return _position_array_m(pos["observer_nm"], "nm").reshape(1, 3)
        if "observer_m" in pos:
            return _position_array_m(pos["observer_m"], "m").reshape(1, 3)
        if "Rx_nm" in pos:
            return horizontal_observer_scan_m(simulation, pos)

    raise ValueError("Define observer_positions_nm/m, observer_grid_nm, or legacy position.Rx_nm.")


def horizontal_observer_scan_m(simulation: Mapping[str, Any], scan_cfg: Mapping[str, Any]) -> np.ndarray:
    """Build observer points from horizontal source-observer separations.

    ``Rx_nm`` is interpreted as an x-offset from the source by default, with
    observer y and z copied from the source.  Explicit ``x0_nm``, ``y_nm`` or
    ``zA_nm``/``zA``/``zA_m`` override those defaults for compatibility with the
    planar-driver style.
    """

    rx_cfg = _get(scan_cfg, "Rx_nm", "rx_nm", "values", default=None)
    if rx_cfg is None:
        raise ValueError("Horizontal observer scans require Rx_nm/rx_nm values in nm.")
    rx_nm = build_grid(rx_cfg)
    source = source_position_m(simulation)

    x0_m = source[0]
    if "x0_nm" in scan_cfg:
        x0_m = float(scan_cfg["x0_nm"]) * 1e-9
    elif "x0_m" in scan_cfg:
        x0_m = float(scan_cfg["x0_m"])

    y_m = source[1]
    if "y_nm" in scan_cfg:
        y_m = float(scan_cfg["y_nm"]) * 1e-9
    elif "y_m" in scan_cfg:
        y_m = float(scan_cfg["y_m"])

    z_m = source[2]
    if "zA_nm" in scan_cfg:
        z_m = float(scan_cfg["zA_nm"]) * 1e-9
    elif "zA" in scan_cfg:
        z_m = float(scan_cfg["zA"])
    elif "zA_m" in scan_cfg:
        z_m = float(scan_cfg["zA_m"])

    return np.column_stack(
        [x0_m + rx_nm * 1e-9, np.full(rx_nm.size, y_m), np.full(rx_nm.size, z_m)]
    )


def emitter_positions_m(simulation: Mapping[str, Any]) -> np.ndarray:
    if "emitter_positions_m" in simulation:
        arr = np.asarray(simulation["emitter_positions_m"], dtype=float)
        return arr.reshape(-1, 3)
    if "emitter_positions_nm" in simulation:
        arr = np.asarray(simulation["emitter_positions_nm"], dtype=float)
        return arr.reshape(-1, 3) * 1e-9
    if "emitters" in simulation:
        emitters = simulation["emitters"]
        if "positions_m" in emitters:
            return np.asarray(emitters["positions_m"], dtype=float).reshape(-1, 3)
        if "positions_nm" in emitters:
            return np.asarray(emitters["positions_nm"], dtype=float).reshape(-1, 3) * 1e-9
    raise ValueError(
        "Pair-layout output requires simulation.emitter_positions_nm/m "
        "or simulation.emitters.positions_nm/m."
    )


def output_layout_from_config(config: Mapping[str, Any]) -> str:
    output_cfg = config.get("output", {})
    simulation = config.get("simulation", {})
    raw = str(
        output_cfg.get(
            "layout",
            simulation.get("output_layout", simulation.get("output_mode", "scan")),
        )
    ).lower()
    aliases = {
        "pair": "pair",
        "pair_tensor": "pair",
        "mqed_pair": "pair",
        "scan": "scan",
        "source_scan": "scan",
        "matlab": "scan",
        "matlab_reproduction": "scan",
    }
    if raw not in aliases:
        raise ValueError("output.layout must be 'pair'/'pair_tensor' or 'scan'/'matlab_reproduction'.")
    return aliases[raw]


def orientation_from_config(config: Mapping[str, Any], *keys: str, default: Sequence[float] | None = None) -> np.ndarray | None:
    """Read and normalize an orientation vector if present."""

    value = _get(config, *keys, default=default)
    if value is None:
        return None
    arr = np.asarray(value, dtype=complex).reshape(3)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError(f"Orientation {keys} must be non-zero.")
    return arr / norm


# -----------------------------------------------------------------------------
#  Dielectric-function resolver
# -----------------------------------------------------------------------------


class MaterialResolver:
    """Resolve region refractive indices from constants, files, or simple models."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._file_cache: dict[Path, dict[str, np.ndarray]] = {}

    def refractive_index(self, material_cfg: Any, energy_eV: float, wavelength_nm: float, omega: float) -> complex:
        """Return complex relative refractive index for one region."""

        if _is_number(material_cfg) or isinstance(material_cfg, str) or (
            isinstance(material_cfg, (list, tuple)) and len(material_cfg) == 2 and all(_is_number(x) for x in material_cfg)
        ):
            # Bare constants are interpreted as relative permittivity for safety,
            # matching most dielectric-function tables.
            return np.sqrt(self._parse_complex(material_cfg))

        if not isinstance(material_cfg, Mapping):
            raise TypeError(f"Unsupported material config: {material_cfg!r}")

        if "n" in material_cfg or "refractive_index" in material_cfg:
            return self._parse_complex(_get(material_cfg, "n", "refractive_index"))
        if "epsilon" in material_cfg or "eps" in material_cfg or "permittivity" in material_cfg:
            return np.sqrt(self._parse_complex(_get(material_cfg, "epsilon", "eps", "permittivity")))
        if "file" in material_cfg or "path" in material_cfg:
            eps = self._epsilon_from_file(material_cfg, energy_eV, wavelength_nm)
            return np.sqrt(eps)
        if str(material_cfg.get("type", "")).lower() == "drude":
            eps = self._epsilon_drude(material_cfg, energy_eV)
            return np.sqrt(eps)
        if "material_config" in material_cfg:
            return self.refractive_index(material_cfg["material_config"], energy_eV, wavelength_nm, omega)

        # Optional compatibility with the existing MQED DataProvider.
        try:
            from mqed.Dyadic_GF.data_provider import DataProvider  # type: ignore

            eps = DataProvider(material_cfg).get_epsilon(omega)
            return np.sqrt(eps)
        except Exception as exc:
            raise ValueError(
                f"Could not resolve material config {material_cfg!r}. "
                "Use n, epsilon, file/path, or type: drude."
            ) from exc

    @staticmethod
    def _parse_complex(value: Any) -> complex:
        if isinstance(value, complex):
            return value
        if _is_number(value):
            return complex(float(value), 0.0)
        if isinstance(value, str):
            return complex(value.replace("i", "j"))
        if isinstance(value, Mapping):
            real = float(_get(value, "real", "re", default=0.0))
            imag = float(_get(value, "imag", "im", default=0.0))
            return complex(real, imag)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return complex(float(value[0]), float(value[1]))
        raise TypeError(f"Cannot parse complex value: {value!r}")

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._file_cache:
            return self._file_cache[path]
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".mat":
            raw = loadmat(path)
            data = {k: np.asarray(v).squeeze() for k, v in raw.items() if not k.startswith("__")}
        elif suffix in {".csv", ".txt", ".dat"}:
            try:
                table = np.genfromtxt(path, delimiter="," if suffix == ".csv" else None, names=True, dtype=float)
                data = {name: np.asarray(table[name]).squeeze() for name in table.dtype.names or []}
            except Exception:
                arr = np.genfromtxt(path, delimiter="," if suffix == ".csv" else None, dtype=float)
                data = {f"col{i}": arr[:, i] for i in range(arr.shape[1])}
        elif suffix == ".npy":
            arr = np.load(path)
            data = {f"col{i}": arr[:, i] for i in range(arr.shape[1])}
        else:
            raise ValueError(f"Unsupported material file type: {path.suffix}")
        self._file_cache[path] = data
        return data

    @staticmethod
    def _choose_key(data: Mapping[str, np.ndarray], preferred: Sequence[str]) -> str:
        for key in preferred:
            if key in data:
                return key
        raise KeyError(f"None of the keys {preferred} were found. Available: {list(data)}")

    @staticmethod
    def _interp_complex(x: np.ndarray, y: np.ndarray, x0: float) -> complex:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=complex).reshape(-1)
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        if x0 < x_sorted[0] or x0 > x_sorted[-1]:
            warnings.warn(
                f"Interpolating material outside tabulated range: x={x0:g}, "
                f"range=[{x_sorted[0]:g}, {x_sorted[-1]:g}]. Clamping to nearest endpoint.",
                RuntimeWarning,
            )
        real = np.interp(x0, x_sorted, np.real(y_sorted))
        imag = np.interp(x0, x_sorted, np.imag(y_sorted))
        return complex(real, imag)

    def _epsilon_from_file(self, material_cfg: Mapping[str, Any], energy_eV: float, wavelength_nm: float) -> complex:
        path = self._resolve_path(str(_get(material_cfg, "file", "path")))
        data = self._load_file(path)

        eps_key = material_cfg.get("epsilon_key")
        if eps_key is None:
            for candidate in ("epsilon", "epsilonmat", "eps", "epsi2"):
                if candidate in data:
                    eps_key = candidate
                    break
        if eps_key is not None:
            y = data[str(eps_key)]
        else:
            re_key = str(_get(material_cfg, "epsilon_real_key", "eps_real_key", "real_key", default="epsilon_real"))
            im_key = str(_get(material_cfg, "epsilon_imag_key", "eps_imag_key", "imag_key", default="epsilon_imag"))
            if re_key not in data or im_key not in data:
                raise KeyError(
                    "Material file needs epsilon_key or real/imag epsilon keys. "
                    f"Available keys: {list(data)}"
                )
            y = data[re_key] + 1j * data[im_key]

        if "energy_eV_key" in material_cfg or "omega0" in data:
            key = str(material_cfg.get("energy_eV_key", "omega0"))
            return self._interp_complex(data[key], y, energy_eV)
        if "wavelength_nm_key" in material_cfg:
            key = str(material_cfg["wavelength_nm_key"])
            return self._interp_complex(data[key], y, wavelength_nm)
        if "x_key" in material_cfg:
            key = str(material_cfg["x_key"])
            x_kind = str(material_cfg.get("x_kind", "energy_eV"))
            x0 = energy_eV if x_kind == "energy_eV" else wavelength_nm
            return self._interp_complex(data[key], y, x0)
        # Common CSV names.
        for key in ("energy_eV", "energy", "E_eV"):
            if key in data:
                return self._interp_complex(data[key], y, energy_eV)
        for key in ("wavelength_nm", "lambda_nm", "lambda"):
            if key in data:
                return self._interp_complex(data[key], y, wavelength_nm)
        raise KeyError(
            "Could not find spectral axis in material file. Specify energy_eV_key, wavelength_nm_key, or x_key."
        )

    @staticmethod
    def _epsilon_drude(material_cfg: Mapping[str, Any], energy_eV: float) -> complex:
        eps_inf = complex(material_cfg.get("eps_inf", material_cfg.get("epsilon_inf", 1.0)))
        omega_p = float(material_cfg["omega_p_eV"])
        gamma = float(material_cfg["gamma_eV"])
        E = complex(energy_eV)
        return eps_inf - omega_p**2 / (E * (E + 1j * gamma))


def region_materials(config: Mapping[str, Any]) -> Sequence[Any]:
    """Return the material configs for each spherical region."""

    materials = config.get("materials", {})
    if "regions" in materials:
        return materials["regions"]
    if "material" in config and "regions" in config["material"]:
        return config["material"]["regions"]
    if "refractive_indices" in config.get("simulation", {}):
        return config["simulation"]["refractive_indices"]
    raise ValueError("Define materials.regions as a list of region material configs.")


def build_refractive_indices(
    config: Mapping[str, Any], resolver: MaterialResolver, energy_eV: float, wavelength_nm: float, omega: float
) -> np.ndarray:
    return np.array(
        [resolver.refractive_index(region, energy_eV, wavelength_nm, omega) for region in region_materials(config)],
        dtype=complex,
    )


# -----------------------------------------------------------------------------
#  Geometry, worker, parallel execution
# -----------------------------------------------------------------------------


def radii_m_from_config(simulation: Mapping[str, Any]) -> np.ndarray:
    geometry = simulation.get("geometry", {})
    if "radii_m" in geometry:
        return np.asarray(geometry["radii_m"], dtype=float).reshape(-1)
    if "radii_nm" in geometry:
        return np.asarray(geometry["radii_nm"], dtype=float).reshape(-1) * 1e-9
    if "radius_m" in geometry:
        return np.array([float(geometry["radius_m"])], dtype=float)
    if "radius_nm" in geometry:
        return np.array([float(geometry["radius_nm"]) * 1e-9], dtype=float)
    if "rbc" in simulation:  # MATLAB-like key in meters
        return np.atleast_1d(np.asarray(simulation["rbc"], dtype=float))
    raise ValueError("Define simulation.geometry.radius_nm/radius_m or radii_nm/radii_m.")


def geometry_name_from_config(simulation: Mapping[str, Any]) -> str:
    geometry = simulation.get("geometry", {})
    return str(geometry.get("boundary", geometry.get("type", simulation.get("BC", "sphere")))).lower()


def _compute_one_energy(
    index: int,
    energy_eV: float,
    wavelength_m: float,
    wavelength_nm: float,
    config: Mapping[str, Any],
    observer_positions: np.ndarray,
    source_position: np.ndarray,
    source_orientation: np.ndarray | None,
    observer_orientation: np.ndarray | None,
    base_dir: Path,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float | None, np.ndarray]:
    """Compute all observer points for one energy.

    The function is top-level and pure enough to be used by joblib workers.
    """

    simulation = config["simulation"]
    omega = energy_eV * eV_to_J / hbar
    resolver = MaterialResolver(base_dir)
    nr = build_refractive_indices(config, resolver, energy_eV, wavelength_nm, omega)
    calculator = MieGreenFunction(
        refractive_indices=nr,
        radii_m=radii_m_from_config(simulation),
        omega=omega,
        nmax=int(simulation["nmax"]),
        geometry=geometry_name_from_config(simulation),
        strict_regions=bool(simulation.get("strict_regions", True)),
    )

    n_obs = observer_positions.shape[0]
    total = np.zeros((n_obs, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    projected = None if source_orientation is None or observer_orientation is None else np.zeros(n_obs, dtype=complex)
    regions = np.zeros(n_obs, dtype=int)

    show_inner = bool(simulation.get("show_observer_progress", False)) and n_obs > 1
    iterator = enumerate(observer_positions)
    if show_inner:
        iterator = enumerate(tqdm(observer_positions, desc=f"Observers @ {energy_eV:.3f} eV", leave=False, ncols=100))

    for obs_index, obs in iterator:
        result = calculator.calculate_components(obs, source_position)
        total[obs_index] = result.total
        vacuum[obs_index] = result.vacuum
        structure[obs_index] = result.structure
        regions[obs_index] = result.observer_region
        if projected is not None:
            projected[obs_index] = observer_orientation @ (result.total @ source_orientation)

    purcell = None
    if bool(simulation.get("compute_purcell", False)):
        if source_orientation is None:
            raise ValueError("compute_purcell=true requires simulation.source_orientation.")
        purcell = calculator.purcell_factor(source_position, source_orientation)

    return index, total, vacuum, structure, projected, purcell, regions


def _compute_one_pair_energy(
    index: int,
    energy_eV: float,
    wavelength_m: float,
    wavelength_nm: float,
    config: Mapping[str, Any],
    emitter_positions: np.ndarray,
    base_dir: Path,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    simulation = config["simulation"]
    omega = energy_eV * eV_to_J / hbar
    resolver = MaterialResolver(base_dir)
    nr = build_refractive_indices(config, resolver, energy_eV, wavelength_nm, omega)
    calculator = MieGreenFunction(
        refractive_indices=nr,
        radii_m=radii_m_from_config(simulation),
        omega=omega,
        nmax=int(simulation["nmax"]),
        geometry=geometry_name_from_config(simulation),
        strict_regions=bool(simulation.get("strict_regions", True)),
    )

    n_emitters = emitter_positions.shape[0]
    total = np.zeros((n_emitters, n_emitters, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    regions = np.zeros((n_emitters, n_emitters), dtype=int)

    for source_index, source in enumerate(emitter_positions):
        for observer_index, observer in enumerate(emitter_positions):
            result = calculator.calculate_components(observer, source)
            total[observer_index, source_index] = result.total
            vacuum[observer_index, source_index] = result.vacuum
            structure[observer_index, source_index] = result.structure
            regions[observer_index, source_index] = result.observer_region

    return index, total, vacuum, structure, regions


def run_sequential(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    observer_positions: np.ndarray,
    source_position: np.ndarray,
    source_orientation: np.ndarray | None,
    observer_orientation: np.ndarray | None,
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    nE = energy_eV.size
    n_obs = observer_positions.shape[0]
    total = np.zeros((nE, n_obs, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    projected = None if source_orientation is None or observer_orientation is None else np.zeros((nE, n_obs), dtype=complex)
    purcell = np.zeros(nE, dtype=float) if bool(config["simulation"].get("compute_purcell", False)) else None
    regions = np.zeros((nE, n_obs), dtype=int)

    for i in tqdm(range(nE), desc="Energies", ncols=100):
        _, tot, vac, st, proj, purc, reg = _compute_one_energy(
            i,
            float(energy_eV[i]),
            float(wavelength_m[i]),
            float(wavelength_nm[i]),
            config,
            observer_positions,
            source_position,
            source_orientation,
            observer_orientation,
            base_dir,
        )
        total[i] = tot
        vacuum[i] = vac
        structure[i] = st
        regions[i] = reg
        if projected is not None and proj is not None:
            projected[i] = proj
        if purcell is not None and purc is not None:
            purcell[i] = purc
    return total, vacuum, structure, projected, purcell, regions, energy_eV


def run_joblib(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    observer_positions: np.ndarray,
    source_position: np.ndarray,
    source_orientation: np.ndarray | None,
    observer_orientation: np.ndarray | None,
    base_dir: Path,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    nE = energy_eV.size
    n_obs = observer_positions.shape[0]
    logger.info(f"Joblib backend: dispatching {nE} energies across {n_jobs} workers")
    raw = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_compute_one_energy)(
            i,
            float(energy_eV[i]),
            float(wavelength_m[i]),
            float(wavelength_nm[i]),
            config,
            observer_positions,
            source_position,
            source_orientation,
            observer_orientation,
            base_dir,
        )
        for i in tqdm(range(nE), desc="Submit energies", ncols=100)
    )

    total = np.zeros((nE, n_obs, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    projected = None if source_orientation is None or observer_orientation is None else np.zeros((nE, n_obs), dtype=complex)
    purcell = np.zeros(nE, dtype=float) if bool(config["simulation"].get("compute_purcell", False)) else None
    regions = np.zeros((nE, n_obs), dtype=int)
    for idx, tot, vac, st, proj, purc, reg in raw:
        total[idx] = tot
        vacuum[idx] = vac
        structure[idx] = st
        regions[idx] = reg
        if projected is not None and proj is not None:
            projected[idx] = proj
        if purcell is not None and purc is not None:
            purcell[idx] = purc
    return total, vacuum, structure, projected, purcell, regions, energy_eV


def _maybe_auto_launch_mpi(parallel_cfg: Mapping[str, Any]) -> None:
    if not bool(parallel_cfg.get("mpi_auto_launch", True)):
        return
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            return
    except ImportError:
        pass
    nproc = int(parallel_cfg.get("mpi_nproc", 4))
    mpi_exec = str(parallel_cfg.get("mpi_exec", "mpiexec"))
    cmd = [mpi_exec, "-n", str(nproc)] + sys.argv
    logger.info("Auto-launching MPI: {}", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def _current_mpi_rank() -> int:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        return 0


def run_mpi(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    observer_positions: np.ndarray,
    source_position: np.ndarray,
    source_orientation: np.ndarray | None,
    observer_orientation: np.ndarray | None,
    base_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nE = energy_eV.size
    n_obs = observer_positions.shape[0]
    local_indices = list(range(rank, nE, size))
    if rank == 0:
        logger.info("MPI backend: {} ranks, {} scan energies", size, nE)

    local_results = []
    for i in local_indices:
        logger.info("[rank {}] Mie scan energy {}/{}: {:.3f} eV", rank, i + 1, nE, energy_eV[i])
        local_results.append(
            _compute_one_energy(
                i,
                float(energy_eV[i]),
                float(wavelength_m[i]),
                float(wavelength_nm[i]),
                config,
                observer_positions,
                source_position,
                source_orientation,
                observer_orientation,
                base_dir,
            )
        )

    all_results = comm.gather(local_results, root=0)
    if rank != 0:
        return None, None, None, None, None, None, energy_eV

    total = np.zeros((nE, n_obs, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    projected = None if source_orientation is None or observer_orientation is None else np.zeros((nE, n_obs), dtype=complex)
    purcell = np.zeros(nE, dtype=float) if bool(config["simulation"].get("compute_purcell", False)) else None
    regions = np.zeros((nE, n_obs), dtype=int)
    for rank_results in all_results:
        for idx, tot, vac, st, proj, purc, reg in rank_results:
            total[idx] = tot
            vacuum[idx] = vac
            structure[idx] = st
            regions[idx] = reg
            if projected is not None and proj is not None:
                projected[idx] = proj
            if purcell is not None and purc is not None:
                purcell[idx] = purc
    return total, vacuum, structure, projected, purcell, regions, energy_eV


def run_pair_sequential(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    emitter_positions: np.ndarray,
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nE = energy_eV.size
    n_emitters = emitter_positions.shape[0]
    total = np.zeros((nE, n_emitters, n_emitters, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    regions = np.zeros((nE, n_emitters, n_emitters), dtype=int)

    for i in tqdm(range(nE), desc="Energies", ncols=100):
        _, tot, vac, st, reg = _compute_one_pair_energy(
            i,
            float(energy_eV[i]),
            float(wavelength_m[i]),
            float(wavelength_nm[i]),
            config,
            emitter_positions,
            base_dir,
        )
        total[i] = tot
        vacuum[i] = vac
        structure[i] = st
        regions[i] = reg
    return total, vacuum, structure, regions, energy_eV


def run_pair_joblib(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    emitter_positions: np.ndarray,
    base_dir: Path,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    nE = energy_eV.size
    n_emitters = emitter_positions.shape[0]
    logger.info(f"Joblib backend: dispatching {nE} pair-tensor energies across {n_jobs} workers")
    raw = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_compute_one_pair_energy)(
            i,
            float(energy_eV[i]),
            float(wavelength_m[i]),
            float(wavelength_nm[i]),
            config,
            emitter_positions,
            base_dir,
        )
        for i in tqdm(range(nE), desc="Submit energies", ncols=100)
    )

    total = np.zeros((nE, n_emitters, n_emitters, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    regions = np.zeros((nE, n_emitters, n_emitters), dtype=int)
    for idx, tot, vac, st, reg in raw:
        total[idx] = tot
        vacuum[idx] = vac
        structure[idx] = st
        regions[idx] = reg
    return total, vacuum, structure, regions, energy_eV


def run_pair_mpi(
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    wavelength_nm: np.ndarray,
    config: Mapping[str, Any],
    emitter_positions: np.ndarray,
    base_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nE = energy_eV.size
    n_emitters = emitter_positions.shape[0]
    local_indices = list(range(rank, nE, size))
    if rank == 0:
        logger.info("MPI backend: {} ranks, {} pair-tensor energies", size, nE)

    local_results = []
    for i in local_indices:
        logger.info("[rank {}] Mie pair energy {}/{}: {:.3f} eV", rank, i + 1, nE, energy_eV[i])
        local_results.append(
            _compute_one_pair_energy(
                i,
                float(energy_eV[i]),
                float(wavelength_m[i]),
                float(wavelength_nm[i]),
                config,
                emitter_positions,
                base_dir,
            )
        )

    all_results = comm.gather(local_results, root=0)
    if rank != 0:
        return None, None, None, None, energy_eV

    total = np.zeros((nE, n_emitters, n_emitters, 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    structure = np.zeros_like(total)
    regions = np.zeros((nE, n_emitters, n_emitters), dtype=int)
    for rank_results in all_results:
        for idx, tot, vac, st, reg in rank_results:
            total[idx] = tot
            vacuum[idx] = vac
            structure[idx] = st
            regions[idx] = reg
    return total, vacuum, structure, regions, energy_eV


# -----------------------------------------------------------------------------
#  HDF5 output
# -----------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_hdf5(
    path: Path,
    total: np.ndarray,
    vacuum: np.ndarray,
    structure: np.ndarray,
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    observer_positions: np.ndarray,
    source_position: np.ndarray,
    regions: np.ndarray,
    config: Mapping[str, Any],
    projected: np.ndarray | None = None,
    purcell: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("G_total", data=total)
        h5.create_dataset("G_vacuum", data=vacuum)
        h5.create_dataset("G_structure", data=structure)
        h5.create_dataset("energy_eV", data=energy_eV)
        h5.create_dataset("wavelength_m", data=wavelength_m)
        h5.create_dataset("wavelength_nm", data=wavelength_m * 1e9)
        h5.create_dataset("observer_positions_m", data=observer_positions)
        h5.create_dataset("source_position_m", data=source_position)
        h5.create_dataset("observer_region", data=regions)
        if projected is not None:
            h5.create_dataset("projected_G", data=projected)
            h5.create_dataset("projected_ImG", data=np.imag(projected))
            h5.create_dataset("projected_abs2", data=np.abs(projected) ** 2)
        if purcell is not None:
            h5.create_dataset("purcell", data=purcell)
        h5.attrs["description"] = "Generalized Mie dyadic Green tensor for spherical dielectrics."
        h5.attrs["config_json"] = json.dumps(config, default=_json_default)
        h5.attrs["units_G"] = "meter^-1 in the Green-tensor convention used by GF_Sommerfeld.py"
        h5.attrs["coordinate_basis"] = "Cartesian x,y,z"


def save_pair_hdf5(
    path: Path,
    total: np.ndarray,
    vacuum: np.ndarray,
    structure: np.ndarray,
    energy_eV: np.ndarray,
    wavelength_m: np.ndarray,
    emitter_positions: np.ndarray,
    regions: np.ndarray,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    emitter_positions_nm = emitter_positions * 1e9
    reference_z = float(emitter_positions[0, 2])
    save_gf_pair_h5(
        str(path),
        total,
        vacuum,
        energy_eV,
        emitter_positions_nm,
        zD=reference_z,
        zA=reference_z,
    )
    with h5py.File(path, "a") as h5:
        h5.create_dataset("green_function_structure", data=structure)
        h5.create_dataset("wavelength_m", data=wavelength_m)
        h5.create_dataset("wavelength_nm", data=wavelength_m * 1e9)
        h5.create_dataset("observer_region", data=regions)
        h5.attrs["description"] = "Generalized Mie pair-indexed dyadic Green tensor."
        h5.attrs["config_json"] = json.dumps(config, default=_json_default)
        h5.attrs["units_G"] = "meter^-1 in the Green-tensor convention used by GF_Sommerfeld.py"
        h5.attrs["coordinate_basis"] = "Cartesian x,y,z"


# -----------------------------------------------------------------------------
#  Example config and CLI
# -----------------------------------------------------------------------------


EXAMPLE_CONFIG = {
    "simulation": {
        "spectral_param": "energy_eV",
        "energy_eV": {"min": 1.8, "max": 2.5, "points": 5},
        "nmax": 15,
        "geometry": {"boundary": "coreshell", "radii_nm": [160.0, 60.0]},
        "emitter_positions_nm": [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]],
        "source_position_nm": [0.0, 0.0, 0.0],
        "position": {"Rx_nm": {"min": 0.0, "max": 120.0, "points": 121}},
        "source_orientation": [0.0, 0.0, 1.0],
        "observer_orientation": [0.0, 0.0, 1.0],
        "compute_purcell": True,
        "strict_regions": True,
    },
    "materials": {
        "regions": [
            {"n": 1.0, "name": "vacuum exterior"},
            {
                "type": "drude",
                "eps_inf": 1.0,
                "omega_p_eV": 12.5,
                "gamma_eV": 0.0621,
                "name": "aluminum shell",
            },
            {"n": 1.0, "name": "vacuum core"},
        ]
    },
    "parallel": {"backend": "sequential", "n_jobs": -1, "mpi_nproc": 4, "mpi_auto_launch": True, "mpi_exec": "mpiexec"},
    "output": {"directory": "outputs", "prefix": "mie_shell_cavity_scan", "layout": "scan"},
}


def output_path_from_config(config: Mapping[str, Any], config_path: Path) -> Path:
    output_cfg = config.get("output", {})
    out_dir = Path(str(output_cfg.get("directory", config_path.parent / "outputs"))).expanduser()
    if not out_dir.is_absolute():
        out_dir = config_path.parent / out_dir
    prefix = str(output_cfg.get("prefix", "mie_green"))
    simulation = config["simulation"]
    energy, _, _ = spectral_grid(simulation)
    return out_dir / f"{prefix}_Emin_{energy[0]:.3f}_Emax_{energy[-1]:.3f}_{energy.size}pts.h5"


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def run_config(config: Mapping[str, Any], config_base_dir: Path, output_override: Path | None = None) -> Path:
    if "simulation" not in config:
        raise ValueError("Config must contain a top-level 'simulation' mapping.")

    simulation = config["simulation"]
    energy_eV, wavelength_m, wavelength_nm = spectral_grid(simulation)
    layout = output_layout_from_config(config)
    parallel = config.get("parallel", {})
    backend = str(parallel.get("backend", "sequential")).lower()
    config_base_dir = config_base_dir.resolve()
    out = output_override.resolve() if output_override is not None else output_path_from_config(config, config_base_dir / "GF_Mie.yaml")

    if backend == "mpi":
        _maybe_auto_launch_mpi(parallel)

    if layout == "pair":
        emitters = emitter_positions_m(simulation)
        logger.info(
            "Grid: {} energies x {}x{} emitter pairs | geometry={} | nmax={}",
            energy_eV.size,
            emitters.shape[0],
            emitters.shape[0],
            geometry_name_from_config(simulation),
            simulation["nmax"],
        )

        if backend == "sequential":
            total, vacuum, structure, regions, _ = run_pair_sequential(
                energy_eV,
                wavelength_m,
                wavelength_nm,
                config,
                emitters,
                config_base_dir,
            )
        elif backend == "joblib":
            total, vacuum, structure, regions, _ = run_pair_joblib(
                energy_eV,
                wavelength_m,
                wavelength_nm,
                config,
                emitters,
                config_base_dir,
                int(parallel.get("n_jobs", -1)),
            )
        elif backend == "mpi":
            total, vacuum, structure, regions, _ = run_pair_mpi(
                energy_eV,
                wavelength_m,
                wavelength_nm,
                config,
                emitters,
                config_base_dir,
            )
            if _current_mpi_rank() != 0:
                return out
        else:
            raise ValueError("parallel.backend must be 'sequential', 'joblib', or 'mpi'.")

        if total is None or vacuum is None or structure is None or regions is None:
            return out

        save_pair_hdf5(out, total, vacuum, structure, energy_eV, wavelength_m, emitters, regions, config)
        logger.success("Mie pair Green-function simulation complete: {}", out)
        return out

    source_pos = source_position_m(simulation)
    observers = observer_positions_m(simulation)
    source_ori = orientation_from_config(simulation, "source_orientation", "donor_orientation", default=None)
    observer_ori = orientation_from_config(simulation, "observer_orientation", "acceptor_orientation", default=None)

    logger.info(
        "Grid: {} energies x {} observer points | geometry={} | nmax={}",
        energy_eV.size,
        observers.shape[0],
        geometry_name_from_config(simulation),
        simulation["nmax"],
    )

    if backend == "sequential":
        total, vacuum, structure, projected, purcell, regions, _ = run_sequential(
            energy_eV,
            wavelength_m,
            wavelength_nm,
            config,
            observers,
            source_pos,
            source_ori,
            observer_ori,
            config_base_dir,
        )
    elif backend == "joblib":
        total, vacuum, structure, projected, purcell, regions, _ = run_joblib(
            energy_eV,
            wavelength_m,
            wavelength_nm,
            config,
            observers,
            source_pos,
            source_ori,
            observer_ori,
            config_base_dir,
            int(parallel.get("n_jobs", -1)),
        )
    elif backend == "mpi":
        total, vacuum, structure, projected, purcell, regions, _ = run_mpi(
            energy_eV,
            wavelength_m,
            wavelength_nm,
            config,
            observers,
            source_pos,
            source_ori,
            observer_ori,
            config_base_dir,
        )
        if _current_mpi_rank() != 0:
            return out
    else:
        raise ValueError("parallel.backend must be 'sequential', 'joblib', or 'mpi'.")

    if total is None or vacuum is None or structure is None or regions is None:
        return out

    save_hdf5(
        out,
        total,
        vacuum,
        structure,
        energy_eV,
        wavelength_m,
        observers,
        source_pos,
        regions,
        config,
        projected=projected,
        purcell=purcell,
    )
    logger.success("Mie Green-function simulation complete: {}", out)
    return out


def run_from_config(config_path: Path, output_override: Path | None = None) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    return run_config(config, config_path.parent, output_override)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalized Mie dyadic Green-function simulation")
    parser.add_argument("--config", type=Path, help="YAML or JSON configuration file")
    parser.add_argument("--output", type=Path, default=None, help="Optional explicit HDF5 output path")
    parser.add_argument(
        "--write-example-config",
        type=Path,
        default=None,
        help="Write an example YAML configuration to this path and exit.",
    )
    return parser.parse_args()


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("Dyadic_GF", __file__)


@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="GF_Mie", version_base=None)
def run_simulation(cfg: DictConfig) -> None:
    setup_loggers_hydra_aware()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("Hydra config must resolve to a mapping.")
    config = dict(config)
    output_cfg = dict(config.get("output", {}) or {})
    output_cfg["directory"] = str(output_dir)
    config["output"] = output_cfg
    run_config(config, Path(HydraConfig.get().runtime.cwd))


def main() -> None:
    args = parse_args()
    if args.write_example_config is not None:
        args.write_example_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_example_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(EXAMPLE_CONFIG, f, sort_keys=False)
        logger.success("Example config written to {}", args.write_example_config.resolve())
        return
    if args.config is None:
        raise SystemExit("Provide --config, or use --write-example-config to create a starting point.")
    run_from_config(args.config, args.output)


if __name__ == "__main__":
    main()
