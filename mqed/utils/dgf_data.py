r"""
HDF5 I/O for dyadic Green's function data.

Two storage layouts are supported, distinguished by the HDF5 attribute
``gf_layout`` on the root group:

Separation-indexed (``gf_layout = "separation"``, legacy default)
    The Green's function is stored as ``(M, K, 3, 3)`` where *M* is the
    number of energy points and *K* is the number of distinct inter-emitter
    separations Rx.  This layout exploits **translational symmetry**: all
    emitter pairs at the same separation share the same tensor.

    Applicable to: planar surfaces (2-layer, N-layer), any geometry with
    full in-plane translational symmetry.

    Datasets::

        green_function_total   (M, K, 3, 3)   complex128
        green_function_vacuum  (M, K, 3, 3)   complex128
        energy_eV              (M,)           float64
        Rx_nm                  (K,)           float64
        position_fixed         group  {zD_meters, zA_meters}

Pair-indexed (``gf_layout = "pair"``)
    The Green's function is stored as ``(M, N, N, 3, 3)`` where *N* is the
    number of emitters.  Entry ``[m, i, j, :, :]`` is the full dyadic
    G(r_i, r_j, ω_m).  No symmetry is assumed.

    Applicable to: nanorods, nanoparticles, arbitrary geometries —
    any case where translational symmetry is broken.

    Datasets::

        green_function_total   (M, N, N, 3, 3)   complex128
        green_function_vacuum  (M, N, N, 3, 3)   complex128
        energy_eV              (M,)              float64
        emitter_positions_nm   (N, 3)            float64
        emitter_orientations   (N, 3)            float64, optional
        position_fixed         group  {zD_meters, zA_meters}

Scan-indexed (``gf_layout = "scan"``)
    The Green's function is stored as ``(M, P, 3, 3)`` where *P* is the
    number of explicit observer positions for one fixed source position.
    No translational symmetry is assumed between observer points.

    Applicable to: spherical particles/cavities or arbitrary point scans.

    Datasets::

        green_function_total   (M, P, 3, 3)   complex128
        green_function_vacuum  (M, P, 3, 3)   complex128
        energy_eV              (M,)           float64
        observer_positions_nm  (P, 3)         float64
        source_position_nm     (3,)           float64
        position_fixed         group  {zD_meters, zA_meters}

Backward compatibility: files written by older planar code (no ``gf_layout``
attribute) are treated as separation-indexed, while older Mie scan files are
recognized from their explicit position datasets and ``G_total`` aliases.
"""
from __future__ import annotations
import h5py
import numpy as np
from typing import Any, Dict
from loguru import logger

from mqed.utils.emitter_geometry import normalize_orientation_vectors


# ── Separation-indexed (legacy) ──────────────────────────────────────

def save_gf_h5(
    h5_path: str,
    Gtot: np.ndarray,
    Gvac: np.ndarray,
    E: np.ndarray,
    Rxnm: np.ndarray,
    zD: float,
    zA: float,
    *,
    Gstructure: np.ndarray | None = None,
    G_scattering_te: np.ndarray | None = None,
    G_scattering_tm: np.ndarray | None = None,
    attrs: Dict[str, Any] | None = None,
) -> None:
    """Save separation-indexed Green's function arrays to HDF5.

    Args:
        h5_path: Output file path.
        Gtot:    Total Green's function, shape ``(M, K, 3, 3)``.
        Gvac:    Vacuum Green's function, shape ``(M, K, 3, 3)``.
        E:       Energy grid in eV, shape ``(M,)``.
        Rxnm:    Separation grid in nm, shape ``(K,)``.
        zD:      Source (donor) z-position in meters.
        zA:      Observer (acceptor) z-position in meters.
        Gstructure: Optional scattering/structure Green tensor, same shape as ``Gtot``.
        G_scattering_te: Optional TE scattering tensor, same shape as ``Gtot``.
        G_scattering_tm: Optional TM scattering tensor, same shape as ``Gtot``.
        attrs: Optional root attributes.
    """
    total = np.asarray(Gtot)
    vacuum = np.asarray(Gvac)
    if total.shape != vacuum.shape:
        raise ValueError(f"Gtot and Gvac must have matching shapes; got {total.shape} and {vacuum.shape}.")
    optional_arrays = {
        "Gstructure": Gstructure,
        "G_scattering_te": G_scattering_te,
        "G_scattering_tm": G_scattering_tm,
    }
    for name, values in optional_arrays.items():
        if values is not None and np.asarray(values).shape != total.shape:
            raise ValueError(f"{name} must have shape {total.shape}; got {np.asarray(values).shape}.")

    with h5py.File(h5_path, "w") as f:
        f.attrs["gf_layout"] = "separation"
        f.create_dataset("green_function_total", data=total)
        f.create_dataset("green_function_vacuum", data=vacuum)
        f.create_dataset("energy_eV", data=E)
        f.create_dataset("Rx_nm", data=Rxnm)
        pos = f.create_group("position_fixed")
        pos.attrs["zD_meters"] = zD
        pos.attrs["zA_meters"] = zA
        _write_optional_common_metadata(
            f,
            Gstructure=None if Gstructure is None else np.asarray(Gstructure),
            wavelength_m=None,
            observer_region=None,
            attrs=attrs,
        )
        if G_scattering_te is not None:
            f.create_dataset("green_function_scattering_te", data=np.asarray(G_scattering_te))
        if G_scattering_tm is not None:
            f.create_dataset("green_function_scattering_tm", data=np.asarray(G_scattering_tm))


# ── Pair-indexed (arbitrary geometry) ────────────────────────────────

def save_gf_pair_h5(
    h5_path: str,
    Gtot: np.ndarray,
    Gvac: np.ndarray,
    E: np.ndarray,
    emitter_positions_nm: np.ndarray,
    zD: float,
    zA: float,
    Gstructure: np.ndarray | None = None,
    wavelength_m: np.ndarray | None = None,
    observer_region: np.ndarray | None = None,
    attrs: Dict[str, Any] | None = None,
    emitter_orientations: np.ndarray | None = None,
) -> None:
    """Save pair-indexed Green's function arrays to HDF5.

    Args:
        h5_path:              Output file path.
        Gtot:                 Total Green's function, shape ``(M, N, N, 3, 3)``.
        Gvac:                 Vacuum Green's function, shape ``(M, N, N, 3, 3)``.
        E:                    Energy grid in eV, shape ``(M,)``.
        emitter_positions_nm: 3D positions of all emitters in nm,
                              shape ``(N, 3)``.
        zD:                   Source z-position in meters (reference height).
        zA:                   Observer z-position in meters (reference height).
        emitter_orientations: Optional normalized emitter dipole orientations,
                              shape ``(N, 3)``.
    """
    positions = np.asarray(emitter_positions_nm, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] == 0:
        raise ValueError("emitter_positions_nm must have shape (N, 3) with N > 0.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("emitter_positions_nm must be finite.")

    orientations = None
    if emitter_orientations is not None:
        orientations = normalize_orientation_vectors(emitter_orientations, positions.shape[0])

    total = np.asarray(Gtot)
    vacuum = np.asarray(Gvac)
    energy = np.asarray(E, dtype=float)
    emitter_count = positions.shape[0]
    expected_green_shape = (energy.size, emitter_count, emitter_count, 3, 3)
    if total.shape != expected_green_shape:
        raise ValueError(
            f"Gtot must have shape {expected_green_shape}; got {total.shape}."
        )
    if vacuum.shape != expected_green_shape:
        raise ValueError(
            f"Gvac must have shape {expected_green_shape}; got {vacuum.shape}."
        )
    if energy.ndim != 1 or energy.size == 0 or not np.all(np.isfinite(energy)):
        raise ValueError("E must be a non-empty finite one-dimensional energy grid.")

    structure = None if Gstructure is None else np.asarray(Gstructure)
    if structure is not None and structure.shape != expected_green_shape:
        raise ValueError(
            f"Gstructure must have shape {expected_green_shape}; got {structure.shape}."
        )
    wavelengths = None if wavelength_m is None else np.asarray(wavelength_m, dtype=float)
    if wavelengths is not None and wavelengths.shape != (energy.size,):
        raise ValueError(
            f"wavelength_m must have shape ({energy.size},); got {wavelengths.shape}."
        )
    regions = None if observer_region is None else np.asarray(observer_region)
    expected_region_shape = (energy.size, emitter_count, emitter_count)
    if regions is not None and regions.shape != expected_region_shape:
        raise ValueError(
            f"observer_region must have shape {expected_region_shape}; got {regions.shape}."
        )

    with h5py.File(h5_path, "w") as f:
        f.attrs["gf_layout"] = "pair"
        f.create_dataset("green_function_total", data=total)
        f.create_dataset("green_function_vacuum", data=vacuum)
        f.create_dataset("energy_eV", data=energy)
        f.create_dataset("emitter_positions_nm", data=positions)
        if orientations is not None:
            f.create_dataset("emitter_orientations", data=orientations)
        pos = f.create_group("position_fixed")
        pos.attrs["zD_meters"] = zD
        pos.attrs["zA_meters"] = zA
        _write_optional_common_metadata(
            f,
            Gstructure=structure,
            wavelength_m=wavelengths,
            observer_region=regions,
            attrs=attrs,
        )


# ── Fixed-source scan layout ─────────────────────────────────────────

def save_gf_scan_h5(
    h5_path: str,
    Gtot: np.ndarray,
    Gvac: np.ndarray,
    E: np.ndarray,
    observer_positions_nm: np.ndarray,
    source_position_nm: np.ndarray,
    zD: float,
    zA: float,
    Gstructure: np.ndarray | None = None,
    wavelength_m: np.ndarray | None = None,
    observer_region: np.ndarray | None = None,
    observer_positions_m: np.ndarray | None = None,
    source_position_m: np.ndarray | None = None,
    projected: np.ndarray | None = None,
    purcell: np.ndarray | None = None,
    attrs: Dict[str, Any] | None = None,
) -> None:
    with h5py.File(h5_path, "w") as f:
        f.attrs["gf_layout"] = "scan"
        total = f.create_dataset("green_function_total", data=Gtot)
        vacuum = f.create_dataset("green_function_vacuum", data=Gvac)
        f.create_dataset("energy_eV", data=E)
        f.create_dataset("observer_positions_nm", data=observer_positions_nm)
        f.create_dataset("source_position_nm", data=source_position_nm)
        pos = f.create_group("position_fixed")
        pos.attrs["zD_meters"] = zD
        pos.attrs["zA_meters"] = zA

        f["G_total"] = total
        f["G_vacuum"] = vacuum
        if observer_positions_m is not None:
            f.create_dataset("observer_positions_m", data=observer_positions_m)
        if source_position_m is not None:
            f.create_dataset("source_position_m", data=source_position_m)
        if projected is not None:
            f.create_dataset("projected_G", data=projected)
            f.create_dataset("projected_ImG", data=np.imag(projected))
            f.create_dataset("projected_abs2", data=np.abs(projected) ** 2)
        if purcell is not None:
            f.create_dataset("purcell", data=purcell)
        _write_optional_common_metadata(
            f,
            Gstructure=Gstructure,
            wavelength_m=wavelength_m,
            observer_region=observer_region,
            attrs=attrs,
        )


def _write_optional_common_metadata(
    h5,
    Gstructure: np.ndarray | None,
    wavelength_m: np.ndarray | None,
    observer_region: np.ndarray | None,
    attrs: Dict[str, Any] | None,
) -> None:
    if Gstructure is not None:
        structure = h5.create_dataset("green_function_structure", data=Gstructure)
        h5["G_structure"] = structure
    if wavelength_m is not None:
        h5.create_dataset("wavelength_m", data=wavelength_m)
        h5.create_dataset("wavelength_nm", data=wavelength_m * 1e9)
    if observer_region is not None:
        h5.create_dataset("observer_region", data=observer_region)
    if attrs is not None:
        for key, value in attrs.items():
            h5.attrs[key] = value


# ── Unified loader ───────────────────────────────────────────────────

def load_gf_h5(h5_path: str) -> Dict[str, np.ndarray]:
    """Load dyadic Green's function from HDF5, auto-detecting layout.

    Returns:
        Dictionary with keys that depend on the layout:

        **Common keys** (both layouts):
            - ``G_total``:  Total Green's function array.
            - ``G_vac``:    Vacuum Green's function array.
            - ``energy_eV``: Energy array, shape ``(M,)``.
            - ``zD``:       Source z-position (meters).
            - ``zA``:       Observer z-position (meters).
            - ``gf_layout``: ``"separation"`` or ``"pair"``.

        **Separation-indexed** adds:
            - ``Rx_nm``: Separation grid, shape ``(K,)``.

        **Pair-indexed** adds:
            - ``emitter_positions_nm``: Emitter coordinates, shape ``(N, 3)``.
            - ``emitter_orientations``: Optional emitter orientations, shape ``(N, 3)``.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            layout = f.attrs.get("gf_layout", "separation")
            if isinstance(layout, bytes):
                layout = layout.decode()
            if "gf_layout" not in f.attrs and "observer_positions_m" in f:
                layout = "scan"

            total_key = "green_function_total" if "green_function_total" in f else "G_total"
            vacuum_key = "green_function_vacuum" if "green_function_vacuum" in f else "G_vacuum"
            Gtot = f[total_key][:]
            Gvac = f[vacuum_key][:]
            E = f["energy_eV"][:].astype(float)
            if "position_fixed" in f:
                pos = f["position_fixed"]
                zD = float(pos.attrs["zD_meters"])
                zA = float(pos.attrs["zA_meters"])
            elif layout == "scan" and "source_position_m" in f and "observer_positions_m" in f:
                zD = float(f["source_position_m"][:][2])
                zA = float(f["observer_positions_m"][:][0, 2])
            else:
                pos = f["position_fixed"]
                zD = float(pos.attrs["zD_meters"])
                zA = float(pos.attrs["zA_meters"])

            result = {
                "G_total": Gtot,
                "G_vac": Gvac,
                "energy_eV": E,
                "zD": zD,
                "zA": zA,
                "gf_layout": layout,
            }

            if "green_function_structure" in f:
                result["G_structure"] = f["green_function_structure"][:]
            elif "G_structure" in f:
                result["G_structure"] = f["G_structure"][:]
            if "green_function_scattering_te" in f:
                result["G_scattering_te"] = f["green_function_scattering_te"][:]
            if "green_function_scattering_tm" in f:
                result["G_scattering_tm"] = f["green_function_scattering_tm"][:]

            if layout == "pair":
                result["emitter_positions_nm"] = f["emitter_positions_nm"][:].astype(float)
                if "emitter_orientations" in f:
                    result["emitter_orientations"] = f["emitter_orientations"][:].astype(float)
                logger.success(
                    f"Loaded pair-indexed GF from {h5_path}: "
                    f"{Gtot.shape[1]} emitters, {len(E)} energies"
                )
            elif layout == "scan":
                if "observer_positions_nm" in f:
                    result["observer_positions_nm"] = f["observer_positions_nm"][:].astype(float)
                else:
                    result["observer_positions_nm"] = f["observer_positions_m"][:].astype(float) * 1e9
                if "source_position_nm" in f:
                    result["source_position_nm"] = f["source_position_nm"][:].astype(float)
                else:
                    result["source_position_nm"] = f["source_position_m"][:].astype(float) * 1e9
                logger.success(
                    f"Loaded scan-indexed GF from {h5_path}: "
                    f"{Gtot.shape[1]} observer positions, {len(E)} energies"
                )
            else:
                result["Rx_nm"] = f["Rx_nm"][:].astype(float)
                logger.success(
                    f"Loaded separation-indexed GF from {h5_path}: "
                    f"{len(result['Rx_nm'])} separations, {len(E)} energies"
                )

    except FileNotFoundError:
        logger.exception(f"HDF5 file not found: {h5_path}")
        raise
    except KeyError as e:
        logger.exception(f"Missing dataset in HDF5 file: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error loading Green's function data: {e}")
        raise

    return result
