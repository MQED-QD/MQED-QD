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
        position_fixed         group  {zD_meters, zA_meters}

Backward compatibility: files written by older code (no ``gf_layout``
attribute) are treated as separation-indexed.
"""
from __future__ import annotations
import h5py
import numpy as np
from typing import Dict
from loguru import logger


# ── Separation-indexed (legacy) ──────────────────────────────────────

def save_gf_h5(
    h5_path: str,
    Gtot: np.ndarray,
    Gvac: np.ndarray,
    E: np.ndarray,
    Rxnm: np.ndarray,
    zD: float,
    zA: float,
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
    """
    with h5py.File(h5_path, "w") as f:
        f.attrs["gf_layout"] = "separation"
        f.create_dataset("green_function_total", data=Gtot)
        f.create_dataset("green_function_vacuum", data=Gvac)
        f.create_dataset("energy_eV", data=E)
        f.create_dataset("Rx_nm", data=Rxnm)
        pos = f.create_group("position_fixed")
        pos.attrs["zD_meters"] = zD
        pos.attrs["zA_meters"] = zA


# ── Pair-indexed (arbitrary geometry) ────────────────────────────────

def save_gf_pair_h5(
    h5_path: str,
    Gtot: np.ndarray,
    Gvac: np.ndarray,
    E: np.ndarray,
    emitter_positions_nm: np.ndarray,
    zD: float,
    zA: float,
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
    """
    with h5py.File(h5_path, "w") as f:
        f.attrs["gf_layout"] = "pair"
        f.create_dataset("green_function_total", data=Gtot)
        f.create_dataset("green_function_vacuum", data=Gvac)
        f.create_dataset("energy_eV", data=E)
        f.create_dataset("emitter_positions_nm", data=emitter_positions_nm)
        pos = f.create_group("position_fixed")
        pos.attrs["zD_meters"] = zD
        pos.attrs["zA_meters"] = zA


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
    """
    try:
        with h5py.File(h5_path, "r") as f:
            layout = f.attrs.get("gf_layout", "separation")
            if isinstance(layout, bytes):
                layout = layout.decode()

            Gtot = f["green_function_total"][:]
            Gvac = f["green_function_vacuum"][:]
            E = f["energy_eV"][:].astype(float)
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

            if layout == "pair":
                result["emitter_positions_nm"] = f["emitter_positions_nm"][:].astype(float)
                logger.success(
                    f"Loaded pair-indexed GF from {h5_path}: "
                    f"{Gtot.shape[1]} emitters, {len(E)} energies"
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
