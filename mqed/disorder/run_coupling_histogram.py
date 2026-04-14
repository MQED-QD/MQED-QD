"""Disorder-averaged nearest-neighbour coupling histogram.

Builds the DDI coupling matrix V for many orientation-disorder
realizations (joblib), collects all nearest-neighbour couplings
V(i, i+1), and produces a histogram.  No quantum dynamics are run.

Usage (CLI via Hydra)
---------------------
Default config (configs/Lindblad/coupling_histogram.yaml)::

    python -m mqed.disorder.run_coupling_histogram

Override parameters on the command line::

    python -m mqed.disorder.run_coupling_histogram \\
        simulation.disorder_sigma_phi_deg=20 \\
        simulation.Nmol=200 \\
        disorder.n_realizations=500 \\
        output.meV=true output.bins=80

Usage (programmatic / notebook)
-------------------------------
::

    from mqed.disorder.run_coupling_histogram import (
        _build_coupling_payload,
        _run_nn_coupling_ensemble,
        plot_nn_coupling_histogram,
    )
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir="<abs_path>/configs/Lindblad"):
        cfg = compose(config_name="coupling_histogram")

    payload = _build_coupling_payload(cfg)
    seeds   = [2025 + i for i in range(100)]
    nn_all  = _run_nn_coupling_ensemble(seeds, payload, n_jobs=-1)
    plot_nn_coupling_histogram(nn_all, meV=True, bins=60)

Outputs
-------
- ``nn_couplings_*.hdf5`` — raw coupling array (n_realizations, Nmol-1) in eV,
  with simulation metadata stored as HDF5 attributes.
  MATLAB: ``data = h5read('nn_couplings_*.hdf5', '/nn_couplings_eV');``
- ``nn_coupling_histogram_*.png`` — density histogram with mean / ±1σ lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import DictConfig

from mqed.Lindblad.ddi_matrix import build_ddi_matrix_from_Gslice
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.joblib_track import tqdm_joblib
from mqed.utils.logging_utils import setup_loggers_hydra_aware

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
#  Lightweight payload – only what the V-matrix builder needs
# ---------------------------------------------------------------------------
def _build_coupling_payload(cfg: DictConfig) -> Dict[str, Any]:
    """Extract the minimal set of parameters needed by ``build_ddi_matrix_from_Gslice``.

    Returns:
        Dictionary with keys: g_slice, Rx_nm, emitter_ev, Nmol, d_nm,
        mu_D_debye, mu_A_debye, theta_deg, phi_deg,
        disorder_sigma_phi_deg, mode.
    """
    data = load_gf_h5(cfg.greens.h5_path)
    g_slice = np.asarray(data["G_total"][0])
    emitter_ev = float(np.asarray(data["energy_eV"])[0])
    rx_nm = np.asarray(data["Rx_nm"])

    return {
        "g_slice": g_slice,
        "Rx_nm": rx_nm,
        "emitter_ev": emitter_ev,
        "Nmol": int(cfg.simulation.Nmol),
        "d_nm": float(cfg.simulation.d_nm),
        "mu_D_debye": float(cfg.simulation.mu_D_debye),
        "mu_A_debye": float(cfg.simulation.mu_A_debye),
        "theta_deg": float(cfg.simulation.theta_deg),
        "phi_deg": cfg.simulation.phi_deg,
        "disorder_sigma_phi_deg": cfg.simulation.get("disorder_sigma_phi_deg", None),
        "mode": str(cfg.simulation.mode),
    }


# ---------------------------------------------------------------------------
#  Single-seed worker
# ---------------------------------------------------------------------------
def _build_V_nn_one(seed: int, payload: Dict[str, Any]) -> np.ndarray:
    """Build the V matrix for one disorder realization and return NN couplings.

    Args:
        seed: RNG seed controlling orientation disorder.
        payload: Output of ``_build_coupling_payload``.

    Returns:
        1-D array of shape ``(Nmol - 1,)`` containing ``V[i, i+1]`` in eV.
    """
    V_eV, _ = build_ddi_matrix_from_Gslice(
        G_slice=payload["g_slice"],
        Rx_nm=payload["Rx_nm"],
        energy_emitter=payload["emitter_ev"],
        N_mol=payload["Nmol"],
        d_nm=payload["d_nm"],
        mu_D_debye=payload["mu_D_debye"],
        mu_A_debye=payload["mu_A_debye"],
        mode=payload["mode"],
        theta_deg=payload["theta_deg"],
        phi_deg=payload["phi_deg"],
        disorder_sigma_phi_deg=payload["disorder_sigma_phi_deg"],
        disorder_seed=seed,
    )
    N = payload["Nmol"]
    nn_couplings = np.array([V_eV[i, i + 1] for i in range(N - 1)])
    return nn_couplings


# ---------------------------------------------------------------------------
#  Ensemble (joblib only)
# ---------------------------------------------------------------------------
def _run_nn_coupling_ensemble(
    seeds: List[int],
    payload: Dict[str, Any],
    n_jobs: int = -1,
) -> np.ndarray:
    """Collect NN couplings from all realizations via joblib.

    Returns:
        2-D array of shape ``(n_realizations, Nmol - 1)`` in eV.
    """
    worker = delayed(_build_V_nn_one)
    if tqdm is not None:
        with tqdm_joblib(tqdm(total=len(seeds), desc="NN coupling realizations")):
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                worker(seed, payload) for seed in seeds
            )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            worker(seed, payload) for seed in seeds
        )
    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
#  Histogram plotter
# ---------------------------------------------------------------------------
def plot_nn_coupling_histogram(
    nn_all: np.ndarray,
    *,
    bins: int = 60,
    title: Optional[str] = None,
    xlabel: str = r"$V_{i,\,i+1}$ (eV)",
    outfile: Optional[Path] = None,
    meV: bool = False,
) -> None:
    """Plot a histogram of pooled nearest-neighbour couplings.

    Args:
        nn_all: 2-D array ``(n_realizations, Nmol - 1)`` or 1-D pooled values, in eV.
        bins: Number of histogram bins.
        title: Figure title (auto-generated if *None*).
        xlabel: x-axis label.
        outfile: If given, the figure is saved here.
        meV: If *True*, convert eV → meV for display.
    """
    values = nn_all.ravel()
    scale = 1000.0 if meV else 1.0
    unit = "meV" if meV else "eV"
    values_plot = values * scale

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values_plot, bins=bins, density=True, alpha=0.75, edgecolor="black", linewidth=0.4)

    mean = np.mean(values_plot)
    std = np.std(values_plot)
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.2, label=f"mean = {mean:.4f} {unit}")
    ax.axvline(mean - std, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(mean + std, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

    if title is None:
        n_real = nn_all.shape[0] if nn_all.ndim == 2 else 1
        n_pairs = values.size
        title = (
            f"NN coupling distribution  "
            f"({n_real} realizations, {n_pairs} pairs total)"
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel.replace("eV", unit) if meV else xlabel, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, dpi=200)
        logger.info(f"Histogram saved to {outfile}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Hydra entrypoint
# ---------------------------------------------------------------------------
@hydra.main(
    config_path="../../configs/Lindblad",
    config_name="coupling_histogram",
    version_base=None,
)
def run_coupling_histogram(cfg: Optional[DictConfig] = None) -> None:
    """Hydra entrypoint: build NN coupling histogram over disorder realizations."""
    if cfg is None:
        raise ValueError("Hydra did not provide configuration.")

    setup_loggers_hydra_aware()
    logger.info("--- NN coupling histogram ---")

    payload = _build_coupling_payload(cfg)
    n = int(cfg.disorder.n_realizations)
    base_seed = int(cfg.disorder.seed)
    seeds = [base_seed + i for i in range(n)]
    n_jobs = int(cfg.disorder.get("n_jobs", -1))

    logger.info(
        f"Nmol={payload['Nmol']}, d_nm={payload['d_nm']}, "
        f"sigma_phi={payload['disorder_sigma_phi_deg']}°, "
        f"n_realizations={n}, n_jobs={n_jobs}"
    )

    nn_all = _run_nn_coupling_ensemble(seeds, payload, n_jobs=n_jobs)
    logger.info(
        f"Collected {nn_all.size} NN couplings "
        f"({nn_all.shape[0]} realizations × {nn_all.shape[1]} pairs)"
    )

    # --- statistics ---
    pooled = nn_all.ravel()
    logger.info(
        f"V_nn stats: mean={np.mean(pooled):.6e} eV, "
        f"std={np.std(pooled):.6e} eV, "
        f"min={np.min(pooled):.6e} eV, max={np.max(pooled):.6e} eV"
    )

    # --- output ---
    outdir = Path(HydraConfig.get().runtime.output_dir)
    use_meV = bool(cfg.output.get("meV", False))
    bins = int(cfg.output.get("bins", 60))

    # Save raw data as HDF5 (MATLAB-friendly)
    data_file = outdir / str(cfg.output.get("data_filename", "nn_couplings.hdf5"))
    with h5py.File(data_file, "w") as f:
        f.create_dataset("nn_couplings_eV", data=nn_all)
        # Store metadata as attributes for easy inspection
        f.attrs["n_realizations"] = n
        f.attrs["Nmol"] = payload["Nmol"]
        f.attrs["d_nm"] = payload["d_nm"]
        f.attrs["disorder_sigma_phi_deg"] = payload["disorder_sigma_phi_deg"]
        f.attrs["phi_deg"] = str(payload["phi_deg"])
        f.attrs["theta_deg"] = payload["theta_deg"]
        f.attrs["base_seed"] = base_seed
        f.attrs["mean_eV"] = float(np.mean(pooled))
        f.attrs["std_eV"] = float(np.std(pooled))
    logger.info(f"Raw NN coupling data saved to {data_file}")

    # Plot histogram
    fig_file = outdir / str(cfg.output.get("fig_filename", "nn_coupling_histogram.png"))
    sigma_str = f"{payload['disorder_sigma_phi_deg']}"
    title = (
        f"NN coupling distribution  "
        rf"($\sigma_\phi = {sigma_str}°$, "
        f"{n} realizations, {pooled.size} pairs)"
    )
    plot_nn_coupling_histogram(
        nn_all, bins=bins, title=title, outfile=fig_file, meV=use_meV,
    )

    logger.success(f"Done. Output directory: {outdir.absolute()}")


if __name__ == "__main__":
    run_coupling_histogram()
