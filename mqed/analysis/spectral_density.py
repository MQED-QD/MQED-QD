"""
Spectral Density from the Dyadic Green's Function
===================================================

Computes the generalized spectral density tensor :math:`J_{\\alpha\\beta}(\\omega)`
from dyadic Green's function data stored in HDF5 files.

Physics Background
------------------

In macroscopic QED, the interaction between quantum emitters and the
electromagnetic environment is fully characterised by the dyadic Green's
function :math:`\\mathbf{G}(\\mathbf{r}_\\alpha, \\mathbf{r}_\\beta, \\omega)`.
The **generalized spectral density** is defined as:

.. math::

    J_{\\alpha\\beta}(\\omega)
    = \\frac{\\omega^{2}}{\\pi\\,\\hbar\\,\\varepsilon_0\\,c^{2}}\\;
      \\boldsymbol{\\mu}_\\alpha
      \\cdot
      \\operatorname{Im}\\!\\left[
          \\mathbf{G}(\\mathbf{r}_\\alpha,\\,\\mathbf{r}_\\beta,\\,\\omega)
      \\right]
      \\cdot
      \\boldsymbol{\\mu}_\\beta

where :math:`\\alpha, \\beta` label molecular emitters at positions
:math:`\\mathbf{r}_\\alpha, \\mathbf{r}_\\beta` with transition dipole moments
:math:`\\boldsymbol{\\mu}_\\alpha, \\boldsymbol{\\mu}_\\beta`.

Physical interpretation
^^^^^^^^^^^^^^^^^^^^^^^

* **Self-term** :math:`J_{\\alpha\\alpha}(\\omega)`:
  Encodes the local photonic density of states projected onto the emitter's
  dipole orientation.  Proportional to the Purcell-enhanced spontaneous
  emission rate:

  .. math::

      \\Gamma_{\\alpha\\alpha}(\\omega) = 2\\pi\\, J_{\\alpha\\alpha}(\\omega)

* **Cross-term** :math:`J_{\\alpha\\beta}(\\omega)` with
  :math:`\\alpha \\neq \\beta`:
  Encodes the environment-mediated dissipative coupling between emitters
  :math:`\\alpha` and :math:`\\beta`.  Related to the Lindblad dissipation
  matrix element:

  .. math::

      \\Gamma_{\\alpha\\beta}(\\omega) = 2\\pi\\, J_{\\alpha\\beta}(\\omega)

Connection to the Markov approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Lindblad master equation used in :mod:`mqed.Lindblad` evaluates
:math:`J_{\\alpha\\beta}` at the single emitter frequency :math:`\\omega_M`:

.. math::

    \\Gamma_{\\alpha\\beta}
    = \\frac{2\\,\\omega_M^{2}}{\\hbar\\,\\varepsilon_0\\,c^{2}}\\;
      \\boldsymbol{\\mu}_\\alpha
      \\cdot
      \\operatorname{Im}\\!\\left[
          \\mathbf{G}(\\mathbf{r}_\\alpha,\\,\\mathbf{r}_\\beta,\\,\\omega_M)
      \\right]
      \\cdot
      \\boldsymbol{\\mu}_\\beta

This is valid when :math:`J_{\\alpha\\beta}(\\omega)` varies slowly near
:math:`\\omega_M`.  Computing the full spectral density allows one to:

1. **Verify the Markov approximation** — check that
   :math:`J_{\\alpha\\beta}(\\omega)` is smooth near :math:`\\omega_M`.
2. **Compute the Casimir–Polder (Lamb) shift** via the principal-value
   integral:

   .. math::

       \\Lambda_\\alpha^{\\mathrm{Sc}}
       = \\mathcal{P}\\!\\int_0^\\infty \\!\\mathrm{d}\\omega\\;
         J_{\\alpha\\alpha}^{\\mathrm{Sc}}(\\omega)\\,
         \\left(
             \\frac{1}{\\omega + \\omega_M}
             - \\frac{1}{\\omega - \\omega_M}
         \\right)

3. **Serve as input for non-Markovian methods** (HEOM, TEDOPA, etc.)
   that require the full bath spectral function.

Data layouts
^^^^^^^^^^^^

This module supports both Green's function storage layouts:

* **Separation-indexed** (``gf_layout='separation'``):
  :math:`G(K, 3, 3)` per energy, indexed by donor–acceptor separation
  :math:`R_x`.  Output shape: ``(K, M)`` where *K* = number of separations
  and *M* = number of energies.

* **Pair-indexed** (``gf_layout='pair'``):
  :math:`G(N, N, 3, 3)` per energy, for all emitter pairs.
  Output shape: ``(N, N, M)`` where *N* = number of emitters.
"""

from pathlib import Path

import h5py
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.orientation import resolve_angle_deg, spherical_to_cartesian_dipole
from mqed.utils.SI_unit import D2CMM, c, eps0, eV_to_J, hbar


# ---------------------------------------------------------------------------
#  Core computation
# ---------------------------------------------------------------------------


def compute_spectral_density_separation(
    G_imag: np.ndarray,
    energy_eV: np.ndarray,
    p_donor: np.ndarray,
    p_acceptor: np.ndarray,
    mu_D_debye: float = 1.0,
    mu_A_debye: float = 1.0,
) -> np.ndarray:
    r"""Compute spectral density for separation-indexed Green's function data.

    For each energy slice :math:`\omega_m` and separation index *k*, evaluates:

    .. math::

        J_k(\omega_m)
        = \frac{|\mu_D|\,|\mu_A|\,\omega_m^{2}}
               {\pi\,\hbar\,\varepsilon_0\,c^{2}}\;
          \hat{\boldsymbol{\mu}}_A
          \cdot
          \operatorname{Im}\!\bigl[
              \mathbf{G}_k(\omega_m)
          \bigr]
          \cdot
          \hat{\boldsymbol{\mu}}_D

    where the dipole moment vectors are
    :math:`\boldsymbol{\mu} = |\mu|\,\hat{\boldsymbol{\mu}}`,
    and the returned quantity has units of **eV** (energy).

    Args:
        G_imag: Imaginary part of the Green's function, shape ``(M, K, 3, 3)``.
            *M* = number of energies, *K* = number of separations.
        energy_eV: Energy grid in eV, shape ``(M,)``.
        p_donor: Donor dipole orientation unit vector, shape ``(3,)``.
        p_acceptor: Acceptor dipole orientation unit vector, shape ``(3,)``.
        mu_D_debye: Donor dipole moment magnitude in Debye.
        mu_A_debye: Acceptor dipole moment magnitude in Debye.

    Returns:
        J: Spectral density array, shape ``(K, M)``, in units of **eV**.
            ``J[k, m]`` is :math:`J_k(\omega_m)`.
    """
    # Convert dipole magnitudes from Debye to SI (C·m)
    mu_D_SI = mu_D_debye * D2CMM
    mu_A_SI = mu_A_debye * D2CMM
    mu2 = mu_D_SI * mu_A_SI  # |mu_D| * |mu_A| in (C·m)^2

    # omega = E / hbar  (SI)
    omega = energy_eV * eV_to_J / hbar  # shape (M,)

    # Prefactor: mu2 * omega^2 / (pi * hbar * eps0 * c^2)
    # Units: (C·m)^2 * (rad/s)^2 / (J·s * C^2/(J·m) * (m/s)^2) → rad/s
    prefactor = mu2 * omega**2 / (np.pi * hbar * eps0 * c**2)  # shape (M,)

    # Project Im[G] onto dipole orientations (unit vectors):
    #   p_A · Im[G(m, k, :, :)] · p_D  →  shape (M, K)
    projected = np.einsum("a,mkab,b->mk", p_acceptor, G_imag, p_donor)

    # J(k, omega_m) = prefactor(m) * projected(m, k)
    # Transpose to (K, M) for output convention: separations × energies
    J = (prefactor[:, np.newaxis] * projected).T  # (K, M)

    # Convert from rad/s to eV: multiply by hbar / eV_to_J
    J_eV = J * hbar / eV_to_J  # (K, M) in eV

    return J_eV


def compute_spectral_density_pair(
    G_imag: np.ndarray,
    energy_eV: np.ndarray,
    p_orientations: np.ndarray,
    mu_debye: float = 1.0,
) -> np.ndarray:
    r"""Compute spectral density for pair-indexed Green's function data.

    For each energy slice :math:`\omega_m` and emitter pair
    :math:`(\alpha, \beta)`, evaluates:

    .. math::

        J_{\alpha\beta}(\omega_m)
        = \frac{|\mu|^{2}\,\omega_m^{2}}
               {\pi\,\hbar\,\varepsilon_0\,c^{2}}\;
          \hat{\boldsymbol{\mu}}_\alpha
          \cdot
          \operatorname{Im}\!\bigl[
              \mathbf{G}(\mathbf{r}_\alpha,\,\mathbf{r}_\beta,\,\omega_m)
          \bigr]
          \cdot
          \hat{\boldsymbol{\mu}}_\beta

    where :math:`\hat{\boldsymbol{\mu}}_\alpha` is the dipole orientation of
    emitter :math:`\alpha` and all emitters share the same magnitude
    :math:`|\mu|`.

    Args:
        G_imag: Imaginary part of the Green's function, shape ``(M, N, N, 3, 5)``.
            *M* = number of energies, *N* = number of emitters.
        energy_eV: Energy grid in eV, shape ``(M,)``.
        p_orientations: Dipole orientation unit vectors for each emitter,
            shape ``(N, 3)``.  For the stationary case where all emitters
            share the same orientation, broadcast a single ``(3,)`` vector.
        mu_debye: Dipole moment magnitude in Debye (same for all emitters).

    Returns:
        J: Spectral density array, shape ``(N, N, M)``, in units of **eV**.
            ``J[alpha, beta, m]`` is :math:`J_{\alpha\beta}(\omega_m)`.
    """
    # Convert dipole magnitude from Debye to SI (C·m)
    mu_SI = mu_debye * D2CMM
    mu2 = mu_SI * mu_SI  # |mu|^2 in (C·m)^2

    omega = energy_eV * eV_to_J / hbar  # (M,)
    prefactor = mu2 * omega**2 / (np.pi * hbar * eps0 * c**2)  # (M,)

    # Project Im[G] onto per-emitter dipole orientations:
    #   p[alpha, a] * Im[G(m, alpha, beta, a, b)] * p[beta, b]
    #   → shape (M, N, N)
    projected = np.einsum(
        "ia,mijab,jb->mij", p_orientations, G_imag, p_orientations
    )  # (M, N, N)

    # J(alpha, beta, omega_m) = prefactor(m) * projected(m, alpha, beta)
    J = prefactor[:, np.newaxis, np.newaxis] * projected  # (M, N, N)

    # Transpose to (N, N, M) and convert to eV
    J_eV = np.transpose(J, (1, 2, 0)) * hbar / eV_to_J  # (N, N, M)

    return J_eV


# ---------------------------------------------------------------------------
#  Hydra CLI entry point
# ---------------------------------------------------------------------------


def _resolve_orientations(cfg):
    """Resolve donor/acceptor dipole orientations from config.

    Follows the same convention as FE.py — supports 'magic' angle and
    explicit theta/phi in degrees.
    """
    theta_D = resolve_angle_deg(cfg.orientations.donor.theta_deg)
    phi_D = resolve_angle_deg(cfg.orientations.donor.phi_deg)
    theta_A = resolve_angle_deg(cfg.orientations.acceptor.theta_deg)
    phi_A = resolve_angle_deg(cfg.orientations.acceptor.phi_deg)

    p_donor = spherical_to_cartesian_dipole(theta_D, phi_D)
    p_acceptor = spherical_to_cartesian_dipole(theta_A, phi_A)

    logger.info(f"Donor orientation:    theta={theta_D:.2f}°, phi={phi_D:.2f}°")
    logger.info(f"Acceptor orientation: theta={theta_A:.2f}°, phi={phi_A:.2f}°")

    return p_donor, p_acceptor


def _save_spectral_density_h5(filepath: Path, data: dict) -> None:
    """Save spectral density results to HDF5.

    Handles numpy arrays as datasets and scalars/strings as attributes.
    Follows the same HDF5 conventions as :func:`mqed.utils.dgf_data.save_gf_h5`.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(key, data=val)
            elif isinstance(val, str):
                f.attrs[key] = val
            elif val is not None:
                f.attrs[key] = val


@hydra.main(
    config_path="../../configs/analysis",
    config_name="spectral_density",
    version_base=None,
)
def compute_and_save_spectral_density(cfg) -> None:
    """Compute the generalized spectral density and save to HDF5.

    This is the Hydra CLI entry point.  Configuration is loaded from
    ``configs/analysis/spectral_density.yaml``.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("Computing generalized spectral density J_αβ(ω)")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Resolve input path ---
    input_path = Path(cfg.input_file)
    if not input_path.is_absolute():
        input_path = Path(hydra.utils.get_original_cwd()) / input_path
    logger.info(f"Loading GF data from: {input_path}")

    # --- Load Green's function data ---
    gf_data = load_gf_h5(str(input_path))
    G_total = gf_data["G_total"]
    energy_eV = gf_data["energy_eV"]
    gf_layout = gf_data["gf_layout"]
    zD_m = gf_data.get("zD", None)
    zD_nm = zD_m * 1e9 if zD_m is not None else None

    logger.info(f"GF layout: {gf_layout}")
    logger.info(f"Energy grid: {energy_eV[0]:.4f} – {energy_eV[-1]:.4f} eV "
                f"({len(energy_eV)} points)")

    G_imag = np.imag(G_total)

    # --- Resolve orientations and dipole magnitudes ---
    p_donor, p_acceptor = _resolve_orientations(cfg)
    mu_D_debye = cfg.get("mu_D_debye", 1.0)
    mu_A_debye = cfg.get("mu_A_debye", 1.0)
    logger.info(f"Dipole magnitudes: mu_D = {mu_D_debye} D, mu_A = {mu_A_debye} D")

    # --- Build dynamic output filename ---
    # Hydra config provides the prefix (e.g. "spec_dens"), Python appends
    # the energy range and point count at runtime.
    output_prefix = cfg.get("output_prefix", "spec_dens")
    E_min = energy_eV[0]
    E_max = energy_eV[-1]
    n_pts = len(energy_eV)
    if zD_nm is not None:
        output_fname = f"{output_prefix}_Emin_{E_min:.2f}_Emax_{E_max:.2f}_{n_pts}pts_height_{zD_nm:.0f}nm.hdf5"
    else:
        output_fname = f"{output_prefix}_Emin_{E_min:.2f}_Emax_{E_max:.2f}_{n_pts}pts_height_unknown.hdf5"

    # --- Compute spectral density ---
    if gf_layout == "separation":
        Rx_nm = gf_data["Rx_nm"]
        logger.info(f"Separation grid: {Rx_nm[0]:.2f} – {Rx_nm[-1]:.2f} nm "
                    f"({len(Rx_nm)} points)")

        J_eV = compute_spectral_density_separation(
            G_imag, energy_eV, p_donor, p_acceptor,
            mu_D_debye=mu_D_debye, mu_A_debye=mu_A_debye,
        )
        logger.success(f"Spectral density computed: shape {J_eV.shape} (K, M)")

        # --- Save results ---
        output_file = output_dir / output_fname
        results = {
            "J_eV": J_eV,
            "energy_eV": energy_eV,
            "Rx_nm": Rx_nm,
            "gf_layout": gf_layout,
            "p_donor": p_donor,
            "p_acceptor": p_acceptor,
            "mu_D_debye": mu_D_debye,
            "mu_A_debye": mu_A_debye,
        }

    elif gf_layout == "pair":
        N = G_total.shape[1]
        logger.info(f"Number of emitters: {N}")

        # For pair layout, use a uniform orientation for all emitters
        # (stationary mode).  A future extension could support per-emitter
        # orientations from a file.
        p_orientations = np.tile(p_donor, (N, 1))  # (N, 3)

        J_eV = compute_spectral_density_pair(
            G_imag, energy_eV, p_orientations,
            mu_debye=mu_D_debye,
        )
        logger.success(f"Spectral density computed: shape {J_eV.shape} (N, N, M)")

        output_file = output_dir / output_fname
        emitter_positions_nm = gf_data.get("emitter_positions_nm", None)
        results = {
            "J_eV": J_eV,
            "energy_eV": energy_eV,
            "gf_layout": gf_layout,
            "p_orientations": p_orientations,
            "mu_debye": mu_D_debye,
        }
        if emitter_positions_nm is not None:
            results["emitter_positions_nm"] = emitter_positions_nm

    else:
        raise ValueError(f"Unknown GF layout: {gf_layout}")

    _save_spectral_density_h5(output_file, results)
    logger.success(f"Saved spectral density to: {output_file}")


if __name__ == "__main__":
    compute_and_save_spectral_density()
