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
from mqed.utils.SI_unit import c, eps0, eV_to_J, hbar


# ---------------------------------------------------------------------------
#  Core computation
# ---------------------------------------------------------------------------


def compute_spectral_density_separation(
    G_imag: np.ndarray,
    energy_eV: np.ndarray,
    p_donor: np.ndarray,
    p_acceptor: np.ndarray,
) -> np.ndarray:
    r"""Compute spectral density for separation-indexed Green's function data.

    For each energy slice :math:`\omega_m` and separation index *k*, evaluates:

    .. math::

        J_k(\omega_m)
        = \frac{\omega_m^{2}}{\pi\,\hbar\,\varepsilon_0\,c^{2}}\;
          \hat{\boldsymbol{\mu}}_A
          \cdot
          \operatorname{Im}\!\bigl[
              \mathbf{G}_k(\omega_m)
          \bigr]
          \cdot
          \hat{\boldsymbol{\mu}}_D
          \;\times\; |\mu_A|\,|\mu_D|

    where the dipole moment vectors are
    :math:`\boldsymbol{\mu} = |\mu|\,\hat{\boldsymbol{\mu}}`,
    and the returned quantity has units of **eV** (energy).

    .. note::

        The unit dipole vectors ``p_donor`` and ``p_acceptor`` should already
        encode the dipole magnitude (i.e.
        :math:`\boldsymbol{\mu} = \mu\,\hat{n}` in SI units, C·m).

    Args:
        G_imag: Imaginary part of the Green's function, shape ``(M, K, 3, 3)``.
            *M* = number of energies, *K* = number of separations.
        energy_eV: Energy grid in eV, shape ``(M,)``.
        p_donor: Donor dipole unit vector, shape ``(3,)``.
        p_acceptor: Acceptor dipole unit vector, shape ``(3,)``.

    Returns:
        J: Spectral density array, shape ``(K, M)``, in units of **eV**.
            ``J[k, m]`` is :math:`J_k(\omega_m)`.
    """
    # omega = E / hbar  (SI)
    omega = energy_eV * eV_to_J / hbar  # shape (M,)

    # Prefactor: omega^2 / (pi * hbar * eps0 * c^2)  [SI, gives 1/s per (C·m)^2]
    # We want J in eV, so divide by eV_to_J at the end.
    prefactor = omega**2 / (np.pi * hbar * eps0 * c**2)  # shape (M,)

    # Project Im[G] onto dipole orientations:
    #   p_A · Im[G(m, k, :, :)] · p_D  →  shape (M, K)
    projected = np.einsum("a,mkab,b->mk", p_acceptor, G_imag, p_donor)

    # J(k, omega_m) = prefactor(m) * projected(m, k)
    # Transpose to (K, M) for output convention: separations × energies
    J = (prefactor[:, np.newaxis] * projected).T  # (K, M)

    # Convert from SI (J/s · s = J ... actually [1/s]) to eV
    # prefactor has units [1/(s · C²·m² · ...)] — let's be precise:
    #   prefactor * <G_proj> has units [1/s] because G has units [1/m³]
    #   Actually omega²/(pi*hbar*eps0*c²) * Im[G] has units [rad/s / (C·m)²]
    #   With dipole projection (unit vectors), J has units [rad/s]
    #   Convert to eV: multiply by hbar (J·s) / eV_to_J
    J_eV = J * hbar / eV_to_J  # (K, M) in eV

    return J_eV


def compute_spectral_density_pair(
    G_imag: np.ndarray,
    energy_eV: np.ndarray,
    p_orientations: np.ndarray,
) -> np.ndarray:
    r"""Compute spectral density for pair-indexed Green's function data.

    For each energy slice :math:`\omega_m` and emitter pair
    :math:`(\alpha, \beta)`, evaluates:

    .. math::

        J_{\alpha\beta}(\omega_m)
        = \frac{\omega_m^{2}}{\pi\,\hbar\,\varepsilon_0\,c^{2}}\;
          \hat{\boldsymbol{\mu}}_\alpha
          \cdot
          \operatorname{Im}\!\bigl[
              \mathbf{G}(\mathbf{r}_\alpha,\,\mathbf{r}_\beta,\,\omega_m)
          \bigr]
          \cdot
          \hat{\boldsymbol{\mu}}_\beta

    where :math:`\hat{\boldsymbol{\mu}}_\alpha` is the dipole orientation of
    emitter :math:`\alpha`.

    Args:
        G_imag: Imaginary part of the Green's function, shape ``(M, N, N, 3, 3)``.
            *M* = number of energies, *N* = number of emitters.
        energy_eV: Energy grid in eV, shape ``(M,)``.
        p_orientations: Dipole orientation unit vectors for each emitter,
            shape ``(N, 3)``.  For the stationary case where all emitters
            share the same orientation, broadcast a single ``(3,)`` vector.

    Returns:
        J: Spectral density array, shape ``(N, N, M)``, in units of **eV**.
            ``J[alpha, beta, m]`` is :math:`J_{\alpha\beta}(\omega_m)`.
    """
    omega = energy_eV * eV_to_J / hbar  # (M,)
    prefactor = omega**2 / (np.pi * hbar * eps0 * c**2)  # (M,)

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

    logger.info(f"GF layout: {gf_layout}")
    logger.info(f"Energy grid: {energy_eV[0]:.4f} – {energy_eV[-1]:.4f} eV "
                f"({len(energy_eV)} points)")

    G_imag = np.imag(G_total)

    # --- Resolve orientations ---
    p_donor, p_acceptor = _resolve_orientations(cfg)

    # --- Compute spectral density ---
    if gf_layout == "separation":
        Rx_nm = gf_data["Rx_nm"]
        logger.info(f"Separation grid: {Rx_nm[0]:.2f} – {Rx_nm[-1]:.2f} nm "
                    f"({len(Rx_nm)} points)")

        J_eV = compute_spectral_density_separation(
            G_imag, energy_eV, p_donor, p_acceptor
        )
        logger.success(f"Spectral density computed: shape {J_eV.shape} (K, M)")

        # --- Save results ---
        output_file = output_dir / cfg.get("output_file", "spectral_density.h5")
        results = {
            "J_eV": J_eV,
            "energy_eV": energy_eV,
            "Rx_nm": Rx_nm,
            "gf_layout": gf_layout,
            "p_donor": p_donor,
            "p_acceptor": p_acceptor,
        }

    elif gf_layout == "pair":
        N = G_total.shape[1]
        logger.info(f"Number of emitters: {N}")

        # For pair layout, use a uniform orientation for all emitters
        # (stationary mode).  A future extension could support per-emitter
        # orientations from a file.
        p_orientations = np.tile(p_donor, (N, 1))  # (N, 3)

        J_eV = compute_spectral_density_pair(
            G_imag, energy_eV, p_orientations
        )
        logger.success(f"Spectral density computed: shape {J_eV.shape} (N, N, M)")

        output_file = output_dir / cfg.get("output_file", "spectral_density.h5")
        emitter_positions_nm = gf_data.get("emitter_positions_nm", None)
        results = {
            "J_eV": J_eV,
            "energy_eV": energy_eV,
            "gf_layout": gf_layout,
            "p_orientations": p_orientations,
        }
        if emitter_positions_nm is not None:
            results["emitter_positions_nm"] = emitter_positions_nm

    else:
        raise ValueError(f"Unknown GF layout: {gf_layout}")

    _save_spectral_density_h5(output_file, results)
    logger.success(f"Saved spectral density to: {output_file}")


if __name__ == "__main__":
    compute_and_save_spectral_density()
