r"""
Dipole–dipole interaction (DDI) matrices for Lindblad dynamics.

Two Green's function input formats are supported:

**Separation-indexed** ``G_slice`` of shape ``(K, 3, 3)``
    One tensor per distinct separation Rx.  Exploits translational symmetry:
    all emitter pairs ``(i, j)`` with the same ``|i−j|`` share the same
    Green's function.  Appropriate for planar surfaces and any geometry with
    in-plane translational symmetry.

**Pair-indexed** ``G_pair`` of shape ``(N, N, 3, 3)``
    One tensor per emitter pair.  No symmetry assumed.  Required for
    nanorods, nanoparticles, or any geometry where the self-term
    (Purcell factor) and inter-site coupling depend on absolute emitter
    position — not just their separation.

The public entry point :func:`build_ddi_matrix` auto-dispatches based on
which argument is provided.  The legacy function
:func:`build_ddi_matrix_from_Gslice` is preserved for backward
compatibility and is called internally for separation-indexed data.
"""
import numpy as np
from typing import Optional, Union

from loguru import logger
from mqed.utils.SI_unit import c, eps0, hbar, eV_to_J, D2CMM
from mqed.utils.orientation import spherical_to_cartesian_dipole, resolve_angle_deg
from mqed.utils.orientation_disorder import phi_wrapped_normal_deg as _phi_wrapped_normal_deg


# ─────────────────────────────────────────────────────────────────────
#  Orientation helpers (shared by both code paths)
# ─────────────────────────────────────────────────────────────────────

def _resolve_orientations(
    mode: str,
    N_mol: int,
    *,
    uD: Optional[np.ndarray] = None,
    uA: Optional[np.ndarray] = None,
    U_list: Optional[np.ndarray] = None,
    theta_deg: Optional[float] = None,
    phi_deg: Optional[float] = None,
    disorder_sigma_phi_deg: Optional[float] = None,
    disorder_seed: Optional[int] = None,
) -> tuple:
    """Resolve dipole orientations for stationary or disorder mode.

    Returns:
        (mode_flag, uD, uA, U):
            For stationary: ``("stationary", uD(3,), uA(3,), None)``
            For disorder:   ``("disorder", None, None, U(N,3))``
    """
    if mode == "stationary":
        if uD is None or uA is None:
            raise ValueError("mode='stationary' requires uD and uA (shape (3,)).")
        uD = np.asarray(uD, dtype=float).reshape(3)
        uA = np.asarray(uA, dtype=float).reshape(3)
        return "stationary", uD, uA, None

    elif mode == "disorder":
        if U_list is not None:
            U = np.asarray(U_list, dtype=float)
            if U.shape != (N_mol, 3):
                raise ValueError("U_list must have shape (N_mol,3).")
        else:
            if phi_deg is None or disorder_sigma_phi_deg is None:
                raise ValueError(
                    "mode='disorder' needs phi_deg and disorder_sigma_phi_deg "
                    "if U_list is not provided."
                )
            phi_deg = resolve_angle_deg(phi_deg)
            phi_deg = _phi_wrapped_normal_deg(
                N_mol, phi_deg, disorder_sigma_phi_deg, seed=disorder_seed,
            )
            U = spherical_to_cartesian_dipole(theta_deg, phi_deg)
            if U.shape != (N_mol, 3):
                raise ValueError("Generated U_list must have shape (N_mol,3).")
        return "disorder", None, None, U

    else:
        raise ValueError("mode must be 'stationary' or 'disorder'.")


# ─────────────────────────────────────────────────────────────────────
#  Pair-indexed builder  (NEW — arbitrary geometry)
# ─────────────────────────────────────────────────────────────────────

def build_ddi_matrix_from_Gpair(
    G_pair: np.ndarray,
    energy_emitter: float,
    N_mol: int,
    mu_D_debye: float,
    mu_A_debye: Optional[float] = None,
    *,
    mode: str = "stationary",
    uD: Optional[np.ndarray] = None,
    uA: Optional[np.ndarray] = None,
    U_list: Optional[np.ndarray] = None,
    theta_deg: Optional[float] = None,
    phi_deg: Optional[float] = None,
    disorder_sigma_phi_deg: Optional[float] = None,
    disorder_seed: Optional[int] = None,
) -> tuple:
    r"""Build DDI matrices from a **pair-indexed** Green's function.

    Each emitter pair ``(i, j)`` has its own ``G(r_i, r_j, ω)`` tensor.
    No translational symmetry is assumed — the diagonal (self-term,
    Purcell factor) can differ for each emitter.

    Physics (same as separation-indexed):

    .. math::

       V_{ij} = -\frac{\omega^2}{\epsilon_0 c^2}\,
                \boldsymbol{\mu}_i \cdot \Re\,G(r_i, r_j) \cdot \boldsymbol{\mu}_j

    .. math::

       \hbar\Gamma_{ij} = \frac{2\omega^2}{\epsilon_0 c^2}\,
                \boldsymbol{\mu}_i \cdot \Im\,G(r_i, r_j) \cdot \boldsymbol{\mu}_j

    Args:
        G_pair:         Shape ``(N, N, 3, 3)`` complex — full pair-indexed
                        dyadic Green's function for a single energy.
        energy_emitter: Emitter transition energy in eV.
        N_mol:          Number of emitters.
        mu_D_debye:     Donor dipole moment (Debye).
        mu_A_debye:     Acceptor dipole moment (Debye); defaults to donor.
        mode:           ``"stationary"`` or ``"disorder"``.
        uD, uA:         Orientation unit vectors (stationary mode).
        U_list, theta_deg, phi_deg, disorder_*:
                        Orientation controls (disorder mode).

    Returns:
        ``(V_eV, hbarGamma_eV)`` — both ``(N, N)`` real arrays.
    """
    G_pair = np.asarray(G_pair)
    if G_pair.shape != (N_mol, N_mol, 3, 3):
        raise ValueError(
            f"G_pair shape {G_pair.shape} does not match expected "
            f"({N_mol}, {N_mol}, 3, 3)."
        )

    mu_A_debye = mu_D_debye if mu_A_debye is None else mu_A_debye
    muA = mu_A_debye * D2CMM
    muD = mu_D_debye * D2CMM
    mu2 = muA * muD
    omega = energy_emitter * eV_to_J / hbar
    pref = (omega ** 2) / (eps0 * c ** 2)

    mode_flag, uD_v, uA_v, U = _resolve_orientations(
        mode, N_mol,
        uD=uD, uA=uA, U_list=U_list,
        theta_deg=theta_deg, phi_deg=phi_deg,
        disorder_sigma_phi_deg=disorder_sigma_phi_deg,
        disorder_seed=disorder_seed,
    )

    G_re = np.real(G_pair)  # (N, N, 3, 3)
    G_im = np.imag(G_pair)

    V_eV = np.zeros((N_mol, N_mol), dtype=np.float64)
    hbarGamma_eV = np.zeros((N_mol, N_mol), dtype=np.float64)

    if mode_flag == "stationary":
        # μ_A · G · μ_D  for all (i,j) at once via einsum:
        #   u_A[a] * G[i,j,a,b] * u_D[b]  → scalar per (i,j)
        val_re = np.einsum("a,ijab,b->ij", uA_v, G_re, uD_v)
        val_im = np.einsum("a,ijab,b->ij", uA_v, G_im, uD_v)
        V_eV = -(pref * mu2 * val_re) / eV_to_J
        hbarGamma_eV = +(2.0 * pref * mu2 * val_im) / eV_to_J

    else:  # disorder
        # U[i,a] * G[i,j,a,b] * U[j,b]  → scalar per (i,j)
        val_re = np.einsum("ia,ijab,jb->ij", U, G_re, U)
        val_im = np.einsum("ia,ijab,jb->ij", U, G_im, U)
        V_eV = -(pref * mu2 * val_re) / eV_to_J
        hbarGamma_eV = +(2.0 * pref * mu2 * val_im) / eV_to_J

    np.fill_diagonal(V_eV, 0.0)
    return V_eV, hbarGamma_eV


# ─────────────────────────────────────────────────────────────────────
#  Separation-indexed builder  (ORIGINAL — planar / translational sym.)
# ─────────────────────────────────────────────────────────────────────

def build_ddi_matrix_from_Gslice(
    G_slice: np.ndarray,
    Rx_nm: np.ndarray,
    energy_emitter: float,
    N_mol: int,
    d_nm: float,
    mu_D_debye: float,
    mu_A_debye=None,
    *,
    mode: str = "stationary",
    uD: Union[np.ndarray, None] = None,
    uA: Union[np.ndarray, None] = None,
    U_list=None,
    theta_deg: Union[None, float] = None,
    phi_deg: Union[None, float] = None,
    disorder_sigma_phi_deg=None,
    disorder_seed=None,
):
    r"""Build DDI matrices from a **separation-indexed** Green's slice.

    Exploits translational symmetry: ``G(r_i, r_j)`` depends only on
    ``|r_i − r_j|``.  All pairs at the same separation share the same
    tensor.  The self-term at ``Rx = 0`` is used for *every* diagonal
    element — valid only when the Purcell factor is site-independent
    (e.g. planar surfaces, N-layer slabs).

    Args:
        G_slice:  Shape ``(K, 3, 3)`` complex — one tensor per separation.
        Rx_nm:    Shape ``(K,)`` — separations in nm.  Must contain
                  ``0, d, 2d, ..., (N-1)d``.
        energy_emitter: Emitter energy in eV.
        N_mol:    Number of molecules.
        d_nm:     Lattice spacing in nm.
        mu_D_debye: Donor dipole moment (Debye).
        mu_A_debye: Acceptor dipole; defaults to donor value.
        mode:     ``"stationary"`` or ``"disorder"``.
        uD, uA:   Orientation vectors (stationary mode).
        U_list, theta_deg, phi_deg, disorder_*:
                  Orientation controls (disorder mode).

    Returns:
        ``(V_eV, hbarGamma_eV)`` — both ``(N, N)`` real arrays.
    """
    Rx_nm = np.asarray(Rx_nm, dtype=float)
    dist_to_idx = {float(r): k for k, r in enumerate(Rx_nm)}
    needed = [float(s * d_nm) for s in range(N_mol)]
    missing = [r for r in needed if r not in dist_to_idx]
    if missing:
        logger.error("Rx_nm must contain all separations 0, d, 2d, ...")
        raise ValueError(
            f"Rx_nm grid must contain all separations 0, d, 2d, ..., (N-1)d in nm. "
            f"Missing: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )

    mu_A_debye = mu_D_debye if mu_A_debye is None else mu_A_debye
    muA = mu_A_debye * D2CMM
    muD = mu_D_debye * D2CMM
    mu2 = muA * muD
    omega = energy_emitter * eV_to_J / hbar
    pref = (omega ** 2) / (eps0 * c ** 2)

    mode_flag, uD_v, uA_v, U = _resolve_orientations(
        mode, N_mol,
        uD=uD, uA=uA, U_list=U_list,
        theta_deg=theta_deg, phi_deg=phi_deg,
        disorder_sigma_phi_deg=disorder_sigma_phi_deg,
        disorder_seed=disorder_seed,
    )

    V_eV = np.zeros((N_mol, N_mol), dtype=np.float64)
    hbarGamma_eV = np.zeros((N_mol, N_mol), dtype=np.float64)

    G_slice = np.asarray(G_slice)
    if G_slice.ndim != 3:
        raise ValueError(f"G_slice must be 3D, got {G_slice.ndim}D {G_slice.shape}")

    if G_slice.shape[-2:] == (3, 3):
        GK33 = G_slice
    elif G_slice.shape[0:2] == (3, 3):
        GK33 = np.transpose(G_slice, (2, 0, 1))
    else:
        raise ValueError(
            f"Unrecognized G_slice shape {G_slice.shape}; "
            f"expected (K,3,3) or (3,3,K)"
        )

    K = GK33.shape[0]
    if len(Rx_nm) != K:
        raise ValueError(f"Rx_nm length {len(Rx_nm)} does not match K={K}")

    G_re_full = np.real(GK33)
    G_im_full = np.imag(GK33)
    idx = np.arange(N_mol)

    for s in range(N_mol):
        k = dist_to_idx[float(s * d_nm)]
        Gre = G_re_full[k]
        Gim = G_im_full[k]
        if Gre.shape != (3, 3) or Gim.shape != (3, 3):
            raise ValueError("G_slice must have shape (K,3,3).")

        if mode_flag == "stationary":
            val_re = float(np.dot(uA_v, Gre @ uD_v))
            val_im = float(np.dot(uA_v, Gim @ uD_v))
            if s == 0:
                V_eV[idx, idx] = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[idx, idx] = +(2.0 * pref * mu2 * val_im) / eV_to_J
            else:
                i = idx[:-s]
                j = idx[s:]
                V_eV[i, j] = -(pref * mu2 * val_re) / eV_to_J
                V_eV[j, i] = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[i, j] = +(2.0 * pref * mu2 * val_im) / eV_to_J
                hbarGamma_eV[j, i] = +(2.0 * pref * mu2 * val_im) / eV_to_J

        else:  # disorder
            Lre = U @ Gre
            Lim = U @ Gim
            if s == 0:
                val_re = np.einsum("ik,ik->i", Lre, U)
                val_im = np.einsum("ik,ik->i", Lim, U)
                V_eV[idx, idx] = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[idx, idx] = +(2.0 * pref * mu2 * val_im) / eV_to_J
            else:
                i = idx[:-s]
                j = idx[s:]
                vij_re = np.einsum("jk,jk->j", Lre[j], U[i])
                vij_im = np.einsum("jk,jk->j", Lim[j], U[i])
                V_eV[i, j] = -(pref * mu2 * vij_re) / eV_to_J
                hbarGamma_eV[i, j] = +(2.0 * pref * mu2 * vij_im) / eV_to_J

                vji_re = np.einsum("ik,ik->i", Lre[i], U[j])
                vji_im = np.einsum("ik,ik->i", Lim[i], U[j])
                V_eV[j, i] = -(pref * mu2 * vji_re) / eV_to_J
                hbarGamma_eV[j, i] = +(2.0 * pref * mu2 * vji_im) / eV_to_J

    np.fill_diagonal(V_eV, 0.0)
    return V_eV, hbarGamma_eV


# ─────────────────────────────────────────────────────────────────────
#  Unified entry point (auto-dispatch)
# ─────────────────────────────────────────────────────────────────────

def build_ddi_matrix(
    energy_emitter: float,
    N_mol: int,
    mu_D_debye: float,
    mu_A_debye: Optional[float] = None,
    *,
    # --- separation-indexed inputs (planar / translational symmetry) ---
    G_slice: Optional[np.ndarray] = None,
    Rx_nm: Optional[np.ndarray] = None,
    d_nm: Optional[float] = None,
    # --- pair-indexed input (arbitrary geometry) ---
    G_pair: Optional[np.ndarray] = None,
    # --- orientation ---
    mode: str = "stationary",
    uD: Optional[np.ndarray] = None,
    uA: Optional[np.ndarray] = None,
    U_list: Optional[np.ndarray] = None,
    theta_deg: Optional[float] = None,
    phi_deg: Optional[float] = None,
    disorder_sigma_phi_deg: Optional[float] = None,
    disorder_seed: Optional[int] = None,
) -> tuple:
    r"""Build DDI matrices, auto-dispatching by Green's function format.

    Provide **either** ``G_pair`` (pair-indexed) **or** the trio
    ``(G_slice, Rx_nm, d_nm)`` (separation-indexed).  If both are given,
    ``G_pair`` takes precedence.

    Args:
        energy_emitter: Emitter transition energy in eV.
        N_mol:          Number of emitters.
        mu_D_debye:     Donor dipole moment (Debye).
        mu_A_debye:     Acceptor dipole; defaults to donor value.
        G_slice:        Separation-indexed ``(K, 3, 3)`` Green's function.
        Rx_nm:          Separation grid ``(K,)`` in nm.
        d_nm:           Lattice spacing in nm.
        G_pair:         Pair-indexed ``(N, N, 3, 3)`` Green's function.
        mode:           ``"stationary"`` or ``"disorder"``.
        (orientation kwargs): see individual builders.

    Returns:
        ``(V_eV, hbarGamma_eV)`` — both ``(N, N)`` real arrays.
    """
    orient_kw = dict(
        mode=mode, uD=uD, uA=uA, U_list=U_list,
        theta_deg=theta_deg, phi_deg=phi_deg,
        disorder_sigma_phi_deg=disorder_sigma_phi_deg,
        disorder_seed=disorder_seed,
    )

    if G_pair is not None:
        logger.info(
            f"Using pair-indexed G ({N_mol}×{N_mol}) — no translational symmetry assumed."
        )
        return build_ddi_matrix_from_Gpair(
            G_pair, energy_emitter, N_mol,
            mu_D_debye, mu_A_debye,
            **orient_kw,
        )

    if G_slice is not None:
        if Rx_nm is None or d_nm is None:
            raise ValueError(
                "Separation-indexed mode requires G_slice, Rx_nm, and d_nm."
            )
        return build_ddi_matrix_from_Gslice(
            G_slice, Rx_nm, energy_emitter, N_mol, d_nm,
            mu_D_debye, mu_A_debye,
            **orient_kw,
        )

    raise ValueError(
        "Must provide either G_pair (pair-indexed) or "
        "G_slice + Rx_nm + d_nm (separation-indexed)."
    )
