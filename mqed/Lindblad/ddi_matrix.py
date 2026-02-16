r"""Dipole-dipole interaction (DDI) helpers for Lindblad dynamics."""
import numpy as np
from typing import Union

from loguru import logger
from mqed.utils.SI_unit import c, eps0, hbar, eV_to_J, D2CMM
from mqed.utils.orientation import spherical_to_cartesian_dipole, resolve_angle_deg
from mqed.utils.orientation_disorder import phi_wrapped_normal_deg as _phi_wrapped_normal_deg


def build_ddi_matrix_from_Gslice(
    G_slice: np.ndarray,           # (K,3,3) complex, K=len(Rx_nm) for one energy
    Rx_nm: np.ndarray,             # (K,) float, distances in nm; must include 0, d_nm, 2*d_nm, ..., (N_mol-1)*d_nm
    energy_emitter: float,         # scalar
    N_mol: int,             # number of molecules
    d_nm: float ,              # intermolecular spacing (nm)
    mu_D_debye: float,        # donor dipole (Debye)
    mu_A_debye=None,   # acceptor dipole (Debye); if None → mu_D_debye
    *,
    mode: str = "stationary",    # "stationary" or "disorder"
    uD: Union[np.ndarray, None]=None,     # used if mode="stationary": 1D (3,)
    uA: Union[np.ndarray, None]=None,     # used if mode="stationary": 1D (3,)
    U_list= None,          # optional per-molecule orientations (N,3); if provided, used in disorder mode
    theta_deg: Union[None,float]=None,     # used only if we need to generate U_list
    phi_deg: Union[None,float]=None,       # used only if we need to generate U_list
    disorder_sigma_phi_deg=None,    # used only if we need to generate U_list
    disorder_seed=None              # used only if we need to generate U_list
):
    r"""Compute coupling ``V`` [eV] and ``\hbar\Gamma`` [eV] from a dyadic Green's slice.

    Physics:

    .. math::

       V_{\alpha\beta} \,=\, -\frac{\omega_\mathrm{M}^2}{\epsilon_0 c^2}\, \boldsymbol{\mu}_\alpha \cdot \Re\,\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega_\mathrm{M}) \cdot \boldsymbol{\mu}_\beta,

    .. math::

       \hbar\,\Gamma_{\alpha\beta} \,=\, \frac{2\,\omega_\mathrm{M}^2}{\epsilon_0 c^2}\, \boldsymbol{\mu}_\alpha \cdot \Im\,\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega_\mathrm{M}) \cdot \boldsymbol{\mu}_\beta.

    Args:
        G_slice: Dyadic Green's function slice (K,3,3) for K separations.
        Rx_nm: Distances (nm), must include ``0, d, 2d, ..., (N-1)d``.
        energy_emitter: Emitter energy in eV.
        N_mol: Number of molecules.
        d_nm: Lattice spacing in nm.
        mu_D_debye: Donor dipole moment (Debye).
        mu_A_debye: Acceptor dipole (Debye); defaults to donor value.
        mode: ``"stationary"`` uses uD/uA; ``"disorder"`` generates orientations.
        uD/uA: Dipole orientation unit vectors (stationary mode).
        U_list/theta_deg/phi_deg/disorder_*: Orientation controls for disorder mode.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(V_eV, hbarGamma_eV)`` matrices.
    """
    # --- sanity on Rx grid: we will look up exact distances s*d_nm
    Rx_nm = np.asarray(Rx_nm, dtype=float)
    dist_to_idx = {float(r): k for k, r in enumerate(Rx_nm)}
    needed = [float(s * d_nm) for s in range(N_mol)]
    missing = [r for r in needed if r not in dist_to_idx]
    if missing:
        logger.error('Rx_nm must contain all separations 0, d, 2d, ...')
        raise ValueError(
            f"Rx_nm grid must contain all separations 0, d, 2d, ..., (N-1)d in nm. "
            f"Missing: {missing[:8]}{'...' if len(missing)>8 else ''}"
        )

    # --- dipoles & prefactor
    mu_A_debye = mu_D_debye if mu_A_debye is None else mu_A_debye
    muA = mu_A_debye * D2CMM
    muD = mu_D_debye * D2CMM
    mu2 = muA * muD
    omega = energy_emitter * eV_to_J / hbar
    pref  = (omega**2) / (eps0 * c**2)  # SI

    # --- orientations
    if mode == "stationary":
        if uD is None or uA is None:
            raise ValueError("mode='stationary' requires uD and uA (shape (3,)).")
        uD = np.asarray(uD, dtype=float).reshape(3)
        uA = np.asarray(uA, dtype=float).reshape(3)
        # U_list_used = None
    elif mode == "disorder":
        if U_list is not None:
            U = np.asarray(U_list, dtype=float)
            if U.shape != (N_mol, 3):
                raise ValueError("U_list must have shape (N_mol,3).")
        else:
            if phi_deg is None or disorder_sigma_phi_deg is None:
                raise ValueError("mode='disorder' needs disorder_mu_phi_deg and disorder_sigma_phi_deg "
                                 "if U_list is not provided.")
            # logger.info('Generating orientation-disordered U_list.')
            phi_deg = resolve_angle_deg(phi_deg) # allow magic angle. 
            phi_deg = _phi_wrapped_normal_deg(N_mol, phi_deg, disorder_sigma_phi_deg, seed=disorder_seed)
            U = spherical_to_cartesian_dipole(theta_deg, phi_deg)  # (N,3)
            if U.shape != (N_mol, 3):
                raise ValueError("U_list must have shape (N_mol,3).")
        # U_list_used = U
    else:
        raise ValueError("mode must be 'stationary' or 'disorder'.")

    # --- allocate outputs
    V_eV         = np.zeros((N_mol, N_mol), dtype=np.float64)
    hbarGamma_eV = np.zeros((N_mol, N_mol), dtype=np.float64)

    G_slice = np.asarray(G_slice)
    if G_slice.ndim != 3:
        raise ValueError(f"G_slice must be 3D, got {G_slice.ndim}D {G_slice.shape}")

    # Normalize to (K,3,3)
    if G_slice.shape[-2:] == (3, 3):
        # (K,3,3) → already good
        GK33 = G_slice
    elif G_slice.shape[0:2] == (3, 3):
        # (3,3,K) → transpose to (K,3,3)
        GK33 = np.transpose(G_slice, (2, 0, 1))
    else:
        raise ValueError(f"Unrecognized G_slice shape {G_slice.shape}; expected (K,3,3) or (3,3,K)")

    K = GK33.shape[0]
    if len(Rx_nm) != K:
        raise ValueError(f"Rx_nm length {len(Rx_nm)} does not match K={K}")
    # pre-split re/im once
    G_re_full = np.real(GK33)
    G_im_full = np.imag(GK33)

    idx = np.arange(N_mol)

    # --- fill by separation s = |i-j|
    for s in range(N_mol):
        k = dist_to_idx[float(s * d_nm)]  # exact Rx index
        Gre = G_re_full[k]                # (3,3)
        Gim = G_im_full[k]
        if Gre.shape != (3, 3) or Gim.shape != (3, 3):
            raise ValueError("G_slice must have shape (K,3,3).")

        if mode == "stationary":
            # same orientation for all pairs
            val_re = float(np.dot(uA, Gre @ uD))  # scalar
            val_im = float(np.dot(uA, Gim @ uD))
            if s == 0:
                V_eV[idx, idx]         = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[idx, idx] = +(2.0 * pref * mu2 * val_im) / eV_to_J
            else:
                i = idx[:-s]; j = idx[s:]
                V_eV[i, j]         = -(pref * mu2 * val_re) / eV_to_J
                V_eV[j, i]         = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[i, j] = +(2.0 * pref * mu2 * val_im) / eV_to_J
                hbarGamma_eV[j, i] = +(2.0 * pref * mu2 * val_im) / eV_to_J

        else:  # disorder: per-molecule U
            # For all acceptors j: L_j = u_j^T G
            Lre = U @ Gre  # (N,3)
            Lim = U @ Gim
            if s == 0:
                # (u_i)^T G(0) u_i
                val_re = np.einsum('ik,ik->i', Lre, U)  # (N,)
                val_im = np.einsum('ik,ik->i', Lim, U)
                V_eV[idx, idx]         = -(pref * mu2 * val_re) / eV_to_J
                hbarGamma_eV[idx, idx] = +(2.0 * pref * mu2 * val_im) / eV_to_J
            else:
                i = idx[:-s]; j = idx[s:]
                # (u_j)^T G_s (u_i)
                vij_re = np.einsum('jk,jk->j', Lre[j], U[i])  # (N-s,)
                vij_im = np.einsum('jk,jk->j', Lim[j], U[i])
                V_eV[i, j]         = -(pref * mu2 * vij_re) / eV_to_J
                hbarGamma_eV[i, j] = +(2.0 * pref * mu2 * vij_im) / eV_to_J

                # (u_i)^T G_s (u_j)
                vji_re = np.einsum('ik,ik->i', Lre[i], U[j])
                vji_im = np.einsum('ik,ik->i', Lim[i], U[j])
                V_eV[j, i]         = -(pref * mu2 * vji_re) / eV_to_J
                hbarGamma_eV[j, i] = +(2.0 * pref * mu2 * vji_im) / eV_to_J

    # match your MATLAB: zero V diagonal
    np.fill_diagonal(V_eV, 0.0)
    return V_eV, hbarGamma_eV
