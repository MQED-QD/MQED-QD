# mqed/Lindblad/coupling_filter.py
import numpy as np

def _mask_within_hops(N: int, hop_radius: int, *, include_on_site: bool) -> np.ndarray:
    """
    Boolean mask M[i,j] = True if two sites i and j are within a given
    *hop distance* on the 1D chain, i.e. |i - j| <= hop_radius.

    hop_radius = 1  → nearest neighbours only (and on-site if include_on_site=True)
    hop_radius = 2  → up to next-nearest neighbours, etc.
    """
    i = np.arange(N)
    M = (np.abs(i[:, None] - i[None, :]) <= hop_radius)
    if not include_on_site:
        np.fill_diagonal(M, False)
    return M

def enforce_coupling_range(
    V: np.ndarray, Gamma: np.ndarray, *,
    V_hop_radius: int | None = None,
    keep_V_on_site: bool = False,
    Gamma_rule: str = "leave",            # "leave" | "same_as_V" | "diagonal_only" | "limit_by_hops"
    Gamma_hop_radius: int | None = None,
    keep_Gamma_on_site: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero out matrix elements beyond a chosen interaction range.

    Parameters:
    ----------
    V, Gamma : (N, N) arrays
        Coherent (V) and dissipative (Γ) coupling matrices in site basis.
    V_hop_radius : int or None
        Limit V to pairs with hop distance |i-j| <= V_hop_radius.
        Example: 1 → nearest-neighbour-only V.
    keep_V_on_site : bool
        If False, set V_ii = 0 (typical for your model).
    Gamma_rule : str
        How to modify Γ relative to V:
          - "leave"           : do nothing to Γ
          - "same_as_V"       : apply *the same* mask used for V
          - "diagonal_only"   : keep only Γ_ii (local decay), zero off-diagonals
          - "limit_by_hops"   : apply a hop-radius mask to Γ (use Gamma_hop_radius)
          - "limit_by_distance": apply a real-space cutoff to Γ (use Gamma_cutoff_nm)
    keep_Gamma_on_site : bool
        If True, always keep Γ_ii.
    Gamma_hop_radius : int or None
        Range parameters for Γ when Gamma_rule is "limit_by_hops"/"limit_by_distance".

    Returns:
        V_filtered, Gamma_filtered : (N, N)
        Filtered copies of V and Γ. Symmetrized to remove tiny numerical asymmetries.
    """
    N = V.shape[0]
    V_new, G_new = V.copy(), Gamma.copy()

    # --- V (coherent)
    if V_hop_radius is not None:
        MV = _mask_within_hops(N, V_hop_radius, include_on_site=keep_V_on_site)
    else:
        MV = np.ones_like(V_new, dtype=bool)
    V_new[~MV] = 0.0

    # --- Γ (dissipative)
    if Gamma_rule == "leave":
        pass
    elif Gamma_rule == "same_as_V":
        G_new[~MV] = 0.0
        if not keep_Gamma_on_site:
            np.fill_diagonal(G_new, 0.0)
    elif Gamma_rule == "diagonal_only":
        G_new[:] = 0.0
        if keep_Gamma_on_site:
            np.fill_diagonal(G_new, np.diag(Gamma))
    elif Gamma_rule == "limit_by_hops":
        if Gamma_hop_radius is None:
            raise ValueError("Gamma_hop_radius required for Gamma_rule='limit_by_hops'.")
        MG = _mask_within_hops(N, Gamma_hop_radius, include_on_site=keep_Gamma_on_site)
        G_new[~MG] = 0.0
    else:
        raise ValueError(f"Unknown Gamma_rule '{Gamma_rule}'.")

    # Symmetrize to clean tiny FP asymmetries
    V_new = 0.5 * (V_new + V_new.T)
    G_new = 0.5 * (G_new + G_new.T)
    return V_new, G_new
