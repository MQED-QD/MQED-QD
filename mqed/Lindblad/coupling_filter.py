# mqed/Lindblad/coupling_filter.py
import numpy as np

def _mask_within_hops(N: int, hop_radius: int, *, include_on_site: bool) -> np.ndarray:
    """Boolean mask for sites within a hop distance on a 1D chain.

    Condition: ``abs(i - j) <= hop_radius``. Example: ``hop_radius=1`` keeps
    nearest neighbours (and on-site when ``include_on_site=True``); ``hop_radius=2``
    keeps up to next-nearest neighbours.
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
    """Zero out V and Γ beyond a chosen hop range.

    Args:
        V: (N, N) coherent coupling matrix.
        Gamma: (N, N) dissipative coupling matrix.
        V_hop_radius: If set, keep pairs with ``abs(i-j) <= V_hop_radius``.
        keep_V_on_site: If False, zero the diagonal of V.
        Gamma_rule: How to mask Γ ("leave", "same_as_V", "diagonal_only", "limit_by_hops").
        Gamma_hop_radius: Hop limit for Γ when ``Gamma_rule="limit_by_hops"``.
        keep_Gamma_on_site: If False, zero the diagonal of Γ where applicable.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered, symmetrized copies of V and Γ.
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
