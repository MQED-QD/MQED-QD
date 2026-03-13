"""Nearest-neighbour 1D chain dynamics with diagonal + off-diagonal disorder.

Physics
-------
Tight-binding Hamiltonian on N sites with nearest-neighbour coupling:
.. math::
    H_{nn'} = \epsilon_n \delta_{nn'} + J_n \delta_{n,n'+1} + J_n^* \delta_{n,n'-1}

Disorder is Gaussian:

    \epsilon_n = \epsilon_0 + \delta\epsilon_n,    \delta\epsilon_n ~ N(0, \sigma_{\epsilon}^2)
    J_n   = J_0   + \delta J_n,    \delta J_n ~ N(0, \sigma_J^2)

Time evolution via Schrödinger equation  dC/dt = -i H C  (ħ = 1 in eV·fs units)
using :func:`scipy.sparse.linalg.expm_multiply` on the tridiagonal sparse matrix.

Units
-----
Energies in eV throughout.  Times in femtoseconds in the config / output;
internally converted to dimensionless τ = t [fs] × eV / ħ [eV·fs].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import expm_multiply
from loguru import logger

from mqed.utils.SI_unit import hbar, eV_to_J

# ── conversion factor: 1 fs in seconds ──
_FS_TO_S: float = 1.0e-15

# τ = t[fs] * eV / ħ   (dimensionless time for H in eV)
_FS_TO_TAU: float = eV_to_J * _FS_TO_S / hbar


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NNChainConfig:
    """Immutable configuration for the NN-chain disorder simulation."""

    # chain
    N_emit: int                         # number of sites
    eps_0_eV: float                     # baseline on-site energy  [eV]
    J_0_eV: float                       # baseline NN coupling     [eV]

    # disorder
    sigma_eps_eV: float = 0.0           # Gaussian σ for on-site disorder [eV]
    sigma_J_eV: float = 0.0            # Gaussian σ for coupling disorder [eV]

    # time grid  (in femtoseconds)
    t_total_fs: float = 1500.0
    n_steps: int = 1500

    # initial state
    initial_state_type: str = "gaussian"  # "gaussian" | "single_site"
    sigma_sites: float = 10.0            # Gaussian width (std-dev in sites)
    k_parallel: float = 1.0             # plane-wave phase (rad / site)
    center_site: Optional[int] = None   # None → middle of the chain

    # observables
    obs_msd: bool = True
    obs_populations: bool = True
    obs_position: bool = True


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class NNChainResult:
    """Container for a single-realisation result."""
    t_fs: np.ndarray                           # (T,)  time grid
    msd: Optional[np.ndarray] = None           # (T,)  MSD
    populations: Optional[np.ndarray] = None   # (N, T) site populations |C_n(t)|²
    position: Optional[np.ndarray] = None      # (T,)  mean position

# ---------------------------------------------------------------------------
# Dynamics class
# ---------------------------------------------------------------------------
class NNChainDynamics:
    """Build and propagate a disordered 1D nearest-neighbour chain.

    Follows the MQED framework pattern (config → build → evolve) but uses
    scipy sparse matrices instead of QuTiP, since the tridiagonal structure
    makes Krylov-based ``expm_multiply`` highly efficient.
    """

    def __init__(self, config: NNChainConfig) -> None:
        self.cfg = config
        self.N = config.N_emit

        # time grid
        self.t_fs = np.linspace(0.0, config.t_total_fs, config.n_steps)
        self.tau = self.t_fs * _FS_TO_TAU          # dimensionless τ

        # centre site
        self._center = (
            config.center_site if config.center_site is not None
            else (self.N - 1) // 2
        )

        # populated by build_hamiltonian()
        self._H_eff: csc_matrix = csc_matrix((self.N, self.N))

    # ---- Hamiltonian --------------------------------------------------------

    def build_hamiltonian(self, seed: Optional[int] = None) -> csc_matrix:
        """Build the tridiagonal Hamiltonian with optional Gaussian disorder.

        Args:
            seed: RNG seed for reproducible disorder realisations.

        Returns:
            H_eff: sparse CSC matrix  ``(-i) * H``  (the generator for dC/dτ = H_eff C).
        """
        rng = np.random.default_rng(seed)
        N = self.N

        # on-site energies
        eps = np.full(N, self.cfg.eps_0_eV, dtype=np.complex128)
        if self.cfg.sigma_eps_eV > 0.0:
            eps += rng.normal(0.0, self.cfg.sigma_eps_eV, size=N)

        # NN couplings  (N-1 values; J[n] couples site n ↔ n+1)
        J = np.full(N - 1, self.cfg.J_0_eV, dtype=np.complex128)
        if self.cfg.sigma_J_eV > 0.0:
            J += rng.normal(0.0, self.cfg.sigma_J_eV, size=N - 1)

        # tridiagonal Hermitian matrix   H = diag(J*, -1) + diag(eps, 0) + diag(J, +1)
        H_phys = diags(
            diagonals=[np.conjugate(J), eps, J],
            offsets=[-1, 0, 1],
            format="csc",
        )
        # generator for  dC/dτ = (-i H) C
        self._H_eff = csc_matrix((-1j) * H_phys)
        return self._H_eff

    # ---- initial state ------------------------------------------------------

    def build_initial_state(self) -> np.ndarray:
        """Construct the normalised initial-state vector C(0).

        Returns:
            C0: complex128 array of shape (N,).
        """
        N = self.N
        n0 = self._center

        if self.cfg.initial_state_type == "single_site":
            C0 = np.zeros(N, dtype=np.complex128)
            C0[n0] = 1.0
            logger.info(f"Initial state: single site |{n0}⟩")

        elif self.cfg.initial_state_type == "gaussian":
            x = np.arange(N, dtype=float) - float(n0)
            envelope = np.exp(-x ** 2 / (2.0 * self.cfg.sigma_sites ** 2))
            phase = np.exp(1j * self.cfg.k_parallel * x)
            C0 = (envelope * phase).astype(np.complex128)
            norm = np.linalg.norm(C0)
            if norm == 0.0:
                raise ValueError(
                    "Gaussian wavepacket has zero norm.  "
                    "Check sigma_sites and center_site."
                )
            C0 /= norm
            logger.info(
                f"Initial state: Gaussian wavepacket  "
                f"(σ={self.cfg.sigma_sites}, k={self.cfg.k_parallel}, "
                f"center={n0})"
            )
        else:
            raise ValueError(
                f"Unknown initial_state_type '{self.cfg.initial_state_type}'.  "
                "Use 'gaussian' or 'single_site'."
            )
        return C0

    # ---- propagation --------------------------------------------------------

    def evolve(self, seed: Optional[int] = None) -> NNChainResult:
        """Run full time evolution for one disorder realisation.

        1. Build Hamiltonian (with disorder drawn from *seed*).
        2. Construct initial state.
        3. Propagate via ``expm_multiply`` step-by-step.
        4. Compute requested observables at each output time.

        Args:
            seed: RNG seed for disorder.

        Returns:
            :class:`NNChainResult` with requested observables.
        """
        self.build_hamiltonian(seed)
        C = self.build_initial_state()

        T = len(self.tau)

        # observable storage
        msd_arr: Optional[np.ndarray] = None
        pop_arr: Optional[np.ndarray] = None
        position_arr: Optional[np.ndarray] = None  # (N,) precompute n for MSD calculation

        # displacement² for MSD (cheap — always compute for clarity)
        x_sq = (np.arange(self.N, dtype=float) - self._center) ** 2
        x = np.arange(self.N, dtype=float) - self._center

        if self.cfg.obs_msd:
            msd_arr = np.empty(T, dtype=float)
        if self.cfg.obs_populations:
            pop_arr = np.empty((self.N, T), dtype=float)
        if self.cfg.obs_position:
            position_arr = np.empty(T, dtype=float)

        # record t=0
        P = np.abs(C) ** 2
        if msd_arr is not None:
            msd_arr[0] = np.dot(P, x_sq)
        if pop_arr is not None:
            pop_arr[:, 0] = P
        if position_arr is not None:
            position_arr[0] = np.dot(P, x)

        # step-by-step propagation (expm_multiply is efficient for sparse)
        for k in range(1, T):
            dtau = self.tau[k] - self.tau[k - 1]
            C = expm_multiply(self._H_eff * dtau, C)

            P = np.abs(C) ** 2
            if msd_arr is not None:
                msd_arr[k] = np.dot(P, x_sq)
            if pop_arr is not None:
                pop_arr[:, k] = P
            if position_arr is not None:
                position_arr[k] = np.dot(P, x)

        return NNChainResult(
            t_fs=self.t_fs,
            msd=msd_arr,
            populations=pop_arr,
            position=position_arr,
        )
