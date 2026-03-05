# MacroscopicQED/mqed/Lindblad/quantum_operator.py
"""Quantum operators (MSD, position, IPR) for dynamics simulations."""
from qutip import Qobj, qeye, projection, expect
import numpy as np


def msd_operator(dim: int, d_nm: float, Nmol: int, init_site_index: int) -> Qobj:
    r"""Mean-square displacement operator (single-excitation manifold).

    Basis: ``|0\rangle`` (ground), ``|1\rangle,...,|N\rangle`` (sites). Positions: ground=0,
    site ``j`` at ``j d``. MSD operator is ``(X - x_0 I)^2``.

    .. math::

       \langle x^2 \rangle - \langle x \rangle^2 = \mathrm{Tr}\big[(X - x_0 I)^2 \, \rho\big].
    """
    positions = np.zeros(dim)
    positions[1:] = d_nm * np.arange(1, Nmol + 1, dtype=float)
    X = Qobj(np.diag(positions), dims=[[dim], [dim]])
    x0 = positions[init_site_index]
    return (X - x0 * qeye(dim)) ** 2

def site_population_operator(dim: int, site:int) -> Qobj:
    r"""Projector onto site ``site`` (single excitation)."""
    e_ops_populations = projection(dim, site+1, site+1)

    return e_ops_populations

def position_operator(dim: int, d_nm: float, Nmol: int, init_site_index: int) -> Qobj:
    r"""Position operator (single excitation), centered at the initial site ``x0``."""
    positions = np.zeros(dim)
    positions[1:] = d_nm * np.arange(1, Nmol + 1, dtype=float)
    X = Qobj(np.diag(positions), dims=[[dim], [dim]])
    x0 = positions[init_site_index]
    return (X - x0 * qeye(dim))

def ipr_callable(t, state, *, Nmol):
    r"""Inverse participation ratio (IPR) at time ``t`` for a state (ket or density matrix).

    .. math::

       \mathrm{IPR} = \frac{\sum_j |c_j|^4}{\left(\sum_j |c_j|^2\right)^2},

    where ``c_j`` are site amplitudes (or populations for a density matrix) over the excited subspace.
    """
    if state.isket:
        amp = state.full().ravel()              # length N+1
        pop = np.abs(amp)**2
    else:
        rho = state.full()
        pop = np.real(np.diag(rho))
    pop_exc = pop[1:1+Nmol]
    s = pop_exc.sum()
    if s <= 0:
        return 0.0
    q = pop_exc / s
    return float(np.dot(q, q))                  # IPR_site

def excited_population_norm(state, *, Nmol: int) -> float:
    """Total excited-manifold population (sum over sites |1>...|N|)."""
    if state.isket:
        amp = state.full().ravel()
        pop = np.abs(amp)**2
    else:
        rho = state.full()
        pop = np.real(np.diag(rho))

    s = float(np.sum(pop[1:1+Nmol]))
    return max(s, 0.0)


def x_shift_conditional_callable(t, state, *, X_shift: Qobj, Nmol: int) -> float:
    r"""Conditional <X-x0I> / P_exc, where P_exc is excited population."""
    s = excited_population_norm(state, Nmol=Nmol)
    if s <= 0.0:
        return 0.0
    val = expect(X_shift, state)
    return float(np.real(val) / s)


def x_shift2_conditional_callable(t, state, *, X_shift2: Qobj, Nmol: int) -> float:
    r"""Conditional <(X-x0I)^2> / P_exc, where P_exc is excited population."""
    s = excited_population_norm(state, Nmol=Nmol)
    if s <= 0.0:
        return 0.0
    val = expect(X_shift2, state)
    return float(np.real(val) / s)