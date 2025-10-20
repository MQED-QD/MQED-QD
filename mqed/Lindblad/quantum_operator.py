from qutip import Qobj, qeye, projection
import numpy as np


def msd_operator(dim: int, d_nm: float, Nmol: int, init_site_index: int) -> Qobj:
    """Mean-square displacement operator in the single-excitation manifold.


    Basis: |0> (ground), |1>,...,|N> (site excitations).
    Positions are 0 for ground and j*d for site j. MSD = (X - x0 I)^2.
    """
    positions = np.zeros(dim)
    positions[1:] = d_nm * np.arange(1, Nmol + 1, dtype=float)
    X = Qobj(np.diag(positions), dims=[[dim], [dim]])
    x0 = positions[init_site_index]
    return (X - x0 * qeye(dim)) ** 2

def excitation_population_operator(dim: int, Nmol:int) -> Qobj:
    """
    This creates a list of projectors:
    |e_j><e_j| for j=1,...,Nmol
    where |e_j> is the state with excitation on molecule j.
    Args:
        dim (int): Dimension of the Hilbert space (Nmol + 1).
        Nmol (int): Number of molecules.
    Returns:
        List[Qobj]: List of projection operators for each molecule's excitation.
    """
    e_ops_populations = [projection(dim, i+1, i+1) for i in range(Nmol)]

    return e_ops_populations