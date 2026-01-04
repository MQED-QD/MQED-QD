from numpy import np


def compute_enhancement(p_donor,p_acceptor,g_total, g_vac):
    """
    Computes the enhancement factor for Resonance Energy Transfer (RET)
    given donor and acceptor dipole orientations and Green's functions.

    Args:
        p_donot (np.ndarray): Donor dipole orientation (3-element array).
        p_acceptor (np.ndarray): Acceptor dipole orientation (3-element array).
        g_total (np.ndarray): Total Green's function array of shape (M, N, 3, 3).
        g_vac (np.ndarray): Vacuum Green's function array of shape (M, N, 3, 3).
    Returns:
        gamma (np.ndarray): Enhancement factor array of shape (M, N).
        E_enhance_real (np.ndarray): Real part of field enhancement array of shape (M, N).
        E_enhance_imag (np.ndarray): Imaginary part of field enhancement array of shape (M, N).
    """
    # Project the Green's functions onto the dipole orientations
    # This is the NumPy equivalent of the MATLAB code
    # g_da = p_A^T * G * p_D
    g_da_total = np.einsum('i,...ij,j->...', p_acceptor, g_total, p_donor)
    g_da_vac = np.einsum('i,...ij,j->...', p_acceptor, g_vac, p_donor)

    # Calculate enhancement factor
    gamma = np.abs(g_da_total / g_da_vac)**2
    E_enhance_real = np.real(g_da_total)/np.real(g_da_vac)
    E_enhance_imag = np.imag(g_da_total)/np.imag(g_da_vac)
    return gamma, E_enhance_real, E_enhance_imag