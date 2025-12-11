'''Utilities for reading BEM data files. 
Includes reading p_eff from CSV and dyadic from Excel.
'''

import numpy as np
import pandas as pd

def read_peff(peff_csv: str, lambda_nm: float) -> complex:
    """Return p_eff at lambda_nm (C*m). If exact lambda not found, use nearest.
    Args:
        peff_csv (str): Path to CSV file with columns: lambda_nm, p_eff_Cm_real, p_eff_Cm_imag
        lambda_nm (float): Wavelength in nm to look up.
    Returns:
        complex: Effective dipole moment p_eff.
        """
    df = pd.read_csv(peff_csv)
    df["lambda_nm"] = df["lambda_nm"].astype(float)
    idx = (df["lambda_nm"] - lambda_nm).abs().idxmin()
    row = df.loc[idx]
    return float(row["p_eff_Cm_real"]) + 1j * float(row["p_eff_Cm_imag"])

def read_bem_dyadic(xlsx_path: str, sheet: str)-> tuple[np.ndarray, np.ndarray]:
    """Read BEM dyadic from Excel. Returns Rx_nm (N,) and G (N,3,3) complex.
    Args:
        xlsx_path (str): Path to Excel file.
        sheet (str): Sheet name in Excel file.
    Returns:
        tuple: (Rx_nm (N,), G (N,3,3) complex) dyadic.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    rx = df["x_nm"].to_numpy(float)

    def ccol(name):
        return df[f"Re_{name}"].to_numpy(float) + 1j * df[f"Im_{name}"].to_numpy(float)

    G = np.zeros((len(rx), 3, 3), dtype=np.complex128)
    G[:,0,0] = ccol("Gxx"); G[:,0,1] = ccol("Gxy"); G[:,0,2] = ccol("Gxz")
    G[:,1,0] = ccol("Gyx"); G[:,1,1] = ccol("Gyy"); G[:,1,2] = ccol("Gyz")
    G[:,2,0] = ccol("Gzx"); G[:,2,1] = ccol("Gzy"); G[:,2,2] = ccol("Gzz")

    return rx, G

def read_purcell_sheet(xlsx_path: str, sheet="G_self") -> tuple[float, float, float, float]:
    '''Read Purcell factors from a BEM Excel sheet.
    Args:
        xlsx_path (str): Path to Excel file.
        sheet (str): Sheet name in Excel file.
    Returns:
        tuple: (lambda_nm, F_x, F_y, F_z)
    '''
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    lam_nm = float(df["lambda_nm"].iloc[0])
    Fx = float(df["Purcell_x"].iloc[0])
    Fy = float(df["Purcell_y"].iloc[0])
    Fz = float(df["Purcell_z"].iloc[0])
    return lam_nm, Fx, Fy, Fz