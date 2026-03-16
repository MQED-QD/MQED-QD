"""Verify BEM vs Fresnel dyadic Green's functions via least-squares scaling.

Reads the CSV produced by compare_BEM_dyadic.py, fits a complex scalar
s_ij for each tensor component such that ||s_ij * G_BEM - G_Fresnel||^2
is minimised, then reports the per-component and averaged scale factors
along with the relative RMS error after scaling.

To use this script: 
python -m mqed.BEM.verify_bem_fresnel path/to/bem_vs_fresnel.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from loguru import logger

COMPONENTS = ["Gxx", "Gxy", "Gxz", "Gyx", "Gyy", "Gyz", "Gzx", "Gzy", "Gzz"]


def fit_scale(re_bem, im_bem, re_fresnel, im_fresnel):
    """Fit complex scalar s minimising ||s * G_BEM - G_Fresnel||^2."""
    Gb = re_bem.to_numpy() + 1j * im_bem.to_numpy()
    Gf = re_fresnel.to_numpy() + 1j * im_fresnel.to_numpy()
    if np.all(Gb == 0):
        logger.warning("BEM data is all zero, cannot fit scale factor.")
        return 0
    else:
        s = np.vdot(Gb, Gf) / np.vdot(Gb, Gb)
    return s


def relative_rms(s, re_bem, im_bem, re_fresnel, im_fresnel):
    """Relative RMS error after applying scale factor s."""
    Gb = re_bem.to_numpy() + 1j * im_bem.to_numpy()
    Gf = re_fresnel.to_numpy() + 1j * im_fresnel.to_numpy()
    return np.linalg.norm(s * Gb - Gf) / np.linalg.norm(Gf)


def main():
    parser = argparse.ArgumentParser(
        description="Verify BEM vs Fresnel dyadic GF from a comparison CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file produced by compare_BEM_dyadic.py",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    scales = {}
    for comp in COMPONENTS:
        re_bem_col = f"Re_{comp}_BEM"
        im_bem_col = f"Im_{comp}_BEM"
        re_fr_col = f"Re_{comp}_Fresnel"
        im_fr_col = f"Im_{comp}_Fresnel"

        # 1. Check if the columns exist
        if re_bem_col not in df.columns:
            logger.warning(f"Component {comp} not found in CSV, skipping.")
            continue
        # 2. Check if the columns are all zeros
        # This checks if the sum of absolute values is zero
        is_bem_zero = (df[re_bem_col].abs().sum() == 0) and (df[im_bem_col].abs().sum() == 0)
        is_fr_zero = (df[re_fr_col].abs().sum() == 0) and (df[im_fr_col].abs().sum() == 0)

        if is_bem_zero or is_fr_zero:
            logger.info(f"Component {comp} contains only zero values, skipping scale fit.")
            continue

        # 3. Proceed with fitting if data exists
        s = fit_scale(df[re_bem_col], df[im_bem_col], df[re_fr_col], df[im_fr_col])
        rms = relative_rms(s, df[re_bem_col], df[im_bem_col], df[re_fr_col], df[im_fr_col])
        scales[comp] = s

        logger.info(f"s_{comp} = {s:.6f}  (rel. RMS after scaling = {rms:.6e})")

    if not scales:
        logger.error("No components found in CSV.")
        return

    s_avg = np.mean([s for s in scales.values() if s != 0])
    logger.info(f"s_avg  = {s_avg:.6f}  (averaged over {len(scales)} components)")

    rms_all = []
    for comp, s in scales.items():
        rms = relative_rms(
            s_avg,
            df[f"Re_{comp}_BEM"], df[f"Im_{comp}_BEM"],
            df[f"Re_{comp}_Fresnel"], df[f"Im_{comp}_Fresnel"],
        )
        rms_all.append(rms)

    logger.success(
        f"Mean rel. RMS error (using s_avg) = {np.mean(rms_all):.6e} "
        f"over {len(rms_all)} components"
    )


if __name__ == "__main__":
    main()
