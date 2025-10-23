from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import h5py


def save_dx_h5(outfile: Path,
               t_ps: np.ndarray,
               dx_mean_nm: np.ndarray,
               *,
               dx_std_nm: np.ndarray | None = None,
               method: str,
               mode: str,
               n_realizations: int,
               expectations: dict | None = None,
               extra_attrs: dict | None = None) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outfile, "w") as f:
        f.create_dataset("t_ps", data=np.asarray(t_ps))
        f.create_dataset("dx_mean_nm", data=np.asarray(dx_mean_nm))
        if dx_std_nm is not None:
            f.create_dataset("dx_std_nm", data=np.asarray(dx_std_nm))
        f.attrs["method"] = method
        f.attrs["mode"] = mode
        f.attrs["n_realizations"] = int(n_realizations)
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v
        if expectations:
            grp = f.create_group("expectations")
            for name, val in expectations.items():
                grp.create_dataset(name, data=np.asarray(val))
