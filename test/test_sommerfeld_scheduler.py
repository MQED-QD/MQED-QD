import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mqed.Dyadic_GF import main


def _sim_params():
    return main.OmegaConf.create(
        {
            "position": {"zD": 5e-9, "zA": 5e-9},
            "integration": {
                "qmax": None,
                "epsabs": 1e-6,
                "epsrel": 1e-6,
                "limit": 20,
                "split_propagating": False,
            },
        }
    )


def test_sommerfeld_backend_default_keeps_complete_energy_rows():
    assert main.scheduler_for_backend(
        {"scheduler": "backend_default"},
        "joblib",
        backend_default_scheduler="energy",
    ) == "energy"
    assert main.scheduler_for_backend(
        {"scheduler": "backend_default"},
        "mpi",
        backend_default_scheduler="energy",
    ) == "energy"


def test_sommerfeld_auto_splits_only_when_energies_are_scarce():
    scarce_tasks = main.task_slices(
        n_energy=1,
        n_rx=7,
        worker_count=3,
        integration_method="direct",
        scheduler="auto",
    )
    filled_tasks = main.task_slices(
        n_energy=3,
        n_rx=7,
        worker_count=3,
        integration_method="direct",
        scheduler="auto",
    )

    assert [indices.tolist() for _, indices in scarce_tasks] == [[0, 1, 2], [3, 4], [5, 6]]
    assert len(filled_tasks) == 3
    assert all(indices.tolist() == list(range(7)) for _, indices in filled_tasks)


def test_sommerfeld_flattened_without_chunk_size_always_splits_rx():
    tasks = main.task_slices(
        n_energy=4,
        n_rx=8,
        worker_count=2,
        integration_method="direct",
        scheduler="flattened",
    )

    assert len(tasks) == 8
    for energy_index in range(4):
        energy_chunks = [indices.tolist() for idx, indices in tasks if idx == energy_index]
        assert energy_chunks == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 2.0])
def test_sommerfeld_rx_chunk_size_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer or null"):
        main.rx_chunk_size({"rx_chunk_size": value})


def test_sommerfeld_joblib_flattened_restores_original_grid(monkeypatch):
    calls = []

    def fake_compute_one_energy(idx, rx_values_m, **kwargs):
        calls.append((idx, rx_values_m.copy()))
        rx_labels = np.rint(rx_values_m / 1e-9).astype(int)
        total = np.asarray(
            [np.eye(3, dtype=complex) * (100 * idx + rx_index) for rx_index in rx_labels]
        )
        vacuum = np.asarray(
            [np.eye(3, dtype=complex) * (200 * idx + rx_index) for rx_index in rx_labels]
        )
        return idx, total, vacuum

    monkeypatch.setattr(main, "_compute_one_energy", fake_compute_one_energy)
    rx_values_m = np.arange(5, dtype=float) * 1e-9

    total, vacuum = main._run_joblib(
        np.array([1.0, 2.0]),
        np.array([600e-9, 500e-9]),
        rx_values_m,
        _sim_params(),
        main.OmegaConf.create({"source_type": "constant", "constant_value": "1+0j"}),
        n_jobs=1,
        parallel_cfg={"scheduler": "flattened", "rx_chunk_size": 2},
    )

    assert len(calls) == 6
    assert [values.tolist() for _, values in calls[:3]] == [
        rx_values_m[0:2].tolist(),
        rx_values_m[2:4].tolist(),
        rx_values_m[4:5].tolist(),
    ]
    assert total.shape == (2, 5, 3, 3)
    assert vacuum.shape == total.shape
    for energy_index in range(2):
        for rx_index in range(5):
            assert np.allclose(
                total[energy_index, rx_index],
                np.eye(3) * (100 * energy_index + rx_index),
            )
            assert np.allclose(
                vacuum[energy_index, rx_index],
                np.eye(3) * (200 * energy_index + rx_index),
            )


def test_sommerfeld_assembly_rejects_duplicate_and_missing_work():
    tensor = np.zeros((1, 3, 3), dtype=complex)
    duplicated = [[
        (0, np.array([0]), tensor, tensor),
        (0, np.array([0]), tensor, tensor),
    ]]

    with pytest.raises(RuntimeError, match="missing=.*duplicate="):
        main.assemble_sliced_results(
            duplicated,
            n_energy=1,
            n_rx=2,
            save_components=False,
            worker_label="Sommerfeld test",
        )


def test_sommerfeld_jobarray_rejects_injected_numeric_field(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda_base = tmp_path / "conda"
    profile_dir = fake_conda_base / "etc" / "profile.d"
    profile_dir.mkdir(parents=True)
    (profile_dir / "conda.sh").write_text("conda() { return 0; }\n")
    (fake_bin / "conda").write_text(
        f"#!/bin/bash\nif [ \"$1\" = info ]; then printf '%s\\n' '{fake_conda_base}'; fi\n"
    )
    (fake_bin / "conda").chmod(0o755)
    (fake_bin / "module").write_text("#!/bin/bash\nexit 0\n")
    (fake_bin / "module").chmod(0o755)
    marker = tmp_path / "injected"
    (fake_bin / "mpirun").write_text(f"#!/bin/bash\ntouch '{marker}'\n")
    (fake_bin / "mpirun").chmod(0o755)

    params = tmp_path / "params.tsv"
    injected_height = f'0);system("touch${{IFS}}{marker}");}}#'
    params.write_text(
        "label\tenergy_min_eV\tenergy_max_eV\tenergy_points\tzD_nm\tzA_nm\tmaterial\n"
        f"case\t1.0\t2.0\t2\t{injected_height}\t5\tAg\n"
    )
    script = Path(__file__).resolve().parents[1] / "mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MQED_REPO_ROOT": str(Path(__file__).resolve().parents[1]),
            "GF_SWEEP_PARAM_FILE": str(params),
            "SGE_TASK_ID": "1",
            "NSLOTS": "1",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode != 0
    assert "must be a finite number" in completed.stderr
    assert not marker.exists()
