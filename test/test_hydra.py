'''
This is a test script to verify if Hydra works well.
'''
# tests/test_hydra_wiring.py
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir, compose
from mqed.Lindblad.run_quantum_dynamics import app_run, build_initial_ket
from mqed.utils.hydra_local import prepare_hydra_config_path

def _compose(cfg_dir: Path, config_name: str, overrides=None):
    overrides = overrides or []
    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def test_config_composes():
    cfg_dir = Path(__file__).parent / "../configs/Lindblad"
    cfg = _compose(cfg_dir, "quantum_dynamics")
    assert "simulation" in cfg
    assert cfg.simulation.Nmol > 0
    assert cfg.solver.method in ("Lindblad", "NonHermitian")

def test_overrides_apply():
    cfg_dir = Path(__file__).parent / "../configs/Lindblad"
    cfg = _compose(
        cfg_dir, "quantum_dynamics",
        overrides=["solver.method=NonHermitian", "simulation.t_ps.stop=12.5", "output.filename=test.h5"]
    )
    assert cfg.solver.method == "NonHermitian"
    assert abs(cfg.simulation.t_ps.stop - 12.5) < 1e-12
    assert cfg.output.filename == "test.h5"

def test_smoke_writes_to_tmp(tmp_path):
    cfg_dir = Path(__file__).parent / "../configs/Lindblad"
    cfg = _compose(cfg_dir, "quantum_dynamics", overrides=["output.filename=test_out.h5",
                                                         f"greens.h5_path={(Path(__file__).resolve().parents[1] / 'test/GF_Sommerfeld_data/Fresnel_GF_planar_Ag_height_2nm_665nm.hdf5')}",
                                                         "simulation.Nmol=5"])
    # IMPORTANT: app_run writes into output_dir we pass
    app_run(cfg, output_dir=tmp_path)
    assert (tmp_path / "test_out.h5").exists()


def test_gaussian_initial_state_normalized():
    cfg_dir = Path(__file__).parent / "../configs/Lindblad"
    cfg = _compose(
        cfg_dir,
        "quantum_dynamics_disorder",
        overrides=[
            "simulation.Nmol=11",
            "initial_state.type=gaussian",
            "initial_state.center_site=6",
            "initial_state.sigma_sites=2.5",
            "initial_state.k_parallel=0.3",
        ],
    )
    ket, ref_site = build_initial_ket(cfg.initial_state, nmol=int(cfg.simulation.Nmol))
    norm = float(ket.norm())

    assert np.isclose(norm, 1.0, atol=1e-12)
    assert ref_site == 6
    assert np.isclose(np.abs(ket.full().ravel()[0]), 0.0, atol=1e-14)


def test_gaussian_initial_state_null_center_uses_site_index():
    cfg_dir = Path(__file__).parent / "../configs/Lindblad"
    cfg = _compose(
        cfg_dir,
        "quantum_dynamics_disorder",
        overrides=[
            "simulation.Nmol=11",
            "initial_state.type=gaussian",
            "initial_state.site_index=4",
            "initial_state.center_site=null",
            "initial_state.sigma_sites=2.0",
        ],
    )
    ket, ref_site = build_initial_ket(cfg.initial_state, nmol=int(cfg.simulation.Nmol))
    norm = float(ket.norm())

    assert ref_site == 4
    assert np.isclose(norm, 1.0, atol=1e-12)


def test_prepare_hydra_config_path_merges_local_configs(tmp_path):
    repo_root = tmp_path / "repo"
    caller_dir = repo_root / "mqed" / "plotting"
    caller_dir.mkdir(parents=True)
    caller_file = caller_dir / "fake_plot.py"
    caller_file.write_text("pass\n")

    shared_dir = repo_root / "configs" / "plots"
    (shared_dir / "hydra").mkdir(parents=True)
    (shared_dir / "hydra" / "default.yaml").write_text("run:\n  dir: outputs/test\n")
    (shared_dir / "msd.yaml").write_text(
        "defaults:\n"
        "  - hydra: default\n"
        "  - _self_\n"
        "hydra:\n"
        "  job:\n"
        "    name: plot_msd\n"
        "plot_settings:\n"
        "  source: shared\n"
    )

    local_dir = repo_root / "local" / "configs" / "plots"
    local_dir.mkdir(parents=True)
    (local_dir / "my_msd.yaml").write_text(
        "defaults:\n"
        "  - hydra: default\n"
        "  - _self_\n"
        "hydra:\n"
        "  job:\n"
        "    name: plot_msd\n"
        "plot_settings:\n"
        "  source: local\n"
    )

    merged_rel = prepare_hydra_config_path("plots", str(caller_file))
    merged_dir = (caller_dir / merged_rel).resolve()

    assert (merged_dir / "hydra" / "default.yaml").exists()
    assert (merged_dir / "msd.yaml").exists()
    assert (merged_dir / "my_msd.yaml").exists()

    with initialize_config_dir(config_dir=str(merged_dir), version_base=None):
        cfg = compose(config_name="my_msd", return_hydra_config=True)

    assert cfg.plot_settings.source == "local"
    assert cfg.hydra.job.name == "plot_msd"
