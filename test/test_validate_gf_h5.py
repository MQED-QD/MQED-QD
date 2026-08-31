import h5py
import numpy as np
import pytest

from mqed.analysis.validate_gf_h5 import main, validate_separation_gf_h5


def _write_separation_file(path, scattering):
    energy_eV = np.arange(scattering.shape[0], dtype=float) + 1.0
    vacuum = np.ones_like(scattering, dtype=complex)
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "separation"
        h5.create_dataset("energy_eV", data=energy_eV)
        h5.create_dataset("Rx_nm", data=np.arange(scattering.shape[1], dtype=float))
        h5.create_dataset("green_function_total", data=vacuum + scattering)
        h5.create_dataset("green_function_vacuum", data=vacuum)


def test_validate_separation_gf_h5_accepts_smooth_finite_data(tmp_path):
    scattering = np.zeros((5, 2, 3, 3), dtype=complex)
    scattering[:, :, 0, 0] = np.arange(1.0, 6.0)[:, None]
    path = tmp_path / "smooth.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path)

    assert report["valid"]
    assert report["errors"] == []
    assert report["spikes"] == []


def test_validate_separation_gf_h5_reports_isolated_scattering_spike(tmp_path):
    scattering = np.ones((5, 2, 3, 3), dtype=complex)
    scattering[2, 1, 0, 2] = 100.0 + 3.0j
    path = tmp_path / "spike.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path, spike_ratio=10.0)

    assert not report["valid"]
    assert report["errors"] == []
    assert len(report["spikes"]) == 1
    assert report["spikes"][0]["energy_eV"] == 3.0
    assert report["spikes"][0]["rx_nm"] == 1.0
    assert report["spikes"][0]["component"] == (0, 2)
    assert report["spikes"][0]["ratio"] > 100.0


def test_validate_separation_gf_h5_rejects_nonfinite_values(tmp_path):
    scattering = np.ones((3, 1, 3, 3), dtype=complex)
    scattering[1, 0, 2, 2] = np.nan
    path = tmp_path / "nonfinite.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path)

    assert not report["valid"]
    assert "contains 1 non-finite values" in report["errors"][0]


def test_validate_separation_gf_h5_detects_spike_in_weaker_component(tmp_path):
    scattering = np.ones((5, 1, 3, 3), dtype=complex)
    scattering[:, 0, 0, 0] = 1e12
    scattering[2, 0, 1, 1] = 100.0
    path = tmp_path / "mixed_scales.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path)

    assert not report["valid"]
    assert report["spikes"][0]["component"] == (1, 1)
    assert report["spikes"][0]["ratio"] == 100.0


def test_validate_separation_gf_h5_rejects_nonfinite_scattering(tmp_path):
    shape = (3, 1, 3, 3)
    total = np.ones(shape, dtype=complex)
    vacuum = np.zeros(shape, dtype=complex)
    total[1, 0, 0, 0] = np.finfo(float).max
    vacuum[1, 0, 0, 0] = -np.finfo(float).max
    path = tmp_path / "overflow.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "separation"
        h5.create_dataset("energy_eV", data=np.arange(3, dtype=float) + 1.0)
        h5.create_dataset("Rx_nm", data=np.array([0.0]))
        h5.create_dataset("green_function_total", data=total)
        h5.create_dataset("green_function_vacuum", data=vacuum)

    report = validate_separation_gf_h5(path)

    assert not report["valid"]
    assert report["nonfinite_scattering"] == 1
    assert "contains 1 non-finite values" in report["errors"][0]


def test_validate_separation_gf_h5_reports_endpoint_spike(tmp_path):
    scattering = np.ones((4, 1, 3, 3), dtype=complex)
    scattering[-1, 0, 2, 2] = 20.0
    path = tmp_path / "endpoint.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path)

    assert report["spikes"][0]["energy_index"] == 3
    assert report["spikes"][0]["component"] == (2, 2)


def test_validate_separation_gf_h5_returns_stable_report_for_missing_data(tmp_path):
    path = tmp_path / "missing.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["gf_layout"] = "pair"

    report = validate_separation_gf_h5(path)

    assert set(report) == {
        "path",
        "valid",
        "errors",
        "nonfinite_total",
        "nonfinite_vacuum",
        "nonfinite_scattering",
        "spike_ratio",
        "spikes",
    }
    assert len(report["errors"]) == 2


def test_validate_separation_gf_h5_accepts_single_energy_and_zero_scattering(tmp_path):
    scattering = np.zeros((1, 1, 3, 3), dtype=complex)
    path = tmp_path / "single_energy.h5"
    _write_separation_file(path, scattering)

    report = validate_separation_gf_h5(path)

    assert report["valid"]
    assert report["spikes"] == []


@pytest.mark.parametrize("spike_ratio", [1.0, 0.0, np.nan, np.inf])
def test_validate_separation_gf_h5_rejects_invalid_spike_ratio(tmp_path, spike_ratio):
    with pytest.raises(ValueError, match="finite and greater than 1"):
        validate_separation_gf_h5(tmp_path / "unused.h5", spike_ratio=spike_ratio)


def test_validate_gf_h5_cli_exit_codes_and_argument_errors(tmp_path, monkeypatch, capsys):
    smooth = np.ones((3, 1, 3, 3), dtype=complex)
    smooth_path = tmp_path / "smooth.h5"
    _write_separation_file(smooth_path, smooth)
    spike = smooth.copy()
    spike[1, 0, 0, 0] = 20.0
    spike_path = tmp_path / "spike.h5"
    _write_separation_file(spike_path, spike)

    monkeypatch.setattr("sys.argv", ["mqed_validate_gf_h5", str(smooth_path)])
    assert main() == 0
    assert "PASS:" in capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv", ["mqed_validate_gf_h5", str(smooth_path), str(spike_path)]
    )
    assert main() == 1
    output = capsys.readouterr().out
    assert "PASS:" in output
    assert "SUSPECT:" in output

    monkeypatch.setattr(
        "sys.argv", ["mqed_validate_gf_h5", "--spike-ratio", "1", str(smooth_path)]
    )
    with pytest.raises(SystemExit, match="2"):
        main()
