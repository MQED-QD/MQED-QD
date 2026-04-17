import numpy as np
import pytest
from omegaconf import OmegaConf

from mqed.Dyadic_GF.data_provider import DataProvider


HBAR_EV_S = 6.582119569e-16


def _omega_from_ev(value_eV):
    return value_eV / HBAR_EV_S


def test_constant_source_returns_configured_value():
    cfg = OmegaConf.create({"source_type": "constant", "constant_value": "9.0+0.2j"})
    provider = DataProvider(cfg)

    epsilon = provider.get_epsilon(1.23e16)

    assert epsilon == complex(9.0, 0.2)


def test_drude_source_uses_case_insensitive_name_and_ev_parameters():
    cfg = OmegaConf.create(
        {
            "source_type": "Drude",
            "drude_config": {
                "eps_inf": 1.5,
                "omega_p_eV": 9.0,
                "gamma_eV": 0.05,
            },
        }
    )
    provider = DataProvider(cfg)
    omega = _omega_from_ev(3.0)

    epsilon = provider.get_epsilon(omega)

    omega_p = _omega_from_ev(9.0)
    gamma = _omega_from_ev(0.05)
    expected = 1.5 - (omega_p**2) / (omega**2 + 1j * gamma * omega)
    assert np.isclose(epsilon.real, expected.real)
    assert np.isclose(epsilon.imag, expected.imag)


def test_drude_lorentz_source_supports_hyphenated_name():
    cfg = OmegaConf.create(
        {
            "source_type": "Drude-Lorentz",
            "drude_lorentz_config": {
                "eps_inf": 2.0,
                "omega_p_eV": 8.0,
                "gamma_eV": 0.08,
                "oscillators": [
                    {"strength": 0.7, "omega_0_eV": 3.8, "gamma_eV": 0.25},
                    {"strength": 0.2, "omega_0_eV": 4.9, "gamma_eV": 0.35},
                ],
            },
        }
    )
    provider = DataProvider(cfg)
    omega = _omega_from_ev(4.2)

    epsilon = provider.get_epsilon(omega)

    omega_p = _omega_from_ev(8.0)
    gamma_d = _omega_from_ev(0.08)
    expected = 2.0 - (omega_p**2) / (omega**2 + 1j * gamma_d * omega)

    for strength, omega_0_ev, gamma_ev in [(0.7, 3.8, 0.25), (0.2, 4.9, 0.35)]:
        omega_0 = _omega_from_ev(omega_0_ev)
        gamma = _omega_from_ev(gamma_ev)
        expected += (strength * omega_0**2) / (omega_0**2 - omega**2 - 1j * gamma * omega)

    assert np.isclose(epsilon.real, expected.real)
    assert np.isclose(epsilon.imag, expected.imag)


def test_drude_requires_exactly_one_unit_key_per_frequency_parameter():
    cfg = OmegaConf.create(
        {
            "source_type": "drude",
            "drude_config": {
                "eps_inf": 1.0,
                "omega_p_eV": 9.0,
                "omega_p_rad_s": 1.0,
                "gamma_eV": 0.05,
            },
        }
    )

    with pytest.raises(ValueError, match="Specify only one"):
        DataProvider(cfg)


def test_drude_supports_legacy_rad_per_second_key_without_suffix():
    cfg = OmegaConf.create(
        {
            "source_type": "drude",
            "drude_config": {
                "eps_inf": 1.0,
                "omega_p": 1.2e16,
                "gamma": 5.0e13,
            },
        }
    )

    provider = DataProvider(cfg)
    epsilon = provider.get_epsilon(2.0e15)

    assert np.isfinite(epsilon.real)
    assert np.isfinite(epsilon.imag)


def test_negative_frequency_parameter_is_rejected():
    cfg = OmegaConf.create(
        {
            "source_type": "drude",
            "drude_config": {
                "eps_inf": 1.0,
                "omega_p_eV": -9.0,
                "gamma_eV": 0.05,
            },
        }
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        DataProvider(cfg)


def test_unknown_source_type_raises_clear_error():
    cfg = OmegaConf.create({"source_type": "my-model"})

    with pytest.raises(ValueError, match="Unknown material source_type"):
        DataProvider(cfg)
