import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from loguru import logger
from importlib import resources
from omegaconf import DictConfig, ListConfig

# Physical constants for converting wavelength to angular frequency
C_METERS_PER_SEC = 2.99792458e8  # Speed of light in m/s
HBAR_EV_S = 6.582119569e-16      # Reduced Planck constant in eV*s

class DataProvider:
    """
    Handles loading and providing material permittivity. It is optimized to create
    interpolation functions only once and performs interpolation in frequency
    space for better numerical accuracy.
    """

    def __init__(self, material_config):
        self.config = material_config
        self.source_type_raw = str(self.config.source_type)
        self.source_type = self._normalize_source_type(self.source_type_raw)
        logger.info(
            f"Initializing DataProvider with source type: '{self.source_type_raw}' "
            f"(normalized='{self.source_type}')"
        )

        if self.source_type == 'excel':
            self._setup_interpolator_from_excel()
        elif self.source_type == 'constant':
            self._setup_constant_epsilon()
        elif self.source_type == 'drude':
            self._setup_drude_model()
        elif self.source_type == 'drude_lorentz':
            self._setup_drude_lorentz_model()
        else:
            raise ValueError(
                f"Unknown material source_type: '{self.source_type_raw}'. "
                "Use one of: excel, constant, Drude, Drude-Lorentz."
            )

    @staticmethod
    def _normalize_source_type(source_type):
        """Normalize source type values from YAML into internal enum-like strings."""
        text = str(source_type).strip().lower().replace("_", "-")
        if text in {"excel", "constant", "drude"}:
            return text
        if text in {"drude-lorentz", "drudelorentz", "drude lorentz"}:
            return "drude_lorentz"
        return text

    @staticmethod
    def _to_omega_from_ev(value_eV):
        """Convert an energy in eV to angular frequency in rad/s."""
        return float(value_eV) / HBAR_EV_S

    @staticmethod
    def _require_finite(value, key_name):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Parameter '{key_name}' must be finite.")
        return numeric

    @staticmethod
    def _require_nonnegative(value, key_name):
        numeric = DataProvider._require_finite(value, key_name)
        if numeric < 0:
            raise ValueError(f"Parameter '{key_name}' must be >= 0.")
        return numeric

    @staticmethod
    def _require_positive(value, key_name):
        numeric = DataProvider._require_finite(value, key_name)
        if numeric <= 0:
            raise ValueError(f"Parameter '{key_name}' must be > 0.")
        return numeric

    def _read_omega_parameter(self, cfg, param_name):
        """Read frequency-like model parameters from either eV or rad/s YAML keys.

        Supported keys for a parameter named `omega_p` are:
          - `omega_p_eV`   (preferred for material-model fits in literature)
          - `omega_p_rad_s`
          - `omega_p`      (legacy fallback interpreted as rad/s)
        """
        ev_key = f"{param_name}_eV"
        rad_key = f"{param_name}_rad_s"
        legacy_key = param_name

        has_ev = ev_key in cfg
        has_rad = rad_key in cfg
        has_legacy = legacy_key in cfg

        if sum([has_ev, has_rad, has_legacy]) > 1:
            raise ValueError(
                f"Specify only one of '{ev_key}', '{rad_key}', or '{legacy_key}' "
                "in material config."
            )
        if has_ev:
            value_eV = self._require_nonnegative(cfg[ev_key], ev_key)
            return self._to_omega_from_ev(value_eV)
        if has_rad:
            return self._require_nonnegative(cfg[rad_key], rad_key)
        if has_legacy:
            return self._require_nonnegative(cfg[legacy_key], legacy_key)

        raise ValueError(
            f"Missing '{param_name}' in material config. "
            f"Provide '{ev_key}', '{rad_key}', or legacy '{legacy_key}'."
        )

    def _setup_interpolator_from_excel(self):
        """Loads data, converts to frequency space, and creates interpolators."""
        try:
            excel_cfg = self.config.excel_config
            filepath = resources.files('mqed') / excel_cfg.filepath
            logger.info(f"Loading dispersive material data from: {filepath}")
            logger.info(f"Excel sheet name: {excel_cfg.sheet_name}")

            df = pd.read_excel(filepath, sheet_name=excel_cfg.sheet_name)
            
            lambda_nm = df.iloc[:, 0].values
            epsilon_real = df.iloc[:, 1].values
            epsilon_imag = df.iloc[:, 2].values
            epsilon_complex = epsilon_real + 1j * epsilon_imag

            # Convert wavelength to angular frequency (omega)
            omega_data = 2 * np.pi * C_METERS_PER_SEC / (lambda_nm * 1e-9)

            # Sort data by omega, as interp1d requires monotonically increasing x-values
            sort_indices = np.argsort(omega_data)
            omega_sorted = omega_data[sort_indices]
            epsilon_sorted = epsilon_complex[sort_indices]
            self.omega_min = float(omega_sorted[0])
            self.omega_max = float(omega_sorted[-1])

            # Create and store interpolation functions once
            self.interp_real = interp1d(
                omega_sorted, epsilon_sorted.real, kind='cubic',
                bounds_error=True,
            )
            self.interp_imag = interp1d(
                omega_sorted, epsilon_sorted.imag, kind='cubic',
                bounds_error=True,
            )
            logger.success("Successfully initialized interpolator from Excel data.")

        except Exception as e:
            logger.error(f"Failed to load or process Excel data: {e}")
            raise

    def _setup_constant_epsilon(self):
        """Sets up the provider for a non-dispersive material."""
        self.constant_epsilon = complex(self.config.constant_value)
        logger.info(f"Using constant, non-dispersive epsilon = {self.constant_epsilon}")

    def _setup_drude_model(self):
        """Set up single-pole Drude parameters.

        Model form used in this code:
            epsilon(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma*omega)
        """
        cfg = self.config.drude_config
        self.drude_eps_inf = self._require_finite(cfg.eps_inf, "drude_config.eps_inf")
        self.drude_omega_p = self._read_omega_parameter(cfg, "omega_p")
        self.drude_gamma = self._read_omega_parameter(cfg, "gamma")
        self.drude_omega_p = self._require_positive(
            self.drude_omega_p,
            "drude_config.omega_p",
        )
        logger.info(
            "Using Drude model with eps_inf={}, omega_p={} rad/s, gamma={} rad/s",
            self.drude_eps_inf,
            self.drude_omega_p,
            self.drude_gamma,
        )

    def _setup_drude_lorentz_model(self):
        """Set up Drude-Lorentz parameters.

        Model form used in this code:
            epsilon(omega) = eps_inf
                             - omega_p^2 / (omega^2 + i*gamma_D*omega)
                             + sum_j [ f_j * omega_0j^2 /
                                      (omega_0j^2 - omega^2 - i*gamma_j*omega) ]

        where each oscillator entry provides:
            - strength (f_j)
            - omega_0 in eV or rad/s
            - gamma in eV or rad/s
        """
        cfg = self.config.drude_lorentz_config
        self.dl_eps_inf = self._require_finite(cfg.eps_inf, "drude_lorentz_config.eps_inf")
        self.dl_omega_p = self._read_omega_parameter(cfg, "omega_p")
        self.dl_gamma = self._read_omega_parameter(cfg, "gamma")
        self.dl_omega_p = self._require_positive(
            self.dl_omega_p,
            "drude_lorentz_config.omega_p",
        )

        oscillators = cfg.oscillators
        if not isinstance(oscillators, (list, ListConfig)) or len(oscillators) == 0:
            raise ValueError("drude_lorentz_config.oscillators must be a non-empty list.")

        self.dl_oscillators = []
        for idx, osc in enumerate(oscillators):
            if not isinstance(osc, (dict, DictConfig)):
                raise ValueError(
                    f"Oscillator #{idx} must be a dict with strength, omega_0_*, and gamma_*."
                )

            parsed = {
                "strength": self._require_finite(
                    osc.strength,
                    f"drude_lorentz_config.oscillators[{idx}].strength",
                ),
                "omega_0": self._read_omega_parameter(osc, "omega_0"),
                "gamma": self._read_omega_parameter(osc, "gamma"),
            }
            parsed["omega_0"] = self._require_positive(
                parsed["omega_0"],
                f"drude_lorentz_config.oscillators[{idx}].omega_0",
            )
            self.dl_oscillators.append(parsed)

        logger.info(
            "Using Drude-Lorentz model with {} oscillator(s)",
            len(self.dl_oscillators),
        )

    def _epsilon_drude(self, omega):
        """Evaluate single-pole Drude permittivity."""
        denominator = omega**2 + 1j * self.drude_gamma * omega
        return self.drude_eps_inf - (self.drude_omega_p**2) / denominator

    def _epsilon_drude_lorentz(self, omega):
        """Evaluate Drude-Lorentz permittivity with configured oscillators."""
        epsilon = self.dl_eps_inf - (self.dl_omega_p**2) / (
            omega**2 + 1j * self.dl_gamma * omega
        )

        for osc in self.dl_oscillators:
            numerator = osc["strength"] * (osc["omega_0"]**2)
            denominator = (osc["omega_0"]**2) - omega**2 - 1j * osc["gamma"] * omega
            epsilon = epsilon + numerator / denominator

        return epsilon

    def get_epsilon(self, omega):
        """
        Returns the complex relative permittivity (ε_r) for a given angular frequency.

        Args:
            omega (float): The target angular frequency in rad/s.

        Returns:
            complex: The complex relative permittivity ε_r(ω).
        """
        if self.source_type == 'excel':
            omega_eval = float(np.clip(float(omega), self.omega_min, self.omega_max))
            real_part = self.interp_real(omega_eval)
            imag_part = self.interp_imag(omega_eval)
            return complex(real_part, imag_part)

        elif self.source_type == 'constant':
            return self.constant_epsilon

        elif self.source_type == 'drude':
            return complex(self._epsilon_drude(float(omega)))

        elif self.source_type == 'drude_lorentz':
            return complex(self._epsilon_drude_lorentz(float(omega)))

        raise ValueError(f"Unsupported source type at evaluation time: '{self.source_type_raw}'")
