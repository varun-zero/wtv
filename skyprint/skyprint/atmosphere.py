"""Atmosphere models for SkyPrint.

Provides:
  * ISA temperature / pressure profiles (with a user temperature offset),
  * altitude-dependent speed of sound and path-effective sound speed,
  * uniform wind vectors (user prior; scenario-perturbed),
  * ISO 9613-1 / ANSI S1.26 pure-tone atmospheric absorption.

References
----------
ISO 9613-1:1996 / ANSI S1.26-2014, "Calculation of the absorption of sound
by the atmosphere" (relaxation frequencies and absorption coefficient).
U.S. Standard Atmosphere, 1976 (temperature and pressure below 51 km).

The ISO 9613-1 constants used here were numerically validated:
  alpha(1000 Hz, 20 C, 50% RH, 101.325 kPa) ~ 4.7 dB/km
  alpha(4000 Hz, 20 C, 50% RH)              ~ 30 dB/km
  psat anchors: 0.611 kPa @ 0 C, 2.332 kPa @ 20 C, 4.23 kPa @ 30 C
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

T01 = 293.15          # K, ISO 9613-1 reference temperature (20 degC)
T_SL = 288.15         # K, ISA sea-level temperature (15 degC)
T_TRIPLE = 273.16     # K, water triple point (vapor pressure anchor)
P_REF = 101.325       # kPa, ISO reference pressure
LAPSE = 0.0065        # K/m, ISA troposphere lapse rate
TROPOPAUSE = 11000.0  # m
GAMMA = 1.4           # ratio of specific heats, dry air
R_D = 287.05287       # J/(kg K), specific gas constant of dry air
G0 = 9.80665          # m/s^2

_EXP_P = G0 / (R_D * LAPSE)      # 5.2559
_SCALE_H = R_D * 216.65 / G0     # ~6341 m, isothermal above tropopause


def isa_temperature(alt_m, offset_k: float = 0.0):
    """Temperature (K) at geometric altitude with an optional offset."""
    a = np.asarray(alt_m, dtype=float)
    base = np.where(a < TROPOPAUSE, T_SL - LAPSE * a, 216.65)
    return base + offset_k


def isa_pressure(alt_m):
    """Pressure (kPa) at geometric altitude (ISA)."""
    a = np.asarray(alt_m, dtype=float)
    trop = np.minimum(a, TROPOPAUSE)
    below = P_REF * (1.0 - LAPSE / T_SL * trop) ** _EXP_P
    above = P_REF * (1.0 - LAPSE * TROPOPAUSE / T_SL) ** _EXP_P * np.exp(
        -(a - TROPOPAUSE) / _SCALE_H
    )
    return np.where(a < TROPOPAUSE, below, above)


def sound_speed(t_k):
    """Speed of sound (m/s) from temperature (K)."""
    return np.sqrt(GAMMA * R_D * np.asarray(t_k, dtype=float))


def saturation_vapor_pressure_kpa(t_k):
    """Saturation vapor pressure (kPa), ANSI S1.26 / ISO 9613-1 form."""
    t_k = np.asarray(t_k, dtype=float)
    return P_REF * 10.0 ** (-6.8346 * (T_TRIPLE / t_k) ** 1.261 + 4.6151)


def absorption_coefficient_db_per_m(f_hz, t_k, rel_humidity, p_kpa):
    """ISO 9613-1 pure-tone absorption coefficient alpha (dB/m).

    f_hz: frequency (Hz); t_k: temperature (K); rel_humidity: fraction
    (0..1); p_kpa: ambient pressure (kPa). Vectorized over f_hz and t_k.
    """
    f = np.asarray(f_hz, dtype=float)
    t = np.asarray(t_k, dtype=float)
    p = np.asarray(p_kpa, dtype=float)
    h = 100.0 * rel_humidity * saturation_vapor_pressure_kpa(t) / p
    h = np.clip(h, 0.0, 100.0)
    pr = p / P_REF
    # Oxygen and nitrogen relaxation frequencies (Hz).
    fr_o = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * np.sqrt(T01 / t) * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T01 / t) ** (1.0 / 3.0) - 1.0))
    )
    tr = t / T01
    alpha = 8.686 * f ** 2 * (
        1.84e-11 * pr * np.sqrt(tr)
        + tr ** (-2.5)
        * (
            0.01275 * np.exp(-2239.1 / t) / (fr_o + f ** 2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / t) / (fr_n + f ** 2 / fr_n)
        )
    )
    return alpha


@dataclass(frozen=True)
class Atmosphere:
    """Steady atmospheric state used by propagation and scoring.

    wind_e_ms / wind_n_ms: uniform east/north wind components (m/s) -- a
    user prior; scenario envelopes perturb it. rel_humidity: fraction.
    temp_offset_k: sea-level temperature anomaly vs ISA (K).

    Wind is treated as uniform with height (a deliberate simplification;
    vertical shear and refraction are covered by the scenario envelope in
    confidence.py rather than modeled deterministically).
    """

    temp_offset_k: float = 0.0
    wind_e_ms: float = 0.0
    wind_n_ms: float = 0.0
    rel_humidity: float = 0.50

    # --- profiles -----------------------------------------------------
    def temperature(self, alt_m):
        return isa_temperature(alt_m, self.temp_offset_k)

    def pressure(self, alt_m):
        return isa_pressure(alt_m)

    def sound_speed_at(self, alt_m):
        return sound_speed(self.temperature(alt_m))

    def wind_speed(self) -> float:
        return float(np.hypot(self.wind_e_ms, self.wind_n_ms))

    def wind_ecef(self, frame) -> np.ndarray:
        """Wind vector in ECEF for a local ENU frame (u=0 component)."""
        return frame.enu_direction_to_ecef(self.wind_e_ms, self.wind_n_ms, 0.0)

    # --- path quantities ----------------------------------------------
    def path_sound_speed(self, alt_rx_m: float, alt_src_m: float) -> float:
        """Effective sound speed (m/s) along a slant path.

        Travel time through a stratified medium along a straight path is
        L * <1/c>, so the effective speed is the harmonic mean of c(h)
        sampled along the path (5 nodes), not the arithmetic mean.
        """
        alts = np.linspace(float(alt_rx_m), float(alt_src_m), 5)
        c = self.sound_speed_at(alts)
        return float(1.0 / np.mean(1.0 / c))

    def path_absorption_db_per_m(self, f_hz, alt_rx_m: float, alt_src_m: float):
        """Absorption (dB/m) at path-mid conditions."""
        alt_mid = 0.5 * (float(alt_rx_m) + float(alt_src_m))
        t_mid = float(self.temperature(alt_mid))
        p_mid = max(float(self.pressure(alt_mid)), 1.0)
        return float(absorption_coefficient_db_per_m(f_hz, t_mid, self.rel_humidity, p_mid))

    # --- scenario machinery --------------------------------------------
    def perturbed(
        self,
        d_wind_e_ms: float = 0.0,
        d_wind_n_ms: float = 0.0,
        d_temp_k: float = 0.0,
        d_rh: float = 0.0,
    ) -> "Atmosphere":
        return Atmosphere(
            temp_offset_k=self.temp_offset_k + d_temp_k,
            wind_e_ms=self.wind_e_ms + d_wind_e_ms,
            wind_n_ms=self.wind_n_ms + d_wind_n_ms,
            rel_humidity=float(np.clip(self.rel_humidity + d_rh, 0.05, 0.99)),
        )


def scenario_atmospheres(base: Atmosphere, wind_ms: float = 10.0, temp_k: float = 10.0):
    """Model-error envelope variants (wind at 4 headings, temperature)."""
    out = [("central", base)]
    for name, de, dn in [
        ("wind+e", wind_ms, 0.0),
        ("wind-e", -wind_ms, 0.0),
        ("wind+n", 0.0, wind_ms),
        ("wind-n", 0.0, -wind_ms),
        ("wind+ne", wind_ms / np.sqrt(2), wind_ms / np.sqrt(2)),
        ("wind+sw", -wind_ms / np.sqrt(2), -wind_ms / np.sqrt(2)),
    ]:
        out.append((name, base.perturbed(d_wind_e_ms=de, d_wind_n_ms=dn)))
    out.append(("hot", base.perturbed(d_temp_k=temp_k)))
    out.append(("cold", base.perturbed(d_temp_k=-temp_k)))
    return out
