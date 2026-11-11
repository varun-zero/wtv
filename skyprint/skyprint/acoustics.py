"""Retarded-time acoustic propagation with wind and altitude-dependent
sound speed.

This module is the physics core. The frequency heard at receive time t is
determined by the source's motion at the *emission* time tau, where
tau + T(tau) = t and T is the travel time. In a uniform wind W the travel
time along the straight ground-frame ray satisfies

    |D(tau) - W T| = c_path T        with D = x_rx - x_s(tau),

a quadratic in T, and differentiating gives

    dt/dtau = 1 - (u . v_s) / (c_path + u . W)
    f_rx    = f0 / (dt/dtau) = f0 * (c_path + u . W) / (c_path + u . W - u . v_s)

where u is the ground-frame ray unit vector from the advected source
position toward the receiver. Setting W = 0 recovers the classical
stationary-medium formula f0 c / (c - u . v_s); a source drifting with the
medium (v_s = W) gives no Doppler shift, as it must. These identities are
enforced by tests (see tests/test_acoustics.py).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .atmosphere import Atmosphere

_MAX_TRAVEL_S = 90.0  # 90 s of sound travel ~ 30 km; hard bracket bound


@dataclass(frozen=True)
class Emission:
    tau: float            # emission time (s, same epoch as the track)
    travel: float         # travel time (s)
    distance: float       # geometric slant distance (m)
    unit: np.ndarray      # ground-frame ray unit vector, source -> receiver
    c_path: float         # path-effective sound speed (m/s)
    dtdtau: float         # dt/dtau = arrival-rate of wavefronts


def travel_time_quadratic(d_vec: np.ndarray, wind_ecef: np.ndarray, c_path: float):
    """Travel time T solving |D - W T| = c T (uniform wind, scalar path c)."""
    d2 = float(np.dot(d_vec, d_vec))
    w2 = float(np.dot(wind_ecef, wind_ecef))
    dw = float(np.dot(d_vec, wind_ecef))
    a = w2 - c_path * c_path  # < 0 for sub-wind speeds
    if a >= -1e-12:  # wind faster than sound: no steady solution
        raise ValueError("wind speed exceeds path sound speed")
    disc = dw * dw - a * d2
    if disc < 0.0:
        raise ValueError("no real travel-time root")
    # Physical root: T = (dw - sqrt(disc)) / a. Checks: W -> 0 gives
    # |D|/c; wind along the ray shortens T to |D|/(c + |W|), against
    # lengthens it to |D|/(c - |W|).
    t = (dw - np.sqrt(disc)) / a  # -> |D|/c when W -> 0
    return float(max(t, 0.0))


def forward_curve(rx_ecef, track, tau_grid, atmos, wind_ecef, rx_alt_m):
    """Vectorized forward model: arrival times + 1/dtdtau on an emission
    grid, plus the CPA slant distance.

    For each emission time tau on the grid, computes the arrival time
    tau + T(tau) and the wavefront arrival-rate dt/dtau exactly (wind
    quadratic + path sound speed), all array-wise. The map tau ->
    t_arr is monotone increasing for subsonic sources, so the returned
    (t_arr, shape) series inverts exactly by interpolation -- no
    per-sample Newton solves. This is the scoring workhorse: candidate
    scoring and clock sweeps reduce to np.interp on this curve.

    Returns (t_arr, shape, slant_min): slant_min is the geometric CPA
    slant over the grid, used by the search's slant-consistency term.
    """
    from .geo import ecef_to_llh

    tau = np.asarray(tau_grid, dtype=float)
    pos, vel = track.eval_many(tau)
    d_vec = rx_ecef[None, :] - pos            # (N,3)
    d2 = (d_vec * d_vec).sum(axis=1)
    dw = d_vec @ wind_ecef
    w2 = float(wind_ecef @ wind_ecef)
    _, _, src_alt = ecef_to_llh(pos)
    # path-effective sound speed (harmonic mean over 5 stratification
    # nodes), vectorized over samples
    alts = float(rx_alt_m) + (src_alt - float(rx_alt_m)) * np.linspace(
        0.0, 1.0, 5
    )[:, None]
    c_prof = atmos.sound_speed_at(alts)        # (5,N)
    c_path = 1.0 / np.mean(1.0 / c_prof, axis=0)
    a = w2 - c_path * c_path
    disc = dw * dw - a * d2
    T = np.full_like(d2, np.nan)
    ok = (a < -1e-12) & (disc >= 0.0)
    T[ok] = (dw[ok] - np.sqrt(disc[ok])) / a[ok]
    m = d_vec - wind_ecef[None, :] * T[:, None]
    mn = np.sqrt((m * m).sum(axis=1))
    u = m / np.maximum(mn, 1e-9)[:, None]
    dtdtau = 1.0 - (u * vel).sum(axis=1) / (c_path + (u * wind_ecef[None, :]).sum(axis=1))
    t_arr = tau + T
    good = np.isfinite(t_arr) & np.isfinite(dtdtau) & (dtdtau > 0.2)
    if not good.any():
        return np.array([]), np.array([]), np.inf
    order = np.argsort(t_arr[good])
    slant_min = float(np.sqrt(np.min(d2)))
    return t_arr[good][order], (1.0 / dtdtau[good])[order], slant_min


def emission_state_at(
    rx_ecef: np.ndarray,
    track,
    tau: float,
    atmos: Atmosphere,
    wind_ecef: np.ndarray,
    rx_alt_m: float,
) -> Emission:
    """Direct emission-state evaluation at a *given* emission time tau
    (no root-finding): used by the renderer, which knows tau exactly."""
    pos, vel = track(float(tau))
    d_vec = rx_ecef - pos
    _, _, src_alt = _llh_safe(pos)
    c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
    tt = travel_time_quadratic(d_vec, wind_ecef, c_path)
    m = d_vec - wind_ecef * tt
    u = m / max(float(np.linalg.norm(m)), 1e-9)
    dtdtau = 1.0 - float(np.dot(u, vel)) / (c_path + float(np.dot(u, wind_ecef)))
    return Emission(
        tau=float(tau),
        travel=float(tt),
        distance=float(np.linalg.norm(d_vec)),
        unit=u,
        c_path=float(c_path),
        dtdtau=float(dtdtau),
    )


def solve_emission(
    rx_ecef: np.ndarray,
    track,
    t_recv: float,
    atmos: Atmosphere,
    wind_ecef: np.ndarray,
    rx_alt_m: float,
) -> Emission:
    """Emission state for sound arriving at t_recv at a stationary receiver.

    Newton iteration on g(tau) = tau + T(tau) - t_recv with the analytic
    Jacobian dt/dtau; g is strictly increasing for subsonic sources, so a
    sign-maintained bracket with bisection fallback guarantees convergence.
    """
    pos0, _ = track(t_recv - 30.0)
    lo = t_recv - _MAX_TRAVEL_S
    hi = t_recv
    tau = float(np.clip(t_recv - float(np.linalg.norm(rx_ecef - pos0)) / 320.0, lo, hi))

    def eval_state(t):
        pos, vel = track(float(t))
        d_vec = rx_ecef - pos
        _, _, src_alt = _llh_safe(pos)
        c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
        tt = travel_time_quadratic(d_vec, wind_ecef, c_path)
        m = d_vec - wind_ecef * tt
        u = m / max(float(np.linalg.norm(m)), 1e-9)
        dtdtau = 1.0 - float(np.dot(u, vel)) / (c_path + float(np.dot(u, wind_ecef)))
        return d_vec, u, c_path, dtdtau, float(t) + tt - t_recv, tt

    g = 1.0
    for _ in range(100):
        d_vec, u, c_path, dtdtau, g, tt = eval_state(tau)
        if abs(g) < 1e-9:
            break
        if g > 0.0:
            hi = tau
        else:
            lo = tau
        if dtdtau > 1e-6:
            step = g / dtdtau
            tau_new = tau - step
        else:  # near-supersonic radial motion: fall back to bisection
            tau_new = 0.5 * (lo + hi)
        if not (lo - 1.0 <= tau_new <= hi + 1.0) or np.isnan(tau_new):
            tau_new = 0.5 * (lo + hi)
        if abs(tau_new - tau) < 1e-12:
            break
        tau = tau_new

    d_vec, u, c_path, dtdtau, g, tt = eval_state(tau)
    if abs(g) > 1e-6:
        raise RuntimeError(f"emission solve failed to converge (|g|={abs(g):.2e})")
    return Emission(
        tau=float(tau),
        travel=float(tt),
        distance=float(np.linalg.norm(d_vec)),
        unit=u,
        c_path=float(c_path),
        dtdtau=float(dtdtau),
    )


def _llh_safe(pos_ecef):
    from .geo import ecef_to_llh

    lat, lon, h = ecef_to_llh(pos_ecef)
    return float(lat), float(lon), float(h)


def received_frequency(f0: float, emis: Emission, atmos: Atmosphere, wind_ecef: np.ndarray):
    """Frequency heard by the stationary receiver for emitted f0."""
    return f0 / emis.dtdtau


def received_level_db(
    source_level_db: float,
    f_hz: float,
    emis: Emission,
    atmos: Atmosphere,
    rx_alt_m: float,
    src_alt_m: float,
):
    """Received level (dB re the same reference as source_level_db@1m).

    Spherical spreading over the geometric slant distance plus ISO 9613-1
    absorption evaluated at path-mid conditions. Wind-induced intensity
    changes (order |W|/c) are ignored -- they are covered by the scenario
    envelope.
    """
    spread = -20.0 * np.log10(max(emis.distance, 1.0))
    abs_db = -float(atmos.path_absorption_db_per_m(f_hz, rx_alt_m, src_alt_m)) * emis.travel * atmos.path_sound_speed(rx_alt_m, src_alt_m)
    return source_level_db + spread + abs_db


def delayed_cpa(
    rx_ecef: np.ndarray, track, atmos: Atmosphere, wind_ecef: np.ndarray, rx_alt_m: float,
    t_lo: float, t_hi: float,
):
    """(delayed receive time of closest approach, geometric slant, tau*).

    Closest approach minimizes the geometric slant distance; the *delayed*
    CPA receive time is tau* + T(tau*). This is the "delayed CPA" of the
    source methodology: the moment an observer actually hears the loudest
    point of the pass, which is what time-offset joins must use.
    """
    ts = np.arange(t_lo, t_hi, 1.0)
    if len(ts) == 0:
        ts = np.array([t_lo])
    dists = []
    for t in ts:
        pos, _ = track(float(t))
        dists.append(np.linalg.norm(rx_ecef - pos))
    i = int(np.argmin(dists))
    t_ref = float(ts[i])
    pos, _ = track(t_ref)
    # golden-section refine on the geometric distance
    a, b = t_ref - 1.0, t_ref + 1.0

    def f(t):
        p, _ = track(float(t))
        return float(np.linalg.norm(rx_ecef - p))

    for _ in range(40):
        m1 = a + 0.382 * (b - a)
        m2 = b - 0.382 * (b - a)
        if f(m1) < f(m2):
            b = m2
        else:
            a = m1
    tau_star = 0.5 * (a + b)
    pos, _ = track(tau_star)
    slant = float(np.linalg.norm(rx_ecef - pos))
    _, _, src_alt = _llh_safe(pos)
    c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
    tt = travel_time_quadratic(rx_ecef - pos, wind_ecef, c_path)
    return tau_star + tt, slant, tau_star


def brute_force_frequency(
    rx_ecef, track, t_grid, f0, atmos, wind_ecef, rx_alt_m,
):
    """Reference implementation: dense emission scan + interpolation.

    Used by the test suite to validate solve_emission independently: it
    computes arrival times for a dense set of emissions and inverts the
    monotone map by interpolation rather than Newton iteration.
    """
    taus = np.linspace(t_grid[0] - _MAX_TRAVEL_S, t_grid[-1], 20001)
    t_arr = np.empty_like(taus)
    for i, tau in enumerate(taus):
        pos, _ = track(float(tau))
        _, _, src_alt = _llh_safe(pos)
        c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
        t_arr[i] = tau + travel_time_quadratic(rx_ecef - pos, wind_ecef, c_path)
    f_arr = np.empty_like(taus)
    for i, tau in enumerate(taus):
        pos, vel = track(float(tau))
        d_vec = rx_ecef - pos
        _, _, src_alt = _llh_safe(pos)
        c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
        tt = t_arr[i] - float(tau)
        m = d_vec - wind_ecef * tt
        u = m / max(np.linalg.norm(m), 1e-9)
        dtdtau = 1.0 - float(np.dot(u, vel)) / (c_path + float(np.dot(u, wind_ecef)))
        f_arr[i] = f0 / dtdtau
    # invert: for each t in grid, find tau bracketing, take f
    out = np.interp(t_grid, t_arr, f_arr)
    return out
