"""Synthetic audio rendering: the world -> recording path used by the
validation suite. Renders each flight's tonal signature at a receiver using
the *same* physics the searcher models (retarded time, path absorption,
spreading), plus noise and optional degradations (MP3/AAC codec round-trip
via ffmpeg, clock drift, stationary interference tones, high-pass loss).

Per flight and harmonic k, the received signal is synthesized by
integrating the received-frequency curve on the sample grid:

    phase(t) = 2*pi*INTEGRAL f_k(t) dt,   s(t) = a_k(t) * sin(phase)

with f_k(t) and a_k(t) interpolated from sparse emission solutions (0.05 s
nodes). This is the physically correct way to render a Doppler-swept tone
and naturally reproduces the receive-time curve asymmetry that made the
demo's measured curves "not fit a perfect Doppler curve".
"""
from __future__ import annotations

import subprocess
import tempfile
import os
import numpy as np

from .acoustics import emission_state_at, received_level_db
from .atmosphere import Atmosphere
from .geo import EnuFrame, ecef_to_llh, llh_to_ecef


def render_recording(
    flights,
    t_lo: float,
    t_hi: float,
    truth_lat: float,
    truth_lon: float,
    truth_alt_m: float,
    atmos: Atmosphere | None = None,
    fs: int = 8000,
    noise_db: float = -14.0,
    harmonic_falloff_db: float = 7.0,   # per-order level falloff
    rpm_wander_hz: float = 0.015,
    seed: int = 7,
    interferences: list | None = None,
    band_limit_hz: float | None = None,
    return_tracks: bool = False,
):
    """Render (samples, truth_events[, track_details]) for the receiver.

    truth_events: one entry per flight that actually produced audible tone
    energy in the window -- used to judge detection recall after analysis.
    """
    atmos = atmos or Atmosphere()
    rng = np.random.default_rng(seed)
    rx = llh_to_ecef(truth_lat, truth_lon, truth_alt_m)
    frame = EnuFrame(float(truth_lat), float(truth_lon))
    wind = atmos.wind_ecef(frame)
    dur = t_hi - t_lo
    n = int(round(dur * fs))
    x = np.zeros(n, dtype=np.float64)

    truth_events = []
    tracks_out = []
    for fl in flights:
        cls = fl.aircraft
        if not cls.tonal:
            continue
        f0_base = 0.5 * (cls.bpf_lo_hz + cls.bpf_hi_hz)
        seg_nodes = []  # per segment: (t_arr_nodes, f_k_nodes, a_k_nodes)
        audible = False
        for seg in fl.segments:
            # emission nodes spanning the segment
            taus = np.arange(seg.t[0], min(seg.t[-1], t_hi + 60.0), 0.05)
            if len(taus) < 2:
                continue
            arr = np.empty(len(taus))
            fks = np.empty((len(taus), cls.max_harmonic))
            aks = np.empty((len(taus), cls.max_harmonic))
            for i, tau in enumerate(taus):
                pos, _ = seg(float(tau))
                dist = float(np.linalg.norm(rx - pos))
                if dist > 32_000.0:
                    arr[i] = np.nan
                    continue
                _, _, src_alt = ecef_to_llh(pos)
                emis = emission_state_at(rx, seg, float(tau), atmos, wind, truth_alt_m)
                t_arr = emis.tau + emis.travel
                arr[i] = t_arr
                f_now = f0_base * (1.0 + rpm_wander_hz * np.sin(0.008 * float(tau)))
                for k in range(1, cls.max_harmonic + 1):
                    fk = k * f_now / emis.dtdtau
                    lvl = received_level_db(
                        cls.source_level_db, min(fk, 0.45 * fs), emis, atmos,
                        truth_alt_m, src_alt,
                    )
                    a = 10 ** ((lvl - 94.0) / 20.0) * 10 ** (-harmonic_falloff_db * (k - 1) / 20.0)
                    fks[i, k - 1] = fk
                    aks[i, k - 1] = a
            ok = np.isfinite(arr)
            if ok.sum() < 2:
                continue
            if float(np.nanmax(aks)) > 1e-4:
                audible = True
            seg_nodes.append((arr[ok], fks[ok], aks[ok]))
        if audible:
            truth_events.append(
                dict(
                    flight=fl.callsign,
                    type=fl.type_designator,
                    f0=f0_base,
                    t_lo=t_lo,
                    t_hi=t_hi,
                )
            )
        for (arr, fks, aks) in seg_nodes:
            for k in range(1, cls.max_harmonic + 1):
                fk = fks[:, k - 1]
                ak = aks[:, k - 1]
                if not np.isfinite(fk).any() or np.nanmax(ak) <= 0.0:
                    continue
                # relative cutoff: render any harmonic within 60 dB of this
                # segment's loudest (absolute cutoffs break under the final
                # peak normalization; audibility is decided by physics, and
                # inaudible far-away harmonics simply render at ~0 amplitude)
                if np.nanmax(ak) < np.nanmax(aks) * 10 ** (-60.0 / 20.0):
                    continue
                t_arr = arr[np.isfinite(arr) & np.isfinite(fk)]
                if len(t_arr) < 2:
                    continue
                in_win = (t_arr >= t_lo - 1.0) & (t_arr <= t_hi + 1.0)
                if not in_win.any():
                    continue
                # interpolate f and a onto the sample grid via arrival times
                tt = np.arange(n) / fs + t_lo
                f_grid = np.interp(tt, arr, fk, left=np.nan, right=np.nan)
                a_grid = np.interp(tt, arr, ak, left=0.0, right=0.0)
                good = np.isfinite(f_grid)
                if not good.any():
                    continue
                f_fill = np.where(good, f_grid, np.nanmedian(f_grid))
                phase = 2 * np.pi * np.cumsum(f_fill) / fs
                x += np.where(good, a_grid, 0.0) * np.sin(phase)
                if return_tracks:
                    tracks_out.append(
                        dict(flight=fl.callsign, order=k, t=tt[good],
                             f=f_grid[good], a=a_grid[good])
                    )
    # noise floor as true dB below the peak tone amplitude
    tone_peak = float(np.abs(x).max())
    noise_std = tone_peak * (10.0 ** (noise_db / 20.0))
    x += noise_std * 0.5 * rng.standard_normal(n)
    x += noise_std * 0.8 * np.convolve(
        rng.standard_normal(n), np.ones(8) / 8, "same"
    )
    if interferences:
        for (f0, amp, t0, t1) in interferences:
            i0, i1 = int((t0 - t_lo) * fs), int((t1 - t_lo) * fs)
            tt = np.arange(max(0, i0), min(n, i1)) / fs
            x[max(0, i0):min(n, i1)] += amp * np.sin(2 * np.pi * f0 * tt)
    if band_limit_hz:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(2, band_limit_hz / (fs / 2), "low", output="sos")
        x = sosfiltfilt(sos, x)
    x = x / max(np.abs(x).max(), 1e-9) * 0.9
    if return_tracks:
        return x.astype(np.float32), truth_events, tracks_out
    return x.astype(np.float32), truth_events


# --------------------------------------------------------------------------
# Degradation utilities (the demo's countermeasure tests)
# --------------------------------------------------------------------------

def codec_roundtrip(samples, fs, codec="libmp3lame", bitrate="48k"):
    """Encode + decode via ffmpeg; returns (samples, fs_out)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        mid = os.path.join(td, "mid.mp3" if codec == "libmp3lame" else "mid.aac")
        out = os.path.join(td, "out.wav")
        sf.write(wav, samples, fs)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", codec,
             "-b:a", bitrate, mid],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", mid, out],
            check=True,
        )
        data, fs_out = sf.read(out, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, fs_out


def apply_clock_drift(samples, ppm: float, fs: int):
    """Resample to simulate recorder clock drift; returns (samples, fs')."""
    n = len(samples)
    t = np.arange(n) / fs
    t_drift = t * (1.0 + ppm * 1e-6)
    out = np.interp(t_drift, t, samples)
    return out.astype(np.float32), int(round(fs / (1.0 + ppm * 1e-6)))


def highpass_filter(samples, fs, cutoff_hz: float, order: int = 4):
    """Aggressive HPF (the demo's first proposed countermeasure)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, cutoff_hz / (fs / 2), "high", output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)
