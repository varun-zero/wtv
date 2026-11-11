"""Cached end-to-end debug run: build/render a 2-flyover world once, cache
audio + flights + detection, then iterate on the search stage quickly.

Usage:
  python e2e_debug.py prep     # render + detect, cache to /tmp/skyprint_cache
  python e2e_debug.py diag     # dump band traces + families near CPA
  python e2e_debug.py search   # run search + decision on cached data
"""
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np

CACHE = os.environ.get("SKYPRINT_CACHE", "/tmp/skyprint_cache")
os.makedirs(CACHE, exist_ok=True)

TRUTH = (37.4, -122.3, 0.0)
DAY0 = 100_000.0
T_WIN = (DAY0 + 900.0, DAY0 + 1300.0)


def build_world():
    from skyprint.flights import synth_flight
    from skyprint.geo import destination
    from skyprint.atmosphere import Atmosphere

    la1, lo1 = destination(TRUTH[0], TRUTH[1], 270.0, 40_000.0)
    fl1 = synth_flight("SKY01", "C172", la1, lo1, 1500.0, 90.0, 100.0,
                       DAY0 + 600.0, DAY0 + 1600.0, dt=4.0)
    la2, lo2 = destination(TRUTH[0], TRUTH[1], 315.0, 60_000.0)
    fl2 = synth_flight("SKY02", "SR22", la2, lo2, 2200.0, 135.0, 110.0,
                       DAY0 + 480.0, DAY0 + 1500.0, dt=4.0)
    flights = [fl1, fl2]
    rng = np.random.default_rng(3)
    for i in range(4):
        b = rng.uniform(0, 360)
        d = rng.uniform(25, 70) * 1000
        la, lo = destination(TRUTH[0], TRUTH[1], b, d)
        flights.append(
            synth_flight(
                f"DIS{i}", str(rng.choice(["C208", "PA34"])),
                la, lo, rng.uniform(1500, 2500), rng.uniform(0, 360),
                rng.uniform(90, 140), DAY0 + rng.uniform(1500, 2500),
                DAY0 + rng.uniform(2500, 3500), dt=4.0,
            )
        )
    atmos = Atmosphere(wind_e_ms=3.0, wind_n_ms=-2.0)
    return flights, atmos


def prep():
    from skyprint.synth import render_recording

    flights, atmos = build_world()
    t0 = time.time()
    x, _ = render_recording(flights, T_WIN[0], T_WIN[1], TRUTH[0], TRUTH[1],
                            TRUTH[2], atmos=atmos, fs=8000)
    print(f"render {time.time()-t0:.1f}s, {len(x)/8000:.0f}s audio")
    with open(f"{CACHE}/world.pkl", "wb") as f:
        pickle.dump(dict(x=x, flights=flights, atmos=atmos), f)
    detect()


def detect():
    from skyprint.cli import detect_events

    with open(f"{CACHE}/world.pkl", "rb") as f:
        w = pickle.load(f)
    t0 = time.time()
    events, spec, freqs, times = detect_events(w["x"], 8000)
    print(f"detect {time.time()-t0:.1f}s -> {len(events)} events")
    for ev in events:
        print(f"  f0={ev.f0_hz:6.1f} v={ev.speed_ms:5.1f} r={ev.slant_m:6.0f} "
              f"tc={ev.t_cpa_s:.0f} snr={ev.snr:5.1f} orders={ev.orders}")
    with open(f"{CACHE}/events.pkl", "wb") as f:
        pickle.dump(events, f)


def load():
    with open(f"{CACHE}/world.pkl", "rb") as f:
        w = pickle.load(f)
    with open(f"{CACHE}/events.pkl", "rb") as f:
        events = pickle.load(f)
    return w["x"], w["flights"], w["atmos"], events


def diag():
    from skyprint.audio import (
        spectrogram, whiten, trace_ridge, trace_band_set, band_plan,
        harmonic_families, fit_event,
    )

    x, flights, atmos, events = load()
    spec, freqs, times = spectrogram(x, 8000, n_fft=4096)
    w = whiten(spec)
    print("== band traces (k=1 region, decimated) ==")
    for (lo, hi) in [(55, 165), (110, 330)]:
        ti, fi, wt = trace_ridge(w, freqs, times, lo, hi)
        if not len(ti):
            print(f"band {lo}-{hi}: EMPTY")
            continue
        t = times[ti]
        f = freqs[fi]
        o = np.argsort(t)
        t, f = t[o], f[o]
        print(f"band {lo}-{hi}: n={len(t)} f[{f.min():.1f}..{f.max():.1f}]")
        for tt, ff in list(zip(t, f))[::40]:
            print(f"    t={tt:6.1f} f={ff:6.1f}")
    print("== families ==")
    ridges = trace_band_set(w, freqs, times, band_plan(55, 1500))
    fams = harmonic_families(ridges)
    for i, fam in enumerate(fams):
        fit = fit_event(fam)
        segs = fam["segments"]
        if not segs:
            continue
        tmin = min(s[0][0] for s in segs)
        tmax = max(s[0][-1] for s in segs)
        s = (f"fam {i}: orders={fam['orders']} t[{tmin:.0f}..{tmax:.0f}]"
             if fit is None else
             f"fam {i}: orders={fam['orders']} t[{tmin:.0f}..{tmax:.0f}] "
             f"f0={fit.f0_hz:.1f} v={fit.speed_ms:.0f} r={fit.slant_m:.0f} "
             f"tc={fit.t_cpa_s:.0f} snr={fit.snr:.0f}")
        print(s)
        for (t, f, wt, k) in segs:
            print(f"    k={k}: t[{t.min():.0f}..{t.max():.0f}] "
                  f"f/k[{(f/k).min():.1f}..{(f/k).max():.1f}] n={len(t)}")


def search_run():
    from skyprint.geo import Bounds
    from skyprint.search import search
    from skyprint.confidence import decide

    x, flights, atmos, events = load()
    evs = list(events)
    for ev in evs:
        ev.freq_samples = np.asarray(ev.freq_samples, float) + T_WIN[0]
        ev.t_cpa_s += T_WIN[0]
    plausible = [e for e in evs if 40.0 <= e.speed_ms <= 200.0
                 and e.slant_m < 15_000.0]
    # distinct events: separated CPA times (>15 s dedupes fragments of one
    # flyover but keeps genuinely distinct passes, even overlapping ones)
    chosen = []
    for f in plausible:
        if all(abs(f.t_cpa_s - c.t_cpa_s) > 15.0 for c in chosen):
            chosen.append(f)
        if len(chosen) == 2:
            break
    if not chosen:
        chosen = plausible[:2]
    print(f"chosen {len(chosen)} events for search:")
    for ev in chosen:
        print(f"  f0={ev.f0_hz:.1f} v={ev.speed_ms:.0f} r={ev.slant_m:.0f} "
              f"tc={ev.t_cpa_s:.0f}")
    bounds = Bounds(-122.55, 37.2, -122.05, 37.6)
    t0 = time.time()
    cands = search(flights, chosen, bounds, atmos, T_WIN, verbose=True)
    print(f"search {time.time()-t0:.1f}s -> {len(cands)} candidates")
    for c in cands[:5]:
        print(f"  {c.lat:.4f},{c.lon:.4f} score={c.score:.2f} "
              f"flights={c.flight_ids} off={c.clock_offset_s:+.1f} "
              f"ppm={c.clock_drift_ppm:+.0f}")
    d = decide(cands, chosen, truth=TRUTH, verbose=True)
    print(f"DECISION: {d.status} radius={d.radius_m:.0f}m")
    for r in d.reasons:
        print(f"   {r}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    if mode == "prep":
        prep()
    elif mode == "detect":
        detect()
    elif mode == "diag":
        diag()
    elif mode == "search":
        search_run()
