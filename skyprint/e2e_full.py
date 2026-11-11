"""Full end-to-end validation: rebuilt worlds + degradations + global sweep.

Cases:
  A  crossing baseline (37.4N 122.3W)     A1 MP3 48k   A2 +10ppm   A3 HPF 80
  B  parallel 26 s (same site)
  G1 Svalbard 78.2N 15.6E (polar)         G2 Fiji dateline (wrapped bounds)
  G3 Southern Ocean 54.5S 3.0E            G4 Equatorial Pacific 0.5N 160W

Writes results incrementally to /tmp/skyprint_cache/results.json so partial
results survive a time cut. Truth error + decision status per case.
"""
import json
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np

CACHE = os.environ.get("SKYPRINT_CACHE", "/tmp/skyprint_cache")
import os
os.makedirs(CACHE, exist_ok=True)
RESULTS = f"{CACHE}/results.json"

DAY0 = 100_000.0


def _crossing_world(truth, sep_s, win=(700.0, 1900.0), distractors=4, seed=3):
    """Two crossing flyovers (eastbound + westbound) + distractors."""
    from skyprint.atmosphere import Atmosphere
    from skyprint.flights import synth_flight
    from skyprint.geo import destination

    lat, lon = truth
    la1, lo1 = destination(lat, lon, 270.0, 40_000.0)
    fl1 = synth_flight("SKY01", "C172", la1, lo1, 1500.0, 90.0, 100.0,
                       DAY0 + 600.0, DAY0 + 1600.0, dt=4.0)
    la2, lo2 = destination(lat, lon, 90.0, 60_000.0)
    t2 = DAY0 + 950.0 + (sep_s - 496.0)
    fl2 = synth_flight("SKY02", "SR22", la2, lo2, 2200.0, 270.0, 110.0,
                       t2, t2 + 1050.0, dt=4.0)
    flights = [fl1, fl2]
    rng = np.random.default_rng(seed)
    for i in range(distractors):
        b = rng.uniform(0, 360)
        d = rng.uniform(25, 70) * 1000
        la, lo = destination(lat, lon, b, d)
        flights.append(
            synth_flight(
                f"DIS{i}", str(rng.choice(["C208", "PA34"])),
                la, lo, rng.uniform(1500, 2500), rng.uniform(0, 360),
                rng.uniform(90, 140), DAY0 + rng.uniform(1500, 2500),
                DAY0 + rng.uniform(2500, 3500), dt=4.0,
            )
        )
    return flights, Atmosphere(wind_e_ms=3.0, wind_n_ms=-2.0), win


def _parallel_world(truth):
    from skyprint.atmosphere import Atmosphere
    from skyprint.flights import synth_flight
    from skyprint.geo import destination

    lat, lon = truth
    la1, lo1 = destination(lat, lon, 270.0, 40_000.0)
    fl1 = synth_flight("SKY01", "C172", la1, lo1, 1500.0, 90.0, 100.0,
                       DAY0 + 600.0, DAY0 + 1600.0, dt=4.0)
    la2, lo2 = destination(lat, lon, 315.0, 60_000.0)
    fl2 = synth_flight("SKY02", "SR22", la2, lo2, 2200.0, 135.0, 110.0,
                       DAY0 + 480.0, DAY0 + 1500.0, dt=4.0)
    flights = [fl1, fl2]
    rng = np.random.default_rng(3)
    for i in range(4):
        b = rng.uniform(0, 360)
        d = rng.uniform(25, 70) * 1000
        la, lo = destination(lat, lon, b, d)
        flights.append(
            synth_flight(
                f"DIS{i}", str(rng.choice(["C208", "PA34"])),
                la, lo, rng.uniform(1500, 2500), rng.uniform(0, 360),
                rng.uniform(90, 140), DAY0 + rng.uniform(1500, 2500),
                DAY0 + rng.uniform(2500, 3500), dt=4.0,
            )
        )
    return flights, Atmosphere(wind_e_ms=3.0, wind_n_ms=-2.0), (900.0, 1300.0)


def run_case(name, x, fs, flights, atmos, t_window, bounds, truth):
    from skyprint.cli import detect_events
    from skyprint.confidence import decide
    from skyprint.geo import great_circle_distance
    from skyprint.search import search

    t0 = time.time()
    events, _, _, _ = detect_events(x, fs)
    for ev in events:
        ev.freq_samples = np.asarray(ev.freq_samples, float) + t_window[0]
        ev.t_cpa_s += t_window[0]
    good = [
        e for e in events
        if 40.0 <= e.speed_ms <= 200.0 and e.slant_m < 20_000.0
        and 30.0 <= e.f0_hz <= 500.0
    ]
    classes = [f.aircraft for f in flights if f.aircraft.tonal]
    good = [
        e for e in good
        if any(c.bpf_lo_hz * 0.6 <= e.f0_hz <= c.bpf_hi_hz * 1.4
               for c in classes)
    ]
    chosen = []
    for f in good:
        if all(abs(f.t_cpa_s - c.t_cpa_s) > 15.0 for c in chosen):
            chosen.append(f)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        chosen = good[:2]
    det_s = time.time() - t0
    if not chosen:
        return dict(name=name, status="inconclusive",
                    reason="no events detected", detect_s=round(det_s, 1))
    t0 = time.time()
    cands = search(flights, chosen, bounds, atmos, t_window)
    search_s = time.time() - t0
    d = decide(cands, chosen)
    err = (
        great_circle_distance(d.top.lat, d.top.lon, truth[0], truth[1])
        if d.top is not None else None
    )
    res = dict(
        name=name,
        status=d.status,
        error_m=round(float(err), 1) if err is not None else None,
        radius_m=round(d.radius_m, 1),
        top=dict(lat=round(d.top.lat, 5), lon=round(d.top.lon, 5),
                 score=round(d.top.score, 2),
                 flights=list(d.top.flight_ids),
                 clock_off=round(d.top.clock_offset_s, 2),
                 clock_ppm=round(d.top.clock_drift_ppm, 1)) if d.top else None,
        reasons=d.reasons,
        n_events=len(chosen),
        event_f0=[round(e.f0_hz, 1) for e in chosen],
        detect_s=round(det_s, 1),
        search_s=round(round(search_s, 1), 1),
    )
    print(f"[{name}] {d.status} err={res.get('error_m')}m "
          f"radius={res['radius_m']}m ({res['detect_s']}s+{res['search_s']}s)",
          flush=True)
    return res


def main():
    from skyprint.geo import Bounds
    from skyprint.synth import (
        apply_clock_drift, codec_roundtrip, highpass_filter, render_recording,
    )

    results = []

    def record(r):
        results.append(r)
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)

    # ---- world A (crossing, 500 s apart) ------------------------------
    truth_a = (37.4, -122.3)
    flights_a, atmos_a, win_a = _crossing_world(truth_a, 500.0)
    x, _ = render_recording(flights_a, DAY0 + win_a[0], DAY0 + win_a[1],
                             truth_a[0], truth_a[1], 0.0, atmos=atmos_a,
                             fs=8000)
    bounds_a = Bounds(-122.55, 37.2, -122.05, 37.6)
    record(run_case("A-crossing-baseline", x, 8000, flights_a, atmos_a,
                    (DAY0 + win_a[0], DAY0 + win_a[1]), bounds_a, truth_a))

    x_mp3, fs_mp3 = codec_roundtrip(x, 8000, "libmp3lame", "48k")
    record(run_case("A1-mp3-48k", x_mp3, fs_mp3, flights_a, atmos_a,
                    (DAY0 + win_a[0], DAY0 + win_a[1]), bounds_a, truth_a))

    x_dr, _ = apply_clock_drift(x, 10.0, 8000)
    record(run_case("A2-clock-drift-10ppm", x_dr, 8000, flights_a, atmos_a,
                    (DAY0 + win_a[0], DAY0 + win_a[1]), bounds_a, truth_a))

    x_hp = highpass_filter(x, 8000, 80.0)
    record(run_case("A3-highpass-80hz", x_hp, 8000, flights_a, atmos_a,
                    (DAY0 + win_a[0], DAY0 + win_a[1]), bounds_a, truth_a))

    # ---- world B (parallel 26 s) --------------------------------------
    truth_b = (37.4, -122.3)
    flights_b, atmos_b, win_b = _parallel_world(truth_b)
    x, _ = render_recording(flights_b, DAY0 + win_b[0], DAY0 + win_b[1],
                             truth_b[0], truth_b[1], 0.0, atmos=atmos_b,
                             fs=8000)
    record(run_case("B-parallel-26s", x, 8000, flights_b, atmos_b,
                    (DAY0 + win_b[0], DAY0 + win_b[1]), bounds_a, truth_b))

    # ---- global sweep ---------------------------------------------------
    worlds = [
        ("G1-polar-svalbard", (78.2, 15.6), Bounds(15.0, 77.8, 16.2, 78.6)),
        ("G2-dateline-fiji", (-17.5, 179.0), Bounds(177.0, -18.5, -179.0, -16.5)),
        ("G3-southern-ocean", (-54.5, 3.0), Bounds(2.0, -55.2, 4.0, -53.8)),
        ("G4-equatorial-pacific", (0.5, -160.0), Bounds(-161.2, -0.4, -158.8, 1.4)),
    ]
    for name, truth, bounds in worlds:
        flights, atmos, win = _crossing_world(
            truth, 350.0, win=(700.0, 1500.0), distractors=2, seed=7
        )
        x, _ = render_recording(flights, DAY0 + win[0], DAY0 + win[1],
                                truth[0], truth[1], 0.0, atmos=atmos,
                                fs=8000)
        record(run_case(name, x, 8000, flights, atmos,
                        (DAY0 + win[0], DAY0 + win[1]), bounds, truth))

    print("\n== FINAL ==")
    for r in results:
        print(f"  {r['name']}: {r['status']} err={r.get('error_m')}m "
              f"radius={r.get('radius_m')}m")


if __name__ == "__main__":
    main()
