"""End-to-end validation suite: multi-world geolocation + degradations.

Worlds:
  A. crossing   -- two flyovers, opposite headings, 500 s apart (located)
  B. parallel   -- two similar-heading flyovers 26 s apart (ambiguous
                   expected: intrinsic corridor degeneracy)
Degradations (applied to world A):
  1. MP3 48 kbps codec round-trip
  2. +10 ppm recorder clock drift (applied to the AUDIO, not the model)
  3. Aggressive 150 Hz high-pass (kills fundamental + k=2 of world A's
     SR22-class tone: must still geolocate via higher harmonics)

Each case reports: status, error-vs-truth, radius. Writes JSON results
for the HTML report generator.
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
TRUTH = (37.4, -122.3, 0.0)
DAY0 = 100_000.0


def load_events(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_case(name, x, fs, flights, atmos, t_window, bounds, expect_status=None):
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
    # drop events whose fundamental matches NO flight's aircraft-class BPF
    # window (sub-harmonic artifacts like an isolated k=3 family read at
    # f0/3): they cannot be assigned and would poison the chooser
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
                    reason="no events detected",
                    detect_s=round(det_s, 1))
    t0 = time.time()
    cands = search(flights, chosen, bounds, atmos, t_window)
    search_s = time.time() - t0
    d = decide(cands, chosen)
    err = (
        great_circle_distance(d.top.lat, d.top.lon, TRUTH[0], TRUTH[1])
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
        search_s=round(search_s, 1),
    )
    print(f"[{name}] {d.status} err={res.get('error_m')}m "
          f"radius={res['radius_m']}m "
          f"({res['detect_s']}s+{res['search_s']}s)")
    return res


def main():
    from skyprint.geo import Bounds
    from skyprint.synth import (
        apply_clock_drift, codec_roundtrip, highpass_filter,
    )

    results = []
    bounds = Bounds(-122.55, 37.2, -122.05, 37.6)

    with open(f"{CACHE}/world2.pkl", "rb") as f:
        w2 = pickle.load(f)
    t_win_a = (DAY0 + 700.0, DAY0 + 1900.0)

    # A: baseline (crossing, 500 s offset)
    results.append(run_case(
        "A-crossing-baseline", w2["x"], 8000, w2["flights"], w2["atmos"],
        t_win_a, bounds,
    ))

    # A1: MP3 48k
    try:
        x_mp3, fs_mp3 = codec_roundtrip(w2["x"], 8000, "libmp3lame", "48k")
        results.append(run_case(
            "A1-mp3-48k", x_mp3, fs_mp3, w2["flights"], w2["atmos"],
            t_win_a, bounds,
        ))
    except Exception as e:
        results.append(dict(name="A1-mp3-48k", status="skipped",
                            reason=str(e)))

    # A2: clock drift +10 ppm on the recording (fs reported stays 8000 ->
    # the search must absorb it via the ppm term)
    x_dr, _ = apply_clock_drift(w2["x"], 10.0, 8000)
    results.append(run_case(
        "A2-clock-drift-10ppm", x_dr, 8000, w2["flights"], w2["atmos"],
        t_win_a, bounds,
    ))

    # A3: 80 Hz high-pass (phone-mic style rolloff: weakens/removes the
    # fundamental; higher harmonics must carry the fix)
    x_hp = highpass_filter(w2["x"], 8000, 80.0)
    results.append(run_case(
        "A3-highpass-80hz", x_hp, 8000, w2["flights"], w2["atmos"],
        t_win_a, bounds,
    ))

    # B: parallel world (cached world.pkl), expect ambiguous w/ good err
    with open(f"{CACHE}/world.pkl", "rb") as f:
        w1 = pickle.load(f)
    t_win_b = (DAY0 + 900.0, DAY0 + 1300.0)
    results.append(run_case(
        "B-parallel-26s", w1["x"], 8000, w1["flights"], w1["atmos"],
        t_win_b, bounds,
    ))

    with open(f"{CACHE}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {CACHE}/results.json")
    for r in results:
        print(f"  {r['name']}: {r['status']} "
              f"err={r.get('error_m')}m radius={r.get('radius_m')}m")


if __name__ == "__main__":
    main()
