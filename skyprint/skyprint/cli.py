"""SkyPrint CLI: analyze a recording (or synthesize a demo world) and
report ranked candidate locations.

Commands:
  analyze <audio>      -- detect events, search, decide, emit report.html
  demo                 -- build a synthetic world, geolocate it end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np

from .atmosphere import Atmosphere
from .audio import (
    spectrogram, whiten, trace_band_set, harmonic_families, fit_event,
    band_plan,
)
from .confidence import decide
from .geo import Bounds
from .search import search


def detect_events(x, fs):
    spec, freqs, times = spectrogram(x, fs, n_fft=4096)
    w = whiten(spec)
    ridges = trace_band_set(w, freqs, times, band_plan(55, min(1500, fs * 0.45)))
    fams = harmonic_families(ridges)
    fits = []
    for fam in fams:
        fit = fit_event(fam)
        if fit is None:
            continue
        fits.append(fit)
    # keep physically plausible events, strongest first
    fits = [f for f in fits if f.snr > 3.0 and f.speed_ms > 25.0]
    fits.sort(key=lambda f: -f.snr)
    return fits, spec, freqs, times


def run_analysis(x, fs, flights, bounds, t_window, atmos,
                 rx_alt_m=0.0, verbose=True):
    t0 = time.time()
    events, spec, freqs, times = detect_events(x, fs)
    # events' freq_samples are window-relative; make them absolute
    for ev in events:
        ev.freq_samples = np.asarray(ev.freq_samples, float) + t_window[0]
        ev.t_cpa_s += t_window[0]
    if verbose:
        print(f"detected {len(events)} candidate events "
              f"({time.time()-t0:.1f}s)")
        for ev in events[:6]:
            print(f"  f0={ev.f0_hz:6.1f} Hz v={ev.speed_ms:5.1f} m/s "
                  f"r={ev.slant_m:6.0f} m snr={ev.snr:5.1f} "
                  f"t_cpa={ev.t_cpa_s:.0f}")
    t0 = time.time()
    # physically plausible events only, then strongest-first: fragments
    # (recede-only or k-contaminated families) carry biased speeds/slants
    # whose masks kill the all-event join
    plausible = [
        f for f in events
        if 40.0 <= f.speed_ms <= 200.0 and f.slant_m < 15_000.0
    ]
    # prefer mutually-consistent pairs (distinct CPA windows)
    chosen = []
    for f in plausible:
        if all(abs(f.t_cpa_s - c.t_cpa_s) > 120.0 for c in chosen):
            chosen.append(f)
        if len(chosen) == 2:
            break
    if not chosen:
        chosen = plausible[:2]
    cands = search(
        flights, chosen, bounds, atmos, t_window,
        rx_alt_m=rx_alt_m, verbose=verbose,
    )
    if verbose:
        print(f"search: {len(cands)} candidates ({time.time()-t0:.1f}s)")
    decision = decide(cands, events, verbose=verbose)
    return events, cands, decision


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skyprint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("audio")
    p_an.add_argument("--bounds", nargs=4, type=float,
                      help="west south east north (deg)")
    p_an.add_argument("--wind-e", type=float, default=0.0)
    p_an.add_argument("--wind-n", type=float, default=0.0)
    p_an.add_argument("--temp-offset", type=float, default=0.0)
    p_an.add_argument("--out", default="report.html")
    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--out", default="demo_report.html")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--region-deg", type=float, default=0.25)
    
    p_ui = sub.add_parser("ui", help="Launch interactive Geolocation Workbench")
    p_ui.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    p_ui.add_argument("--no-browser", action="store_true", help="Do not launch browser")

    p_serve = sub.add_parser("serve", help="Alias for ui")
    p_serve.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    p_serve.add_argument("--no-browser", action="store_true", help="Do not launch browser")

    args = ap.parse_args(argv)

    if args.cmd in ("ui", "serve"):
        from .ui_server import run_server
        run_server(port=args.port, open_browser=not getattr(args, "no_browser", False))
        return 0

    if args.cmd == "demo":
        return _demo(args)
    # analyze path requires flights data etc. -- delegated to library use
    print("analyze requires a flight-data bundle; see demo for the "
          "end-to-end workflow")
    return 2


def _demo(args):
    from .flights import synth_flight
    from .geo import destination
    from .synth import render_recording
    from . import report

    rng = np.random.default_rng(args.seed)
    truth = (37.4, -122.3, 0.0)
    day0 = 100_000.0
    region = args.region_deg
    bounds = Bounds(
        truth[1] - region, truth[0] - region * 0.8,
        truth[1] + region, truth[0] + region * 0.8,
    )
    t_window = (day0 + 900.0, day0 + 1300.0)

    # two clean flyovers (the 920 s style pair) + distractors
    def fstart(bearing, dist):
        return destination(truth[0], truth[1], bearing, dist)

    la1, lo1 = fstart(270.0, 40_000.0)
    fl1 = synth_flight("SKY01", "C172", la1, lo1, 1500.0, 90.0, 100.0,
                       day0 + 600.0, day0 + 1600.0, dt=4.0)
    # second flyover: pass over truth ~920 s after the first (the source
    # demonstration's offset), i.e. CPA arrival ~ day0+1025
    la2, lo2 = fstart(315.0, 60_000.0)
    fl2 = synth_flight("SKY02", "SR22", la2, lo2, 2200.0, 135.0, 110.0,
                       day0 + 480.0, day0 + 1500.0, dt=4.0)
    flights = [fl1, fl2]
    for i in range(8):
        b = rng.uniform(0, 360)
        d = rng.uniform(15, 70) * 1000
        la, lo = destination(truth[0], truth[1], b, d)
        flights.append(
            synth_flight(
                f"DIS{i:02d}", str(rng.choice(["C208", "PA34", "DA42"])),
                la, lo, rng.uniform(800, 2800), rng.uniform(0, 360),
                rng.uniform(70, 150), day0 + rng.uniform(-500, 1200),
                day0 + rng.uniform(1600, 2400), dt=4.0,
                turn_rate=rng.uniform(-0.01, 0.01),
            )
        )
    atmos = Atmosphere(wind_e_ms=3.0, wind_n_ms=-2.0)
    x, truth_events = render_recording(
        flights, t_window[0], t_window[1], truth[0], truth[1], truth[2],
        atmos=atmos, fs=8000,
    )
    print(f"rendered {len(x)/8000:.0f}s of audio at truth "
          f"{truth[0]:.4f},{truth[1]:.4f}")
    events, cands, decision = run_analysis(
        x, 8000, flights, bounds, t_window, atmos
    )
    print(f"decision: {decision.status}")
    if decision.top:
        print(f"top: {decision.top.lat:.4f},{decision.top.lon:.4f} "
              f"score={decision.top.score:.2f}")
    report.write_report(
        args.out, events, cands, decision, truth, x, 8000,
    )
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
