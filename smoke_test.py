"""Smoke test: physics + audio on a small synthetic world.

Verifies: ECEF/ENU round trips, travel-time quadratic, retarded-time
inversion vs brute force, wind identities, absorption anchors, renderer
produces a detectable Doppler event that the tracer + harmonic grouper +
event fitter recover with correct parameters.
"""
import sys

sys.path.insert(0, ".")
import numpy as np

from skyprint.atmosphere import (
    Atmosphere, absorption_coefficient_db_per_m, isa_temperature,
    isa_pressure, sound_speed,
)
from skyprint.geo import (
    EnuFrame, Bounds, llh_to_ecef, ecef_to_llh, destination,
    great_circle_distance,
)
from skyprint.acoustics import (
    travel_time_quadratic, solve_emission, emission_state_at,
    brute_force_frequency,
)
from skyprint.flights import synth_flight
from skyprint.synth import render_recording
from skyprint.audio import (
    spectrogram, whiten, trace_band_set, harmonic_families, fit_event,
    band_plan, receive_doppler,
)

ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


print("== geodesy ==")
lat, lon, h = 37.5, -122.3, 12.0
xyz = llh_to_ecef(lat, lon, h)
lat2, lon2, h2 = ecef_to_llh(xyz)
check("llh->ecef->llh", abs(lat2 - lat) < 1e-9 and abs(lon2 - lon) < 1e-9 and abs(h2 - h) < 1e-4,
      f"dlat={abs(lat2-lat):.2e}")
f = EnuFrame(37.5, -122.3)
ecef2 = f.to_ecef(1000.0, 2000.0, 50.0)
e, n, u = f.from_ecef(ecef2)
check("enu round trip", abs(e - 1000) < 1e-6 and abs(n - 2000) < 1e-6 and abs(u - 50) < 1e-6)
la2, lo2 = destination(37.5, -122.3, 90.0, 111_000.0)
d = great_circle_distance(37.5, -122.3, la2, lo2)
check("destination ~111km/deg lon", abs(d - 111_000) < 600, f"{d:.0f} m")
b = Bounds(-179.5, 35.0, 179.0, 40.0)  # wraps dateline
tiles = b.tiles(2.0)
check("dateline tiles exist", len(tiles) > 0, f"{len(tiles)} tiles")
b2 = Bounds(10.0, 45.0, 12.0, 47.0)
t2 = b2.tiles(1.0)
check("normal tiles 4", len(t2) == 4, f"{len(t2)}")

print("== atmosphere ==")
a = absorption_coefficient_db_per_m(np.array([1000.0]), 293.15, 0.5, 101.325)[0]
check("absorption ~4.7 dB/km @1kHz", abs(a * 1000 - 4.7) < 0.8, f"{a*1000:.2f}")
a4 = absorption_coefficient_db_per_m(np.array([4000.0]), 293.15, 0.5, 101.325)[0]
check("absorption ~30 dB/km @4kHz", abs(a4 * 1000 - 30) < 6, f"{a4*1000:.1f}")
at = Atmosphere()
check("ISA T @0m = 288.15", abs(float(isa_temperature(0)) - 288.15) < 1e-6)
check("ISA T @11km = 216.65", abs(float(isa_temperature(11_000)) - 216.65) < 1e-6)
check("ISA p @0m ~101.325", abs(float(isa_pressure(0)) - 101.325) < 0.01)
check("ISA p @11km ~22.6", abs(float(isa_pressure(11_000)) - 22.632) < 0.05)
c0 = float(sound_speed(288.15))
check("c(15C) ~340.3", abs(c0 - 340.3) < 0.2, f"{c0:.1f}")
c3k = float(at.sound_speed_at(3000))
check("c at 3km ~328", abs(c3k - 328.4) < 1.0, f"{c3k:.1f}")

print("== travel-time quadratic ==")
d_vec = np.array([3000.0, 0.0, 0.0])
tt = travel_time_quadratic(d_vec, np.zeros(3), 343.0)
check("no-wind travel 3000/343", abs(tt - 3000.0 / 343.0) < 1e-9, f"{tt:.6f}")
# wind along the ray (tailwind): sound advected with the medium
w = np.array([10.0, 0.0, 0.0])
tt2 = travel_time_quadratic(d_vec, w, 343.0)
check("tailwind 3000/353", abs(tt2 - 3000.0 / 353.0) < 0.02, f"{tt2:.4f}")
w = np.array([-10.0, 0.0, 0.0])
tt3 = travel_time_quadratic(d_vec, w, 343.0)
check("headwind 3000/333", abs(tt3 - 3000.0 / 333.0) < 0.02, f"{tt3:.4f}")

print("== retarded-time identities ==")
# straight flight passing 4 km abeam, 100 m/s
fl = synth_flight("TST", "C172", 37.4, -122.3, 1500.0, 90.0, 100.0,
                  1000.0, 1400.0, dt=4.0)
seg = fl.segments[0]
rx = llh_to_ecef(37.5, -122.3, 0.0)  # ~11 km abeam? adjust below
# place receiver right under the path: path goes east from (37.4,-122.3)
rx = llh_to_ecef(37.4, -122.3, 0.0)
# no wind
atm = Atmosphere()
wind0 = np.zeros(3)
t_recv = 1200.0
emis = solve_emission(rx, seg, t_recv, atm, wind0, 0.0)
# brute force reference: received frequency for f0
f0 = 100.0
t_grid = np.array([1140.0, 1180.0, 1200.0, 1220.0, 1260.0])
f_bf = brute_force_frequency(rx, seg, t_grid, f0, atm, wind0, 0.0)
f_newton = np.array([f0 / solve_emission(rx, seg, t, atm, wind0, 0.0).dtdtau
                     for t in t_grid])
err = np.max(np.abs(f_bf - f_newton))
check("Newton == brute force freq", err < 0.05, f"max {err:.4f} Hz")
# wind identity: source drifting WITH the medium heard by a ground-fixed
# receiver = the moving-observer formula: dtdtau = c / (c + u.W).
# (A ground-fixed receiver moves at -W through the medium, so it DOES
# hear a shift; no-Doppler requires the receiver to co-move with the
# medium as well.)
atm_w = Atmosphere(wind_e_ms=8.0, wind_n_ms=0.0)
windv = atm_w.wind_ecef(f)
flw = synth_flight("TSTW", "C172", 37.4, -122.3, 1500.0, 90.0, 8.0,
                   1000.0, 1400.0, dt=4.0)
segw = flw.segments[0]
em_w = solve_emission(rx, segw, 1200.0, atm_w, windv, 0.0)
c_eff = em_w.c_path
expected = c_eff / (c_eff + float(np.dot(em_w.unit, windv)))
check("drift-with-medium = moving-observer formula",
      abs(em_w.dtdtau - expected) < 2e-3,
      f"dtdtau={em_w.dtdtau:.5f} expected={expected:.5f}")

print("== render + detect ==")
# geometry: flight starts 40 km WEST of receiver, flying east, so the
# window [900,1300] captures approach -> CPA (arrival ~1004) -> recede
from skyprint.geo import destination as _dest

st_lat, st_lon = _dest(37.4, -122.3, 270.0, 40_000.0)
fl = synth_flight("TST", "C172", st_lat, st_lon, 1500.0, 90.0, 100.0,
                  600.0, 1600.0, dt=4.0)
flights = [fl]
t_lo, t_hi = 900.0, 1100.0  # 200 s window: CPA ~99 s keeps margin both sides
x, truth = render_recording(flights, t_lo, t_hi, 37.4, -122.3, 0.0, fs=8000)
check("render non-silent", np.abs(x).max() > 0.4, f"peak {np.abs(x).max():.2f}")
check("truth event recorded", len(truth) >= 1)
spec, freqs, times = spectrogram(x, 8000, n_fft=4096)
w_spec = whiten(spec)
ridges = trace_band_set(w_spec, freqs, times, band_plan(55, 1500))
check("ridges found", len(ridges) >= 1, f"{len(ridges)} ridges")
fams = harmonic_families(ridges)
check("families found", len(fams) >= 1, f"{len(fams)}")
# The TRUE event: multi-order family whose fit is physically consistent
multi = [f for f in fams if len(set(f["orders"])) >= 2]
check("multi-order family exists", len(multi) >= 1, f"{len(multi)}")
fits = []
for fam in multi:
    fit = fit_event(fam)
    if fit is not None:
        fits.append(fit)
best = None
for fit in fits:
    if best is None or fit.snr > best.snr:
        best = fit
check("event fitted", best is not None)
if best:
    print(f"  fit: f0={best.f0_hz:.1f} Hz v={best.speed_ms:.1f} m/s "
          f"r={best.slant_m:.0f} m tc={best.t_cpa_s:.1f} "
          f"snr={best.snr:.1f} orders={best.orders}")
    check("f0 near class BPF (~85 Hz)", 70 <= best.f0_hz <= 105,
          f"{best.f0_hz:.1f}")
    check("speed near 100", abs(best.speed_ms - 100) < 25, f"{best.speed_ms:.1f}")
    check("slant near 1500", 1000 < best.slant_m < 2600, f"{best.slant_m:.0f}")
    check("harmonics grouped (>=2 orders)", len(best.orders) >= 2,
          f"orders={best.orders}")
    # tc approximates the CPA-emission arrival (~99-110 s window time)
    check("CPA time near swept-curve center",
          85 <= best.t_cpa_s <= 125, f"tc={best.t_cpa_s:.1f}")
    # traced receive curve: f=f0 at the CPA-emission arrival (~104.5 window
    # time); fitter tc convention matches (validated against bisection)
    # traced receive curve: f=f0 at the CPA-emission arrival; the model's
    # t=0 is the CPA-EMISSION instant (tau=0), whose sound arrives at r/c
    f_true = receive_doppler(1500.0 / 343.0, 85.0, 100.0, 1500.0)
    check("receive model: f=f0 at CPA-arrival instant", abs(f_true - 85.0) < 0.5,
          f"{f_true:.4f}")

print(f"\nTOTAL: {ok} pass, {fail} fail")
sys.exit(1 if fail else 0)
