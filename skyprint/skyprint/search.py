"""Geolocation search.

Methodology (following the source demonstration, with the improvements
motivated by its difficulties):

1. CORRIDOR PRE-FILTER: for each flight, only candidate receiver points
   within the CPA-distance window of the measured events can hear the
   flyover at the right level; each flight's audibility corridor is a
   swept annulus along its spline. This is flight-centric: every flight
   is processed exactly once (no per-tile re-querying).
2. TIME-OFFSET JOIN: for a receiver to explain events (i, j), its delayed
   CPA times for those flights must match the measured inter-event offset
   (with tolerance). The video's 920 s join -- generalized to any number
   of events.
3. SILENCE FILTER: between the matched events, no OTHER flight may have
   produced a clearly-audible flyover at the candidate point (the video's
   "what you don't hear is a signal" -- implemented with an audibility
   threshold from the propagation model, not a hearing heuristic).
4. BPF PRIOR: the measured fundamental must lie inside the flight's
   aircraft-class blade-pass range (collapses the f0 degeneracy).
5. FORWARD-MODEL SCORING: the exact retarded-time Doppler curve for each
   (candidate, flight) pair is synthesized from the flight's own spline
   and scored against the measured f(t) samples under a joint clock
   (offset + ppm drift) shared by all events, refined per-candidate.
6. REFINEMENT: the top candidates are re-scored on a finer grid.
7. DECISION: rank by joint score; report uncertainty from the clock/lambda
   sweeps and scenario envelope (wind, temperature); abstain when the
   runner-up is within the near-tie threshold or evidence is single-event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .acoustics import forward_curve, travel_time_quadratic
from .aircraft import AircraftClass
from .atmosphere import Atmosphere, scenario_atmospheres
from .audio import EventFit, SPEED_OF_SOUND
from .geo import (
    Bounds, EnuFrame, ecef_to_llh, great_circle_distance, llh_to_ecef,
    grid_in_bounds,
)


@dataclass
class Candidate:
    lat: float
    lon: float
    flight_ids: tuple          # per event
    score: float = 0.0
    clock_offset_s: float = 0.0
    clock_drift_ppm: float = 0.0
    detail: dict = field(default_factory=dict)


def _flight_cpa_sweep(seg, rx_list, t_window):
    """Per-flight CPA sweep: emission-CPA time (tau*) + geometric slant
    for each receiver, on a 2 s lattice once per flight."""
    ts = np.arange(seg.t[0], min(seg.t[-1], t_window[1]) + 60.0, 2.0)
    if len(ts) < 2:
        return None
    pos = np.empty((len(ts), 3))
    for i, t in enumerate(ts):
        pos[i] = seg(float(t))[0]
    out_t = np.empty(len(rx_list))
    out_d = np.empty(len(rx_list))
    for j, rx in enumerate(rx_list):
        d2 = ((pos - rx) ** 2).sum(axis=1)
        k = int(np.argmin(d2))
        lo = max(k - 1, 0)
        hi = min(k + 1, len(ts) - 1)
        a, b = ts[lo], ts[hi]

        def dist(t):
            p, _ = seg(float(t))
            return float(np.linalg.norm(rx - p))

        for _ in range(30):
            m1 = a + 0.382 * (b - a)
            m2 = b - 0.382 * (b - a)
            if dist(m1) < dist(m2):
                b = m2
            else:
                a = m1
        tau_star = 0.5 * (a + b)
        p, _ = seg(float(tau_star))
        out_t[j] = tau_star
        out_d[j] = float(np.linalg.norm(rx - p))
    return out_t, out_d


def _audible_slant(fit: EventFit):
    """Audibility annulus (lo, hi) from the fitted CPA slant + uncertainty."""
    lo = max(200.0, fit.slant_m - 3.0 * fit.slant_sigma)
    hi = fit.slant_m + 3.0 * fit.slant_sigma
    return lo, hi


def search(
    flights,
    events: list,
    bounds: Bounds,
    atmos: Atmosphere,
    t_window: tuple,
    rx_alt_m: float = 0.0,
    grid_m: float = 2000.0,
    refine_grid_m: float = 250.0,
    silence_margin_db: float = 8.0,
    time_tol_s: float = 8.0,
    max_candidates: int = 300,
    verbose: bool = False,
    _grid_override=None,
):
    """Full geolocation search. Returns a ranked list of Candidates."""
    if not events:
        return []
    lat_c, lon_c = bounds.centroid()
    frame = EnuFrame(lat_c, lon_c)
    if _grid_override is not None:
        e_grid, n_grid = _grid_override
    else:
        e_grid, n_grid = grid_in_bounds(bounds, grid_m, frame)
    EE, NN = np.meshgrid(e_grid, n_grid)
    rx_ecef = frame.to_ecef(EE.ravel(), NN.ravel(), np.zeros(EE.size))
    rx_list = [rx_ecef[i] for i in range(rx_ecef.shape[0])]
    n_rx = len(rx_list)
    if verbose:
        print(f"grid: {n_rx} candidates in {bounds.west:.2f},{bounds.south:.2f}"
              f" -> {bounds.east:.2f},{bounds.north:.2f}")

    wind = atmos.wind_ecef(frame)
    tonal = [f for f in flights if f.aircraft.tonal]

    # ---- 1. per-flight corridors (flight-centric, once per flight) ----
    flight_data = {}
    lo_a, hi_a = 1e9, 0.0
    for ev in events:
        l, h = _audible_slant(ev)
        lo_a, hi_a = min(lo_a, l), max(hi_a, h)
    for fl in tonal:
        best = None
        for seg in fl.segments:
            sw = _flight_cpa_sweep(seg, rx_list, t_window)
            if sw is None:
                continue
            tau_stars, slants = sw
            in_band = (slants >= lo_a) & (slants <= hi_a)
            if not in_band.any():
                continue
            if best is None or in_band.sum() > best[2].sum():
                # delayed CPA receive time with wind + path sound speed
                recv = np.empty(n_rx)
                for j, rx in enumerate(rx_list):
                    p, _ = seg(float(tau_stars[j]))
                    _, _, src_alt = ecef_to_llh(p)
                    c_path = atmos.path_sound_speed(rx_alt_m, src_alt)
                    recv[j] = tau_stars[j] + travel_time_quadratic(
                        rx - p, wind, c_path
                    )
                best = (recv, slants, in_band, seg)
        if best is not None:
            flight_data[fl.callsign] = dict(
                flight=fl, seg=best[3], recv=best[0], slants=best[1],
                in_band=best[2],
            )
    if verbose:
        print(f"corridors: {len(flight_data)} flights with in-band receivers")

    # ---- 2. event->flight masks + assignment --------------------------
    ev_assign = []
    for ev in events:
        # The event fit can be badly biased when only part of the S-curve
        # was traced (recede-only families fit v low, k-contaminated
        # families fit v high); the prior window must be wide enough to
        # admit the true flight -- the exact forward-model score does the
        # real discrimination.
        v_lo = max(20.0, ev.speed_ms - max(60.0, 3.5 * ev.speed_sigma))
        v_hi = ev.speed_ms + max(60.0, 3.5 * ev.speed_sigma)
        per_flight = {}
        for cs, fd in flight_data.items():
            cls = fd["flight"].aircraft
            # BPF prior: fundamental (or its harmonic multiple) vs class range
            if not _f0_bpf_compatible(ev.f0_hz, cls):
                continue
            # speed prior: flight's median speed within event bounds
            seg = fd["seg"]
            ts_mid = np.arange(seg.t[0], seg.t[-1], 20.0)
            sp = [float(np.linalg.norm(seg(float(t))[1])) for t in ts_mid]
            sp = float(np.median(sp)) if sp else 0.0
            if not (v_lo <= sp <= v_hi):
                continue
            per_flight[cs] = fd
        ev_assign.append(per_flight)

    ev_masks = []
    for i, ev in enumerate(events):
        mask_any = np.zeros(n_rx, dtype=bool)
        for cs, fd in ev_assign[i].items():
            mask_any |= fd["in_band"]
        ev_masks.append(mask_any)
    possible = np.logical_and.reduce(ev_masks)
    idx_candidates = np.where(possible)[0]
    if verbose:
        print(f"join: {len(idx_candidates)} grid points pass all-event masks")

    # silence thresholds (approximate level model, per flight)
    silence_audible = {}
    for cs, fd in flight_data.items():
        cls = fd["flight"].aircraft
        lvl = cls.source_level_db - 20.0 * np.log10(np.maximum(fd["slants"], 50.0))
        silence_audible[cs] = lvl

    t_first, t_last = events[0].t_cpa_s, events[-1].t_cpa_s
    scored = []
    for ci in idx_candidates[: max_candidates * 20]:
        for combo in _assign_events(events, ev_assign, ci, time_tol_s):
            ok = True
            for cs, fd in flight_data.items():
                if cs in combo:
                    continue
                if t_first - 60.0 < fd["recv"][ci] < t_last + 60.0:
                    if silence_audible[cs][ci] > 55.0 + silence_margin_db:
                        ok = False
                        break
            if not ok:
                continue
            lat, lon, _ = ecef_to_llh(rx_list[ci])
            scored.append(Candidate(float(lat), float(lon), tuple(combo)))
            if len(scored) >= max_candidates * 5:
                break
        if len(scored) >= max_candidates * 5:
            break
    if not scored:
        return []

    # ---- 3. forward-model scoring under a joint clock -----------------
    for cand in scored[: max_candidates]:
        rx = llh_to_ecef(cand.lat, cand.lon, rx_alt_m)
        cand.score, cand.clock_offset_s, cand.clock_drift_ppm, cand.detail = (
            _score_candidate(
                rx, cand.flight_ids, events, flights, atmos, t_window, rx_alt_m
            )
        )
    scored = [c for c in scored if c.score > -1e8]
    scored.sort(key=lambda c: -c.score)

    # ---- 4. refinement -------------------------------------------------
    # Refine around the top candidates of EACH distinct basin (not just the
    # global top-8, which usually all sit in one basin -- refining only
    # those starves rival basins of grid resolution and manufactures
    # near-ties or false margins).
    from .confidence import cluster_candidates

    seeds = []
    for cl in cluster_candidates(scored):
        seeds.append(cl["best"])
        if len(seeds) >= 4:
            break
    if not seeds:
        seeds = scored[: min(4, len(scored))]
    refined = list(scored[: max_candidates])  # keep unrefined pool as floor

    def _refine_pass(seed_list, half_span, step):
        added = []
        for cand in seed_list:
            f = EnuFrame(cand.lat, cand.lon)
            offs = np.arange(-half_span, half_span + step / 2, step)
            for de in offs:
                for dn in offs:
                    rx = f.to_ecef(float(de), float(dn), rx_alt_m)
                    s, off, ppm, det = _score_candidate(
                        rx, cand.flight_ids, events, flights, atmos,
                        t_window, rx_alt_m,
                    )
                    lat, lon, _ = ecef_to_llh(rx)
                    added.append(
                        Candidate(float(lat), float(lon), cand.flight_ids,
                                  s, off, ppm, det)
                    )
        return added

    # coarse pass (250 m rungs), then a fine pass (50 m rungs) around the
    # NEW best of each surviving basin: two levels keep the cost bounded
    # while resolving the peak below the 250 m grid quantization
    refined += _refine_pass(seeds, 2.0 * refine_grid_m, refine_grid_m)
    refined.sort(key=lambda c: -c.score)
    seeds2 = []
    for cl in cluster_candidates([c for c in refined if c.score > -1e8]):
        seeds2.append(cl["best"])
        if len(seeds2) >= 4:
            break
    if seeds2:
        refined += _refine_pass(seeds2, 0.4 * refine_grid_m,
                                0.1 * refine_grid_m)
    refined.sort(key=lambda c: -c.score)
    return refined[:max_candidates]


def _f0_bpf_compatible(f0_hz, cls):
    """BPF prior with harmonic-multiple/submultiple acceptance.

    A family that lost its fundamental to a high-pass or codec boundary is
    reconstructed from the surviving harmonics: their k's renumber from 1,
    so the fitted 'fundamental' is the true f0 divided by an integer (the
    missing lowest order) -- e.g. an SR22 at 115 Hz filtered below 80 Hz
    fits as f0 = 39. Conversely a lone high harmonic with a mis-scaled k
    can OVERestimate f0. Accept the class window against f0 itself
    (relaxed, as before, for measurement scatter) or against integer
    multiples/divisors of f0 (tight window: a derived fundamental must
    land cleanly inside a real class range, not its slack zone).
    """
    lo, hi = cls.bpf_lo_hz, cls.bpf_hi_hz
    if lo * 0.6 <= f0_hz <= hi * 1.4:
        return True
    for m in (2, 3, 4, 5):
        if lo <= m * f0_hz <= hi or lo <= f0_hz / m <= hi:
            return True
    return False


def _assign_events(events, ev_assign, ci, time_tol_s):
    """Enumerate (flight-id tuple) assignments whose delayed-CPA offsets
    match the measured inter-event offsets within tolerance."""
    opts = []
    for i, ev in enumerate(events):
        lst = [(cs, float(fd["recv"][ci])) for cs, fd in ev_assign[i].items()]
        opts.append(lst)
    if any(len(o) == 0 for o in opts):
        return []
    combos = []

    def dfs(i, cur):
        if len(combos) > 200:
            return
        if i == len(opts):
            combos.append(tuple(cs for cs, _ in cur))
            return
        for cs, recv in opts[i]:
            if cur:
                m_off = events[i].t_cpa_s - events[0].t_cpa_s
                d = (recv - cur[0][1]) - m_off
                if abs(d) > time_tol_s * (i + 1):
                    continue
            dfs(i + 1, cur + [(cs, recv)])

    dfs(0, [])
    return combos


def _score_candidate(rx, flight_ids, events, flights, atmos, t_window,
                     rx_alt_m, clock_ppm_bounds=25.0):
    """Joint forward-model score for one receiver + event->flight map.

    The exact forward curve (arrival time -> 1/dtdtau) is built ONCE per
    (event, flight) pair via acoustics.forward_curve; every clock
    hypothesis (offset + ppm drift, shared across events) is then just an
    interpolation of measured times onto that curve. This is what makes
    the 45-point clock grid + refinement affordable per candidate.
    """
    by_id = {f.callsign: f for f in flights}
    wind = _wind_for(atmos, rx)
    curves = []
    for ev, cs in zip(events, flight_ids):
        fl = by_id.get(cs)
        seg = _segment_for(fl, ev, t_window) if fl is not None else None
        if seg is None:
            curves.append(None)
            continue
        t_lo = float(np.min(ev.freq_samples)) - 90.0
        t_hi = float(np.max(ev.freq_samples)) + 30.0
        taus = np.arange(max(seg.t[0], t_lo - 150.0),
                         min(seg.t[-1], t_hi + 10.0), 0.5)
        if len(taus) < 4:
            curves.append(None)
            continue
        curves.append(forward_curve(rx, seg, taus, atmos, wind, rx_alt_m))
    if all(c is None for c in curves):
        return -1e9, 0.0, 0.0, {}

    def total(off, ppm):
        s = 0.0
        det = {}
        for ev, curve in zip(events, curves):
            if curve is None:
                continue
            t_arr, shape, slant_model = curve
            if len(t_arr) < 4:
                s -= 1e6
                continue
            t_true = (np.asarray(ev.freq_samples, float) - off) / (
                1.0 + ppm * 1e-6
            )
            fm = np.interp(t_true, t_arr, shape, left=np.nan, right=np.nan)
            okm = np.isfinite(fm)
            if okm.sum() < 8:
                s -= 1e6
                continue
            fme = np.asarray(ev.freq_values, float)[okm]
            ww = np.asarray(ev.freq_weights, float)[okm]
            fmm = fm[okm]
            # weighted linear fit fme ~= alpha * fmm + beta (alpha absorbs
            # the flight's unknown f0; beta absorbs slow RPM wander), then
            # score the RMS residual in Hz. A correlation score saturates
            # near 1.0 and cannot separate candidates a few hundred meters
            # apart; an Hz residual grows linearly with misfit.
            sw = float(np.sum(ww))
            swx = float(np.sum(ww * fmm))
            swxx = float(np.sum(ww * fmm * fmm))
            swy = float(np.sum(ww * fme))
            swxy = float(np.sum(ww * fmm * fme))
            det0 = sw * swxx - swx * swx
            if abs(det0) < 1e-12:
                s -= 1e6
                continue
            beta = (swy * swxx - swx * swxy) / det0
            alpha = (sw * swxy - swx * swy) / det0
            if alpha <= 0.0:
                s -= 1e6
                continue
            resid = fme - (alpha * fmm + beta)
            rms = float(np.sqrt(np.average(resid * resid, weights=ww)))
            s += 30.0 * float(np.exp(-rms / 2.0))
            # NOTE: a sweep-width (std-ratio) term was evaluated here and
            # removed: when the traced window spans plateau-to-plateau the
            # measured std saturates and carries no position information;
            # only the Hz residual discriminates at the sub-km scale.
            # slant-consistency, DOWN-WEIGHTED: the event fit's slant
            # carries a systematic bias (bin quantization + partial-curve
            # fits measured +100-200 m on synthetic truth), so this term
            # must steer, not decide -- full weight made biased-fit rivals
            # outscore the true point whose RMS was 2x better.
            d_slant = abs(slant_model - float(ev.slant_m))
            s += 2.0 * float(
                np.exp(-d_slant / (2.0 * float(ev.slant_sigma)))
            )
            det[len(det)] = int(okm.sum())
        # mild prior on clock drift: consumer oscillators are usually
        # within a few ppm; large |ppm| needs correspondingly better
        # residuals to win
        s -= 0.05 * abs(ppm)
        return s, det

    best = (-np.inf, 0.0, 0.0, {})
    for off_g in np.linspace(-8.0, 8.0, 9):
        for ppm_g in (-20.0, -10.0, 0.0, 10.0, 20.0):
            s, det = total(off_g, ppm_g)
            if s > best[0]:
                best = (s, float(off_g), float(ppm_g), det)
    off0, ppm0 = best[1], best[2]
    # shrinking-step clock refinement down to 0.05 s / 0.1 ppm: the offset
    # resolution maps directly to along-corridor position error (0.05 s of
    # sound travel ~ 17 m), so the last two levels are what tighten the
    # final fix below the coarse 0.25 s rung
    for d_off, d_ppm in ((2.0, 5.0), (0.5, 1.0), (0.25, 0.5),
                         (0.1, 0.2), (0.05, 0.1)):
        improved = True
        while improved:
            improved = False
            for sgn in (+1, -1):
                off2 = off0 + sgn * d_off
                s, det = total(off2, ppm0)
                if s > best[0]:
                    best = (s, off2, ppm0, det)
                    off0 = off2
                    improved = True
            for sgn in (+1, -1):
                ppm2 = ppm0 + sgn * d_ppm
                if abs(ppm2) > clock_ppm_bounds:
                    continue
                s, det = total(off0, ppm2)
                if s > best[0]:
                    best = (s, off0, ppm2, det)
                    ppm0 = ppm2
                    improved = True
    return best


def _segment_for(fl, ev, t_window):
    """Pick the flight segment covering the event's receive window."""
    best = None
    for seg in fl.segments:
        if seg.t[-1] < t_window[0] - 180 or seg.t[0] > t_window[1] + 180:
            continue
        if best is None or (seg.t[-1] - seg.t[0]) > (best.t[-1] - best.t[0]):
            best = seg
    return best


def _wind_for(atmos, rx_ecef):
    lat, lon, _ = ecef_to_llh(rx_ecef)
    return atmos.wind_ecef(EnuFrame(float(lat), float(lon)))
