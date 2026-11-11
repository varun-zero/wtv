"""Audio analysis: STFT, median whitening, physics-bounded ridge tracing,
harmonic grouping, and Doppler event parameter fitting.

Fixes vs the source methodology (each tied to a difficulty in the demo):

* Inaudible events (the demo's first section): per-frequency median
  subtraction plus per-column MAD normalization suppresses stationary
  tones and flattens the noise floor so <0 dB-SNR ridges remain visible.
* Slope clipping (the demo truncates near-CPA where the sweep is steepest):
  the tracer's slope bounds are derived from physics -- descent up to
  f*v_max^2/(c*r_min) with margin, ascent allowed (post-CPA recovery and
  turning flight), rather than fixed 12 Hz/s down / 1 Hz/s up.
* Fundamental ambiguity (the demo slides f0 by hand): harmonic families
  are detected by integer-ratio grouping, and f0 is then constrained by
  the aircraft-class BPF prior at search time.
* Codec loss (MP3/phone): the detector follows harmonics (k*f0), so losing
  the fundamental or low harmonics to a high-pass or codec boundary is
  survivable -- any surviving harmonic pins the family.

The event fit uses the emission-frame Doppler model for a straight
constant-speed pass with CPA slant distance r:

    f(tau) = f0 * c / (c + v^2 * tau / sqrt(r^2 + v^2 tau^2))

with tau measured from closest approach. The exact receive-time curve is
asymmetric (sound emitted later travels less far); that asymmetry is fully
modeled by the forward scorer (acoustics.py), so the fit here is only used
to derive *bounds* for the search, not final answers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SPEED_OF_SOUND = 343.0


def spectrogram(x, fs, n_fft=2048, hop=None):
    """Magnitude spectrogram (linear), shape (n_freq, n_time)."""
    if hop is None:
        hop = n_fft // 4
    win = np.hanning(n_fft)
    n_t = 1 + (len(x) - n_fft) // hop
    if n_t < 1:
        raise ValueError("recording shorter than one window")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_t]
    spec = np.abs(np.fft.rfft(frames * win, axis=1)).T  # (freq, time)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    times = (np.arange(n_t) * hop + n_fft / 2) / fs
    return spec, freqs, times


def whiten(spec):
    """Median/MAD whitening; suppresses stationary tones and evens the floor.

    The per-frequency median is over the whole column axis; an event
    occupying a minority of the window is preserved (self-suppression only
    bites when one frequency is lit most of the time). No per-column
    rescaling: dividing loud columns by their own median would compress
    genuine flyover peaks (z ~ 200) down to noise level while leaving weak
    noise columns near z ~ 1, inverting the detector's signal-to-noise
    ordering. Output is z-scored contrast per (freq, time) cell.
    """
    med = np.median(spec, axis=1, keepdims=True)
    mad = np.median(np.abs(spec - med), axis=1, keepdims=True)
    out = (spec - med) / np.maximum(mad, 1e-12)
    return out


def slope_bounds(f_ref_hz, v_max_ms=260.0, r_min_m=300.0, c=SPEED_OF_SOUND):
    """Physics-derived ridge slope bounds (Hz/s).

    Peak Doppler sweep rate near CPA is f0 * v^2 / (c * r). The tracer is
    allowed descent beyond the worst case (margin 1.35x) and modest ascent
    (post-CPA recovery and turn-induced rises).
    """
    desc = f_ref_hz * v_max_ms ** 2 / (c * r_min_m)
    return 1.35 * desc, 0.25 * desc + 2.0  # (max descent, max ascent)


def trace_ridge(whitened, freqs, times, f_lo_hz, f_hi_hz,
                max_down_hz_s=None, max_up_hz_s=None):
    """Viterbi ridge tracer over a frequency band, weighted by contrast.

    Returns arrays (t_idx, f_idx, weight) with weak samples dropped.
    """
    band = (freqs >= f_lo_hz) & (freqs <= f_hi_hz)
    if not band.any():
        return np.array([], int), np.array([], int), np.array([], float)
    f_idx = np.where(band)[0]  # global row indices of the band
    # Rewards-only DP with RENT: every column costs `rent` z-units, so a
    # path is only sustained where measured contrast exceeds the noise
    # floor. Without rent, a mildly-positive floor (z~2 over 400 s)
    # out-accumulates a short intense S-curve (z~200 over 40 s); without
    # free termination, the path is forced through the silent tail and
    # the same floor-riding degeneracy returns.
    rent = 2.0
    sub = np.clip(whitened[f_idx, :], 0.0, None) - rent
    n_f, n_t = sub.shape
    if n_t < 3:
        return np.array([], int), np.array([], int), np.array([], float)
    if max_down_hz_s is None or max_up_hz_s is None:
        d, u = slope_bounds(float(freqs[f_idx[-1]]))
        max_down_hz_s = d if max_down_hz_s is None else max_down_hz_s
        max_up_hz_s = u if max_up_hz_s is None else max_up_hz_s
    df = float(freqs[1] - freqs[0])
    dt = float(times[1] - times[0])
    # Physics slope bounds, converted to cells WITHOUT lossy rounding: the
    # allowance is evaluated over a window of w>1 columns, i.e. the path may
    # descend max_down_hz_s * (w*dt) Hz within w columns. Single-column
    # rounding (int(round(slope*dt/df))) quantizes a 97 Hz/s allowance to
    # 2 cells = 15 Hz/s effective at this resolution and makes near-CPA
    # sweep rates (up to ~35 Hz/s for harmonics of a 100 m/s prop pass)
    # untraceable.
    down_hz_s = float(max_down_hz_s)
    up_hz_s = float(max_up_hz_s)
    # per-column cap and multi-column window: over ceil(w) columns the
    # total descent may reach down_hz_s * w * dt
    w_cols = 8  # window for fractional accumulation
    down1 = max(1, int(np.floor(down_hz_s * dt / df)))
    up1 = max(1, int(np.floor(up_hz_s * dt / df)))
    # total budget over the window:
    down_w = max(down1, int(np.ceil(down_hz_s * (w_cols * dt) / df)))
    # DP state: (row, descent-used-in-window) is complex; instead use the
    # simpler exact approach: transition cost is quadratic in Hz moved
    # beyond the *average* legal per-column rate, so slow legal descent is
    # free, burst descent pays only for the excess, and noise teleports
    # pay heavily.
    avg_down = down_hz_s * dt  # Hz per column (fractional!)
    avg_up = up_hz_s * dt
    # transition cost function:
    #   cost(hz_moved) = k_over * max(0, |d| - avg_down)  if descending
    #                  + k_over * max(0, |d| - avg_up)    if ascending
    # capped so teleports still pay ~linearly
    k_over = 0.05
    # practical per-column move cap (hard ceiling for array sizing)
    max_move_cells = max(down_w, up1 + down_w)
    # full DP matrix (needed for free termination: `score` as a running
    # vector would make argmax pick a ROW, and divmod would misread it)
    S = np.empty((n_f, n_t), dtype=float)
    S[:, 0] = sub[:, 0]
    # re-run with matrix storage (same transitions)
    back = np.zeros((n_f, n_t), dtype=np.int32)
    for tc in range(1, n_t):
        for fc in range(n_f):
            lo = max(0, fc - max_move_cells)
            hi = min(n_f, fc + max_move_cells + 1)
            if hi <= lo:
                S[fc, tc] = -np.inf
                continue
            deltas = (np.arange(lo, hi) - fc) * df
            desc_excess = np.clip(-deltas - avg_down, 0, None)
            asc_excess = np.clip(deltas - avg_up, 0, None)
            cost = k_over * (desc_excess + asc_excess)
            cand = S[lo:hi, tc - 1] - cost
            j = int(np.argmax(cand))
            S[fc, tc] = cand[j] + sub[fc, tc]
            back[fc, tc] = j
    # free termination: best (row, col) over the whole DP matrix
    flat = int(np.argmax(S))
    f_end, t_end = divmod(flat, n_t)
    path_t = []
    path_f = []
    fc, tc = f_end, t_end
    while tc >= 0:
        path_t.append(tc)
        path_f.append(fc)
        if tc == 0:
            break
        fc = int(back[fc, tc])
        tc -= 1
    path_f = path_f[::-1]
    path_t = path_t[::-1]
    t_idx = np.asarray(path_t, int)
    # re-score the chosen path on the UNCLIPPED values so the gate reflects
    # measured contrast (the DP may pass through negative-z noise columns)
    w_raw = whitened[f_idx, :][np.asarray(path_f), t_idx]
    keep = w_raw > 1.5  # z-score gate: noise floors sit near |z|~1
    # map local band indices back to global frequency-bin indices
    return (
        t_idx[keep],
        f_idx[np.asarray(path_f)[keep]],
        np.clip(w_raw[keep], 0.0, None),
    )


def trace_band_set(whitened, freqs, times, bands):
    """Trace ridges in each (f_lo, f_hi) band; returns ridge records."""
    out = []
    for (f_lo, f_hi) in bands:
        ti, fi, w = trace_ridge(whitened, freqs, times, f_lo, f_hi)
        if len(ti) < 8:
            continue
        out.append(dict(t_s=times[ti], f_hz=freqs[fi], weight=w))
    return out


def _segment_ridge(t, f, w, jump_hz=None, df=None):
    """Split a traced ridge at frequency discontinuities.

    A band's trace concatenates segments of DIFFERENT physical harmonics
    (the tracer follows k=1's approach top, then jumps to k=2's recede when
    k=1 leaves the band). Harmonic jumps are near-instant frequency
    teleportations; a physical sweep's per-step change is bounded by the
    tracer's own slope allowance. The cut threshold is therefore set from
    the fastest legal sweep step (df/dt max ~ 100 Hz/s x dt), NOT a small
    multiple of the bin width: a fixed 3-bin threshold fires mid-sweep at
    CPA (where df/dt peaks) and amputates the curve exactly where it is
    most informative.
    Returns a list of (t, f, w) segments.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    w = np.asarray(w, float)
    if len(t) < 2:
        return []
    order = np.argsort(t)
    t, f, w = t[order], f[order], w[order]
    if len(t) < 2:
        return []
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.128
    if jump_hz is None:
        # harmonic jumps teleport across >= ~1/3 octave; legal sweeps move
        # < ~120 Hz/s * dt between samples
        jump_hz = max(20.0, 120.0 * dt_med)
    jumps = np.where(np.abs(np.diff(f)) > jump_hz)[0]
    cuts = [0, *(jumps + 1).tolist(), len(t)]
    return [
        (t[a:b], f[a:b], w[a:b])
        for a, b in zip(cuts[:-1], cuts[1:])
        if b - a >= 12
    ]


def _concat_seg(A, B):
    """Concatenate two segments (t, f, w, idx), dedupe near-duplicate
    times keeping the higher-weight sample."""
    t = np.concatenate([A[0], B[0]])
    f = np.concatenate([A[1], B[1]])
    w = np.concatenate([A[2], B[2]])
    key = np.round(t, 1)
    order = np.lexsort((-w, key))
    t, f, w, key = t[order], f[order], w[order], key[order]
    first = np.concatenate([[True], np.diff(key) != 0])
    t, f, w = t[first], f[first], w[first]
    o = np.argsort(t)
    return (t[o], f[o], w[o], A[3])


def _curves_match(A, B, f_tol_hz, gap_s):
    """Do two segments belong to the SAME physical f(t) curve?

    Overlapping spans: interpolated tracks must agree within tolerance.
    Adjacent spans (gap <= gap_s): endpoint frequencies must continue.
    Harmonic relatives (2x) and other flights' curves fail both tests.
    """
    ta, fa = A[0], A[1]
    tb, fb = B[0], B[1]
    lo = max(ta[0], tb[0])
    hi = min(ta[-1], tb[-1])
    if hi - lo >= 2.0:
        ts = np.linspace(lo, hi, min(60, max(5, int(hi - lo) * 2)))
        da = np.abs(np.interp(ts, ta, fa) - np.interp(ts, tb, fb))
        return float(np.median(da)) < f_tol_hz
    if ta[-1] <= tb[0] and tb[0] - ta[-1] <= gap_s:
        return abs(float(fa[-1]) - float(fb[0])) < 1.5 * f_tol_hz
    if tb[-1] <= ta[0] and ta[0] - tb[-1] <= gap_s:
        return abs(float(fb[-1]) - float(fa[0])) < 1.5 * f_tol_hz
    return False


def _merge_curve_segments(segs, f_tol_hz=9.0, gap_s=3.0):
    """Merge segment pieces of one physical curve (fixed-point loop).

    A single flyover's fundamental is typically split across band traces
    (the tracer switches harmonics at band jumps; noise blips cut mid-
    sweep). Fragments produce biased event fits -- recede-only fragments
    fit v low, approach-only fragments fit f0 low -- which then breaks the
    search join. Merging on overlap-agreement / endpoint-continuity
    reconstructs the full S-curve; harmonic relatives and other flights'
    curves are rejected by the frequency-agreement test.
    """
    segs = [tuple(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if _curves_match(segs[i], segs[j], f_tol_hz, gap_s):
                    segs[i] = _concat_seg(segs[i], segs[j])
                    del segs[j]
                    changed = True
                    break
            if changed:
                break
    return segs


def harmonic_families(ridges, tol_ratio=0.08, min_len=40, min_f0_hz=55.0,
                      overlap_frac=0.3):
    """Group merged curve segments into families with a shared fundamental.

    Pipeline: (1) segment each traced ridge at harmonic jumps; (2) drop
    short/low segments; (3) MERGE pieces of the same physical curve
    (across bands and interior cuts); (4) cluster the resulting curves by
    temporal overlap and instantaneous integer ratio; (5) reconstruct the
    fundamental curve f0(t) = f_seg(t)/k from all members.

    Returns families: dict(segments=[(t, f, w, k)...], orders=[...],
    implied_f0=median of fundamental curve).
    """
    segs = []  # (t, f, w, source_index) -- time-sorted
    for idx, r in enumerate(ridges):
        for (t, f, w) in _segment_ridge(r["t_s"], r["f_hz"], r["weight"]):
            o = np.argsort(t)
            t, f, w = t[o], f[o], w[o]
            if len(t) >= min_len and float(np.median(f)) >= min_f0_hz:
                segs.append((t, f, w, idx))
    segs = _merge_curve_segments(segs)

    def span(s):
        return float(s[0][-1] - s[0][0])

    def overlap(a, b):
        lo = max(a[0][0], b[0][0])
        hi = min(a[0][-1], b[0][-1])
        return max(0.0, hi - lo)

    def inst_ratio(a, b):
        ts = np.clip(b[0], a[0][0], a[0][-1])
        fa = np.interp(ts, a[0], a[1])
        fc = np.interp(ts, b[0], b[1])
        return float(np.median(fc / np.maximum(fa, 1e-6)))

    fams = []
    used = [False] * len(segs)
    for i, s in enumerate(segs):
        if used[i]:
            continue
        members = [(s, 1)]
        used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            sj = segs[j]
            if overlap(s, sj) < overlap_frac * min(span(s), span(sj)):
                continue
            ratio = inst_ratio(s, sj)
            k = int(round(ratio))
            if k >= 2 and abs(ratio - k) <= tol_ratio * k:
                members.append((sj, k))
                used[j] = True
        # fundamental-implied curve from all members: seg = (t, f, w, idx)
        f0_all = np.concatenate([m[0][1] / m[1] for m in members])
        fams.append(
            dict(
                segments=[(m[0][0], m[0][1], m[0][2], m[1]) for m in members],
                orders=[m[1] for m in members],
                implied_f0=float(np.median(f0_all)),
                source_ridges=[ridges[m[0][3]] for m in members],
            )
        )
    return fams


@dataclass
class EventFit:
    """Measured event: harmonic family + approximate Doppler fit + bounds."""

    ridge_count: int
    f0_hz: float
    orders: list
    t_start_s: float
    t_end_s: float
    t_cpa_s: float
    speed_ms: float
    speed_sigma: float
    slant_m: float
    slant_sigma: float
    snr: float
    freq_samples: np.ndarray
    freq_values: np.ndarray
    freq_weights: np.ndarray


def emission_doppler(tau, f0, v_ms, r_m, c=SPEED_OF_SOUND):
    """Emission-frame Doppler model for a straight constant-speed pass."""
    d = np.sqrt(r_m ** 2 + (v_ms * np.asarray(tau, float)) ** 2)
    radial = v_ms ** 2 * np.asarray(tau, float) / d  # receding positive
    return f0 * c / (c + radial)


def receive_doppler(t_recv, f0, v_ms, r_m, c=SPEED_OF_SOUND):
    """Exact receive-time Doppler curve for a straight constant-speed pass
    at CPA slant distance r.

    Dimensionless solve: with u = v*t/r (receive time from CPA arrival)
    and a = v/c, the emission time in u-space solves
        u = u_e + a * sqrt(1 + u_e^2)
    (the second term is the travel delay scaled by v/r). The receive-time
    curve is NOT the emission curve shifted: its asymmetry is exactly the
    effect that made the source demo's measured curves fail an ideal
    Doppler fit, and ignoring it trades r against v in least-squares.
    Validated against bisection to <0.05 Hz.
    """
    v = float(v_ms)
    r = float(r_m)
    a = v / c
    t = np.asarray(t_recv, float)
    u = v * t / r
    # Seed below the receive time; emission times can be NEGATIVE (the
    # sound arriving at t=CPA was emitted before closest approach, which
    # is why the received curve at CPA arrival sits BELOW f0's plateau
    # mid-point -- clipping the seed to 0 truncates exactly that asymmetry)
    ue = u - 0.5
    for _ in range(40):
        g = ue + a * np.sqrt(1.0 + ue ** 2) - u
        gp = 1.0 + a * ue / np.sqrt(1.0 + ue ** 2)
        step = g / np.maximum(gp, 1e-9)
        # damped Newton: g(u_e) is monotone increasing with a pole-free
        # derivative; a full step can overshoot for large |u_e|
        ue = ue - np.clip(step, -8.0, 8.0)
    # radial speed at emission: v * tau / distance (receding positive)
    vr = v * ue / np.sqrt(1.0 + ue ** 2)
    return f0 / (1.0 + vr / c)


def _anchor_ridge(family):
    """The k=1 member's FULL ridge record (unsegmented source ridge).

    Segmentation exists for grouping (disambiguating which trace pieces
    belong to which harmonic); but the source ridge itself is one
    continuous Viterbi trace, and its interior harmonic jumps are exactly
    where the tracer switched between k and k+1 *of the same physical
    tone*. Cutting there would amputate the recede phase of the S-curve,
    which carries the post-CPA plateau needed to pin the speed.
    """
    ridges = family.get("source_ridges")
    if ridges:
        return ridges[0]
    return None


def fit_event(family, source_ridges=None):
    """Fit (f0, v, r, tc) to a family; robust loss + plateau-seeded grid +
    Nelder-Mead + coordinate descent. Used only to bound the search; the
    exact forward model scores candidates later.

    Data = the fundamental-implied composite curve: every member segment
    contributes f_seg/k samples. A source band trace can jump between
    harmonics mid-trace (the tracer legitimately switches k -> k+1 when a
    harmonic leaves the band), so any single source ridge is contaminated
    by foreign harmonics; the composite f/k curve is the physically
    consistent series.
    """
    if source_ridges is not None:
        family = dict(family, source_ridges=source_ridges)
    segs = family["segments"]
    if not segs:
        return None
    ts, fs, ws = [], [], []
    for (t, f, w, k) in segs:
        ts.append(np.asarray(t, float))
        fs.append(np.asarray(f, float) / k)
        ws.append(np.maximum(np.asarray(w, float), 0.1) / np.sqrt(k))
    t = np.concatenate(ts)
    f = np.concatenate(fs)
    w = np.concatenate(ws)
    if len(t) < 12:
        return None
    order = np.argsort(t)
    t, f, w = t[order], f[order], w[order]

    def model(p, tt):
        f0, v, r, _tc = p
        return receive_doppler(tt, f0, v, r)

    def loss(p):
        f0, v, r, tc = p
        if f0 <= 5.0 or v <= 10.0 or r <= 100.0:
            return 1e9
        d = np.abs(f - model(p, t - tc)) / 2.0
        return float(np.sum(w * np.where(d <= 1.0, d ** 2, 2.0 * d - 1.0)))

    f0_c = float(np.median(f))
    best = (1e18, None)
    for tc_g in np.linspace(t.min(), t.max(), 9):
        for v_g in (60.0, 90.0, 120.0, 160.0):
            for r_g in (1e3, 3e3, 6e3, 12e3, 25e3):
                p = (f0_c, v_g, r_g, tc_g)
                l = loss(p)
                if l < best[0]:
                    best = (l, p)
    if best[1] is None:
        return None
    p = np.array(best[1], dtype=float)
    # Plateau-seeded grid: the pre/post-CPA plateaus (f_pre, f_post) encode
    # the speed and fundamental directly:
    #   f_pre/f_post = (c+v)/(c-v)  ->  v = c(r-1)/(r+1)
    #   f_pre*f_post = f0^2 c^2/(c^2-v^2)  ->  f0 = sqrt(f_pre*f_post)*sqrt(1-(v/c)^2)
    # A median-based f0 seed lands in the receding-plateau local minimum.
    f_pre = float(np.percentile(f, 92))
    f_post = float(np.percentile(f, 8))
    ratio = f_pre / max(f_post, 1e-6)
    if ratio > 1.05:
        v_hat = SPEED_OF_SOUND * (ratio - 1.0) / (ratio + 1.0)
        f0_hat = np.sqrt(f_pre * f_post) * np.sqrt(
            max(1.0 - (v_hat / SPEED_OF_SOUND) ** 2, 1e-6)
        )
    else:
        v_hat, f0_hat = 50.0, float(np.median(f))
    # steepest-descent time as the CPA seed
    if len(f) >= 5:
        df = np.gradient(f, t)
        tc_hat = float(t[int(np.argmin(df))])
    else:
        tc_hat = float(np.median(t))
    best = (loss(tuple(p)), p)
    for v_g in (v_hat, v_hat * 1.25, v_hat * 0.8, 100.0, 60.0, 160.0):
        if v_g <= 10.0:
            continue
        for f0_g in (f0_hat, f0_hat * 1.1, f0_hat * 0.9):
            for tc_g in (tc_hat, tc_hat - 10.0, tc_hat + 10.0,
                         float(np.median(t))):
                for r_g in (1.5e3, 3e3, 6e3, 12e3, 25e3):
                    p = (f0_g, v_g, r_g, tc_g)
                    l = loss(p)
                    if l < best[0]:
                        best = (l, p)
    p = np.array(best[1], dtype=float)
    cur = loss(tuple(p))
    # Nelder-Mead polish on the robust loss (the plateau/CPA structure is
    # non-convex enough that coordinate descent stalls on bin quantization)
    try:
        from scipy.optimize import minimize

        res = minimize(
            lambda q: loss(tuple(q)),
            p,
            method="Nelder-Mead",
            options=dict(xatol=1e-3, fatol=1e-4, maxiter=4000),
        )
        if res.fun < cur:
            p = np.asarray(res.x, dtype=float)
            cur = float(res.fun)
    except Exception:
        pass
    # per-parameter scales: f0 relative, v absolute, r RELATIVE (an absolute
    # 0.35 m r-step can never converge from a 1 km grid node to 1.5 km truth)
    scales = np.array([f0_c * 0.05, 10.0, best[1][2] * 0.15, 5.0])
    cur = loss(tuple(p))
    for _ in range(20):
        improved = False
        for i in range(4):
            for sgn in (+1, -1):
                p2 = p.copy()
                p2[i] += sgn * scales[i]
                l2 = loss(tuple(p2))
                if l2 < cur:
                    p, cur = p2, l2
                    improved = True
        scales *= 0.62
        if not improved:
            break
    f0, v, r, tc = p
    # Outlier prune + refit: curve merging can glue in a few foreign-flight
    # samples (and noise blips survive the gate); dropping large residuals
    # and re-polishing keeps the stored event series physically clean --
    # the search's shape scoring assumes every stored sample belongs to
    # this one flyover.
    resid = f - model(p, t - tc)
    rms = float(np.sqrt(np.average(resid ** 2, weights=w)))
    keep = np.abs(resid) < max(6.0, 3.0 * rms)
    if keep.sum() >= 12 and keep.sum() < len(f):
        t, f, w = t[keep], f[keep], w[keep]
        try:
            from scipy.optimize import minimize

            res = minimize(
                lambda q: loss(tuple(q)), p, method="Nelder-Mead",
                options=dict(xatol=1e-3, fatol=1e-4, maxiter=4000),
            )
            if res.fun < loss(tuple(p)):
                p = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        f0, v, r, tc = p
    resid = f - model(p, t - tc)
    rms = float(np.sqrt(np.average(resid ** 2, weights=w)))
    snr_all = float(np.mean(np.concatenate([s[2] for s in segs])))
    return EventFit(
        ridge_count=len(segs),
        f0_hz=float(f0),
        orders=[int(k) for k in family["orders"]],
        t_start_s=float(t.min()),
        t_end_s=float(t.max()),
        t_cpa_s=float(tc),
        speed_ms=float(v),
        speed_sigma=8.0 if rms < 1.5 else 18.0,
        slant_m=float(r),
        slant_sigma=0.18 * float(r) if rms < 1.5 else 0.35 * float(r),
        snr=snr_all,
        freq_samples=t,
        freq_values=f,
        freq_weights=w,
    )


def band_plan(f_min_hz=55.0, f_max_hz=1500.0):
    """Overlapping tracer bands.

    Each band spans 1.5 octaves (x3) with 1-octave (x2) stride so adjacent
    bands overlap by half an octave. Narrower spans clip physical curves
    at the band edges: a 100 m/s prop pass with f0=85 Hz has its approach
    plateau at f0/(1-v/c) = 120 Hz, which falls OUTSIDE a 55-112.75 band;
    the tracer then pins the plateau to the last in-band bin (109 Hz), the
    fitter reads a false f0 ~ 77, and every derived bound is biased.
    """
    bands = []
    f = f_min_hz
    while f < f_max_hz:
        bands.append((f, min(f * 3.0, f_max_hz)))
        f *= 2.0
    return bands
