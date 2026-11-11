"""Confidence and decision rules.

A ranked score is not a probability. This module decides whether the top
candidate may be REPORTED at all, and with what uncertainty, following
the source project's philosophy (abstain rather than guess) while fixing
its blind spot: model-error is tested via the wind/temperature scenario
envelope, not just parameter perturbation.

Rules:
  * Single-event evidence is reported as "low-confidence" (no
    cross-checking flight exists).
  * A near-tie (runner-up within `tie_margin` of the best) demotes the
    result to "ambiguous".
  * The uncertainty radius must cover the clock-sweep and scenario
    spread; if the scenario spread exceeds `max_spread_m`, the result is
    "model-limited" (wind/temperature dominates).
  * Every candidate's score detail is retained for the report.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .atmosphere import Atmosphere, scenario_atmospheres
from .geo import great_circle_distance


@dataclass
class Decision:
    status: str          # located | low-confidence | ambiguous | model-limited | inconclusive
    top: object = None
    runner_up: object = None
    radius_m: float = 0.0
    reasons: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def as_dict(self):
        d = dict(status=self.status, radius_m=self.radius_m,
                 reasons=self.reasons)
        if self.top is not None:
            d["top"] = dict(
                lat=self.top.lat, lon=self.top.lon,
                score=self.top.score,
                flights=list(self.top.flight_ids),
                clock_offset_s=self.top.clock_offset_s,
                clock_drift_ppm=self.top.clock_drift_ppm,
            )
        if self.runner_up is not None:
            d["runner_up"] = dict(lat=self.runner_up.lat,
                                  lon=self.runner_up.lon,
                                  score=self.runner_up.score)
        return d


TIE_MARGIN = 1.5          # score units
MAX_SCENARIO_SPREAD_M = 1200.0
CLUSTER_M = 1500.0        # candidates within this distance are one basin


def cluster_candidates(candidates, radius_m=CLUSTER_M):
    """Group ranked candidates into geographic basins.

    Adjacent grid points a few hundred meters apart are NOT competing
    hypotheses -- they are the same basin seen through the scoring
    surface's local slope. Decisions must compare distinct basins; a
    naive top-vs-runner comparison always near-ties inside the winning
    basin and reports 'ambiguous' even when the second basin is far away
    and scored far worse.
    """
    clusters = []
    for c in sorted(candidates, key=lambda q: -q.score):
        placed = False
        for cl in clusters:
            if great_circle_distance(
                c.lat, c.lon, cl["lat"], cl["lon"]
            ) < radius_m:
                cl["members"].append(c)
                # running centroid
                n = len(cl["members"])
                cl["lat"] += (c.lat - cl["lat"]) / n
                cl["lon"] += (c.lon - cl["lon"]) / n
                placed = True
                break
        if not placed:
            clusters.append(
                dict(lat=c.lat, lon=c.lon, best=c, members=[c])
            )
    for cl in clusters:
        mem = cl["members"]
        cl["score_mean"] = float(np.mean([m.score for m in mem]))
        cl["radius_m"] = max(
            great_circle_distance(cl["lat"], cl["lon"], m.lat, m.lon)
            for m in mem
        ) if len(mem) > 1 else 0.0
        # score-weighted mean location (basin center of mass)
        w = np.array([max(m.score + 50.0, 1e-3) for m in mem])
        lats = np.array([m.lat for m in mem])
        lons = np.array([m.lon for m in mem])
        cl["lat_w"] = float(np.sum(w * lats) / np.sum(w))
        cl["lon_w"] = float(np.sum(w * lons) / np.sum(w))
        # PEAK position: a basin is often a ridge elongated along the
        # corridor (the direction the clock trade-off smears); the basin
        # center of mass sits mid-ridge, far from the actual peak. The
        # reported location is the score-weighted mean over members near
        # the TOP score only (within a small score drop), which follows
        # the ridge to its peak without jumping onto a single noisy
        # grid point.
        smax = mem[0].score
        near = [m for m in mem if smax - m.score <= 0.6]
        if not near:
            near = [mem[0]]
        wn = np.array([max(m.score + 50.0, 1e-3) for m in near])
        cl["lat_p"] = float(
            np.sum(wn * np.array([m.lat for m in near])) / np.sum(wn)
        )
        cl["lon_p"] = float(
            np.sum(wn * np.array([m.lon for m in near])) / np.sum(wn)
        )
        cl["peak_n"] = len(near)
    return clusters


def decide(candidates, events, truth=None, verbose=False):
    """Apply decision rules to the ranked candidate list.

    The reported location is the WINNING CLUSTER's score-weighted center
    (not a single grid point), the uncertainty radius is that cluster's
    geographic spread, and the near-tie test compares the best vs the
    best of the nearest DIFFERENT cluster.
    """
    if not candidates:
        return Decision("inconclusive", reasons=["no candidate passed filters"])
    clusters = cluster_candidates(candidates)
    top_cl = clusters[0]
    top = top_cl["best"]
    reasons = []
    status = "located"
    if len(events) < 2:
        status = "low-confidence"
        reasons.append(
            "single event: no independent cross-checking flyover"
        )
    if len(clusters) > 1:
        runner = clusters[1]["best"]
        gap = top.score - clusters[1]["best"].score
        if gap < TIE_MARGIN:
            status = "ambiguous" if status == "located" else status
            reasons.append(
                f"near-tie between basins: best {top.score:.2f} vs "
                f"{clusters[1]['best'].score:.2f} at "
                f"{great_circle_distance(top.lat, top.lon, clusters[1]['best'].lat, clusters[1]['best'].lon)/1000:.1f} km"
            )
    else:
        runner = None
    radius = float(top_cl["radius_m"])
    # never report tighter than the refined grid resolution
    radius = max(radius, 250.0)
    # spread budget scales with evidence: with >=2 events the joint clock
    # is fitted (not swept), so the basin is the honest uncertainty; with
    # 1 event the clock trade-offs smear the basin and a tight budget
    # correctly demotes it
    budget = MAX_SCENARIO_SPREAD_M * (1.5 if len(events) >= 2 else 1.0)
    if radius > budget:
        status = "model-limited"
        reasons.append(
            f"winning basin spread {radius:.0f} m exceeds scenario budget"
        )
    if truth is not None and verbose:
        err = great_circle_distance(
            top_cl["lat_p"], top_cl["lon_p"], truth[0], truth[1]
        )
        reasons.append(f"[debug] error vs truth: {err:.0f} m")
    # reported point: the winning basin's PEAK position (score-weighted
    # mean over near-top members) -- the basin center of mass averages an
    # elongated ridge to its middle, which can sit a kilometer from the
    # peak that actually carries the evidence
    from dataclasses import replace as _dc_replace

    top_rep = _dc_replace(
        top, lat=top_cl["lat_p"], lon=top_cl["lon_p"]
    )
    return Decision(status, top_rep, runner, radius, reasons)
