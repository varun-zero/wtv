"""Flight trajectories: cubic-Hermite splines in ECEF with 30-s gap
segmentation, plus the synthetic world generator used by the test suite
and demo.

A Track is callable: track(t) -> (position_ecef (3,), velocity_ecef (3,)).
Positions come from ADS-B (lat/lon/alt + ground speed/track) or from the
synthetic generator; velocities are taken from the reported values when
present (nicer than numeric differentiation) and refined by the Hermite
spline's derivative otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .aircraft import AircraftClass, classify
from .atmosphere import Atmosphere
from .geo import EnuFrame, destination, ecef_to_llh, llh_to_ecef

GAP_S = 30.0


@dataclass
class Flight:
    callsign: str
    type_designator: str
    aircraft: AircraftClass
    segments: list  # list of Track (ordered in time)

    @property
    def t0(self):
        return min(s.t[0] for s in self.segments)

    @property
    def t1(self):
        return max(s.t[-1] for s in self.segments)

    def sample(self, t):
        """Position/velocity at time t (nan-filled outside support)."""
        out_p = np.full(3, np.nan)
        out_v = np.full(3, np.nan)
        for s in self.segments:
            if s.t[0] <= t <= s.t[-1]:
                return s(t)
        return out_p, out_v

    def active(self, t):
        return any(s.t[0] <= t <= s.t[-1] for s in self.segments)


class Track:
    """One gap-free trajectory piece; cubic Hermite in ECEF.

    Hermite interpolation with ADS-B-reported velocities preserves turns
    (banked arcs) far better than linear interpolation; the spline is
    also C1, so retarded-time Newton iteration is well-posed.
    """

    __slots__ = ("t", "p", "v", "_dt", "_hp", "_hv")

    def __init__(self, t, p, v):
        self.t = np.asarray(t, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.v = np.asarray(v, dtype=float)
        if self.p.ndim != 2 or self.p.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")
        if self.v.shape != self.p.shape:
            raise ValueError("velocities must be (N, 3)")

    def __call__(self, t):
        t = float(t)
        i = np.searchsorted(self.t, t) - 1
        i = int(np.clip(i, 0, len(self.t) - 2))
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        if not (0.0 <= s <= 1.0):  # outside support
            if s < 0 and i == 0:
                return self.p[0].copy(), self.v[0].copy()
            if s > 1 and i == len(self.t) - 2:
                return self.p[-1].copy(), self.v[-1].copy()
        u = 2 * s ** 3 - 3 * s ** 2 + 1
        a = s ** 3 - 2 * s ** 2 + s
        b = -2 * s ** 3 + 3 * s ** 2
        c = s ** 3 - s ** 2
        pos = u * self.p[i] + a * h * self.v[i] + b * self.p[i + 1] + c * h * self.v[i + 1]
        du = (6 * s ** 2 - 6 * s) / h
        da = (3 * s ** 2 - 4 * s + 1)
        db = (-6 * s ** 2 + 6 * s) / h
        dc = (3 * s ** 2 - 2 * s)
        vel = du * self.p[i] + da * self.v[i] + db * self.p[i + 1] + dc * self.v[i + 1]
        return pos, vel

    def eval_many(self, t):
        """Vectorized Hermite evaluation for an array of times.

        Positions/velocities outside the support are clamped to the
        endpoints. Returns (pos (N,3), vel (N,3)). Used by the vectorized
        forward model in acoustics.forward_curve; calling __call__ per
        sample in Python is orders of magnitude slower.
        """
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2)
        h = self.t[i + 1] - self.t[i]
        s = np.clip((t - self.t[i]) / h, 0.0, 1.0)
        s2 = s * s
        s3 = s2 * s
        u = 2 * s3 - 3 * s2 + 1
        a = s3 - 2 * s2 + s
        b = -2 * s3 + 3 * s2
        c = s3 - s2
        pos = (
            u[:, None] * self.p[i]
            + (a * h)[:, None] * self.v[i]
            + b[:, None] * self.p[i + 1]
            + (c * h)[:, None] * self.v[i + 1]
        )
        du = (6 * s2 - 6 * s) / h
        da = 3 * s2 - 4 * s + 1
        db = (-6 * s2 + 6 * s) / h
        dc = 3 * s2 - 2 * s
        vel = (
            du[:, None] * self.p[i]
            + da[:, None] * self.v[i]
            + db[:, None] * self.p[i + 1]
            + dc[:, None] * self.v[i + 1]
        )
        return pos, vel


def build_flight(
    callsign: str,
    type_designator: str,
    times,
    lat,
    lon,
    alt_m,
    ground_speed_ms=None,
    track_deg=None,
    vertical_rate_ms=None,
):
    """Assemble a Flight from ADS-B-style records, splitting at gaps."""
    t = np.asarray(times, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt_m, dtype=float)
    order = np.argsort(t)
    t, lat, lon, alt = t[order], lat[order], lon[order], alt[order]
    xyz = llh_to_ecef(lat, lon, alt)
    # velocities from ground speed + track heading (true), in ENU -> ECEF
    if ground_speed_ms is None:
        ground_speed_ms = np.full_like(t, np.nan)
    if track_deg is None:
        track_deg = np.full_like(t, np.nan)
    gs = np.asarray(ground_speed_ms, dtype=float)
    trk = np.asarray(track_deg, dtype=float)
    lat_r, lon_r = lat * np.pi / 180.0, lon * np.pi / 180.0
    # ENU velocity from heading
    ve = gs * np.sin(np.radians(np.nan_to_num(trk, nan=0.0)))
    vn = gs * np.cos(np.radians(np.nan_to_num(trk, nan=0.0)))
    if vertical_rate_ms is None:
        vu = np.gradient(alt, t)
    else:
        vu = np.asarray(vertical_rate_ms, dtype=float)
    # rotate ENU -> ECEF per sample
    vxyz = np.empty_like(xyz)
    for i in range(len(t)):
        f = EnuFrame(float(lat[i]), float(lon[i]))
        vxyz[i] = f.enu_direction_to_ecef(ve[i], vn[i], float(vu[i]))
    # fix unreported velocities by finite differences (robust)
    if np.isnan(gs).any() or np.isnan(trk).any():
        have = ~(np.isnan(gs) | np.isnan(trk))
        if have.sum() >= 2:
            scale = np.nanmean(np.linalg.norm(vxyz[have], axis=1))
        else:
            scale = 0.0
        for i in np.where(~have)[0]:
            j = i + 1 if i + 1 < len(t) else i - 1
            d = xyz[j] - xyz[i]
            dtv = abs(t[j] - t[i]) or 1.0
            vxyz[i] = d / dtv * (scale / max(np.linalg.norm(d) / dtv, 1e-6) if scale else 1.0)
    # gap split
    gaps = np.where(np.diff(t) > GAP_S)[0]
    cuts = [0, *(gaps + 1).tolist(), len(t)]
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            segments.append(Track(t[a:b], xyz[a:b], vxyz[a:b]))
    return Flight(
        callsign=callsign,
        type_designator=type_designator,
        aircraft=classify(type_designator),
        segments=segments,
    )


# --------------------------------------------------------------------------
# Synthetic world generator (validation + demo). Every synthetic flight is a
# realistic straight-arc cruise with reported ground speed/heading so the
# ingest path is exercised exactly as with real ADS-B data.
# --------------------------------------------------------------------------

def _synth_segment(
    lat0, lon0, alt0, bearing_deg, speed_ms, t_start, t_end, turn_rate=0.0,
    climb_ms=0.0, dt=4.0,
):
    """Generate ADS-B-style records for one flight piece (great-circle)."""
    t = np.arange(t_start, t_end, dt)
    lat, lon = float(lat0), float(lon0)
    lats, lons, trks = [], [], []
    brg = float(bearing_deg)
    step = speed_ms * dt
    for _ in t:
        lats.append(lat)
        lons.append(lon)
        trks.append(brg)
        lat, lon = destination(lat, lon, brg, step)
        if turn_rate:
            brg = (brg + turn_rate * dt) % 360.0
    lats, lons, trks = np.array(lats), np.array(lons), np.array(trks)
    alts = alt0 + climb_ms * (t - t[0])
    gs = np.full_like(t, speed_ms)
    vr = np.full_like(t, climb_ms)
    return t, lats, lons, alts, gs, trks, vr


def synth_flight(
    callsign, type_designator, lat0, lon0, alt0, bearing_deg, speed_ms,
    t_start, t_end, turn_rate=0.0, climb_ms=0.0, dt=4.0,
):
    t, lat, lon, alt, gs, trk, vr = _synth_segment(
        lat0, lon0, alt0, bearing_deg, speed_ms, t_start, t_end, turn_rate, climb_ms, dt
    )
    return build_flight(
        callsign, type_designator, t, lat, lon, alt, gs, trk, vr
    )


def synth_world(
    truth_lat, truth_lon, truth_alt,
    day_start_s: float,
    flyovers: list,          # dicts: callsign,type,lat0,lon0,alt,bearing,speed,turn,climb, t_off
    distract_flights: int = 40,
    seed: int = 7,
    region_deg: float = 1.4,
):
    """Create flights around a truth location for validation.

    flyovers: list of dicts with keys t_off (offset from day_start_s when the
    pass is heard), plus synth_flight args. distract_flights random flights
    provide competing tonal candidates and silence-filter pressure.
    """
    rng = np.random.default_rng(seed)
    flights = []
    for spec in flyovers:
        f = synth_flight(
            spec["callsign"], spec["type"], spec["lat0"], spec["lon0"],
            spec["alt"], spec["bearing"], spec["speed"], spec["t_start"],
            spec["t_end"], spec.get("turn", 0.0), spec.get("climb", 0.0),
        )
        flights.append(f)
    for i in range(distractor_count := distract_flights):
        typ = rng.choice(["C172", "SR22", "PA34", "C208", "R44", "DA42", "TBM9"])
        b = rng.uniform(0, 360)
        d = rng.uniform(8, 60) * 1000
        lat0, lon0 = destination(truth_lat, truth_lon, b, d)
        t_start = day_start_s + rng.uniform(-1800, 7200)
        dur = rng.uniform(600, 2400)
        flights.append(
            synth_flight(
                f"DIS{i:02d}", str(typ), lat0, lon0,
                rng.uniform(600, 3000), rng.uniform(0, 360),
                rng.uniform(60, 160), t_start, t_start + dur,
                turn_rate=rng.uniform(-0.02, 0.02),
                climb_ms=rng.uniform(-2, 2),
            )
        )
    return flights
