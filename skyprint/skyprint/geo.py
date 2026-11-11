"""Geodesy for global searches: WGS84 ECEF, local ENU frames, dateline-safe
region grids.

Grids use local tangent (ENU) frames per tile, so accuracy is uniform at any
latitude including polar regions; geometry is always evaluated in true ECEF
so there is no projection error in the physics. Longitude wrap (dateline)
is handled by allowing longitudes beyond +/-180 during grid generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2 = F_WGS84 * (2.0 - F_WGS84)
RAD = np.pi / 180.0


def llh_to_ecef(lat_deg, lon_deg, h_m):
    """Geodetic (deg, deg, m) -> ECEF (m). Vectorized."""
    lat = np.asarray(lat_deg, dtype=float) * RAD
    lon = np.asarray(lon_deg, dtype=float) * RAD
    h = np.asarray(h_m, dtype=float)
    s2 = np.sin(lat) ** 2
    n = A_WGS84 / np.sqrt(1.0 - E2 * s2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h) * np.sin(lat)
    if x.ndim == 0:
        return np.array([float(x), float(y), float(z)])
    return np.stack([x, y, z], axis=-1)


def ecef_to_llh(xyz):
    """ECEF (m) -> (lat_deg, lon_deg, h_m) via Bowring's method."""
    v = np.asarray(xyz, dtype=float)
    single = v.ndim == 1
    if single:
        v = v[None, :]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lon = np.arctan2(y, x) / RAD
    p = np.hypot(x, y)
    b = A_WGS84 * (1.0 - F_WGS84)
    e2p = (A_WGS84 ** 2 - b ** 2) / b ** 2
    th = np.arctan2(z * A_WGS84, p * b)
    lat = np.arctan2(z + e2p * b * np.sin(th) ** 3, p - E2 * A_WGS84 * np.cos(th) ** 3) / RAD
    s2 = np.sin(lat * RAD) ** 2
    n = A_WGS84 / np.sqrt(1.0 - E2 * s2)
    h = p / np.cos(lat * RAD + 1e-30) - n
    h = np.where(np.abs(lat) > 89.9, v[..., 2] / np.sin(lat * RAD + 1e-30) - n * (1 - E2), h)
    out = (lat, lon, h)
    if single:
        return float(lat[0]), float(lon[0]), float(h[0])
    return out


def normalize_lon(lon: float) -> float:
    """Wrap to (-180, 180]."""
    return ((lon + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class EnuFrame:
    """Local east-north-up frame anchored at (lat0, lon0)."""

    lat0_deg: float
    lon0_deg: float

    def __post_init__(self):
        object.__setattr__(self, "origin", llh_to_ecef(self.lat0_deg, self.lon0_deg, 0.0))
        lat = self.lat0_deg * RAD
        lon = self.lon0_deg * RAD
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        # columns: east, north, up
        object.__setattr__(self, "_R", np.array(
            [
                [-so, -sl * co, cl * co],
                [co, -sl * so, cl * so],
                [0.0, cl, sl],
            ]
        ))

    def to_ecef(self, e_m, n_m, u_m):
        """Local ENU offsets -> ECEF. Vectorized over (e,n,u) triples."""
        enu = np.stack(
            [np.asarray(e_m, float), np.asarray(n_m, float), np.asarray(u_m, float)], axis=-1
        )
        if enu.ndim == 1:
            return self.origin + self._R @ enu
        return self.origin + enu @ self._R.T

    def from_ecef(self, xyz):
        """ECEF -> local ENU offsets. Vectorized."""
        d = np.asarray(xyz, dtype=float)
        single = d.ndim == 1
        if single:
            d = d[None, :]
        rel = d - self.origin
        enu = rel @ self._R
        if single:
            return enu[0]
        return enu

    def enu_direction_to_ecef(self, e, n, u):
        v = np.array([e, n, u], dtype=float)
        return self._R @ v


@dataclass(frozen=True)
class Bounds:
    """Geographic bounds, west/south/east/north in degrees (lon, lat, lon, lat)."""

    west: float
    south: float
    east: float
    north: float

    def centroid(self):
        # lon centroid with dateline handling
        if self.east < self.west:  # wraps dateline
            span = (self.east + 360.0 - self.west)
            lon = normalize_lon(self.west + span / 2.0)
        else:
            lon = 0.5 * (self.west + self.east)
        return 0.5 * (self.south + self.north), lon

    def tiles(self, tile_deg: float):
        """Split into lat/lon tiles with per-tile ENU frames.

        Dateline-wrapping bounds are handled by allowing lons beyond 180
        in the internal enumeration, then normalizing for display.
        """
        lo, hi = self.west, self.east
        wrap = hi < lo
        if wrap:
            hi += 360.0
        lat_steps = np.arange(self.south, self.north + 1e-9, tile_deg)
        lon_steps = np.arange(lo, hi + 1e-9, tile_deg)
        tiles = []
        for i, la in enumerate(lat_steps[:-1]):
            la2 = min(lat_steps[i + 1], self.north)
            for j, lo2 in enumerate(lon_steps[:-1]):
                lo3 = min(lon_steps[j + 1], hi)
                tiles.append(
                    Bounds(west=lo2, south=la, east=lo3, north=la2)
                )
        return tiles


def destination(lat_deg, lon_deg, bearing_deg, dist_m):
    """Great-circle destination (used by the synthetic world generator)."""
    lat = np.asarray(lat_deg, float) * RAD
    lon = np.asarray(lon_deg, float) * RAD
    brg = np.asarray(bearing_deg, float) * RAD
    d = np.asarray(dist_m, float) / A_WGS84
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(brg))
    lon2 = lon + np.arctan2(
        np.sin(brg) * np.sin(d) * np.cos(lat),
        np.cos(d) - np.sin(lat) * np.sin(lat2),
    )
    return lat2 / RAD, normalize_lon(lon2 / RAD)


def great_circle_distance(lat1, lon1, lat2, lon2):
    """Haversine distance (m)."""
    p1, l1, p2, l2 = (
        np.asarray(lat1, float) * RAD,
        np.asarray(lon1, float) * RAD,
        np.asarray(lat2, float) * RAD,
        np.asarray(lon2, float) * RAD,
    )
    dp = p2 - p1
    dl = l2 - l1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * A_WGS84 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def grid_in_bounds(bounds: Bounds, spacing_m: float, frame: EnuFrame):
    """Regular grid over bounds in a tile's ENU frame.

    Returns (e_m, n_m) 1-D coordinate arrays. Grid lines follow the ENU
    axes of the tile frame; candidate geometry is always converted back to
    ECEF, so any projection non-uniformity only affects sampling density
    (bounded by ~d^2/(8R^2) relative error), never the physics.
    """
    lat_c, lon_c = bounds.centroid()
    # unwrap longitudes so a dateline-wrapping bounds (west > east, e.g.
    # 175 -> -175) sweeps the SHORT way across the line: linspace(175,
    # -175) would enumerate the 350 deg arc through Greenwich and place
    # every grid candidate on the wrong side of the globe
    lo = bounds.west
    hi = bounds.east
    if hi < lo:
        hi += 360.0
    # convert bounds corners to the frame
    lons = np.linspace(lo, hi, 5)
    lats = np.linspace(bounds.south, bounds.north, 5)
    pts = llh_to_ecef(
        np.repeat(lats, 5), np.tile(lons, 5), np.zeros(25)
    )
    enu = frame.from_ecef(pts)
    e_lo, e_hi = enu[:, 0].min(), enu[:, 0].max()
    n_lo, n_hi = enu[:, 1].min(), enu[:, 1].max()
    e = np.arange(e_lo, e_hi + spacing_m, spacing_m)
    n = np.arange(n_lo, n_hi + spacing_m, spacing_m)
    return e, n
