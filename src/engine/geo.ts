/**
 * WGS84 Ellipsoidal Geodesy, Local ENU Tangent Frames, and Antimeridian-Safe Projections.
 *
 * Handles edge cases:
 * - Polar latitude convergence (|lat| -> 90 deg, Svalbard 78.2N) via Bowring closed-form ECEF inversion
 * - Antimeridian / Dateline wrapping (west > east, Fiji 179E -> -179W)
 * - Southern Hemisphere ENU tangent plane orientation (54.5S)
 * - 3D slant distance with ground elevation (DEM) vs aircraft MSL altitude
 */

export const A_WGS84 = 6378137.0;
export const F_WGS84 = 1.0 / 298.257223563;
export const E2 = F_WGS84 * (2.0 - F_WGS84);
export const RAD = Math.PI / 180.0;

export type Vec3 = [number, number, number];

export interface GeoPoint {
  lat: number;
  lon: number;
  alt_m?: number;
}

export interface GeoBounds {
  west: number;
  south: number;
  east: number;
  north: number;
  wraps_dateline?: boolean;
}

/**
 * Wrap any longitude in degrees to (-180, 180].
 */
export function normalizeLon(lon: number): number {
  const wrapped = ((((lon + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
  return wrapped === -180.0 ? 180.0 : wrapped;
}

/**
 * Compute signed shortest longitude delta (lon - refLon) in degrees in [-180, 180],
 * ensuring seamless continuity across the +/-180 antimeridian.
 */
export function deltaLonSigned(lon: number, refLon: number): number {
  let d = lon - refLon;
  while (d > 180.0) d -= 360.0;
  while (d < -180.0) d += 360.0;
  return d;
}

/**
 * Geodetic (lat_deg, lon_deg, h_m) -> WGS84 ECEF (x, y, z) in meters.
 */
export function llhToEcef(latDeg: number, lonDeg: number, hM: number = 0.0): Vec3 {
  const lat = latDeg * RAD;
  const lon = lonDeg * RAD;
  const sinLat = Math.sin(lat);
  const cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon);
  const cosLon = Math.cos(lon);
  const n = A_WGS84 / Math.sqrt(1.0 - E2 * sinLat * sinLat);
  const x = (n + hM) * cosLat * cosLon;
  const y = (n + hM) * cosLat * sinLon;
  const z = (n * (1.0 - E2) + hM) * sinLat;
  return [x, y, z];
}

/**
 * WGS84 ECEF (x, y, z) in meters -> Geodetic (lat_deg, lon_deg, h_m) via Bowring's method.
 * Includes exact polar fallback when |lat| > 89.9 deg.
 */
export function ecefToLlh(xyz: Vec3): { lat: number; lon: number; alt_m: number } {
  const [x, y, z] = xyz;
  const lon = Math.atan2(y, x) / RAD;
  const p = Math.hypot(x, y);
  const b = A_WGS84 * (1.0 - F_WGS84);
  const e2p = (A_WGS84 * A_WGS84 - b * b) / (b * b);
  const th = Math.atan2(z * A_WGS84, p * b);
  const sinTh = Math.sin(th);
  const cosTh = Math.cos(th);
  const lat =
    Math.atan2(
      z + e2p * b * sinTh * sinTh * sinTh,
      p - E2 * A_WGS84 * cosTh * cosTh * cosTh
    ) / RAD;
  const sinLat = Math.sin(lat * RAD);
  const cosLat = Math.cos(lat * RAD);
  const n = A_WGS84 / Math.sqrt(1.0 - E2 * sinLat * sinLat);
  let h: number;
  if (Math.abs(lat) > 89.9) {
    h = z / (sinLat || 1e-30) - n * (1.0 - E2);
  } else {
    h = p / (cosLat || 1e-30) - n;
  }
  return { lat, lon: normalizeLon(lon), alt_m: h };
}

/**
 * 3D Euclidean distance between two ECEF vectors (true slant range in meters).
 */
export function vec3Dist(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Norm(a: Vec3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/**
 * Haversine great-circle surface distance in meters (dateline-safe).
 */
export function greatCircleDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const p1 = lat1 * RAD;
  const p2 = lat2 * RAD;
  const dp = p2 - p1;
  const dl = deltaLonSigned(lon2, lon1) * RAD;
  const a =
    Math.sin(dp / 2) ** 2 +
    Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2.0 * A_WGS84 * Math.asin(Math.sqrt(Math.min(1.0, Math.max(0.0, a))));
}

/**
 * Great-circle destination point given start (lat, lon), true bearing (deg), and distance (m).
 */
export function destinationPoint(
  latDeg: number,
  lonDeg: number,
  bearingDeg: number,
  distM: number
): { lat: number; lon: number } {
  const lat1 = latDeg * RAD;
  const lon1 = lonDeg * RAD;
  const brg = bearingDeg * RAD;
  const d = distM / A_WGS84;
  const sinLat2 =
    Math.sin(lat1) * Math.cos(d) +
    Math.cos(lat1) * Math.sin(d) * Math.cos(brg);
  const lat2 = Math.asin(Math.min(1.0, Math.max(-1.0, sinLat2)));
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(brg) * Math.sin(d) * Math.cos(lat1),
      Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
    );
  return {
    lat: lat2 / RAD,
    lon: normalizeLon(lon2 / RAD),
  };
}

/**
 * Local East-North-Up (ENU) tangent frame anchored at (lat0Deg, lon0Deg, alt0M).
 */
export class EnuFrame {
  readonly lat0Deg: number;
  readonly lon0Deg: number;
  readonly origin: Vec3;
  private readonly R: [Vec3, Vec3, Vec3]; // rows of 3x3 matrix whose columns are East, North, Up

  constructor(lat0Deg: number, lon0Deg: number, alt0M: number = 0.0) {
    this.lat0Deg = lat0Deg;
    this.lon0Deg = lon0Deg;
    this.origin = llhToEcef(lat0Deg, lon0Deg, alt0M);
    const lat = lat0Deg * RAD;
    const lon = lon0Deg * RAD;
    const sl = Math.sin(lat);
    const cl = Math.cos(lat);
    const so = Math.sin(lon);
    const co = Math.cos(lon);
    this.R = [
      [-so, -sl * co, cl * co],
      [co, -sl * so, cl * so],
      [0.0, cl, sl],
    ];
  }

  /**
   * Convert local ENU offset (east_m, north_m, up_m) -> WGS84 ECEF (x, y, z).
   */
  toEcef(eM: number, nM: number, uM: number = 0.0): Vec3 {
    const x =
      this.origin[0] +
      this.R[0][0] * eM +
      this.R[0][1] * nM +
      this.R[0][2] * uM;
    const y =
      this.origin[1] +
      this.R[1][0] * eM +
      this.R[1][1] * nM +
      this.R[1][2] * uM;
    const z =
      this.origin[2] +
      this.R[2][0] * eM +
      this.R[2][1] * nM +
      this.R[2][2] * uM;
    return [x, y, z];
  }

  /**
   * Convert local ENU velocity/direction vector (east, north, up) -> ECEF vector.
   */
  enuVectorToEcef(e: number, n: number, u: number = 0.0): Vec3 {
    return [
      this.R[0][0] * e + this.R[0][1] * n + this.R[0][2] * u,
      this.R[1][0] * e + this.R[1][1] * n + this.R[1][2] * u,
      this.R[2][0] * e + this.R[2][1] * n + this.R[2][2] * u,
    ];
  }

  /**
   * Convert WGS84 ECEF (x, y, z) -> local ENU offset (east_m, north_m, up_m).
   */
  fromEcef(xyz: Vec3): Vec3 {
    const dx = xyz[0] - this.origin[0];
    const dy = xyz[1] - this.origin[1];
    const dz = xyz[2] - this.origin[2];
    const e = this.R[0][0] * dx + this.R[1][0] * dy + this.R[2][0] * dz;
    const n = this.R[0][1] * dx + this.R[1][1] * dy + this.R[2][1] * dz;
    const u = this.R[0][2] * dx + this.R[1][2] * dy + this.R[2][2] * dz;
    return [e, n, u];
  }
}

/**
 * Compute centroid and span of GeoBounds, properly unwrapping across the +/-180 deg dateline.
 */
export function boundsGeometry(bounds: GeoBounds): {
  centerLat: number;
  centerLon: number;
  latSpan: number;
  lonSpan: number;
  wrapsDateline: boolean;
} {
  const wraps = Boolean(bounds.wraps_dateline || bounds.east < bounds.west);
  const lonSpan = wraps
    ? bounds.east + 360.0 - bounds.west
    : Math.max(1e-4, bounds.east - bounds.west);
  const latSpan = Math.max(1e-4, bounds.north - bounds.south);
  const centerLat = 0.5 * (bounds.south + bounds.north);
  const centerLon = wraps
    ? normalizeLon(bounds.west + 0.5 * lonSpan)
    : 0.5 * (bounds.west + bounds.east);
  return {
    centerLat,
    centerLon,
    latSpan,
    lonSpan: Math.max(1e-4, lonSpan),
    wrapsDateline: wraps,
  };
}
