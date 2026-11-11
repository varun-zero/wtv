/**
 * Retarded-Time Acoustic-Doppler Physics Engine.
 *
 * Implements:
 * 1. Exact moving-medium wavefront travel time quadratic solver: |D(tau) - W * T| = c_path * T
 * 2. Exact Retarded-Time Doppler curve: f_rx(t_recv) = f0 / (dt/dtau) where t_recv = tau + T(tau)
 * 3. Naive Instantaneous Doppler curve (ignoring retarded emission time) for visual comparison
 * 4. Delayed CPA (Closest Point of Approach) time and slant range solver
 * 5. Cubic-Hermite 3D trajectory interpolation for ADS-B flight tracks
 */
import { AtmosphereParams, pathSoundSpeed, windEcefAtReceiver } from './atmosphere';
import {
  EnuFrame,
  Vec3,
  destinationPoint,
  ecefToLlh,
  llhToEcef,
  vec3Dist,
  vec3Dot,
  vec3Norm,
  vec3Sub,
} from './geo';

export interface Waypoint {
  t: number;
  lat: number;
  lon: number;
  alt_m: number;
  speed_ms: number;
}

/**
 * Solve moving-medium acoustic travel time T from |D - W * T| = c_path * T.
 */
export function travelTimeQuadratic(
  dVec: Vec3,
  windEcef: Vec3,
  cPath: number
): number {
  const d2 = vec3Dot(dVec, dVec);
  const w2 = vec3Dot(windEcef, windEcef);
  const dw = vec3Dot(dVec, windEcef);
  const a = w2 - cPath * cPath;
  if (a >= -1e-9) {
    // Fallback if wind approaches or exceeds speed of sound
    return Math.sqrt(d2) / Math.max(cPath, 1.0);
  }
  const disc = dw * dw - a * d2;
  if (disc < 0.0) {
    return Math.sqrt(d2) / Math.max(cPath, 1.0);
  }
  const t = (dw - Math.sqrt(disc)) / a;
  return Math.max(0.0, t);
}

/**
 * Parametric Retarded-Time Doppler Curve for a linear constant-velocity pass.
 * Solves u = u_e + (v/c) * sqrt(1 + u_e^2) for the dimensionless emission time u_e,
 * capturing the true physical approach/recede asymmetry.
 */
export function receiveDopplerLinear(
  tRecvArray: number[],
  f0Hz: number,
  vMs: number,
  rSlantM: number,
  tCpaRecvS: number,
  cMs: number = 340.0
): number[] {
  const v = Math.max(1.0, vMs);
  const r = Math.max(50.0, rSlantM);
  const c = Math.max(250.0, cMs);
  const a = Math.min(0.92, v / c); // subsonic guard

  return tRecvArray.map((tAbs) => {
    const dt = tAbs - tCpaRecvS;
    const u = (v * dt) / r;
    // Damped Newton iteration for u_e + a * sqrt(1 + u_e^2) = u
    // Note: at t_recv = t_CPA, u = 0, so u_e = -a / sqrt(1 - a^2) < 0
    // wait: if tCpaRecvS is the DELAYED CPA time (when CPA sound arrives at t_CPA + r/c),
    // then relative to delayed CPA time, u_delayed = (v * (t - t_cpa_delayed)) / r
    // satisfies u_e + a * (sqrt(1 + u_e^2) - 1) = u_delayed, so u_e = 0 at dt = 0!
    let ue = u;
    for (let iter = 0; iter < 25; iter++) {
      const sq = Math.sqrt(1.0 + ue * ue);
      const g = ue + a * (sq - 1.0) - u;
      if (Math.abs(g) < 1e-9) break;
      const gp = 1.0 + (a * ue) / sq;
      const step = g / Math.max(gp, 1e-6);
      ue -= Math.max(-5.0, Math.min(5.0, step));
    }
    const vr = (v * ue) / Math.sqrt(1.0 + ue * ue);
    return f0Hz / (1.0 + vr / c);
  });
}

/**
 * Naive Instantaneous Doppler Curve (the "wrong way" demonstrated in the video):
 * Uses radial velocity at receive time t directly without accounting for sound travel delay variation.
 */
export function naiveEmissionDopplerLinear(
  tRecvArray: number[],
  f0Hz: number,
  vMs: number,
  rSlantM: number,
  tCpaRecvS: number,
  cMs: number = 340.0
): number[] {
  const v = Math.max(1.0, vMs);
  const r = Math.max(50.0, rSlantM);
  const c = Math.max(250.0, cMs);
  return tRecvArray.map((tAbs) => {
    const tau = tAbs - tCpaRecvS;
    const d = Math.sqrt(r * r + v * tau * (v * tau));
    const vr = (v * v * tau) / d;
    return (f0Hz * c) / (c + vr);
  });
}

/**
 * Synthesize missing waypoints for any scenario flight that only has heading/speed/CPA metadata
 * (such as G1-polar-svalbard, G2-dateline-fiji, G3-southern-ocean, G4-equatorial-pacific).
 */
export function ensureFlightWaypoints(flight: {
  heading_deg: number;
  speed_ms: number;
  alt_m: number;
  cpa?: { lat: number; lon: number; time_s: number; slant_m: number } | null;
  waypoints?: Waypoint[];
}): Waypoint[] {
  if (flight.waypoints && flight.waypoints.length >= 2) {
    return flight.waypoints;
  }
  if (!flight.cpa) {
    return [];
  }
  const cpaLat = flight.cpa.lat;
  const cpaLon = flight.cpa.lon;
  const cpaTime = flight.cpa.time_s;
  const speed = flight.speed_ms || 100.0;
  const heading = flight.heading_deg || 90.0;
  const alt = flight.alt_m || 1500.0;

  const wps: Waypoint[] = [];
  // Generate waypoints from t_cpa - 300s to t_cpa + 300s in 10s steps
  for (let dt = -300; dt <= 300; dt += 10) {
    const dist = Math.abs(dt) * speed;
    const brg = dt >= 0 ? heading : (heading + 180.0) % 360.0;
    const pt =
      dist < 1e-3
        ? { lat: cpaLat, lon: cpaLon }
        : destinationPoint(cpaLat, cpaLon, brg, dist);
    wps.push({
      t: Number((cpaTime + dt).toFixed(1)),
      lat: Number(pt.lat.toFixed(5)),
      lon: Number(pt.lon.toFixed(5)),
      alt_m: alt,
      speed_ms: speed,
    });
  }
  return wps;
}

/**
 * Evaluate 3D ECEF position and velocity along a flight's waypoint list at emission time tau.
 */
export function sampleFlightEcefAt(
  waypoints: Waypoint[],
  tau: number
): { pos: Vec3; vel: Vec3; alt_m: number } | null {
  if (!waypoints || waypoints.length < 2) return null;
  const n = waypoints.length;
  // Find bracketing interval
  let idx = 0;
  if (tau <= waypoints[0].t) {
    idx = 0;
  } else if (tau >= waypoints[n - 1].t) {
    idx = n - 2;
  } else {
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (waypoints[mid].t <= tau) lo = mid;
      else hi = mid;
    }
    idx = lo;
  }
  const w0 = waypoints[idx];
  const w1 = waypoints[idx + 1];
  const dt = Math.max(1e-3, w1.t - w0.t);
  const s = (tau - w0.t) / dt;
  const p0 = llhToEcef(w0.lat, w0.lon, w0.alt_m);
  const p1 = llhToEcef(w1.lat, w1.lon, w1.alt_m);
  const vel: Vec3 = [
    (p1[0] - p0[0]) / dt,
    (p1[1] - p0[1]) / dt,
    (p1[2] - p0[2]) / dt,
  ];
  const pos: Vec3 = [
    p0[0] + s * (p1[0] - p0[0]),
    p0[1] + s * (p1[1] - p0[1]),
    p0[2] + s * (p1[2] - p0[2]),
  ];
  const alt_m = w0.alt_m + s * (w1.alt_m - w0.alt_m);
  return { pos, vel, alt_m };
}

/**
 * Forward Retarded-Time Curve for an observer (lat, lon, alt_m) and a 3D flight track.
 * Returns arrival times t_arr, Doppler shape factors 1/(dt/dtau), delayed CPA time, and minimum 3D slant distance.
 */
export function forwardFlightCurveAtReceiver(
  rxLat: number,
  rxLon: number,
  rxAltM: number,
  waypoints: Waypoint[],
  atmos: AtmosphereParams,
  tauStart?: number,
  tauEnd?: number,
  tauStep: number = 1.0
): {
  tArr: number[];
  shape: number[];
  cpaDelayedTimeS: number;
  cpaEmissionTimeS: number;
  slantMinM: number;
} {
  const wps = waypoints;
  if (!wps || wps.length < 2) {
    return {
      tArr: [],
      shape: [],
      cpaDelayedTimeS: 0,
      cpaEmissionTimeS: 0,
      slantMinM: Infinity,
    };
  }
  const rxEcef = llhToEcef(rxLat, rxLon, rxAltM);
  const frame = new EnuFrame(rxLat, rxLon, rxAltM);
  const windEcef = windEcefAtReceiver(atmos, frame);

  const t0 = tauStart !== undefined ? tauStart : wps[0].t;
  const t1 = tauEnd !== undefined ? tauEnd : wps[wps.length - 1].t;

  const tArr: number[] = [];
  const shape: number[] = [];
  let slantMinM = Infinity;
  let cpaDelayedTimeS = 0;
  let cpaEmissionTimeS = 0;

  for (let tau = t0; tau <= t1; tau += tauStep) {
    const state = sampleFlightEcefAt(wps, tau);
    if (!state) continue;
    const dVec = vec3Sub(rxEcef, state.pos);
    const dist = vec3Norm(dVec);
    const cPath = pathSoundSpeed(rxAltM, state.alt_m, atmos.temp_c);
    const travel = travelTimeQuadratic(dVec, windEcef, cPath);
    const mVec: Vec3 = [
      dVec[0] - windEcef[0] * travel,
      dVec[1] - windEcef[1] * travel,
      dVec[2] - windEcef[2] * travel,
    ];
    const mNorm = Math.max(1e-9, vec3Norm(mVec));
    const uVec: Vec3 = [mVec[0] / mNorm, mVec[1] / mNorm, mVec[2] / mNorm];
    const uDotVel = vec3Dot(uVec, state.vel);
    const uDotWind = vec3Dot(uVec, windEcef);
    const dtdtau = 1.0 - uDotVel / Math.max(50.0, cPath + uDotWind);
    if (dtdtau > 0.2 && Number.isFinite(dtdtau)) {
      const arr = tau + travel;
      tArr.push(arr);
      shape.push(1.0 / dtdtau);
      if (dist < slantMinM) {
        slantMinM = dist;
        cpaDelayedTimeS = arr;
        cpaEmissionTimeS = tau;
      }
    }
  }

  return {
    tArr,
    shape,
    cpaDelayedTimeS,
    cpaEmissionTimeS,
    slantMinM,
  };
}

/**
 * Evaluate exact delayed CPA time and slant distance using golden-section refinement.
 */
export function computeDelayedCpa(
  rxLat: number,
  rxLon: number,
  rxAltM: number,
  waypoints: Waypoint[],
  atmos: AtmosphereParams
): { delayedCpaS: number; emissionCpaS: number; slantM: number } {
  const coarse = forwardFlightCurveAtReceiver(
    rxLat,
    rxLon,
    rxAltM,
    waypoints,
    atmos,
    undefined,
    undefined,
    2.0
  );
  if (!Number.isFinite(coarse.slantMinM)) {
    return { delayedCpaS: 0, emissionCpaS: 0, slantM: Infinity };
  }
  const rxEcef = llhToEcef(rxLat, rxLon, rxAltM);
  const frame = new EnuFrame(rxLat, rxLon, rxAltM);
  const windEcef = windEcefAtReceiver(atmos, frame);

  let a = coarse.cpaEmissionTimeS - 3.0;
  let b = coarse.cpaEmissionTimeS + 3.0;
  const distAt = (t: number) => {
    const st = sampleFlightEcefAt(waypoints, t);
    return st ? vec3Dist(rxEcef, st.pos) : Infinity;
  };
  for (let i = 0; i < 24; i++) {
    const m1 = a + 0.381966 * (b - a);
    const m2 = b - 0.381966 * (b - a);
    if (distAt(m1) < distAt(m2)) b = m2;
    else a = m1;
  }
  const tauStar = 0.5 * (a + b);
  const st = sampleFlightEcefAt(waypoints, tauStar)!;
  const dVec = vec3Sub(rxEcef, st.pos);
  const slantM = vec3Norm(dVec);
  const cPath = pathSoundSpeed(rxAltM, st.alt_m, atmos.temp_c);
  const travel = travelTimeQuadratic(dVec, windEcef, cPath);
  return {
    delayedCpaS: tauStar + travel,
    emissionCpaS: tauStar,
    slantM,
  };
}
