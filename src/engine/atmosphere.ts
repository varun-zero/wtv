/**
 * Stratified Atmosphere, Wind Vector, and ISO 9613-1 / ANSI S1.26 Acoustic Absorption Model.
 */
import { EnuFrame, Vec3 } from './geo';

export const T01 = 293.15; // K (20 C ISO reference)
export const T_SL = 288.15; // K (15 C ISA sea level)
export const T_TRIPLE = 273.16; // K (water triple point)
export const P_REF = 101.325; // kPa
export const LAPSE = 0.0065; // K/m troposphere lapse rate
export const TROPOPAUSE = 11000.0; // m
export const GAMMA = 1.4;
export const R_D = 287.05287; // J/(kg K)
export const G0 = 9.80665; // m/s^2

const EXP_P = G0 / (R_D * LAPSE);
const SCALE_H = (R_D * 216.65) / G0;

export interface AtmosphereParams {
  temp_c: number; // ground temperature (deg C)
  wind_e_ms: number; // east wind component (m/s)
  wind_n_ms: number; // north wind component (m/s)
  rel_humidity?: number; // 0..1 (default 0.5)
}

export function isaTemperatureK(altM: number, groundTempC: number = 15.0): number {
  const offsetK = groundTempC - 15.0;
  const base = altM < TROPOPAUSE ? T_SL - LAPSE * altM : 216.65;
  return Math.max(180.0, base + offsetK);
}

export function isaPressureKpa(altM: number): number {
  const trop = Math.min(Math.max(0.0, altM), TROPOPAUSE);
  if (altM < TROPOPAUSE) {
    return P_REF * Math.pow(1.0 - (LAPSE / T_SL) * trop, EXP_P);
  }
  const pTrop = P_REF * Math.pow(1.0 - (LAPSE * TROPOPAUSE) / T_SL, EXP_P);
  return pTrop * Math.exp(-(altM - TROPOPAUSE) / SCALE_H);
}

export function soundSpeedAtTempK(tempK: number): number {
  return Math.sqrt(GAMMA * R_D * Math.max(150.0, tempK));
}

/**
 * Harmonic-mean effective sound speed (m/s) along a 5-node stratified slant path
 * between observer altitude and source aircraft altitude.
 */
export function pathSoundSpeed(
  altRxM: number,
  altSrcM: number,
  groundTempC: number = 15.0
): number {
  let invSum = 0.0;
  const nodes = 5;
  for (let i = 0; i < nodes; i++) {
    const frac = i / (nodes - 1);
    const h = altRxM + frac * (altSrcM - altRxM);
    const c = soundSpeedAtTempK(isaTemperatureK(h, groundTempC));
    invSum += 1.0 / c;
  }
  return nodes / invSum;
}

export function saturationVaporPressureKpa(tempK: number): number {
  return (
    P_REF * Math.pow(10.0, -6.8346 * Math.pow(T_TRIPLE / tempK, 1.261) + 4.6151)
  );
}

/**
 * ISO 9613-1 pure-tone atmospheric absorption coefficient alpha (dB/m).
 */
export function absorptionCoefficientDbPerM(
  fHz: number,
  tempK: number,
  relHumidity: number = 0.5,
  pKpa: number = P_REF
): number {
  const f = Math.max(1.0, fHz);
  const t = Math.max(200.0, tempK);
  const p = Math.max(1.0, pKpa);
  const h = Math.min(
    100.0,
    Math.max(0.0, (100.0 * relHumidity * saturationVaporPressureKpa(t)) / p)
  );
  const pr = p / P_REF;
  const frO = pr * (24.0 + (4.04e4 * h * (0.02 + h)) / (0.391 + h));
  const frN =
    pr *
    Math.sqrt(T01 / t) *
    (9.0 + 280.0 * h * Math.exp(-4.17 * (Math.pow(T01 / t, 1.0 / 3.0) - 1.0)));
  const tr = t / T01;
  const alpha =
    8.686 *
    f *
    f *
    (1.84e-11 * pr * Math.sqrt(tr) +
      Math.pow(tr, -2.5) *
        ((0.01275 * Math.exp(-2239.1 / t)) / (frO + (f * f) / frO) +
          (0.1068 * Math.exp(-3352.0 / t)) / (frN + (f * f) / frN)));
  return alpha;
}

export function windEcefAtReceiver(
  atmos: AtmosphereParams,
  frame: EnuFrame
): Vec3 {
  return frame.enuVectorToEcef(atmos.wind_e_ms, atmos.wind_n_ms, 0.0);
}
