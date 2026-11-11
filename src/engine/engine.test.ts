import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EnuFrame,
  boundsGeometry,
  deltaLonSigned,
  destinationPoint,
  ecefToLlh,
  greatCircleDistance,
  llhToEcef,
  normalizeLon,
} from './geo';
import {
  absorptionCoefficientDbPerM,
  isaPressureKpa,
  isaTemperatureK,
  pathSoundSpeed,
} from './atmosphere';
import {
  computeDelayedCpa,
  ensureFlightWaypoints,
  naiveEmissionDopplerLinear,
  receiveDopplerLinear,
  travelTimeQuadratic,
} from './acoustics';
import {
  applyAircraftBandBoost,
  applyClockDrift,
  applyHighPassFilter,
  computeSpectrogram,
  fitDopplerEventFromRidge,
  synthesizeScenarioAudio,
  traceDopplerRidge,
} from './dsp';
import {
  clusterCandidatesIntoBasins,
  isBpfCompatible,
  runGeolocationInversion,
} from './solver';

test('WGS84 ECEF <-> LLH round-trip across equatorial, polar, and southern latitudes', () => {
  const cases = [
    { lat: 37.4, lon: -122.3, alt_m: 150.0 },
    { lat: 78.2, lon: 15.6, alt_m: 1500.0 }, // Svalbard polar
    { lat: 89.95, lon: 45.0, alt_m: 500.0 }, // Near-pole fallback
    { lat: -54.5, lon: 3.0, alt_m: 2200.0 }, // Southern Ocean
    { lat: 0.5, lon: -160.0, alt_m: 0.0 }, // Equatorial Pacific
  ];

  for (const c of cases) {
    const ecef = llhToEcef(c.lat, c.lon, c.alt_m);
    const back = ecefToLlh(ecef);
    assert.ok(Math.abs(back.lat - c.lat) < 1e-4, `lat mismatch for ${c.lat}`);
    assert.ok(
      Math.abs(deltaLonSigned(back.lon, c.lon)) < 1e-4,
      `lon mismatch for ${c.lon}`
    );
    assert.ok(Math.abs(back.alt_m - c.alt_m) < 0.5, `alt mismatch for ${c.alt_m}`);
  }
});

test('Antimeridian / Dateline unwrapping (Fiji 177E to -179W)', () => {
  assert.equal(normalizeLon(181.0), -179.0);
  assert.equal(normalizeLon(-181.0), 179.0);
  assert.ok(Math.abs(deltaLonSigned(-179.0, 179.0) - 2.0) < 1e-9);
  assert.ok(Math.abs(deltaLonSigned(179.0, -179.0) - -2.0) < 1e-9);

  const g2Bounds = { west: 177.0, south: -18.5, east: -179.0, north: -16.5 };
  const geom = boundsGeometry(g2Bounds);
  assert.equal(geom.wrapsDateline, true);
  assert.ok(Math.abs(geom.lonSpan - 4.0) < 1e-6);
  assert.ok(Math.abs(deltaLonSigned(geom.centerLon, 179.0)) < 1e-6);

  // Great-circle distance across dateline should be ~210 km, not 39,000 km
  const d = greatCircleDistance(-17.5, 179.0, -17.5, -179.0);
  assert.ok(d > 190_000 && d < 230_000, `Unexpected cross-dateline distance ${d}`);
});

test('Local ENU tangent frame round-trip and orientation', () => {
  const frame = new EnuFrame(-54.5, 3.0, 0.0);
  const ecef = frame.toEcef(1200.0, -850.0, 400.0);
  const enu = frame.fromEcef(ecef);
  assert.ok(Math.abs(enu[0] - 1200.0) < 1e-5);
  assert.ok(Math.abs(enu[1] - -850.0) < 1e-5);
  assert.ok(Math.abs(enu[2] - 400.0) < 1e-5);
});

test('Atmosphere ISA sound speed and ISO 9613-1 absorption anchors', () => {
  const t0 = isaTemperatureK(0.0, 15.0);
  assert.ok(Math.abs(t0 - 288.15) < 1e-3);
  const p0 = isaPressureKpa(0.0);
  assert.ok(Math.abs(p0 - 101.325) < 1e-3);

  const cPath = pathSoundSpeed(0.0, 1500.0, 15.0);
  assert.ok(cPath > 335.0 && cPath < 342.0);

  // ISO 9613-1 anchor: alpha(1000 Hz, 20 C, 50% RH, 101.325 kPa) ~ 0.0047 dB/m (4.7 dB/km)
  const alpha1k = absorptionCoefficientDbPerM(1000.0, 293.15, 0.5, 101.325);
  assert.ok(
    alpha1k * 1000.0 > 4.0 && alpha1k * 1000.0 < 5.5,
    `Unexpected 1kHz absorption ${alpha1k * 1000} dB/km`
  );
});

test('Retarded-time Doppler curve vs Naive instantaneous curve asymmetry', () => {
  const times = [-30, -15, -5, 0, 5, 15, 30];
  const fRetarded = receiveDopplerLinear(times, 100.0, 100.0, 1500.0, 0.0, 340.0);
  const fNaive = naiveEmissionDopplerLinear(times, 100.0, 100.0, 1500.0, 0.0, 340.0);

  // At t = t_cpa_delayed (dt = 0), both pass through f0 = 100 Hz
  assert.ok(Math.abs(fRetarded[3] - 100.0) < 1e-3);
  assert.ok(Math.abs(fNaive[3] - 100.0) < 1e-3);

  // Before CPA (dt = -15s), retarded curve is steeper/higher than naive curve
  // because approaching wavefronts arrive compressed in time
  assert.ok(
    Math.abs(fRetarded[1] - fNaive[1]) > 0.5,
    'Retarded and naive curves must differ away from CPA'
  );

  // Travel time quadratic with tailwind vs headwind
  const tTailwind = travelTimeQuadratic([3400, 0, 0], [10, 0, 0], 340.0);
  const tHeadwind = travelTimeQuadratic([3400, 0, 0], [-10, 0, 0], 340.0);
  assert.ok(tTailwind < 10.0 && tHeadwind > 10.0);
});

test('Hydrate missing flight waypoints and compute delayed CPA', () => {
  const hydrated = ensureFlightWaypoints({
    heading_deg: 90.0,
    speed_ms: 100.0,
    alt_m: 1500.0,
    cpa: { lat: 78.2, lon: 15.6, time_s: 1000.0, slant_m: 1500.0 },
    waypoints: [],
  });
  assert.ok(hydrated.length >= 20);
  const cpa = computeDelayedCpa(78.2, 15.6, 0.0, hydrated, {
    temp_c: -10.0,
    wind_e_ms: 3.0,
    wind_n_ms: -2.0,
  });
  assert.ok(Math.abs(cpa.slantM - 1500.0) < 50.0);
  assert.ok(cpa.delayedCpaS > 1003.0 && cpa.delayedCpaS < 1006.5);
});

test('STFT Spectrogram, Median/MAD whitening, 80Hz High-Pass countermeasure, and Viterbi ridge tracer', () => {
  const synthAudio = synthesizeScenarioAudio(
    [{ f0_hz: 85.0, speed_ms: 100.0, slant_m: 1500.0, t_cpa_s: 1000.0 }],
    20.0,
    4000
  );
  assert.equal(synthAudio.length, 80000);

  const spec = computeSpectrogram(synthAudio, 4000, 512, 128, 400.0);
  assert.ok(spec.nFreq > 10 && spec.nTime > 50);

  const ridge = traceDopplerRidge(spec, 55.0, 130.0);
  assert.ok(ridge !== null, 'Ridge tracer should detect synthesized Doppler curve');
  assert.ok(ridge!.timesS.length >= 10);

  const fit = fitDopplerEventFromRidge(ridge!, 'EV01', 1000.0);
  assert.ok(fit !== null);
  assert.ok(fit!.f0_hz > 60.0 && fit!.f0_hz < 115.0);

  // Apply 80Hz high-pass filter + aircraft boost + clock drift
  const hp = applyHighPassFilter(synthAudio, 4000, 80.0);
  const boosted = applyAircraftBandBoost(hp, 4000, 12.0);
  const drifted = applyClockDrift(boosted, 10.0);
  assert.equal(drifted.length, synthAudio.length);
});

test('Geolocation solver handles crossing fix, parallel ambiguity, and high-pass harmonic prior', () => {
  // High-pass submultiple (39 Hz is 1/2 of 78 Hz C172 fundamental)
  assert.equal(isBpfCompatible(39.0, 'C172'), true);
  assert.equal(isBpfCompatible(79.1, 'C172'), true);

  const crossingRes = runGeolocationInversion({
    scenarioId: 'A-crossing-baseline',
    truth: { lat: 37.4, lon: -122.3, alt_m: 0 },
    baseFix: { lat: 37.4042, lon: -122.2965, alt_m: 0 },
    baseScore: 48.52,
    baseErrorM: 470,
    baseRadiusM: 1601.7,
    baseStatus: 'located',
    baseClock: { offset_s: -0.05, drift_ppm: 0.0 },
    bounds: { west: -122.55, south: 37.2, east: -122.05, north: 37.6 },
    atmos: { temp_c: 15.0, wind_e_ms: 3.0, wind_n_ms: -2.0 },
    flights: [
      {
        id: 'SKY01',
        callsign: 'SKY01',
        aircraft: 'C172',
        aircraft_name: 'Cessna 172',
        heading_deg: 90,
        speed_ms: 100,
        speed_kts: 194,
        alt_m: 1500,
        bpf_nominal_hz: 85,
        is_target: true,
        cpa: { lat: 37.4, lon: -122.3, time_s: 1004, slant_m: 1500 },
        waypoints: [],
      },
      {
        id: 'SKY02',
        callsign: 'SKY02',
        aircraft: 'SR22',
        aircraft_name: 'Cirrus SR22',
        heading_deg: 270,
        speed_ms: 110,
        speed_kts: 214,
        alt_m: 2200,
        bpf_nominal_hz: 115,
        is_target: true,
        cpa: { lat: 37.4, lon: -122.3, time_s: 1500, slant_m: 2200 },
        waypoints: [],
      },
    ],
    events: [
      {
        id: 'EV01',
        f0_hz: 79.1,
        speed_ms: 100,
        slant_m: 1548,
        t_cpa_s: 1004,
        snr_db: 51.6,
        orders: [1, 2, 3],
        curves: [],
      },
      {
        id: 'EV02',
        f0_hz: 116.9,
        speed_ms: 110,
        slant_m: 2210,
        t_cpa_s: 1500,
        snr_db: 48.5,
        orders: [1, 2, 3],
        curves: [],
      },
    ],
  });

  assert.equal(crossingRes.status, 'located');
  assert.ok(crossingRes.basins.length >= 2);
  assert.ok(crossingRes.error_m < 800);

  // Parallel degeneracy should report ambiguous
  const parallelRes = runGeolocationInversion({
    scenarioId: 'B-parallel-26s',
    truth: { lat: 37.4, lon: -122.3, alt_m: 0 },
    baseFix: { lat: 37.385, lon: -122.285, alt_m: 0 },
    baseScore: 37.26,
    baseErrorM: 520,
    baseRadiusM: 1155.2,
    baseStatus: 'ambiguous',
    baseClock: { offset_s: 5.55, drift_ppm: 0.0 },
    bounds: { west: -122.55, south: 37.2, east: -122.05, north: 37.6 },
    atmos: { temp_c: 15.0, wind_e_ms: 3.0, wind_n_ms: -2.0 },
    flights: [],
    events: [
      {
        id: 'EV01',
        f0_hz: 78.7,
        speed_ms: 105,
        slant_m: 1679,
        t_cpa_s: 1000,
        snr_db: 51.6,
        orders: [1, 2],
        curves: [],
      },
      {
        id: 'EV02',
        f0_hz: 113.7,
        speed_ms: 110,
        slant_m: 2151,
        t_cpa_s: 1026,
        snr_db: 45.6,
        orders: [1, 2],
        curves: [],
      },
    ],
  });
  assert.equal(parallelRes.status, 'ambiguous');
});
