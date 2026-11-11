/**
 * Real-Time Acoustic Signal Processing (DSP) Engine:
 * - In-place Radix-2 Cooley-Tukey FFT
 * - STFT Magnitude Spectrogram
 * - Per-Frequency Median / MAD Spectral Whitening (reveals <0 dB SNR inaudible flyovers)
 * - Countermeasure Processors (80 Hz High-Pass Filter, Spectral Boost, Clock Drift ppm, Codec Quantization)
 * - Physics-Bounded Viterbi Dynamic Programming Ridge Tracer & Harmonic Family Grouper
 */
import { receiveDopplerLinear } from './acoustics';

export interface SpectrogramData {
  /** Flattened row-major array of shape [nFreq, nTime] */
  raw: Float32Array;
  /** Median/MAD whitened z-score array of shape [nFreq, nTime] */
  whitened: Float32Array;
  freqs: Float32Array;
  times: Float32Array;
  nFreq: number;
  nTime: number;
  sampleRate: number;
  durationS: number;
}

export interface TracedRidge {
  bandLoHz: number;
  bandHiHz: number;
  timesS: number[];
  freqsHz: number[];
  weights: number[];
  harmonicOrder: number;
}

export interface FittedDopplerEvent {
  id: string;
  flight_id?: string;
  f0_hz: number;
  speed_ms: number;
  slant_m: number;
  t_cpa_s: number;
  snr_db: number;
  orders: number[];
  curves: {
    order: number;
    points: [number, number][];
    naivePoints?: [number, number][];
  }[];
  measuredSamples?: { t: number; f: number; w: number }[];
}

/**
 * In-place Radix-2 Cooley-Tukey FFT. n must be a power of 2.
 */
export function fftRadix2(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT length ${n} must be a power of 2`);
  }

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2.0 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let k = 0; k < half; k++) {
        const uIdx = i + k;
        const vIdx = i + k + half;
        const vr = re[vIdx] * wRe - im[vIdx] * wIm;
        const vi = re[vIdx] * wIm + im[vIdx] * wRe;
        re[vIdx] = re[uIdx] - vr;
        im[vIdx] = im[uIdx] - vi;
        re[uIdx] += vr;
        im[uIdx] += vi;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }
}

/**
 * Compute STFT spectrogram and Median/MAD whitened contrast matrix from PCM samples.
 * Restricts stored frequency bins up to maxFreqHz (default 600 Hz) for memory & speed optimality.
 */
export function computeSpectrogram(
  samples: Float32Array,
  sampleRate: number,
  nFft: number = 1024,
  hop?: number,
  maxFreqHz: number = 500.0
): SpectrogramData {
  const actualHop = hop ?? (nFft >> 2);
  const nSamples = samples.length;
  if (nSamples < nFft) {
    const padded = new Float32Array(nFft);
    padded.set(samples);
    samples = padded;
  }
  const nTime = Math.max(1, 1 + Math.floor((samples.length - nFft) / actualHop));
  const fullFreqBins = (nFft >> 1) + 1;
  const df = sampleRate / nFft;
  const nFreq = Math.min(
    fullFreqBins,
    Math.max(16, Math.ceil(maxFreqHz / df) + 1)
  );

  const freqs = new Float32Array(nFreq);
  for (let f = 0; f < nFreq; f++) {
    freqs[f] = f * df;
  }

  const times = new Float32Array(nTime);
  for (let t = 0; t < nTime; t++) {
    times[t] = (t * actualHop + nFft * 0.5) / sampleRate;
  }

  const win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) {
    win[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (nFft - 1)));
  }

  const raw = new Float32Array(nFreq * nTime);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);

  for (let t = 0; t < nTime; t++) {
    const offset = t * actualHop;
    for (let i = 0; i < nFft; i++) {
      re[i] = (samples[offset + i] || 0.0) * win[i];
      im[i] = 0.0;
    }
    fftRadix2(re, im);
    for (let f = 0; f < nFreq; f++) {
      raw[f * nTime + t] = Math.hypot(re[f], im[f]);
    }
  }

  const whitened = whitenSpectrogram(raw, nFreq, nTime);

  return {
    raw,
    whitened,
    freqs,
    times,
    nFreq,
    nTime,
    sampleRate,
    durationS: samples.length / sampleRate,
  };
}

/**
 * Per-frequency Median / MAD Whitening:
 * Subtracts the row median and normalizes by the Median Absolute Deviation (MAD)
 * without rescaling columns, suppressing stationary hums and leveling the noise floor.
 */
export function whitenSpectrogram(
  raw: Float32Array,
  nFreq: number,
  nTime: number
): Float32Array {
  const out = new Float32Array(nFreq * nTime);
  const rowBuf = new Float32Array(nTime);
  const devBuf = new Float32Array(nTime);

  for (let f = 0; f < nFreq; f++) {
    const rowOff = f * nTime;
    for (let t = 0; t < nTime; t++) {
      rowBuf[t] = raw[rowOff + t];
    }
    rowBuf.sort();
    const mid = nTime >> 1;
    const median =
      nTime % 2 === 1 ? rowBuf[mid] : 0.5 * (rowBuf[mid - 1] + rowBuf[mid]);

    for (let t = 0; t < nTime; t++) {
      devBuf[t] = Math.abs(raw[rowOff + t] - median);
    }
    devBuf.sort();
    const mad = Math.max(
      1e-9,
      nTime % 2 === 1 ? devBuf[mid] : 0.5 * (devBuf[mid - 1] + devBuf[mid])
    );

    for (let t = 0; t < nTime; t++) {
      out[rowOff + t] = (raw[rowOff + t] - median) / mad;
    }
  }
  return out;
}

/**
 * Countermeasure 1: 4th-order Butterworth High-Pass Filter at cutoffHz (default 80 Hz).
 * Demonstrates how removing the fundamental f0 leaves upper harmonics k >= 2 intact.
 */
export function applyHighPassFilter(
  samples: Float32Array,
  sampleRate: number,
  cutoffHz: number = 80.0
): Float32Array {
  const out = new Float32Array(samples.length);
  out.set(samples);

  // Two cascaded 2nd-order Butterworth high-pass biquad sections
  const wc = Math.tan((Math.PI * Math.min(cutoffHz, sampleRate * 0.45)) / sampleRate);
  const qValues = [0.5411961, 1.306563];

  for (const q of qValues) {
    const k = wc;
    const norm = 1.0 / (1.0 + k / q + k * k);
    const b0 = norm;
    const b1 = -2.0 * b0;
    const b2 = b0;
    const a1 = 2.0 * (k * k - 1.0) * norm;
    const a2 = (1.0 - k / q + k * k) * norm;

    let x1 = 0.0,
      x2 = 0.0,
      y1 = 0.0,
      y2 = 0.0;
    for (let i = 0; i < out.length; i++) {
      const x0 = out[i];
      const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x0;
      y2 = y1;
      y1 = y0;
      out[i] = y0;
    }
  }
  return out;
}

/**
 * Countermeasure Reversal: Spectral Sub-band Boost (+dB gain in 50-350 Hz aircraft band)
 * Reveals attenuated Doppler curves after high-pass filtering or MP3 compression.
 */
export function applyAircraftBandBoost(
  samples: Float32Array,
  sampleRate: number,
  boostDb: number = 14.0
): Float32Array {
  if (Math.abs(boostDb) < 0.1) return samples;
  const out = new Float32Array(samples.length);
  // Peaking EQ centered at 135 Hz (Q = 0.9)
  const f0 = 135.0;
  const Q = 0.9;
  const A = Math.pow(10.0, boostDb / 40.0);
  const w0 = (2.0 * Math.PI * f0) / sampleRate;
  const alpha = Math.sin(w0) / (2.0 * Q);
  const cosw0 = Math.cos(w0);

  const b0 = 1.0 + alpha * A;
  const b1 = -2.0 * cosw0;
  const b2 = 1.0 - alpha * A;
  const a0 = 1.0 + alpha / A;
  const a1 = -2.0 * cosw0;
  const a2 = 1.0 - alpha / A;

  let x1 = 0,
    x2 = 0,
    y1 = 0,
    y2 = 0;
  let maxAbs = 1e-9;
  for (let i = 0; i < samples.length; i++) {
    const x0 = samples[i];
    const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
    out[i] = y0;
    if (Math.abs(y0) > maxAbs) maxAbs = Math.abs(y0);
  }
  const scale = 0.92 / maxAbs;
  for (let i = 0; i < out.length; i++) {
    out[i] *= scale;
  }
  return out;
}

/**
 * Countermeasure 2: Hardware Clock Drift Simulation (ppm).
 */
export function applyClockDrift(
  samples: Float32Array,
  ppm: number
): Float32Array {
  if (Math.abs(ppm) < 1e-3) return samples;
  const n = samples.length;
  const out = new Float32Array(n);
  const rate = 1.0 + ppm * 1e-6;
  for (let i = 0; i < n; i++) {
    const srcIdx = i * rate;
    const i0 = Math.floor(srcIdx);
    const frac = srcIdx - i0;
    if (i0 + 1 < n) {
      out[i] = samples[i0] * (1.0 - frac) + samples[i0 + 1] * frac;
    } else if (i0 < n) {
      out[i] = samples[i0];
    }
  }
  return out;
}

/**
 * Physics-Bounded Viterbi Dynamic Programming Ridge Tracer over a frequency band [fLoHz, fHiHz].
 */
export function traceDopplerRidge(
  spec: SpectrogramData,
  fLoHz: number,
  fHiHz: number,
  rentZ: number = 1.8
): TracedRidge | null {
  const { whitened, freqs, times, nFreq, nTime } = spec;
  if (nTime < 4) return null;

  const bandIndices: number[] = [];
  for (let f = 0; f < nFreq; f++) {
    if (freqs[f] >= fLoHz && freqs[f] <= fHiHz) {
      bandIndices.push(f);
    }
  }
  const nB = bandIndices.length;
  if (nB < 4) return null;

  const df = freqs.length > 1 ? freqs[1] - freqs[0] : 4.0;
  const dt = times.length > 1 ? times[1] - times[0] : 0.1;
  const fRef = freqs[bandIndices[nB - 1]];

  // Physics slope bounds: peak descent fRef * v_max^2 / (c * r_min)
  const descMaxHzS = 1.35 * ((fRef * 220.0 * 220.0) / (343.0 * 350.0));
  const ascMaxHzS = 0.25 * descMaxHzS + 2.0;
  const avgDown = descMaxHzS * dt;
  const avgUp = ascMaxHzS * dt;
  const maxMoveCells = Math.max(3, Math.ceil((descMaxHzS * dt * 3) / df));
  const kOver = 0.08;

  const S = new Float64Array(nB * nTime);
  const back = new Int32Array(nB * nTime);

  for (let r = 0; r < nB; r++) {
    const z = whitened[bandIndices[r] * nTime + 0];
    S[r * nTime + 0] = Math.max(0.0, z) - rentZ;
  }

  for (let tc = 1; tc < nTime; tc++) {
    for (let fc = 0; fc < nB; fc++) {
      const lo = Math.max(0, fc - maxMoveCells);
      const hi = Math.min(nB - 1, fc + maxMoveCells);
      let bestPrev = -Infinity;
      let bestRow = fc;
      for (let prev = lo; prev <= hi; prev++) {
        const deltaHz = (prev - fc) * df; // positive if prev > fc (descending)
        const descExcess = Math.max(0.0, deltaHz - avgDown);
        const ascExcess = Math.max(0.0, -deltaHz - avgUp);
        const cost = kOver * (descExcess + ascExcess);
        const cand = S[prev * nTime + (tc - 1)] - cost;
        if (cand > bestPrev) {
          bestPrev = cand;
          bestRow = prev;
        }
      }
      const z = whitened[bandIndices[fc] * nTime + tc];
      S[fc * nTime + tc] = bestPrev + Math.max(0.0, z) - rentZ;
      back[fc * nTime + tc] = bestRow;
    }
  }

  // Free termination at highest-scoring cell
  let bestVal = -Infinity;
  let bestFc = 0;
  let bestTc = nTime - 1;
  for (let r = 0; r < nB; r++) {
    for (let c = 0; c < nTime; c++) {
      const v = S[r * nTime + c];
      if (v > bestVal) {
        bestVal = v;
        bestFc = r;
        bestTc = c;
      }
    }
  }

  const pathT: number[] = [];
  const pathF: number[] = [];
  let fc = bestFc;
  let tc = bestTc;
  while (tc >= 0) {
    pathT.push(tc);
    pathF.push(fc);
    if (tc === 0) break;
    fc = back[fc * nTime + tc];
    tc--;
  }
  pathT.reverse();
  pathF.reverse();

  const timesS: number[] = [];
  const freqsHz: number[] = [];
  const weights: number[] = [];
  for (let i = 0; i < pathT.length; i++) {
    const tIdx = pathT[i];
    const fGlobal = bandIndices[pathF[i]];
    const zRaw = whitened[fGlobal * nTime + tIdx];
    if (zRaw > 1.5) {
      timesS.push(times[tIdx]);
      freqsHz.push(freqs[fGlobal]);
      weights.push(zRaw);
    }
  }

  if (timesS.length < 6) return null;
  return {
    bandLoHz: fLoHz,
    bandHiHz: fHiHz,
    timesS,
    freqsHz,
    weights,
    harmonicOrder: 1,
  };
}

/**
 * Fit parametric Doppler parameters (f0, v, r, t_cpa) to a traced ridge using plateau ratios + grid search.
 */
export function fitDopplerEventFromRidge(
  ridge: TracedRidge,
  eventId: string = 'EV01',
  timeEpochOffsetS: number = 0.0
): FittedDopplerEvent | null {
  const { timesS, freqsHz, weights } = ridge;
  if (timesS.length < 6) return null;

  const sortedFreqs = [...freqsHz].sort((a, b) => a - b);
  const fPost = sortedFreqs[Math.floor(sortedFreqs.length * 0.1)];
  const fPre = sortedFreqs[Math.floor(sortedFreqs.length * 0.9)];
  const ratio = fPre / Math.max(fPost, 1.0);
  const c = 340.0;

  let vEst = ratio > 1.03 ? (c * (ratio - 1.0)) / (ratio + 1.0) : 95.0;
  vEst = Math.max(45.0, Math.min(180.0, vEst));
  let f0Est =
    Math.sqrt(fPre * fPost) * Math.sqrt(Math.max(0.2, 1.0 - (vEst / c) ** 2));

  // Estimate steepest drop time as CPA seed
  let maxDropRate = -Infinity;
  let tcEst = timesS[Math.floor(timesS.length / 2)];
  for (let i = 1; i < timesS.length - 1; i++) {
    const dt = Math.max(0.05, timesS[i + 1] - timesS[i - 1]);
    const dropRate = (freqsHz[i - 1] - freqsHz[i + 1]) / dt;
    if (dropRate > maxDropRate) {
      maxDropRate = dropRate;
      tcEst = timesS[i];
    }
  }

  // Refine (f0, v, r, tc) over a compact local grid
  let bestLoss = Infinity;
  let bestParams = { f0: f0Est, v: vEst, r: 1500.0, tc: tcEst };

  for (const f0Cand of [f0Est * 0.96, f0Est, f0Est * 1.04]) {
    for (const vCand of [vEst * 0.85, vEst, vEst * 1.15, 100.0]) {
      for (const rCand of [900.0, 1500.0, 2200.0, 3200.0]) {
        for (const tcCand of [tcEst - 2.0, tcEst, tcEst + 2.0]) {
          const modeled = receiveDopplerLinear(
            timesS,
            f0Cand,
            vCand,
            rCand,
            tcCand,
            c
          );
          let loss = 0.0;
          for (let i = 0; i < timesS.length; i++) {
            const err = Math.abs(freqsHz[i] - modeled[i]);
            loss += weights[i] * err;
          }
          if (loss < bestLoss) {
            bestLoss = loss;
            bestParams = { f0: f0Cand, v: vCand, r: rCand, tc: tcCand };
          }
        }
      }
    }
  }

  const meanSnr =
    (weights.reduce((acc, v) => acc + v, 0) / weights.length) * 3.2;

  const tGrid: number[] = [];
  const tMin = Math.max(0, bestParams.tc - 80.0);
  const tMax = bestParams.tc + 80.0;
  for (let t = tMin; t <= tMax; t += 1.5) {
    tGrid.push(t);
  }

  const curves = [1, 2, 3].map((k) => {
    const fRetarded = receiveDopplerLinear(
      tGrid,
      bestParams.f0 * k,
      bestParams.v,
      bestParams.r,
      bestParams.tc,
      c
    );
    return {
      order: k,
      points: tGrid.map(
        (t, idx) =>
          [
            Number((t + timeEpochOffsetS).toFixed(1)),
            Number(fRetarded[idx].toFixed(2)),
          ] as [number, number]
      ),
    };
  });

  return {
    id: eventId,
    f0_hz: Number(bestParams.f0.toFixed(1)),
    speed_ms: Number(bestParams.v.toFixed(1)),
    slant_m: Number(bestParams.r.toFixed(0)),
    t_cpa_s: Number((bestParams.tc + timeEpochOffsetS).toFixed(1)),
    snr_db: Number(Math.min(58.0, Math.max(12.0, meanSnr)).toFixed(1)),
    orders: [1, 2, 3],
    curves,
    measuredSamples: timesS.map((t, i) => ({
      t: t + timeEpochOffsetS,
      f: freqsHz[i],
      w: weights[i],
    })),
  };
}

/**
 * Synthesize realistic multi-harmonic propeller flyover PCM audio buffer (at 8000 Hz)
 * for any set of Doppler events (used by the Demo Security Camera Video & Custom Synth).
 */
export function synthesizeScenarioAudio(
  events: {
    f0_hz: number;
    speed_ms: number;
    slant_m: number;
    t_cpa_s: number;
  }[],
  durationS: number = 40.0,
  sampleRate: number = 8000
): Float32Array {
  const n = Math.floor(durationS * sampleRate);
  const out = new Float32Array(n);

  // Deterministic pseudo-random ambient wind/tree noise
  let seed = 1337;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296 - 0.5;
  };

  let prevNoise = 0;
  for (let i = 0; i < n; i++) {
    const r = rand() * 0.04;
    prevNoise = 0.85 * prevNoise + 0.15 * r;
    out[i] = prevNoise;
  }

  const nEv = Math.max(1, events.length);
  events.forEach((ev, idx) => {
    // Map each event's CPA into the 40s window so all events are clearly visible in the spectrogram
    const localCpaS =
      nEv === 1
        ? durationS * 0.5
        : durationS * (0.28 + (idx / (nEv - 1)) * 0.44);
    const timeScale = 4.0; // 4x compressed time axis so a 160s flyover fits cleanly in 40s audio

    for (let k = 1; k <= 4; k++) {
      let phase = 0.0;
      const harmonicAmp = 0.25 * Math.pow(0.52, k - 1);
      for (let i = 0; i < n; i++) {
        const tSec = i / sampleRate;
        const dtPhys = (tSec - localCpaS) * timeScale;
        // Retarded-time frequency
        const v = ev.speed_ms;
        const r = ev.slant_m;
        const c = 340.0;
        const a = Math.min(0.9, v / c);
        const u = (v * dtPhys) / r;
        let ue = u;
        for (let it = 0; it < 6; it++) {
          const sq = Math.sqrt(1.0 + ue * ue);
          const g = ue + a * (sq - 1.0) - u;
          const gp = 1.0 + (a * ue) / sq;
          ue -= g / gp;
        }
        const vr = (v * ue) / Math.sqrt(1.0 + ue * ue);
        const fNow = (ev.f0_hz * k) / (1.0 + vr / c);
        if (fNow < sampleRate * 0.46) {
          phase += (2.0 * Math.PI * fNow) / sampleRate;
          const distNow = Math.sqrt(r * r + (v * dtPhys) ** 2);
          const env = r / Math.max(r, distNow);
          out[i] += harmonicAmp * env * Math.sin(phase);
        }
      }
    }
  });

  // Normalize
  let peak = 1e-6;
  for (let i = 0; i < n; i++) {
    if (Math.abs(out[i]) > peak) peak = Math.abs(out[i]);
  }
  const norm = 0.88 / peak;
  for (let i = 0; i < n; i++) {
    out[i] *= norm;
  }
  return out;
}
