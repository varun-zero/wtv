import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  FittedDopplerEvent,
  SpectrogramData,
  TracedRidge,
  computeEventAcousticMetrics,
  fitDopplerEventFromPoints,
} from '../engine/dsp';
import {
  naiveEmissionDopplerLinear,
  receiveDopplerLinear,
} from '../engine/acoustics';
import { Crosshair, Plus, Ruler, Scissors, Sparkles, Trash2 } from 'lucide-react';

export type SpectrogramToolMode =
  | 'inspect'
  | 'harmonic_comb'
  | 'caliper'
  | 'pick_points'
  | 'notch';

interface SpectrogramViewportProps {
  spectrogram: SpectrogramData | null;
  events: FittedDopplerEvent[];
  tracedRidges: TracedRidge[];
  currentTime: number;
  duration: number;
  onSeek: (timeS: number) => void;
  showNaiveComparison: boolean;
  onToggleNaiveComparison: () => void;
  showWhitened: boolean;
  onToggleWhitened: () => void;
  activeEventIndex: number;
  onSelectEventIndex: (idx: number) => void;
  highPass80HzActive: boolean;
  bandBoostDb: number;
  soundSpeedMs?: number;
  contrastGateZ?: number;
  notchFreqHz?: number;
  onChangeNotchFreqHz?: (freqHz: number) => void;
  rejectGroundTraffic?: boolean;
  onFitCustomEvent?: (fitted: FittedDopplerEvent) => void;
  onUpdateActiveF0?: (nextF0Hz: number) => void;
}

export const SpectrogramViewport: React.FC<SpectrogramViewportProps> = ({
  spectrogram,
  events,
  tracedRidges,
  currentTime,
  duration,
  onSeek,
  showNaiveComparison,
  onToggleNaiveComparison,
  showWhitened,
  onToggleWhitened,
  activeEventIndex,
  onSelectEventIndex,
  highPass80HzActive,
  bandBoostDb,
  soundSpeedMs = 340.0,
  contrastGateZ = 0.0,
  notchFreqHz = 0.0,
  onChangeNotchFreqHz,
  rejectGroundTraffic = true,
  onFitCustomEvent,
  onUpdateActiveF0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const spectrumCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [maxFreqHz, setMaxFreqHz] = useState<number>(360);
  const [toolMode, setToolMode] = useState<SpectrogramToolMode>('inspect');
  const [showHarmonics, setShowHarmonics] = useState<boolean>(true);
  const [hoverInfo, setHoverInfo] = useState<{
    timeS: number;
    freqHz: number;
    wavelengthM: number;
    periodMs: number;
    zScore: number;
    radialVelMs: number;
  } | null>(null);

  // Caliper Point A -> Point B state for Δt / Δf / λ measurement
  const [caliperPtA, setCaliperPtA] = useState<{ t: number; f: number } | null>(
    null
  );
  const [caliperPtB, setCaliperPtB] = useState<{ t: number; f: number } | null>(
    null
  );

  // Interactive Point-Pick Curve Fitter state
  const [pickedPoints, setPickedPoints] = useState<{ t: number; f: number }[]>(
    []
  );

  const activeEv =
    events[Math.min(activeEventIndex, Math.max(0, events.length - 1))];

  const activeEvMetrics = useMemo(() => {
    if (!activeEv) return null;
    return computeEventAcousticMetrics(activeEv, soundSpeedMs);
  }, [activeEv, soundSpeedMs]);

  const caliperMetrics = useMemo(() => {
    if (!caliperPtA || !caliperPtB) return null;
    const dt = Math.abs(caliperPtB.t - caliperPtA.t);
    const df = caliperPtB.f - caliperPtA.f;
    const slopeHzS = dt > 0.05 ? df / dt : 0;
    const fHigh = Math.max(caliperPtA.f, caliperPtB.f);
    const fLow = Math.max(5.0, Math.min(caliperPtA.f, caliperPtB.f));
    const ratio = fHigh / fLow;
    const vEst =
      ratio > 1.01 ? (soundSpeedMs * (ratio - 1.0)) / (ratio + 1.0) : 0.0;
    const fCenter = Math.sqrt(
      fHigh * fLow * Math.max(0.1, 1.0 - (vEst / soundSpeedMs) ** 2)
    );
    const wavelengthCenterM = soundSpeedMs / Math.max(5.0, fCenter);
    const rEst =
      Math.abs(slopeHzS) > 0.1 && vEst > 5.0
        ? (fCenter * vEst * vEst) / (soundSpeedMs * Math.abs(slopeHzS))
        : 1000.0;

    return {
      dt,
      df,
      slopeHzS,
      fCenter,
      wavelengthCenterM,
      vEst,
      rEst,
    };
  }, [caliperPtA, caliperPtB, soundSpeedMs]);

  // Draw Main STFT Spectrogram + Overlays + Interactive Tools
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, rect.width);
    const h = Math.max(180, rect.height);

    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const padLeft = 56;
    const padRight = 56;
    const padTop = 14;
    const padBottom = 26;
    const plotW = Math.max(100, w - padLeft - padRight);
    const plotH = Math.max(80, h - padTop - padBottom);

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = '#050505';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    const hasContent = Boolean(
      spectrogram || events.length > 0 || pickedPoints.length > 0 || caliperPtA
    );
    const tWindow = duration > 0 ? duration : 60.0;
    const tStartWindow = 0.0;

    if (hasContent && spectrogram) {
      const imgW = Math.min(520, Math.floor(plotW));
      const imgH = Math.min(280, Math.floor(plotH));
      const offscreen = document.createElement('canvas');
      offscreen.width = imgW;
      offscreen.height = imgH;
      const offCtx = offscreen.getContext('2d');

      if (offCtx) {
        const imgData = offCtx.createImageData(imgW, imgH);
        const data = imgData.data;

        for (let py = 0; py < imgH; py++) {
          const freq = maxFreqHz * (1.0 - py / imgH);
          const hpAtten =
            highPass80HzActive && freq < 82.0
              ? Math.pow(Math.max(0.05, freq / 82.0), 4)
              : 1.0;
          const notchAtten =
            notchFreqHz > 10.0 && Math.abs(freq - notchFreqHz) < 10.0
              ? Math.pow(Math.abs(freq - notchFreqHz) / 10.0, 2)
              : 1.0;
          const boostGain =
            bandBoostDb > 0.5 ? 1.0 + (bandBoostDb / 18.0) * 1.4 : 1.0;

          for (let px = 0; px < imgW; px++) {
            const uTime = px / imgW;
            let val = 0.0;

            if (spectrogram.nTime > 1 && spectrogram.nFreq > 1) {
              const tIdx = Math.min(
                spectrogram.nTime - 1,
                Math.floor(uTime * spectrogram.nTime)
              );
              const maxSpecFreq =
                spectrogram.freqs[spectrogram.nFreq - 1] || 500;
              const fIdx = Math.min(
                spectrogram.nFreq - 1,
                Math.max(
                  0,
                  Math.floor((freq / maxSpecFreq) * spectrogram.nFreq)
                )
              );
              const cellIdx = fIdx * spectrogram.nTime + tIdx;
              if (showWhitened) {
                const z = Math.max(
                  0.0,
                  spectrogram.whitened[cellIdx] - contrastGateZ
                );
                val = z * 0.16;
              } else {
                val = Math.log10(1.0 + spectrogram.raw[cellIdx] * 2.5) * 0.28;
              }
            }

            val = Math.max(
              0.0,
              Math.min(1.0, val * hpAtten * notchAtten * boostGain)
            );

            const idx = (py * imgW + px) * 4;
            if (val < 0.25) {
              const t = val / 0.25;
              data[idx] = Math.round(5 + t * 8);
              data[idx + 1] = Math.round(5 + t * 28);
              data[idx + 2] = Math.round(5 + t * 20);
            } else if (val < 0.65) {
              const t = (val - 0.25) / 0.4;
              data[idx] = Math.round(13 + t * 3);
              data[idx + 1] = Math.round(33 + t * 152);
              data[idx + 2] = Math.round(25 + t * 104);
            } else {
              const t = (val - 0.65) / 0.35;
              data[idx] = Math.round(16 + t * 229);
              data[idx + 1] = Math.round(185 + t * 65);
              data[idx + 2] = Math.round(129 + t * 110);
            }
            data[idx + 3] = 255;
          }
        }
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, padLeft, padTop, plotW, plotH);
      }
    }

    // Draw grid lines (Frequency Y-axis on Left, Wavelength λ on Right, & Time X-axis)
    ctx.strokeStyle = '#1f1f22';
    ctx.lineWidth = 1;
    const freqStep = maxFreqHz <= 300 ? 50 : maxFreqHz <= 500 ? 75 : 100;
    for (let f = 0; f <= maxFreqHz; f += freqStep) {
      const y = padTop + plotH - (f / maxFreqHz) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#737373';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${f} Hz`, padLeft - 6, y + 3);

      // Right-hand axis: exact acoustic wavelength λ = c / f
      if (f > 0) {
        const lambdaM = soundSpeedMs / f;
        ctx.textAlign = 'left';
        ctx.fillStyle = '#52525b';
        ctx.fillText(`λ=${lambdaM.toFixed(2)}m`, padLeft + plotW + 5, y + 3);
      }
    }

    const timeStep = Math.max(5, Math.round(tWindow / 8));
    for (let tRel = 0; tRel <= tWindow; tRel += timeStep) {
      const x = padLeft + (tRel / tWindow) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.fillStyle = '#737373';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${Math.round(tStartWindow + tRel)}s`, x, padTop + plotH + 16);
    }

    if (highPass80HzActive) {
      const y80 = padTop + plotH - (80.0 / maxFreqHz) * plotH;
      if (y80 >= padTop && y80 <= padTop + plotH) {
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(padLeft, y80);
        ctx.lineTo(padLeft + plotW, y80);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#f43f5e';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText('80 Hz HPF', padLeft + 8, y80 - 5);
      }
    }

    // Surgical Notch Filter Line
    if (notchFreqHz > 10.0) {
      const yNotch = padTop + plotH - (notchFreqHz / maxFreqHz) * plotH;
      if (yNotch >= padTop && yNotch <= padTop + plotH) {
        ctx.fillStyle = 'rgba(244, 63, 94, 0.14)';
        ctx.fillRect(padLeft, yNotch - 6, plotW, 12);
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(padLeft, yNotch);
        ctx.lineTo(padLeft + plotW, yNotch);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#f43f5e';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(
          `NOTCH ${notchFreqHz.toFixed(1)} Hz (λ=${(
            soundSpeedMs / notchFreqHz
          ).toFixed(2)}m)`,
          padLeft + 8,
          yNotch - 5
        );
      }
    }

    // Draw Traced Viterbi Ridge Samples if present
    tracedRidges.forEach((ridge) => {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.85)';
      const rDur = Math.max(1.0, tWindow);
      for (let i = 0; i < ridge.timesS.length; i++) {
        const px = padLeft + (ridge.timesS[i] / rDur) * plotW;
        const py = padTop + plotH - (ridge.freqsHz[i] / maxFreqHz) * plotH;
        if (py >= padTop && py <= padTop + plotH) {
          ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
        }
      }
    });

    // Draw all Doppler Events (highlighting active curve & flagging ground traffic/car curves)
    events.forEach((ev, evIdx) => {
      const isSelected = evIdx === activeEventIndex;
      const evMetrics = computeEventAcousticMetrics(ev, soundSpeedMs);
      const isTrafficCar =
        ev.rejectedAsTraffic === true || evMetrics.sourceClass !== 'aircraft';

      const f0 = ev.f0_hz;
      const v = ev.speed_ms;
      const r = ev.slant_m;
      const tc = ev.t_cpa_s;

      const tSamples: number[] = [];
      for (let s = 0; s <= 120; s++) {
        tSamples.push(tStartWindow + (s / 120) * tWindow);
      }

      const harmonicOrders =
        isSelected && showHarmonics && !isTrafficCar ? [1, 2, 3] : [1];
      const harmonicColors: Record<number, string> = {
        1: isTrafficCar ? '#f43f5e' : isSelected ? '#10b981' : '#52525b',
        2: '#f59e0b',
        3: '#fafafa',
      };

      harmonicOrders.forEach((k) => {
        const retardedFreqs = receiveDopplerLinear(
          tSamples,
          f0 * k,
          v,
          r,
          tc,
          soundSpeedMs
        );

        ctx.strokeStyle = harmonicColors[k] || '#10b981';
        ctx.lineWidth = isSelected ? (k === 1 ? 2.0 : 1.4) : 1.2;
        if (isTrafficCar && rejectGroundTraffic) {
          ctx.setLineDash([4, 3]);
        }
        ctx.beginPath();
        let started = false;
        tSamples.forEach((tVal, idx) => {
          const fVal = retardedFreqs[idx];
          const x = padLeft + ((tVal - tStartWindow) / tWindow) * plotW;
          const y = padTop + plotH - (fVal / maxFreqHz) * plotH;
          if (y >= padTop && y <= padTop + plotH) {
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);

        if (isSelected) {
          const lastF = retardedFreqs[retardedFreqs.length - 1];
          const lastY = padTop + plotH - (lastF / maxFreqHz) * plotH;
          if (lastY >= padTop + 10 && lastY <= padTop + plotH - 6) {
            ctx.fillStyle = harmonicColors[k] || '#10b981';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            const labelText = isTrafficCar
              ? `${ev.id} [FILTERED CAR/TRAFFIC · r₀=${r.toFixed(0)}m, v=${v.toFixed(0)}m/s]`
              : `k=${k} (${(f0 * k).toFixed(0)}Hz · λ=${(
                  soundSpeedMs /
                  (f0 * k)
                ).toFixed(2)}m)`;
            ctx.fillText(labelText, padLeft + plotW - 6, lastY - 5);
          }
        }
      });

      if (isSelected && showNaiveComparison) {
        const naiveFreqs = naiveEmissionDopplerLinear(
          tSamples,
          f0,
          v,
          r,
          tc,
          soundSpeedMs
        );
        ctx.strokeStyle = '#ff4f00';
        ctx.lineWidth = 1.8;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        let started = false;
        tSamples.forEach((tVal, idx) => {
          const fVal = naiveFreqs[idx];
          const x = padLeft + ((tVal - tStartWindow) / tWindow) * plotW;
          const y = padTop + plotH - (fVal / maxFreqHz) * plotH;
          if (y >= padTop && y <= padTop + plotH) {
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (isSelected) {
        const cpaX = padLeft + ((tc - tStartWindow) / tWindow) * plotW;
        if (cpaX >= padLeft && cpaX <= padLeft + plotW) {
          ctx.strokeStyle = isTrafficCar
            ? 'rgba(244, 63, 94, 0.7)'
            : 'rgba(16, 185, 129, 0.7)';
          ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(cpaX, padTop);
          ctx.lineTo(cpaX, padTop + plotH);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = isTrafficCar ? '#f43f5e' : '#10b981';
          ctx.font = '10px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(
            `${isTrafficCar ? 'CAR CPA' : 'CPA'} ${tc.toFixed(1)}s`,
            cpaX,
            padTop + 12
          );
        }
      }
    });

    // Draw Interactive Caliper (A -> B) if active
    if (caliperPtA) {
      const ax = padLeft + (caliperPtA.t / tWindow) * plotW;
      const ay = padTop + plotH - (caliperPtA.f / maxFreqHz) * plotH;
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(ax - 3, ay - 3, 6, 6);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        `A (${caliperPtA.t.toFixed(1)}s, ${caliperPtA.f.toFixed(1)}Hz)`,
        ax + 6,
        ay - 4
      );

      if (caliperPtB) {
        const bx = padLeft + (caliperPtB.t / tWindow) * plotW;
        const by = padTop + plotH - (caliperPtB.f / maxFreqHz) * plotH;
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(bx - 3, by - 3, 6, 6);
        ctx.fillText(
          `B (${caliperPtB.t.toFixed(1)}s, ${caliperPtB.f.toFixed(1)}Hz)`,
          bx + 6,
          by - 4
        );
      }
    }

    // Draw Interactive Picked Points if active
    if (pickedPoints.length > 0) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      pickedPoints.forEach((pt, idx) => {
        const px = padLeft + (pt.t / tWindow) * plotW;
        const py = padTop + plotH - (pt.f / maxFreqHz) * plotH;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      pickedPoints.forEach((pt, idx) => {
        const px = padLeft + (pt.t / tWindow) * plotW;
        const py = padTop + plotH - (pt.f / maxFreqHz) * plotH;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(px - 3, py - 3, 6, 6);
        ctx.fillStyle = '#10b981';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`P${idx + 1}`, px + 5, py - 3);
      });
    }

    // Draw Live Harmonic Comb Cursor when in 'harmonic_comb' mode
    if (toolMode === 'harmonic_comb' && hoverInfo) {
      const fBase = Math.max(10.0, hoverInfo.freqHz);
      for (let k = 1; k <= 5; k++) {
        const fk = fBase * k;
        if (fk > maxFreqHz) break;
        const yk = padTop + plotH - (fk / maxFreqHz) * plotH;
        ctx.strokeStyle = k === 1 ? '#10b981' : 'rgba(245, 158, 11, 0.75)';
        ctx.lineWidth = k === 1 ? 1.5 : 1.0;
        ctx.setLineDash(k === 1 ? [] : [3, 3]);
        ctx.beginPath();
        ctx.moveTo(padLeft, yk);
        ctx.lineTo(padLeft + plotW, yk);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = k === 1 ? '#10b981' : '#f59e0b';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(
          `${k}f₀ = ${fk.toFixed(1)} Hz (λ=${(soundSpeedMs / fk).toFixed(2)}m)`,
          padLeft + 8,
          yk - 4
        );
      }
    }

    // Playhead Scrubber Line
    if (hasContent && tWindow > 0) {
      const playX = padLeft + (currentTime / tWindow) * plotW;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playX, padTop);
      ctx.lineTo(playX, padTop + plotH);
      ctx.stroke();
    }

    ctx.strokeStyle = '#262626';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    if (!hasContent) {
      ctx.fillStyle = '#737373';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        'NO ACOUSTIC TELEMETRY LOADED — UPLOAD MEDIA, PICK POINTS, OR ADD DOPPLER CURVE',
        padLeft + plotW * 0.5,
        padTop + plotH * 0.5
      );
    }

    ctx.restore();
  }, [
    spectrogram,
    events,
    tracedRidges,
    currentTime,
    duration,
    maxFreqHz,
    showNaiveComparison,
    showWhitened,
    showHarmonics,
    activeEventIndex,
    highPass80HzActive,
    bandBoostDb,
    soundSpeedMs,
    contrastGateZ,
    notchFreqHz,
    rejectGroundTraffic,
    toolMode,
    hoverInfo,
    caliperPtA,
    caliperPtB,
    pickedPoints,
  ]);

  // Draw Bottom Instantaneous FFT Spectrum Slice & Wavelength Bar
  useEffect(() => {
    const canvas = spectrumCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, rect.width);
    const h = Math.max(48, rect.height);

    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 56;
    const padRight = 56;
    const plotW = Math.max(100, w - padLeft - padRight);
    const plotH = h - 16;

    // Grid ticks along frequency axis (0 .. maxFreqHz)
    ctx.strokeStyle = '#1f1f22';
    ctx.lineWidth = 1;
    for (let f = 0; f <= maxFreqHz; f += 60) {
      const x = padLeft + (f / maxFreqHz) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, 4 + plotH);
      ctx.stroke();

      ctx.fillStyle = '#737373';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${f}Hz (${f > 0 ? (soundSpeedMs / f).toFixed(1) + 'm' : '∞'})`,
        x,
        h - 3
      );
    }

    // Plot instantaneous spectrum slice at hover time or playhead time
    const targetTime = hoverInfo ? hoverInfo.timeS : currentTime;
    const tWindow = duration > 0 ? duration : 60.0;

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i <= 180; i++) {
      const freq = (i / 180) * maxFreqHz;
      let mag = 0.08;

      if (spectrogram && spectrogram.nTime > 0 && spectrogram.nFreq > 0) {
        const uT = Math.max(0, Math.min(1, targetTime / tWindow));
        const tIdx = Math.min(
          spectrogram.nTime - 1,
          Math.floor(uT * spectrogram.nTime)
        );
        const maxSpecF = spectrogram.freqs[spectrogram.nFreq - 1] || 500;
        const fIdx = Math.min(
          spectrogram.nFreq - 1,
          Math.floor((freq / maxSpecF) * spectrogram.nFreq)
        );
        const z = spectrogram.whitened[fIdx * spectrogram.nTime + tIdx] || 0;
        mag = Math.min(1.0, Math.max(0.05, z / 8.0));
      } else if (activeEv) {
        // Synthesize theoretical harmonic peaks at targetTime for active Doppler curve
        for (const k of [1, 2, 3]) {
          const fInst = receiveDopplerLinear(
            [targetTime],
            activeEv.f0_hz * k,
            activeEv.speed_ms,
            activeEv.slant_m,
            activeEv.t_cpa_s,
            soundSpeedMs
          )[0];
          const distHz = Math.abs(freq - fInst);
          mag += (0.85 / k) * Math.exp(-0.5 * Math.pow(distHz / 3.5, 2));
        }
        mag = Math.min(1.0, mag);
      }

      const x = padLeft + (freq / maxFreqHz) * plotW;
      const y = 4 + plotH - mag * (plotH - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.restore();
  }, [
    spectrogram,
    activeEv,
    hoverInfo,
    currentTime,
    duration,
    maxFreqHz,
    soundSpeedMs,
  ]);

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const padLeft = 56;
    const padRight = 56;
    const padTop = 14;
    const padBottom = 26;
    const plotW = Math.max(100, rect.width - padLeft - padRight);
    const plotH = Math.max(80, rect.height - padTop - padBottom);

    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    if (
      mx < padLeft ||
      mx > padLeft + plotW ||
      my < padTop ||
      my > padTop + plotH
    ) {
      return null;
    }

    const uX = (mx - padLeft) / plotW;
    const uY = 1.0 - (my - padTop) / plotH;
    const tWindow = duration > 0 ? duration : 60.0;
    return {
      uX,
      uY,
      timeS: Number((uX * tWindow).toFixed(2)),
      freqHz: Number(Math.max(5.0, uY * maxFreqHz).toFixed(1)),
    };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = getCanvasCoords(e);
    if (!pt) {
      setHoverInfo(null);
      return;
    }

    const { uX, timeS, freqHz } = pt;
    const f0 = activeEv ? activeEv.f0_hz : 85.0;
    const radialVelMs = soundSpeedMs * (f0 / Math.max(15.0, freqHz) - 1.0);
    const wavelengthM = soundSpeedMs / Math.max(5.0, freqHz);
    const periodMs = 1000.0 / Math.max(5.0, freqHz);

    let zScore = 0.0;
    if (spectrogram && spectrogram.nTime > 0 && spectrogram.nFreq > 0) {
      const tIdx = Math.min(
        spectrogram.nTime - 1,
        Math.floor(uX * spectrogram.nTime)
      );
      const fIdx = Math.min(
        spectrogram.nFreq - 1,
        Math.floor(
          (freqHz / (spectrogram.freqs[spectrogram.nFreq - 1] || 500)) *
            spectrogram.nFreq
        )
      );
      zScore = spectrogram.whitened[fIdx * spectrogram.nTime + tIdx] || 0.0;
    }

    setHoverInfo({
      timeS,
      freqHz,
      wavelengthM,
      periodMs,
      zScore,
      radialVelMs,
    });
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = getCanvasCoords(e);
    if (!pt) return;

    if (toolMode === 'caliper') {
      if (!caliperPtA || (caliperPtA && caliperPtB)) {
        setCaliperPtA({ t: pt.timeS, f: pt.freqHz });
        setCaliperPtB(null);
      } else {
        setCaliperPtB({ t: pt.timeS, f: pt.freqHz });
      }
      return;
    }

    if (toolMode === 'pick_points') {
      setPickedPoints((prev) => [...prev, { t: pt.timeS, f: pt.freqHz }]);
      return;
    }

    if (toolMode === 'notch') {
      if (onChangeNotchFreqHz) {
        onChangeNotchFreqHz(pt.freqHz);
      }
      return;
    }

    if (toolMode === 'harmonic_comb') {
      if (onUpdateActiveF0) {
        onUpdateActiveF0(pt.freqHz);
      }
      return;
    }

    if (duration > 0) {
      onSeek(pt.timeS);
    }
  };

  const handleFitFromPickedPoints = () => {
    if (pickedPoints.length < 2 || !onFitCustomEvent) return;
    const fitted = fitDopplerEventFromPoints(
      pickedPoints,
      `EV0${events.length + 1}`,
      soundSpeedMs
    );
    if (fitted) {
      onFitCustomEvent(fitted);
      setPickedPoints([]);
    }
  };

  const handleFitFromCaliper = () => {
    if (!caliperPtA || !caliperPtB || !onFitCustomEvent) return;
    const midT = 0.5 * (caliperPtA.t + caliperPtB.t);
    const midF = Math.sqrt(caliperPtA.f * caliperPtB.f);
    const fitted = fitDopplerEventFromPoints(
      [caliperPtA, { t: midT, f: midF }, caliperPtB],
      `EV0${events.length + 1}`,
      soundSpeedMs
    );
    if (fitted) {
      onFitCustomEvent(fitted);
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#000000] relative overflow-hidden">
      {/* Row 1 Toolbar: Event Selector + Display Overlays */}
      <div className="h-10 px-4 bg-[#0A0A0A] border-b border-[#1F1F1F] flex items-center justify-between gap-2 shrink-0 overflow-x-auto">
        <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626] shrink-0">
          {events.map((ev, idx) => {
            const m = computeEventAcousticMetrics(ev, soundSpeedMs);
            const isCar =
              ev.rejectedAsTraffic === true || m.sourceClass !== 'aircraft';
            return (
              <button
                key={ev.id}
                type="button"
                onClick={() => onSelectEventIndex(idx)}
                className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  activeEventIndex === idx
                    ? 'bg-[#18181B] text-[#FAFAFA]'
                    : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
                }`}
              >
                <span
                  className="w-1.5 h-1.5 shrink-0"
                  style={{
                    backgroundColor: isCar
                      ? '#f43f5e'
                      : activeEventIndex === idx
                      ? '#10b981'
                      : '#3f3f46',
                  }}
                />
                <span>
                  {ev.id} ({ev.f0_hz.toFixed(1)}Hz · λ=
                  {m.wavelength0_m.toFixed(2)}m{isCar ? ' · CAR' : ''})
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Interactive Spectrogram Analysis Mode Selector */}
          <div
            data-testid="spectrogram-tool-selector"
            className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]"
          >
            <button
              type="button"
              data-testid="spec-tool-inspect"
              onClick={() => setToolMode('inspect')}
              className={`h-7 px-2 text-[10px] font-mono uppercase tracking-wider rounded-none flex items-center gap-1 cursor-pointer ${
                toolMode === 'inspect'
                  ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Inspect frequency, wavelength (λ), and radial velocity"
            >
              <Crosshair className="w-3 h-3" />
              Inspect
            </button>
            <button
              type="button"
              data-testid="spec-tool-harmonic"
              onClick={() => setToolMode('harmonic_comb')}
              className={`h-7 px-2 text-[10px] font-mono uppercase tracking-wider rounded-none flex items-center gap-1 cursor-pointer ${
                toolMode === 'harmonic_comb'
                  ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Harmonic Comb Caliper (1f0..5f0 & Wavelength)"
            >
              <Sparkles className="w-3 h-3" />
              Harmonic Comb
            </button>
            <button
              type="button"
              data-testid="spec-tool-caliper"
              onClick={() => setToolMode('caliper')}
              className={`h-7 px-2 text-[10px] font-mono uppercase tracking-wider rounded-none flex items-center gap-1 cursor-pointer ${
                toolMode === 'caliper'
                  ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Click two points A and B to measure Δt, Δf, slope, wavelength λ, and speed"
            >
              <Ruler className="w-3 h-3" />
              Caliper Δf/λ
            </button>
            <button
              type="button"
              data-testid="spec-tool-pick"
              onClick={() => setToolMode('pick_points')}
              className={`h-7 px-2 text-[10px] font-mono uppercase tracking-wider rounded-none flex items-center gap-1 cursor-pointer ${
                toolMode === 'pick_points'
                  ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Click points along a trace on the spectrogram to fit a Doppler curve"
            >
              <Plus className="w-3 h-3" />
              Pick Curve
            </button>
            <button
              type="button"
              data-testid="spec-tool-notch"
              onClick={() => setToolMode('notch')}
              className={`h-7 px-2 text-[10px] font-mono uppercase tracking-wider rounded-none flex items-center gap-1 cursor-pointer ${
                toolMode === 'notch'
                  ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Click a frequency band to place a surgical band-stop notch filter"
            >
              <Scissors className="w-3 h-3" />
              Notch
            </button>
          </div>

          {/* Flush Monochrome Segmented Toggle Group with 6px × 6px Square Status LEDs */}
          <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]">
            <button
              type="button"
              onClick={onToggleWhitened}
              aria-pressed={showWhitened}
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showWhitened
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: showWhitened ? '#10b981' : '#3f3f46',
                }}
              />
              <span>{showWhitened ? 'Whitened' : 'Raw STFT'}</span>
            </button>

            <button
              type="button"
              onClick={() => setShowHarmonics((v) => !v)}
              aria-pressed={showHarmonics}
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showHarmonics
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: showHarmonics ? '#f59e0b' : '#3f3f46',
                }}
              />
              <span>Harmonics</span>
            </button>

            <button
              type="button"
              onClick={onToggleNaiveComparison}
              aria-pressed={showNaiveComparison}
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showNaiveComparison
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: showNaiveComparison ? '#ff4f00' : '#3f3f46',
                }}
              />
              <span>Naive Curve</span>
            </button>
          </div>

          <select
            value={maxFreqHz}
            onChange={(e) => setMaxFreqHz(Number(e.target.value))}
            aria-label="Maximum frequency"
            className="h-7 bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] text-[11px] font-mono border border-[#262626] rounded-none px-2 outline-none cursor-pointer"
          >
            <option value={240}>0–240 Hz</option>
            <option value={360}>0–360 Hz</option>
            <option value={500}>0–500 Hz</option>
          </select>
        </div>
      </div>

      {/* Contextual Action Bar for Caliper / Pick Points / Notch */}
      {(caliperMetrics || pickedPoints.length > 0 || notchFreqHz > 10) && (
        <div className="h-8 px-4 bg-[#0A0A0A] border-b border-[#1F1F1F] flex items-center justify-between text-[11px] font-mono tabular-nums text-[#FAFAFA] shrink-0">
          {caliperMetrics && (
            <div className="flex items-center gap-3">
              <span className="text-[#F59E0B] font-semibold">
                CALIPER A→B:
              </span>
              <span>Δt={caliperMetrics.dt.toFixed(2)}s</span>
              <span>Δf={caliperMetrics.df.toFixed(1)}Hz</span>
              <span>df/dt={caliperMetrics.slopeHzS.toFixed(2)}Hz/s</span>
              <span>
                f₀≈{caliperMetrics.fCenter.toFixed(1)}Hz (λ≈
                {caliperMetrics.wavelengthCenterM.toFixed(2)}m)
              </span>
              <span>v≈{caliperMetrics.vEst.toFixed(0)}m/s</span>
              <span>r₀≈{caliperMetrics.rEst.toFixed(0)}m</span>
              <button
                type="button"
                onClick={handleFitFromCaliper}
                className="h-6 px-2 bg-[#FFFFFF] text-[#000000] font-medium rounded-none cursor-pointer"
              >
                + Fit Curve from Caliper
              </button>
              <button
                type="button"
                onClick={() => {
                  setCaliperPtA(null);
                  setCaliperPtB(null);
                }}
                className="h-6 px-2 bg-[#111111] border border-[#262626] text-[#A1A1AA] hover:text-[#FAFAFA] rounded-none cursor-pointer"
              >
                Clear
              </button>
            </div>
          )}

          {pickedPoints.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-[#10B981] font-semibold">
                PICKED TRACE POINTS ({pickedPoints.length}):
              </span>
              <button
                type="button"
                disabled={pickedPoints.length < 2}
                onClick={handleFitFromPickedPoints}
                className="h-6 px-2 bg-[#FFFFFF] disabled:opacity-40 text-[#000000] font-medium rounded-none cursor-pointer"
              >
                Fit Doppler Curve ({pickedPoints.length} pts)
              </button>
              <button
                type="button"
                onClick={() => setPickedPoints([])}
                className="h-6 px-2 bg-[#111111] border border-[#262626] text-[#A1A1AA] hover:text-[#F43F5E] rounded-none flex items-center gap-1 cursor-pointer"
              >
                <Trash2 className="w-3 h-3" />
                Clear Points
              </button>
            </div>
          )}

          {notchFreqHz > 10 && !caliperMetrics && pickedPoints.length === 0 && (
            <div className="flex items-center gap-3">
              <span className="text-[#F43F5E] font-semibold">
                ACTIVE NOTCH FILTER: {notchFreqHz.toFixed(1)} Hz (λ=
                {(soundSpeedMs / notchFreqHz).toFixed(2)} m)
              </span>
              {onChangeNotchFreqHz && (
                <button
                  type="button"
                  onClick={() => onChangeNotchFreqHz(0)}
                  className="h-6 px-2 bg-[#111111] border border-[#262626] text-[#A1A1AA] hover:text-[#FAFAFA] rounded-none cursor-pointer"
                >
                  Disable Notch
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Main STFT Canvas */}
      <div className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          data-testid="spectrogram-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverInfo(null)}
          onClick={handleCanvasClick}
          className="w-full h-full block cursor-crosshair"
        />

        {hoverInfo && (
          <div
            data-testid="spectrogram-hover-readout"
            className="absolute bottom-3 left-16 bg-[#0A0A0A] border border-[#262626] px-2.5 py-1 rounded-none text-[11px] font-mono tabular-nums text-[#FAFAFA] pointer-events-none flex items-center gap-2.5"
          >
            <span>t={hoverInfo.timeS.toFixed(2)}s</span>
            <span className="text-[#737373]">·</span>
            <span className="text-[#10B981]">
              f={hoverInfo.freqHz.toFixed(1)} Hz
            </span>
            <span className="text-[#737373]">·</span>
            <span className="text-[#F59E0B]">
              λ={hoverInfo.wavelengthM.toFixed(3)} m
            </span>
            <span className="text-[#737373]">·</span>
            <span>T={hoverInfo.periodMs.toFixed(1)} ms</span>
            <span className="text-[#737373]">·</span>
            <span>z={hoverInfo.zScore.toFixed(1)}σ</span>
            <span className="text-[#737373]">·</span>
            <span>
              v_r={hoverInfo.radialVelMs >= 0 ? '+' : ''}
              {hoverInfo.radialVelMs.toFixed(1)} m/s
            </span>
          </div>
        )}
      </div>

      {/* Bottom Real-Time FFT Spectrum Slice & Wavelength / Traffic Analyzer Bar */}
      <div
        data-testid="wavelength-analyzer-bar"
        className="h-24 bg-[#0A0A0A] border-t border-[#1F1F1F] flex flex-col shrink-0"
      >
        <div className="h-7 px-4 border-b border-[#1F1F1F] flex items-center justify-between text-[10px] font-mono tabular-nums">
          <div className="flex items-center gap-4">
            <span className="uppercase tracking-[0.1em] text-[#737373] font-semibold">
              SPECTRUM & WAVELENGTH ANALYZER (c={soundSpeedMs.toFixed(1)} m/s)
            </span>
            {activeEv && activeEvMetrics && (
              <>
                <span className="text-[#A1A1AA]">
                  FUNDAMENTAL:{' '}
                  <strong className="text-[#FAFAFA]">
                    f₀={activeEv.f0_hz.toFixed(1)} Hz
                  </strong>
                </span>
                <span className="text-[#A1A1AA]">
                  WAVELENGTH:{' '}
                  <strong className="text-[#10B981]">
                    λ₀={activeEvMetrics.wavelength0_m.toFixed(3)} m
                  </strong>{' '}
                  (λ_app={activeEvMetrics.wavelengthApp_m.toFixed(2)}m / λ_rec=
                  {activeEvMetrics.wavelengthRec_m.toFixed(2)}m)
                </span>
                <span className="text-[#A1A1AA]">
                  DOPPLER SPAN:{' '}
                  <strong className="text-[#FAFAFA]">
                    Δf={activeEvMetrics.dopplerShiftSpan_hz.toFixed(1)} Hz
                  </strong>
                </span>
                <span className="text-[#A1A1AA]">
                  PROP RPM (2B/3B):{' '}
                  <strong className="text-[#FAFAFA]">
                    {activeEvMetrics.rpm2Blade.toFixed(0)} /{' '}
                    {activeEvMetrics.rpm3Blade.toFixed(0)}
                  </strong>
                </span>
              </>
            )}
          </div>

          {activeEv && activeEvMetrics && (
            <div
              data-testid="source-classification-badge"
              className="flex items-center gap-1.5"
            >
              <span
                className="w-1.5 h-1.5"
                style={{
                  backgroundColor:
                    activeEvMetrics.sourceClass === 'aircraft' &&
                    !activeEv.rejectedAsTraffic
                      ? '#10b981'
                      : '#f43f5e',
                }}
              />
              <span
                className={
                  activeEvMetrics.sourceClass === 'aircraft' &&
                  !activeEv.rejectedAsTraffic
                    ? 'text-[#10B981] font-semibold'
                    : 'text-[#F43F5E] font-semibold'
                }
              >
                {activeEvMetrics.sourceClass === 'aircraft' &&
                !activeEv.rejectedAsTraffic
                  ? 'SOURCE: AIRBORNE PROPELLER AIRCRAFT'
                  : 'SOURCE: GROUND TRAFFIC / CAR PASS-BY (FILTERED)'}
              </span>
            </div>
          )}
        </div>

        <div className="flex-1 relative min-h-0">
          <canvas
            ref={spectrumCanvasRef}
            className="w-full h-full block"
          />
        </div>
      </div>
    </div>
  );
};
