import React, { useEffect, useRef, useState } from 'react';
import {
  FittedDopplerEvent,
  SpectrogramData,
  TracedRidge,
} from '../engine/dsp';
import {
  naiveEmissionDopplerLinear,
  receiveDopplerLinear,
} from '../engine/acoustics';

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
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [maxFreqHz, setMaxFreqHz] = useState<number>(360);
  const [hoverInfo, setHoverInfo] = useState<{
    timeS: number;
    freqHz: number;
    zScore: number;
    radialVelMs: number;
  } | null>(null);
  const [showHarmonics, setShowHarmonics] = useState<boolean>(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, rect.width);
    const h = Math.max(200, rect.height);

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

    const padLeft = 52;
    const padRight = 18;
    const padTop = 14;
    const padBottom = 28;
    const plotW = Math.max(100, w - padLeft - padRight);
    const plotH = Math.max(80, h - padTop - padBottom);

    ctx.fillStyle = '#05080f';
    ctx.fillRect(0, 0, w, h);

    const hasContent = Boolean(spectrogram || events.length > 0);
    const tWindow = duration > 0 ? duration : 60.0;
    const tStartWindow = 0.0;
    const activeEv =
      events[Math.min(activeEventIndex, Math.max(0, events.length - 1))];

    if (hasContent) {
      const imgW = Math.min(480, Math.floor(plotW));
      const imgH = Math.min(260, Math.floor(plotH));
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
          const boostGain =
            bandBoostDb > 0.5 ? 1.0 + (bandBoostDb / 18.0) * 1.4 : 1.0;

          for (let px = 0; px < imgW; px++) {
            const uTime = px / imgW;
            let val = 0.0;

            if (spectrogram && spectrogram.nTime > 1 && spectrogram.nFreq > 1) {
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
                val = spectrogram.whitened[cellIdx] * 0.16;
              } else {
                val = Math.log10(1.0 + spectrogram.raw[cellIdx] * 2.5) * 0.28;
              }
            }

            val = Math.max(0.0, Math.min(1.0, val * hpAtten * boostGain));

            const idx = (py * imgW + px) * 4;
            if (val < 0.25) {
              const t = val / 0.25;
              data[idx] = Math.round(6 + t * 14);
              data[idx + 1] = Math.round(10 + t * 36);
              data[idx + 2] = Math.round(20 + t * 68);
            } else if (val < 0.6) {
              const t = (val - 0.25) / 0.35;
              data[idx] = Math.round(20 + t * 18);
              data[idx + 1] = Math.round(46 + t * 128);
              data[idx + 2] = Math.round(88 + t * 135);
            } else {
              const t = (val - 0.6) / 0.4;
              data[idx] = Math.round(38 + t * 212);
              data[idx + 1] = Math.round(174 + t * 72);
              data[idx + 2] = Math.round(223 - t * 90);
            }
            data[idx + 3] = 255;
          }
        }
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, padLeft, padTop, plotW, plotH);
      }
    }

    // Draw grid lines (Frequency Y-axis & Time X-axis)
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
    ctx.lineWidth = 1;
    const freqStep = maxFreqHz <= 300 ? 50 : maxFreqHz <= 500 ? 75 : 100;
    for (let f = 0; f <= maxFreqHz; f += freqStep) {
      const y = padTop + plotH - (f / maxFreqHz) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${f} Hz`, padLeft - 6, y + 3);
    }

    const timeStep = Math.max(5, Math.round(tWindow / 8));
    for (let tRel = 0; tRel <= tWindow; tRel += timeStep) {
      const x = padLeft + (tRel / tWindow) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
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

    // Draw Traced Viterbi Ridge Samples if present
    tracedRidges.forEach((ridge) => {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.75)';
      const rDur = Math.max(1.0, tWindow);
      for (let i = 0; i < ridge.timesS.length; i++) {
        const px = padLeft + (ridge.timesS[i] / rDur) * plotW;
        const py = padTop + plotH - (ridge.freqsHz[i] / maxFreqHz) * plotH;
        if (py >= padTop && py <= padTop + plotH) {
          ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
        }
      }
    });

    // Draw Retarded-Time Doppler Curves for active event
    if (activeEv) {
      const f0 = activeEv.f0_hz;
      const v = activeEv.speed_ms;
      const r = activeEv.slant_m;
      const tc = activeEv.t_cpa_s;

      const tSamples: number[] = [];
      for (let s = 0; s <= 120; s++) {
        tSamples.push(tStartWindow + (s / 120) * tWindow);
      }

      const harmonicOrders = showHarmonics ? [1, 2, 3] : [1];
      const harmonicColors: Record<number, string> = {
        1: '#38bdf8',
        2: '#a855f7',
        3: '#ec4899',
      };

      harmonicOrders.forEach((k) => {
        const retardedFreqs = receiveDopplerLinear(
          tSamples,
          f0 * k,
          v,
          r,
          tc,
          340.0
        );

        ctx.strokeStyle = harmonicColors[k] || '#38bdf8';
        ctx.lineWidth = k === 1 ? 2.2 : 1.6;
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

        const lastF = retardedFreqs[retardedFreqs.length - 1];
        const lastY = padTop + plotH - (lastF / maxFreqHz) * plotH;
        if (lastY >= padTop + 10 && lastY <= padTop + plotH - 6) {
          ctx.fillStyle = harmonicColors[k] || '#38bdf8';
          ctx.font = '10px "JetBrains Mono", monospace';
          ctx.textAlign = 'right';
          ctx.fillText(
            `k=${k} (${(f0 * k).toFixed(0)}Hz)`,
            padLeft + plotW - 6,
            lastY - 5
          );
        }
      });

      if (showNaiveComparison) {
        const naiveFreqs = naiveEmissionDopplerLinear(
          tSamples,
          f0,
          v,
          r,
          tc,
          340.0
        );
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.0;
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

      // CPA vertical line
      const cpaX = padLeft + ((tc - tStartWindow) / tWindow) * plotW;
      if (cpaX >= padLeft && cpaX <= padLeft + plotW) {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.65)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cpaX, padTop);
        ctx.lineTo(cpaX, padTop + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#10b981';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`CPA ${tc.toFixed(1)}s`, cpaX, padTop + 12);
      }
    }

    // Playhead Scrubber Line
    if (hasContent && tWindow > 0) {
      const playX = padLeft + (currentTime / tWindow) * plotW;
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playX, padTop);
      ctx.lineTo(playX, padTop + plotH);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    if (!hasContent) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        'Upload a video or audio file to compute spectrogram, or add a Doppler event on the left.',
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
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const padLeft = 52;
    const padRight = 18;
    const padTop = 14;
    const padBottom = 28;
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
      setHoverInfo(null);
      return;
    }

    const uX = (mx - padLeft) / plotW;
    const uY = 1.0 - (my - padTop) / plotH;
    const freqHz = uY * maxFreqHz;
    const tWindow = duration > 0 ? duration : 60.0;
    const timeS = uX * tWindow;
    const activeEv =
      events[Math.min(activeEventIndex, Math.max(0, events.length - 1))];
    const f0 = activeEv ? activeEv.f0_hz : 85.0;
    const radialVelMs = 340.0 * (f0 / Math.max(20.0, freqHz) - 1.0);

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
      zScore,
      radialVelMs,
    });
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const padLeft = 52;
    const padRight = 18;
    const plotW = Math.max(100, rect.width - padLeft - padRight);
    const mx = e.clientX - rect.left;
    if (mx >= padLeft && mx <= padLeft + plotW) {
      const uX = (mx - padLeft) / plotW;
      onSeek(uX * duration);
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#05080f] relative overflow-hidden">
      {/* Toolbar */}
      <div className="h-10 px-4 bg-[#0b101b] border-b border-slate-800/80 flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-1.5">
          {events.map((ev, idx) => (
            <button
              key={ev.id}
              type="button"
              onClick={() => onSelectEventIndex(idx)}
              className={`px-2.5 py-1 text-[11px] font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeEventIndex === idx
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              }`}
            >
              {ev.id} ({ev.f0_hz.toFixed(1)} Hz)
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleWhitened}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showWhitened
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}
          >
            {showWhitened ? 'Whitened' : 'Raw STFT'}
          </button>

          <button
            type="button"
            onClick={() => setShowHarmonics((v) => !v)}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showHarmonics
                ? 'bg-purple-950/60 text-purple-300 border-purple-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}
          >
            Harmonics
          </button>

          <button
            type="button"
            onClick={onToggleNaiveComparison}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showNaiveComparison
                ? 'bg-amber-950/70 text-amber-300 border-amber-500/50'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            Naive Curve
          </button>

          <select
            value={maxFreqHz}
            onChange={(e) => setMaxFreqHz(Number(e.target.value))}
            aria-label="Maximum frequency"
            className="bg-slate-900 text-slate-200 text-[11px] font-mono border border-slate-700 rounded px-2 py-1 outline-none cursor-pointer"
          >
            <option value={240}>0–240 Hz</option>
            <option value={360}>0–360 Hz</option>
            <option value={500}>0–500 Hz</option>
          </select>
        </div>
      </div>

      {/* Canvas */}
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
          <div className="absolute bottom-3 left-16 bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded text-[11px] font-mono tabular-nums text-slate-300 pointer-events-none">
            {hoverInfo.timeS.toFixed(1)}s · {hoverInfo.freqHz.toFixed(1)} Hz · z=
            {hoverInfo.zScore.toFixed(1)} · v_r=
            {hoverInfo.radialVelMs >= 0 ? '+' : ''}
            {hoverInfo.radialVelMs.toFixed(1)} m/s
          </div>
        )}
      </div>
    </div>
  );
};
