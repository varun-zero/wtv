import React, { useEffect, useRef, useState } from 'react';
import {
  GeoBounds,
  boundsGeometry,
  deltaLonSigned,
  greatCircleDistance,
  normalizeLon,
} from '../engine/geo';
import {
  BasinSummary,
  CandidatePoint,
  ScenarioFlight,
} from '../engine/solver';
import { ensureFlightWaypoints } from '../engine/acoustics';
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';

interface GeodeticMapViewportProps {
  bounds: GeoBounds;
  fix: { lat: number; lon: number; alt_m: number };
  radiusM: number;
  status: string;
  flights: ScenarioFlight[];
  candidates: CandidatePoint[];
  basins: BasinSummary[];
  selectedBasin: BasinSummary | null;
  onSelectBasin: (basin: BasinSummary) => void;
  currentTime: number;
  duration: number;
}

export const GeodeticMapViewport: React.FC<GeodeticMapViewportProps> = ({
  bounds,
  fix,
  radiusM,
  status,
  flights,
  candidates,
  basins,
  selectedBasin,
  onSelectBasin,
  currentTime,
  duration,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragOrigin, setDragOrigin] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });
  const [showCorridors, setShowCorridors] = useState<boolean>(true);
  const [cursorGeo, setCursorGeo] = useState<{
    lat: number;
    lon: number;
  } | null>(null);

  useEffect(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, [bounds.west, bounds.south, bounds.east, bounds.north]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(300, rect.width);
    const h = Math.max(220, rect.height);

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

    ctx.fillStyle = '#05080f';
    ctx.fillRect(0, 0, w, h);

    const geom = boundsGeometry(bounds);
    const mapScaleW = w * 0.76 * zoom;
    const mapScaleH = h * 0.76 * zoom;

    const toScreen = (lat: number, lon: number) => {
      const dLon = deltaLonSigned(lon, geom.centerLon);
      const dLat = lat - geom.centerLat;
      const x = w * 0.5 + (dLon / geom.lonSpan) * mapScaleW + pan.x;
      const y = h * 0.5 - (dLat / geom.latSpan) * mapScaleH + pan.y;
      return { x, y };
    };

    // Draw WGS84 Coordinate Graticule
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.lineWidth = 1;
    const nGrid = 6;
    for (let i = 0; i <= nGrid; i++) {
      const latVal = bounds.south + (i / nGrid) * geom.latSpan;
      const p1 = toScreen(latVal, bounds.west);
      const p2 = toScreen(latVal, normalizeLon(bounds.west + geom.lonSpan));
      ctx.beginPath();
      ctx.moveTo(0, p1.y);
      ctx.lineTo(w, p2.y);
      ctx.stroke();

      if (p1.y >= 20 && p1.y <= h - 20) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${latVal.toFixed(2)}°`, 10, p1.y - 4);
      }
    }

    for (let i = 0; i <= nGrid; i++) {
      const lonVal = normalizeLon(bounds.west + (i / nGrid) * geom.lonSpan);
      const p1 = toScreen(bounds.south, lonVal);
      ctx.beginPath();
      ctx.moveTo(p1.x, 0);
      ctx.lineTo(p1.x, h);
      ctx.stroke();

      if (p1.x >= 40 && p1.x <= w - 40) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${lonVal.toFixed(2)}°`, p1.x, h - 10);
      }
    }

    // Draw Candidate Points
    candidates.forEach((c) => {
      const pt = toScreen(c.lat, c.lon);
      if (pt.x < -10 || pt.x > w + 10 || pt.y < -10 || pt.y > h + 10) return;

      const normScore = Math.min(1.0, Math.max(0.05, c.score / 52.0));
      if (c.passedSilence === false) {
        ctx.fillStyle = 'rgba(100, 116, 139, 0.22)';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, Math.max(2.0, 2.5 * Math.sqrt(zoom)), 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle =
          normScore > 0.72
            ? `rgba(16, 185, 129, ${0.35 + normScore * 0.55})`
            : `rgba(245, 158, 11, ${0.14 + normScore * 0.52})`;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, Math.max(2.5, 4.2 * Math.sqrt(zoom)), 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw Flight Splines & Audibility Corridors
    flights.forEach((fl) => {
      const wps = ensureFlightWaypoints(fl);
      if (wps.length < 2) return;

      if (fl.is_target && showCorridors) {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.11)';
        ctx.lineWidth = Math.max(16, 34 * zoom);
        ctx.lineCap = 'round';
        ctx.beginPath();
        wps.forEach((wp, idx) => {
          const pt = toScreen(wp.lat, wp.lon);
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }

      ctx.strokeStyle = fl.is_target
        ? '#06b6d4'
        : 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = fl.is_target ? 2.2 : 1.2;
      if (!fl.is_target) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      wps.forEach((wp, idx) => {
        const pt = toScreen(wp.lat, wp.lon);
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      const progress =
        duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0.5;
      const wpFloatIdx = progress * (wps.length - 1);
      const idx0 = Math.floor(wpFloatIdx);
      const idx1 = Math.min(wps.length - 1, idx0 + 1);
      const frac = wpFloatIdx - idx0;
      const curLat = wps[idx0].lat + frac * (wps[idx1].lat - wps[idx0].lat);
      const curLon = normalizeLon(
        wps[idx0].lon + frac * deltaLonSigned(wps[idx1].lon, wps[idx0].lon)
      );
      const acPt = toScreen(curLat, curLon);

      ctx.fillStyle = fl.is_target ? '#38bdf8' : '#64748b';
      ctx.beginPath();
      ctx.arc(acPt.x, acPt.y, fl.is_target ? 4.5 : 3.0, 0, Math.PI * 2);
      ctx.fill();

      const labelIdx = fl.is_target
        ? Math.floor(wps.length * 0.14)
        : Math.floor(wps.length * 0.82);
      const midWp = wps[Math.min(wps.length - 1, Math.max(0, labelIdx))];
      const midPt = toScreen(midWp.lat, midWp.lon);
      const labelText = `${fl.callsign} (${fl.aircraft})`;
      ctx.font = '10px "JetBrains Mono", monospace';
      const lw = ctx.measureText(labelText).width + 8;
      ctx.fillStyle = 'rgba(5, 8, 15, 0.82)';
      ctx.fillRect(midPt.x + 6, midPt.y - 16, lw, 14);
      ctx.fillStyle = fl.is_target ? '#38bdf8' : '#94a3b8';
      ctx.textAlign = 'left';
      ctx.fillText(labelText, midPt.x + 10, midPt.y - 6);
    });

    // Draw Rival Basins (#2, #3...)
    basins.slice(1).forEach((b) => {
      const bPt = toScreen(b.lat, b.lon);
      const isSelected = selectedBasin?.rank === b.rank;
      ctx.strokeStyle = isSelected ? '#f43f5e' : 'rgba(245, 158, 11, 0.75)';
      ctx.lineWidth = isSelected ? 2.0 : 1.3;
      ctx.beginPath();
      ctx.arc(bPt.x, bPt.y, isSelected ? 8 : 5.0, 0, Math.PI * 2);
      ctx.stroke();

      if (b.rank <= 3 || isSelected) {
        const bText = `#${b.rank} (${b.score.toFixed(1)})`;
        ctx.font = '10px "JetBrains Mono", monospace';
        const bw = ctx.measureText(bText).width + 8;
        ctx.fillStyle = 'rgba(5, 8, 15, 0.85)';
        ctx.fillRect(bPt.x + 8, bPt.y - 7, bw, 14);
        ctx.fillStyle = isSelected ? '#fda4af' : '#fbbf24';
        ctx.textAlign = 'left';
        ctx.fillText(bText, bPt.x + 12, bPt.y + 3);
      }
    });

    // Draw Estimated Fix Reticle when solved
    if (basins.length > 0 && status !== 'inconclusive') {
      const fixPt = toScreen(fix.lat, fix.lon);
      const mPerDegLat = 111139.0;
      const radiusPx = Math.max(
        14,
        (radiusM / mPerDegLat / geom.latSpan) * mapScaleH
      );

      const fixColor = status === 'ambiguous' ? '#f59e0b' : '#10b981';
      ctx.strokeStyle =
        status === 'ambiguous'
          ? 'rgba(245, 158, 11, 0.45)'
          : 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(fixPt.x, fixPt.y, radiusPx, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = fixColor;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(fixPt.x, fixPt.y, 5.5, 0, Math.PI * 2);
      ctx.stroke();

      const fixLabel = `Fix ${fix.lat.toFixed(4)}°, ${fix.lon.toFixed(4)}°`;
      ctx.font = 'bold 10.5px "JetBrains Mono", monospace';
      const fw = ctx.measureText(fixLabel).width + 10;
      ctx.fillStyle = 'rgba(5, 8, 15, 0.88)';
      ctx.fillRect(fixPt.x + 9, fixPt.y - 21, fw, 16);
      ctx.fillStyle = fixColor;
      ctx.textAlign = 'left';
      ctx.fillText(fixLabel, fixPt.x + 14, fixPt.y - 10);
    }

    if (flights.length === 0 && candidates.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        'No flight tracks loaded yet. Add or import flight tracks in the Flight Tracks tab.',
        w * 0.5,
        h * 0.5
      );
    }

    ctx.restore();
  }, [
    bounds,
    fix,
    radiusM,
    status,
    flights,
    candidates,
    basins,
    selectedBasin,
    zoom,
    pan,
    showCorridors,
    currentTime,
    duration,
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(300, rect.width);
    const h = Math.max(220, rect.height);

    if (isDragging) {
      setPan({
        x: e.clientX - dragOrigin.x,
        y: e.clientY - dragOrigin.y,
      });
    }

    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const geom = boundsGeometry(bounds);
    const mapScaleW = w * 0.76 * zoom;
    const mapScaleH = h * 0.76 * zoom;

    const lon = normalizeLon(
      geom.centerLon + ((mx - w * 0.5 - pan.x) / mapScaleW) * geom.lonSpan
    );
    const lat =
      geom.centerLat - ((my - h * 0.5 - pan.y) / mapScaleH) * geom.latSpan;

    setCursorGeo({ lat, lon });
  };

  const handleCanvasClick = () => {
    if (!cursorGeo || basins.length === 0) return;
    let closestBasin: BasinSummary | null = null;
    let minD = Infinity;
    for (const b of basins) {
      const d = greatCircleDistance(
        cursorGeo.lat,
        cursorGeo.lon,
        b.lat,
        b.lon
      );
      if (d < minD) {
        minD = d;
        closestBasin = b;
      }
    }
    if (closestBasin && minD < 6000) {
      onSelectBasin(closestBasin);
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#05080f] relative overflow-hidden">
      {/* Toolbar */}
      <div className="h-10 px-4 bg-[#0b101b] border-b border-slate-800/80 flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowCorridors((v) => !v)}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showCorridors
                ? 'bg-cyan-950/60 text-cyan-300 border-cyan-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}
          >
            Corridors
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(6.0, z * 1.25))}
            aria-label="Zoom in map"
            className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 cursor-pointer"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.5, z / 1.25))}
            aria-label="Zoom out map"
            className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 cursor-pointer"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              setZoom(1.0);
              setPan({ x: 0, y: 0 });
            }}
            aria-label="Reset map pan and zoom"
            className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 flex items-center gap-1 cursor-pointer"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          data-testid="geodetic-map-canvas"
          onMouseDown={(e) => {
            setIsDragging(true);
            setDragOrigin({ x: e.clientX - pan.x, y: e.clientY - pan.y });
          }}
          onMouseMove={handleMouseMove}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => {
            setIsDragging(false);
            setCursorGeo(null);
          }}
          onClick={handleCanvasClick}
          onWheel={(e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.87;
            setZoom((z) => Math.min(6.0, Math.max(0.5, z * factor)));
          }}
          className="w-full h-full block cursor-crosshair"
        />

        {cursorGeo && (
          <div className="absolute bottom-3 left-4 bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded text-[11px] font-mono tabular-nums text-slate-300 pointer-events-none">
            {cursorGeo.lat.toFixed(4)}°, {cursorGeo.lon.toFixed(4)}° ·{' '}
            {zoom.toFixed(1)}x
          </div>
        )}
      </div>
    </div>
  );
};
