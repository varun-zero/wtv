import React, { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { GeoBounds, deltaLonSigned, normalizeLon } from '../engine/geo';
import {
  BasinSummary,
  CandidatePoint,
  ScenarioFlight,
  computeEstimated3dTrajectory,
} from '../engine/solver';
import { ensureFlightWaypoints } from '../engine/acoustics';
import {
  Box,
  Crosshair,
  Globe,
  Layers,
  Moon,
  Mountain,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

export type MapStyleId = 'dark' | 'terrain' | 'satellite';

interface MapStyleConfig {
  id: MapStyleId;
  label: string;
  shortDesc: string;
  url: string;
  attribution: string;
  maxZoom: number;
  isDarkTileFilter?: boolean;
}

export const MAP_STYLES: Record<MapStyleId, MapStyleConfig> = {
  dark: {
    id: 'dark',
    label: 'Dark Mode',
    shortDesc: 'OSM Tactical Dark',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
    isDarkTileFilter: true,
  },
  terrain: {
    id: 'terrain',
    label: 'Terrain',
    shortDesc: 'USGS / SRTM Topo DEM',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; USGS, Esri, NASA SRTM, OpenStreetMap',
    maxZoom: 19,
  },
  satellite: {
    id: 'satellite',
    label: 'Satellite',
    shortDesc: 'USGS / Esri Optical',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, USGS, USDA',
    maxZoom: 19,
  },
};

interface GeodeticMapViewportProps {
  bounds: GeoBounds;
  onChangeBounds?: (nextBounds: GeoBounds) => void;
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
  selectedFlightId?: string | null;
  onSelectFlight?: (flightId: string) => void;
  show3dTrajectory?: boolean;
  onToggle3dTrajectory?: () => void;
}

export const GeodeticMapViewport: React.FC<GeodeticMapViewportProps> = ({
  bounds,
  onChangeBounds,
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
  selectedFlightId = null,
  onSelectFlight,
  show3dTrajectory = false,
  onToggle3dTrajectory,
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const overlayGroupRef = useRef<L.LayerGroup | null>(null);

  const [mapStyle, setMapStyle] = useState<MapStyleId>('dark');
  const [showCorridors, setShowCorridors] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [cursorGeo, setCursorGeo] = useState<{
    lat: number;
    lon: number;
  } | null>(null);

  const selectedFlight = useMemo(() => {
    if (flights.length === 0) return null;
    if (selectedFlightId) {
      return flights.find((f) => f.id === selectedFlightId) || flights[0];
    }
    return flights[0];
  }, [flights, selectedFlightId]);

  const referenceReceiver = useMemo(() => {
    if (basins.length > 0 && status !== 'inconclusive') {
      return fix;
    }
    return {
      lat: (bounds.south + bounds.north) * 0.5,
      lon: (bounds.west + bounds.east) * 0.5,
      alt_m: 0,
    };
  }, [basins.length, status, fix, bounds]);

  const active3dTrajectory = useMemo(() => {
    if (!show3dTrajectory || !selectedFlight) return null;
    return computeEstimated3dTrajectory(selectedFlight, referenceReceiver);
  }, [show3dTrajectory, selectedFlight, referenceReceiver]);

  // Initialize Leaflet Map once
  useEffect(() => {
    const container = mapContainerRef.current;
    if (!container || mapInstanceRef.current) return;

    const centerLat = (bounds.south + bounds.north) * 0.5;
    const centerLon = (bounds.west + bounds.east) * 0.5;

    const map = L.map(container, {
      center: [centerLat, centerLon],
      zoom: 11,
      zoomControl: false,
      attributionControl: true,
    });

    const cfg = MAP_STYLES[mapStyle];
    const tileLayer = L.tileLayer(cfg.url, {
      attribution: cfg.attribution,
      maxZoom: cfg.maxZoom,
    }).addTo(map);

    const overlayGroup = L.layerGroup().addTo(map);

    map.on('mousemove', (e: L.LeafletMouseEvent) => {
      setCursorGeo({
        lat: e.latlng.lat,
        lon: normalizeLon(e.latlng.lng),
      });
    });

    map.on('mouseout', () => {
      setCursorGeo(null);
    });

    mapInstanceRef.current = map;
    tileLayerRef.current = tileLayer;
    overlayGroupRef.current = overlayGroup;

    // Fit initial bounds
    map.fitBounds(
      [
        [bounds.south, bounds.west],
        [bounds.north, bounds.east],
      ],
      { padding: [20, 20] }
    );

    const resizeObserver = new ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapInstanceRef.current = null;
      tileLayerRef.current = null;
      overlayGroupRef.current = null;
    };
  }, []);

  // Switch basemap tile layer when mapStyle changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (tileLayerRef.current) {
      tileLayerRef.current.remove();
    }
    const cfg = MAP_STYLES[mapStyle];
    const nextLayer = L.tileLayer(cfg.url, {
      attribution: cfg.attribution,
      maxZoom: cfg.maxZoom,
    }).addTo(map);
    nextLayer.bringToBack();
    tileLayerRef.current = nextLayer;
  }, [mapStyle]);

  // Render search bounds, flights, 3D trajectory extrusion, candidates, basins, and fix
  useEffect(() => {
    const group = overlayGroupRef.current;
    if (!group) return;
    group.clearLayers();

    const isBrightBasemap = mapStyle === 'terrain';

    // 1. Search Region Bounding Box
    L.rectangle(
      [
        [bounds.south, bounds.west],
        [bounds.north, bounds.east],
      ],
      {
        color: isBrightBasemap ? '#0284c7' : '#38bdf8',
        weight: isBrightBasemap ? 1.8 : 1.2,
        dashArray: '5, 5',
        fillColor: '#38bdf8',
        fillOpacity: isBrightBasemap ? 0.05 : 0.03,
        interactive: false,
      }
    ).addTo(group);

    // 2. Candidate Grid Points
    if (showGrid) {
      candidates.forEach((c) => {
        const normScore = Math.min(1.0, Math.max(0.05, c.score / 52.0));
        const color =
          c.passedSilence === false
            ? '#64748b'
            : normScore > 0.72
            ? '#10b981'
            : '#f59e0b';
        const opacity =
          c.passedSilence === false ? 0.25 : 0.3 + normScore * 0.55;

        L.circleMarker([c.lat, c.lon], {
          radius: c.passedSilence === false ? 2.5 : 3.5,
          color: isBrightBasemap ? '#0f172a' : 'transparent',
          weight: isBrightBasemap ? 0.75 : 0,
          fillColor: color,
          fillOpacity: opacity,
          interactive: false,
        }).addTo(group);
      });
    }

    // 3. ADS-B Flight Trajectories & Slant Corridors
    flights.forEach((fl) => {
      const wps = ensureFlightWaypoints(fl);
      if (wps.length < 2) return;
      const latlngs: [number, number][] = wps.map((wp) => [wp.lat, wp.lon]);
      const isSelected = selectedFlight?.id === fl.id;

      if (fl.is_target && showCorridors) {
        L.polyline(latlngs, {
          color: isBrightBasemap ? '#0284c7' : '#06b6d4',
          weight: 28,
          opacity: isBrightBasemap ? 0.2 : 0.14,
          lineCap: 'round',
          interactive: false,
        }).addTo(group);
      }

      // High-contrast dark casing on Terrain/Satellite for crisp legibility
      if (mapStyle !== 'dark') {
        L.polyline(latlngs, {
          color: '#05080f',
          weight: fl.is_target ? 4.5 : 3,
          opacity: 0.65,
          interactive: false,
        }).addTo(group);
      }

      const trackLine = L.polyline(latlngs, {
        color: isSelected && show3dTrajectory
          ? '#94a3b8'
          : fl.is_target
          ? '#22d3ee'
          : '#94a3b8',
        weight: fl.is_target ? 2.5 : 1.5,
        dashArray:
          isSelected && show3dTrajectory
            ? '3, 5'
            : fl.is_target
            ? undefined
            : '4, 4',
        opacity: fl.is_target ? 0.95 : 0.65,
      }).addTo(group);

      if (onSelectFlight) {
        trackLine.on('click', () => onSelectFlight(fl.id));
      }

      // Place callsign label near the outer entry point of the track so it never collides with the central Fix
      const entryWp = wps[Math.min(wps.length - 1, 2)];
      const entryMarker = L.circleMarker([entryWp.lat, entryWp.lon], {
        radius: 2,
        color: 'transparent',
        fillOpacity: 0,
        interactive: false,
      }).addTo(group);

      entryMarker.bindTooltip(
        `${fl.callsign} · ${fl.aircraft} (${fl.speed_ms.toFixed(0)} m/s)`,
        {
          permanent: true,
          direction: 'top',
          className: 'skyprint-tooltip',
        }
      );

      // Animated aircraft marker along flight spline
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

      L.circleMarker([curLat, curLon], {
        radius: fl.is_target ? 5 : 3.5,
        color: '#05080f',
        weight: 1.5,
        fillColor: fl.is_target ? '#38bdf8' : '#64748b',
        fillOpacity: 1,
      }).addTo(group);
    });

    // 3B. Estimated 3D Flight Path Trajectory Overlay for Selected Flight
    if (active3dTrajectory && active3dTrajectory.points.length >= 2) {
      const pts = active3dTrajectory.points;
      const groundCoords: [number, number][] = pts.map((p) => [p.lat, p.lon]);
      const elevatedCoords: [number, number][] = pts.map((p) => [
        p.elevatedLat,
        p.elevatedLon,
      ]);

      // 3D Vertical Curtain Polygon between ground track and elevated 3D flight path
      const curtainRing: [number, number][] = [
        ...groundCoords,
        ...[...elevatedCoords].reverse(),
      ];
      L.polygon(curtainRing, {
        color: 'transparent',
        fillColor: '#a855f7',
        fillOpacity: mapStyle === 'dark' ? 0.18 : 0.25,
        interactive: false,
      }).addTo(group);

      // Vertical Altitude Drop-Pillars connecting 3D trajectory down to ground shadow
      pts.forEach((p, idx) => {
        if (idx % 3 === 0 || idx === pts.length - 1) {
          L.polyline(
            [
              [p.lat, p.lon],
              [p.elevatedLat, p.elevatedLon],
            ],
            {
              color: '#c084fc',
              weight: 1.2,
              dashArray: '2, 3',
              opacity: 0.75,
              interactive: false,
            }
          ).addTo(group);
        }
      });

      // Elevated 3D Flight Path Casing & Ribbon
      L.polyline(elevatedCoords, {
        color: '#05080f',
        weight: 5,
        opacity: 0.75,
        interactive: false,
      }).addTo(group);

      L.polyline(elevatedCoords, {
        color: '#c084fc',
        weight: 3,
        opacity: 1,
        interactive: false,
      }).addTo(group);

      // 3D Acoustic Slant-Range Vector from 3D CPA Point to Ground Receiver
      const cpaPt = active3dTrajectory.cpaPoint;
      L.polyline(
        [
          [cpaPt.elevatedLat, cpaPt.elevatedLon],
          [referenceReceiver.lat, referenceReceiver.lon],
        ],
        {
          color: '#f59e0b',
          weight: 2,
          dashArray: '4, 4',
          opacity: 0.9,
          interactive: false,
        }
      ).addTo(group);

      L.circleMarker([cpaPt.elevatedLat, cpaPt.elevatedLon], {
        radius: 6,
        color: '#05080f',
        weight: 2,
        fillColor: '#c084fc',
        fillOpacity: 1,
        interactive: false,
      }).addTo(group);

      const labelPt = pts[Math.min(pts.length - 1, Math.floor(pts.length * 0.78))];
      const label3dMarker = L.circleMarker(
        [labelPt.elevatedLat, labelPt.elevatedLon],
        {
          radius: 2,
          color: 'transparent',
          fillOpacity: 0,
          interactive: false,
        }
      ).addTo(group);

      label3dMarker.bindTooltip(
        `3D · ${active3dTrajectory.callsign} (${cpaPt.alt_m.toFixed(0)} m MSL)`,
        {
          permanent: true,
          direction: 'right',
          className: 'skyprint-tooltip',
        }
      );
    }

    // 4. Rival Candidate Basins (#2, #3...)
    basins.slice(1).forEach((b) => {
      const isSelected = selectedBasin?.rank === b.rank;
      const marker = L.circleMarker([b.lat, b.lon], {
        radius: isSelected ? 7.5 : 5.5,
        color: isSelected ? '#f43f5e' : '#f59e0b',
        weight: 2,
        fillColor: '#05080f',
        fillOpacity: 0.85,
      }).addTo(group);

      marker.on('click', () => onSelectBasin(b));

      if (b.rank <= 3 || isSelected) {
        marker.bindTooltip(`#${b.rank} (${b.score.toFixed(1)})`, {
          permanent: true,
          direction: 'right',
          className: 'skyprint-tooltip',
        });
      }
    });

    // 5. Solved Estimated Fix + 95% Uncertainty Circle
    if (basins.length > 0 && status !== 'inconclusive') {
      const fixColor = status === 'ambiguous' ? '#f59e0b' : '#10b981';

      L.circle([fix.lat, fix.lon], {
        radius: Math.max(120, radiusM),
        color: fixColor,
        weight: 1.8,
        dashArray: '4, 4',
        fillColor: fixColor,
        fillOpacity: mapStyle === 'dark' ? 0.08 : 0.14,
        interactive: false,
      }).addTo(group);

      const fixMarker = L.circleMarker([fix.lat, fix.lon], {
        radius: 6.5,
        color: '#05080f',
        weight: 2,
        fillColor: fixColor,
        fillOpacity: 1,
      }).addTo(group);

      fixMarker.bindTooltip(
        `Fix ${fix.lat.toFixed(4)}°, ${fix.lon.toFixed(4)}°`,
        {
          permanent: true,
          direction: 'top',
          className: 'skyprint-tooltip',
        }
      );
    }
  }, [
    bounds,
    fix,
    radiusM,
    status,
    flights,
    candidates,
    basins,
    selectedBasin,
    onSelectBasin,
    showCorridors,
    showGrid,
    mapStyle,
    currentTime,
    duration,
    selectedFlight,
    show3dTrajectory,
    active3dTrajectory,
    referenceReceiver,
    onSelectFlight,
  ]);

  const handleFitSearchBounds = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.fitBounds(
      [
        [bounds.south, bounds.west],
        [bounds.north, bounds.east],
      ],
      { padding: [24, 24] }
    );
  };

  const handleSetBoundsFromMapView = () => {
    const map = mapInstanceRef.current;
    if (!map || !onChangeBounds) return;
    const b = map.getBounds();
    onChangeBounds({
      south: Number(b.getSouth().toFixed(4)),
      north: Number(b.getNorth().toFixed(4)),
      west: Number(b.getWest().toFixed(4)),
      east: Number(b.getEast().toFixed(4)),
    });
  };

  const styleIcon = (id: MapStyleId) => {
    if (id === 'dark') return <Moon className="w-3.5 h-3.5" />;
    if (id === 'terrain') return <Mountain className="w-3.5 h-3.5" />;
    return <Globe className="w-3.5 h-3.5" />;
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#05080f] relative overflow-hidden">
      {/* Top Toolbar */}
      <div className="h-11 px-4 bg-[#0b101b] border-b border-slate-800/80 flex items-center justify-between gap-2 shrink-0 z-10">
        {/* Layer Selector: Dark Mode | Terrain | Satellite */}
        <div className="flex items-center gap-2">
          <div
            data-testid="map-layer-selector"
            role="group"
            aria-label="Map style selector"
            className="inline-flex items-center bg-slate-950/90 p-0.5 rounded-md border border-slate-800"
          >
            {(Object.keys(MAP_STYLES) as MapStyleId[]).map((id) => {
              const cfg = MAP_STYLES[id];
              const active = mapStyle === id;
              return (
                <button
                  key={id}
                  type="button"
                  data-testid={`layer-btn-${id}`}
                  aria-pressed={active}
                  onClick={() => setMapStyle(id)}
                  title={`${cfg.label} (${cfg.shortDesc})`}
                  className={`px-2.5 py-1 text-[11px] font-mono rounded flex items-center gap-1.5 transition-colors whitespace-nowrap cursor-pointer ${
                    active
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-medium'
                      : 'text-slate-400 hover:text-slate-200 border border-transparent'
                  }`}
                >
                  {styleIcon(id)}
                  <span>{cfg.label}</span>
                </button>
              );
            })}
          </div>

          <div className="h-4 w-px bg-slate-800 mx-0.5" />

          {/* Overlay Toggles */}
          <button
            type="button"
            onClick={() => setShowCorridors((v) => !v)}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showCorridors
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            Corridors
          </button>

          <button
            type="button"
            onClick={() => setShowGrid((v) => !v)}
            className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showGrid
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            Grid
          </button>

          {onToggle3dTrajectory && (
            <button
              type="button"
              data-testid="map-toggle-3d-trajectory"
              aria-pressed={show3dTrajectory}
              onClick={onToggle3dTrajectory}
              className={`px-2.5 py-1 text-[11px] font-mono rounded border flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                show3dTrajectory
                  ? 'bg-purple-950/70 text-purple-300 border-purple-500/50'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              <Box className="w-3 h-3" />
              <span>3D Trajectory</span>
            </button>
          )}
        </div>

        {/* Viewport Controls */}
        <div className="flex items-center gap-1.5">
          {onChangeBounds && (
            <button
              type="button"
              onClick={handleSetBoundsFromMapView}
              className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-700 flex items-center gap-1 cursor-pointer whitespace-nowrap"
              title="Set search bounding box to current map viewport"
            >
              <Crosshair className="w-3 h-3" />
              Set Bounds to View
            </button>
          )}
          <button
            type="button"
            onClick={() => mapInstanceRef.current?.zoomIn()}
            aria-label="Zoom in map"
            className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 cursor-pointer"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => mapInstanceRef.current?.zoomOut()}
            aria-label="Zoom out map"
            className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 cursor-pointer"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={handleFitSearchBounds}
            aria-label="Fit search bounds"
            className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 flex items-center gap-1 cursor-pointer"
          >
            <RotateCcw className="w-3 h-3" />
            Fit Bounds
          </button>
        </div>
      </div>

      {/* Real Leaflet Open-Source Map Container */}
      <div
        data-map-style={mapStyle}
        className={`flex-1 relative min-h-0 ${
          MAP_STYLES[mapStyle].isDarkTileFilter ? 'osm-dark-tiles' : ''
        }`}
      >
        <div
          ref={mapContainerRef}
          data-testid="geodetic-map-canvas"
          className="w-full h-full z-0"
        />

        {/* Bottom-Left HUD: Active Layer Indicator + Live Cursor Coordinates */}
        <div className="absolute bottom-3 left-3 z-[400] flex items-center gap-2 pointer-events-none">
          <div className="bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 flex items-center gap-1.5">
            <Layers className="w-3 h-3 text-cyan-400" />
            <span className="text-slate-100">{MAP_STYLES[mapStyle].label}</span>
            <span className="text-slate-500">·</span>
            <span className="text-slate-400">
              {MAP_STYLES[mapStyle].shortDesc}
            </span>
          </div>

          {cursorGeo && (
            <div className="bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded text-[11px] font-mono tabular-nums text-slate-300">
              {cursorGeo.lat.toFixed(4)}°, {cursorGeo.lon.toFixed(4)}°
            </div>
          )}
        </div>

        {/* Bottom-Right HUD: Estimated 3D Flight Trajectory Profile */}
        {active3dTrajectory && (
          <div
            data-testid="trajectory-3d-hud"
            className="absolute bottom-6 right-3 z-[400] w-72 bg-slate-950/95 border border-purple-500/40 rounded-md p-3 shadow-xl pointer-events-auto"
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-2">
              <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-purple-300">
                <Box className="w-3.5 h-3.5 text-purple-400" />
                <span>3D Trajectory · {active3dTrajectory.callsign}</span>
              </div>
              <span className="text-[10px] font-mono text-slate-400">
                {active3dTrajectory.aircraft}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-[11px] font-mono tabular-nums mb-2">
              <div>
                <div className="text-[10px] text-slate-500">MSL Alt</div>
                <div className="text-slate-100">
                  {active3dTrajectory.cpaPoint.alt_m.toFixed(0)} m
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-500">3D Slant</div>
                <div className="text-amber-300">
                  {active3dTrajectory.minSlant3dM.toFixed(0)} m
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-500">Peak Elev</div>
                <div className="text-purple-300">
                  {active3dTrajectory.maxElevationDeg.toFixed(1)}°
                </div>
              </div>
            </div>

            {/* Mini 3D Slant & Altitude Profile SVG */}
            <svg
              viewBox="0 0 240 54"
              className="w-full h-14 bg-[#070a12] rounded border border-slate-800/90 overflow-visible"
            >
              {(() => {
                const pts = active3dTrajectory.points;
                const maxSlant = Math.max(
                  ...pts.map((p) => p.slantDist3dM),
                  2500
                );
                const minSlant = Math.min(
                  ...pts.map((p) => p.slantDist3dM),
                  200
                );
                const span = Math.max(200, maxSlant - minSlant);
                const polyPoints = pts
                  .map((p, i) => {
                    const x = 8 + (i / Math.max(1, pts.length - 1)) * 224;
                    const y = 46 - ((p.slantDist3dM - minSlant) / span) * 34;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');

                return (
                  <>
                    <line
                      x1="8"
                      y1="22"
                      x2="232"
                      y2="22"
                      stroke="#334155"
                      strokeDasharray="2,2"
                      strokeWidth="1"
                    />
                    <polyline
                      fill="none"
                      stroke="#c084fc"
                      strokeWidth="2"
                      points={polyPoints}
                    />
                  </>
                );
              })()}
            </svg>
          </div>
        )}
      </div>
    </div>
  );
};
