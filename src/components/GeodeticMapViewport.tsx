import React, { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { GeoBounds, deltaLonSigned, normalizeLon, greatCircleDistance } from '../engine/geo';
import {
  BasinSummary,
  CandidatePoint,
  HeatmapGridPoint,
  ScenarioFlight,
  computeEstimated3dTrajectory,
  heatmapColorGreenToRed,
} from '../engine/solver';
import { ensureFlightWaypoints } from '../engine/acoustics';
import {
  Box,
  Crosshair,
  Globe,
  Grid,
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
    shortDesc: 'Achromatic Carbon C2',
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
  heatmapGrid?: HeatmapGridPoint[];
  basins: BasinSummary[];
  selectedBasin: BasinSummary | null;
  onSelectBasin: (basin: BasinSummary) => void;
  currentTime: number;
  duration: number;
  selectedFlightId?: string | null;
  onSelectFlight?: (flightId: string) => void;
  show3dTrajectory?: boolean;
  onToggle3dTrajectory?: () => void;
  groundTruth?: { lat: number; lon: number } | null;
}

export const GeodeticMapViewport: React.FC<GeodeticMapViewportProps> = ({
  bounds,
  onChangeBounds,
  fix,
  radiusM,
  status,
  flights,
  candidates,
  heatmapGrid = [],
  basins,
  selectedBasin,
  onSelectBasin,
  currentTime,
  duration,
  selectedFlightId = null,
  onSelectFlight,
  show3dTrajectory = false,
  onToggle3dTrajectory,
  groundTruth = null,
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const overlayGroupRef = useRef<L.LayerGroup | null>(null);
  const canvasRendererRef = useRef<L.Canvas | null>(null);

  const [mapStyle, setMapStyle] = useState<MapStyleId>('dark');
  const [show100mHeatmap, setShow100mHeatmap] = useState<boolean>(true);
  const [invertHeatmapRamp, setInvertHeatmapRamp] = useState<boolean>(false);
  const [showCorridors, setShowCorridors] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(false);
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

    const canvasRenderer = L.canvas({ padding: 0.5 });
    canvasRendererRef.current = canvasRenderer;

    const map = L.map(container, {
      center: [centerLat, centerLon],
      zoom: 12,
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
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
      canvasRendererRef.current = null;
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

  // Render search bounds, 100m heatmap dot-grid, flights, 3D trajectory extrusion, basins, and fix
  useEffect(() => {
    const group = overlayGroupRef.current;
    if (!group) return;
    group.clearLayers();

    const isBrightBasemap = mapStyle === 'terrain';
    const canvasRenderer = canvasRendererRef.current || undefined;

    // 1. Search Region Bounding Box
    L.rectangle(
      [
        [bounds.south, bounds.west],
        [bounds.north, bounds.east],
      ],
      {
        color: isBrightBasemap ? '#18181b' : '#737373',
        weight: isBrightBasemap ? 1.5 : 1.0,
        dashArray: '4, 4',
        fillColor: '#ffffff',
        fillOpacity: isBrightBasemap ? 0.04 : 0.02,
        interactive: false,
      }
    ).addTo(group);

    // 2A. Coarse Regional Candidate Grid Points (optional background grid)
    if (showGrid) {
      candidates.forEach((c) => {
        const normScore = Math.min(1.0, Math.max(0.05, c.score / 52.0));
        const color =
          c.passedSilence === false
            ? '#52525b'
            : normScore > 0.72
            ? '#10b981'
            : '#f59e0b';
        const opacity =
          c.passedSilence === false ? 0.25 : 0.35 + normScore * 0.55;

        L.circleMarker([c.lat, c.lon], {
          renderer: canvasRenderer,
          radius: c.passedSilence === false ? 2.5 : 3.5,
          color: isBrightBasemap ? '#000000' : 'transparent',
          weight: isBrightBasemap ? 0.75 : 0,
          fillColor: color,
          fillOpacity: opacity,
          interactive: false,
        }).addTo(group);
      });
    }

    // 2B. 100-Meter Spacing Green-to-Red Dot-Grid Probability Heatmap
    if (show100mHeatmap && heatmapGrid.length > 0 && status !== 'inconclusive') {
      heatmapGrid.forEach((pt) => {
        const dotColor = heatmapColorGreenToRed(
          pt.normalizedProb,
          invertHeatmapRamp
        );
        const isCore = pt.normalizedProb >= 0.75;
        const dotMarker = L.circleMarker([pt.lat, pt.lon], {
          renderer: canvasRenderer,
          radius: isCore ? 4.2 : 3.2,
          color: isBrightBasemap ? '#000000' : '#050505',
          weight: 0.8,
          fillColor: dotColor,
          fillOpacity: 0.38 + pt.normalizedProb * 0.58,
        }).addTo(group);

        dotMarker.bindTooltip(
          `100m Grid [${pt.gridRow >= 0 ? '+' : ''}${pt.gridRow * 100}m N, ${
            pt.gridCol >= 0 ? '+' : ''
          }${pt.gridCol * 100}m E] · P=${(pt.normalizedProb * 100).toFixed(
            0
          )}% · Score ${pt.score.toFixed(1)}`,
          {
            direction: 'top',
            className: 'skyprint-tooltip',
          }
        );
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
          color: '#10b981',
          weight: 28,
          opacity: isBrightBasemap ? 0.14 : 0.08,
          lineCap: 'butt',
          interactive: false,
        }).addTo(group);
      }

      if (mapStyle !== 'dark') {
        L.polyline(latlngs, {
          color: '#000000',
          weight: fl.is_target ? 4.5 : 3,
          opacity: 0.75,
          interactive: false,
        }).addTo(group);
      }

      const trackLine = L.polyline(latlngs, {
        color:
          isSelected && show3dTrajectory
            ? '#737373'
            : fl.is_target
            ? '#fafafa'
            : '#737373',
        weight: fl.is_target ? 2.0 : 1.5,
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
        radius: fl.is_target ? 4.5 : 3.5,
        color: '#000000',
        weight: 1.5,
        fillColor: fl.is_target ? '#ffffff' : '#737373',
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

      const curtainRing: [number, number][] = [
        ...groundCoords,
        ...[...elevatedCoords].reverse(),
      ];
      L.polygon(curtainRing, {
        color: 'transparent',
        fillColor: '#ff4f00',
        fillOpacity: mapStyle === 'dark' ? 0.16 : 0.22,
        interactive: false,
      }).addTo(group);

      pts.forEach((p, idx) => {
        if (idx % 3 === 0 || idx === pts.length - 1) {
          L.polyline(
            [
              [p.lat, p.lon],
              [p.elevatedLat, p.elevatedLon],
            ],
            {
              color: '#ff4f00',
              weight: 1.2,
              dashArray: '2, 3',
              opacity: 0.75,
              interactive: false,
            }
          ).addTo(group);
        }
      });

      L.polyline(elevatedCoords, {
        color: '#000000',
        weight: 5,
        opacity: 0.8,
        interactive: false,
      }).addTo(group);

      L.polyline(elevatedCoords, {
        color: '#ff4f00',
        weight: 2.5,
        opacity: 1,
        interactive: false,
      }).addTo(group);

      const cpaPt = active3dTrajectory.cpaPoint;
      L.polyline(
        [
          [cpaPt.elevatedLat, cpaPt.elevatedLon],
          [referenceReceiver.lat, referenceReceiver.lon],
        ],
        {
          color: '#f59e0b',
          weight: 1.8,
          dashArray: '4, 4',
          opacity: 0.9,
          interactive: false,
        }
      ).addTo(group);

      L.circleMarker([cpaPt.elevatedLat, cpaPt.elevatedLon], {
        radius: 5.5,
        color: '#000000',
        weight: 2,
        fillColor: '#ff4f00',
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
        radius: isSelected ? 7.0 : 5.0,
        color: isSelected ? '#ffffff' : '#f59e0b',
        weight: 2,
        fillColor: '#000000',
        fillOpacity: 0.9,
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
      const fixColor = status === 'ambiguous' ? '#f59e0b' : '#ffffff';

      L.circle([fix.lat, fix.lon], {
        radius: Math.max(120, radiusM),
        color: '#ffffff',
        weight: 1.5,
        dashArray: '4, 4',
        fillColor: '#ef4444',
        fillOpacity: 0.05,
        interactive: false,
      }).addTo(group);

      const fixMarker = L.circleMarker([fix.lat, fix.lon], {
        radius: 6.5,
        color: '#000000',
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

    // 6. Ground Truth Target Reticle & Accuracy Error Vector
    if (groundTruth) {
      const truthMarker = L.circleMarker([groundTruth.lat, groundTruth.lon], {
        radius: 8,
        color: '#ffffff',
        weight: 2,
        fillColor: '#10b981',
        fillOpacity: 1,
      }).addTo(group);

      truthMarker.bindTooltip(
        `<span style="color:#10b981;font-weight:700">● GROUND TRUTH</span><br/>${groundTruth.lat.toFixed(4)}°, ${groundTruth.lon.toFixed(4)}°`,
        {
          permanent: true,
          direction: 'top',
          className: 'skyprint-tooltip',
          offset: [0, -8],
        }
      );

      if (basins.length > 0 && status !== 'inconclusive') {
        const errorLine = L.polyline(
          [
            [fix.lat, fix.lon],
            [groundTruth.lat, groundTruth.lon],
          ],
          {
            color: '#10b981',
            weight: 2,
            dashArray: '4, 4',
          }
        ).addTo(group);

        const distM = greatCircleDistance(
          fix.lat,
          fix.lon,
          groundTruth.lat,
          groundTruth.lon
        );
        errorLine.bindTooltip(`Accuracy Error: ${distM.toFixed(0)} m`, {
          permanent: true,
          direction: 'center',
          className: 'skyprint-tooltip',
        });
      }
    }
  }, [
    bounds,
    fix,
    radiusM,
    status,
    flights,
    candidates,
    heatmapGrid,
    basins,
    selectedBasin,
    onSelectBasin,
    show100mHeatmap,
    invertHeatmapRamp,
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
    groundTruth,
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

  const handleZoomToHeatmapFix = () => {
    const map = mapInstanceRef.current;
    if (!map || basins.length === 0) return;
    map.setView([fix.lat, fix.lon], 14, { animate: true });
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
    if (id === 'dark') return <Moon className="w-3 h-3" />;
    if (id === 'terrain') return <Mountain className="w-3 h-3" />;
    return <Globe className="w-3 h-3" />;
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#000000] relative overflow-hidden">
      {/* Top Toolbar: Consolidated Flush 28px Segmented Button Groups */}
      <div className="h-10 px-4 bg-[#0A0A0A] border-b border-[#1F1F1F] flex items-center justify-between gap-2 shrink-0 z-10 overflow-x-auto">
        <div className="flex items-center gap-2 shrink-0">
          {/* Segmented Group 1: Dark Mode | Terrain | Satellite */}
          <div
            data-testid="map-layer-selector"
            role="group"
            aria-label="Map style selector"
            className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]"
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
                  className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-1.5 transition-colors whitespace-nowrap cursor-pointer ${
                    active
                      ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                      : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA]'
                  }`}
                >
                  {styleIcon(id)}
                  <span>{cfg.label}</span>
                </button>
              );
            })}
          </div>

          {/* Segmented Group 2: 100m Heatmap | Corridors | Regional Grid | 3D Trajectory */}
          <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]">
            <button
              type="button"
              data-testid="map-toggle-100m-heatmap"
              onClick={() => setShow100mHeatmap((v) => !v)}
              aria-pressed={show100mHeatmap}
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                show100mHeatmap
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
              title="Toggle 100m-spaced Green-to-Red dot-grid probability heatmap around estimated fix"
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: show100mHeatmap ? '#ef4444' : '#3f3f46',
                }}
              />
              <Grid className="w-3 h-3" />
              <span>100m Heatmap</span>
            </button>

            <button
              type="button"
              onClick={() => setShowCorridors((v) => !v)}
              aria-pressed={showCorridors}
              aria-label="Toggle flight acoustic corridors overlay"
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showCorridors
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: showCorridors ? '#10b981' : '#3f3f46',
                }}
              />
              <span>Corridors</span>
            </button>

            <button
              type="button"
              onClick={() => setShowGrid((v) => !v)}
              aria-pressed={showGrid}
              aria-label="Toggle coarse geodetic grid"
              className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showGrid
                  ? 'bg-[#161616] text-[#FAFAFA]'
                  : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
              }`}
            >
              <span
                className="w-1.5 h-1.5 shrink-0"
                style={{
                  backgroundColor: showGrid ? '#10b981' : '#3f3f46',
                }}
              />
              <span>Coarse Grid</span>
            </button>

            {onToggle3dTrajectory && (
              <button
                type="button"
                data-testid="map-toggle-3d-trajectory"
                aria-pressed={show3dTrajectory}
                onClick={onToggle3dTrajectory}
                className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  show3dTrajectory
                    ? 'bg-[#161616] text-[#FAFAFA]'
                    : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
                }`}
              >
                <span
                  className="w-1.5 h-1.5 shrink-0"
                  style={{
                    backgroundColor: show3dTrajectory ? '#ff4f00' : '#3f3f46',
                  }}
                />
                <Box className="w-3 h-3" />
                <span>3D Trajectory</span>
              </button>
            )}
          </div>
        </div>

        {/* Segmented Group 3: Viewport Controls */}
        <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626] shrink-0">
          {basins.length > 0 && status !== 'inconclusive' && (
            <button
              type="button"
              data-testid="zoom-to-heatmap-btn"
              onClick={handleZoomToHeatmapFix}
              aria-label="Zoom to 100m dot-grid heatmap around estimated fix"
              className="h-7 px-2.5 text-[11px] font-mono rounded-none bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] flex items-center gap-1.5 cursor-pointer whitespace-nowrap transition-colors"
              title="Zoom to 100m dot-grid heatmap around estimated fix"
            >
              <span className="w-1.5 h-1.5 bg-[#EF4444]" />
              Focus 100m Grid
            </button>
          )}
          {onChangeBounds && (
            <button
              type="button"
              onClick={handleSetBoundsFromMapView}
              aria-label="Set search bounding box to current map viewport"
              className="h-7 px-2.5 text-[11px] font-mono rounded-none bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] flex items-center gap-1.5 cursor-pointer whitespace-nowrap transition-colors"
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
            className="h-7 w-7 rounded-none bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center justify-center cursor-pointer transition-colors"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => mapInstanceRef.current?.zoomOut()}
            aria-label="Zoom out map"
            className="h-7 w-7 rounded-none bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center justify-center cursor-pointer transition-colors"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={handleFitSearchBounds}
            aria-label="Fit search bounds"
            className="h-7 px-2.5 text-[11px] font-mono rounded-none bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center gap-1.5 cursor-pointer transition-colors"
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

        {/* Bottom-Left HUD: 100m Dot-Grid Heatmap Legend + Live Cursor Coordinates */}
        <div className="absolute bottom-3 left-3 z-[400] flex flex-col gap-1.5 pointer-events-auto">
          {show100mHeatmap &&
            heatmapGrid.length > 0 &&
            status !== 'inconclusive' && (
              <div
                data-testid="heatmap-100m-hud"
                className="bg-[#0A0A0A] border border-[#262626] px-3 py-2 rounded-none text-[10px] font-mono tabular-nums text-[#FAFAFA] flex flex-col gap-1.5 w-64"
              >
                <div className="flex items-center justify-between border-b border-[#1F1F1F] pb-1">
                  <span className="uppercase tracking-[0.08em] text-[#A1A1AA] font-semibold">
                    100M DOT-GRID HEATMAP ({heatmapGrid.length} PTS)
                  </span>
                  <button
                    type="button"
                    onClick={() => setInvertHeatmapRamp((v) => !v)}
                    aria-label="Invert Green/Red heatmap polarity"
                    className="px-1.5 py-0.5 bg-[#111111] hover:bg-[#161616] border border-[#262626] text-[9px] text-[#FAFAFA] cursor-pointer"
                    title="Invert Green/Red heatmap polarity"
                  >
                    {invertHeatmapRamp ? 'RED→GRN' : 'GRN→RED'}
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[#737373]">
                    {invertHeatmapRamp ? 'LOW' : 'LOW'}
                  </span>
                  <div
                    className="flex-1 h-2 border border-[#262626]"
                    style={{
                      background: invertHeatmapRamp
                        ? 'linear-gradient(90deg, #ef4444 0%, #f97316 25%, #eab308 50%, #84cc16 75%, #10b981 100%)'
                        : 'linear-gradient(90deg, #10b981 0%, #84cc16 25%, #eab308 50%, #f97316 75%, #ef4444 100%)',
                    }}
                  />
                  <span className="text-[#FAFAFA]">PEAK FIX</span>
                </div>
                <div className="flex justify-between text-[9.5px] text-[#737373]">
                  <span>PITCH: 100.0 m × 100.0 m</span>
                  <span>SPAN: 2.4 km × 2.4 km</span>
                </div>
              </div>
            )}

          {cursorGeo && (
            <div className="bg-[#0A0A0A] border border-[#262626] px-2.5 py-1 rounded-none text-[11px] font-mono tabular-nums text-[#FAFAFA] pointer-events-none self-start">
              {cursorGeo.lat.toFixed(4)}°, {cursorGeo.lon.toFixed(4)}°
            </div>
          )}
        </div>

        {/* Bottom-Right HUD: Estimated 3D Flight Trajectory Profile */}
        {active3dTrajectory && (
          <div
            data-testid="trajectory-3d-hud"
            className="absolute bottom-6 right-3 z-[400] w-72 max-w-[calc(100%-24px)] bg-[#0A0A0A] border border-[#262626] rounded-none p-3 pointer-events-auto"
          >
            <div className="flex items-center justify-between border-b border-[#1F1F1F] pb-1.5 mb-2 gap-2">
              <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold text-[#FAFAFA] truncate min-w-0">
                <span className="w-1.5 h-1.5 bg-[#FF4F00] inline-block shrink-0" />
                <span className="truncate">3D Trajectory · {active3dTrajectory.callsign}</span>
              </div>
              <span className="text-[10px] font-mono text-[#737373] shrink-0">
                {active3dTrajectory.aircraft}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-[11px] font-mono tabular-nums mb-2">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#737373]">
                  MSL ALT
                </div>
                <div className="text-[#FAFAFA]">
                  {active3dTrajectory.cpaPoint.alt_m.toFixed(0)} m
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#737373]">
                  3D SLANT
                </div>
                <div className="text-[#F59E0B]">
                  {active3dTrajectory.minSlant3dM.toFixed(0)} m
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#737373]">
                  PEAK ELEV
                </div>
                <div className="text-[#FF4F00]">
                  {active3dTrajectory.maxElevationDeg.toFixed(1)}°
                </div>
              </div>
            </div>

            {/* Mini 3D Slant & Altitude Profile SVG */}
            <svg
              viewBox="0 0 240 54"
              className="w-full h-14 bg-[#000000] rounded-none border border-[#1F1F1F] overflow-visible"
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
                      stroke="#262626"
                      strokeDasharray="2,2"
                      strokeWidth="1"
                    />
                    <polyline
                      fill="none"
                      stroke="#ff4f00"
                      strokeWidth="1.75"
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
