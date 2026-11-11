import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SpectrogramData,
  TracedRidge,
  FittedDopplerEvent,
  applyAircraftBandBoost,
  applyClockDrift,
  applyHighPassFilter,
  computeSpectrogram,
  fitDopplerEventFromRidge,
  traceDopplerRidge,
} from './engine/dsp';
import {
  BasinSummary,
  FlightSortDirection,
  FlightSortField,
  InversionResult,
  ScenarioFlight,
  runGeolocationInversion,
  sortFlights,
} from './engine/solver';
import { GeoBounds } from './engine/geo';
import { pathSoundSpeed } from './engine/atmosphere';
import { ForensicVideoMonitor } from './components/ForensicVideoMonitor';
import { SpectrogramViewport } from './components/SpectrogramViewport';
import { GeodeticMapViewport } from './components/GeodeticMapViewport';
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Box,
  Download,
  MapPin,
  Plus,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';

type WorkspaceTab = 'spectrogram' | 'geolocation' | 'flights';

export function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('spectrogram');

  // User Doppler Events (starts empty — no demo data)
  const [events, setEvents] = useState<FittedDopplerEvent[]>([]);
  const [activeEventIndex, setActiveEventIndex] = useState<number>(0);

  // User ADS-B Flight Tracks (starts empty — no demo data)
  const [flights, setFlights] = useState<ScenarioFlight[]>([]);

  // Search Region & Atmospheric Priors
  const [bounds, setBounds] = useState<GeoBounds>({
    west: -122.6,
    south: 37.2,
    east: -122.0,
    north: 37.6,
  });
  const [tempC, setTempC] = useState<number>(15.0);
  const [windEMs, setWindEMs] = useState<number>(0.0);
  const [windNMs, setWindNMs] = useState<number>(0.0);

  // Audio DSP Controls
  const [showNaiveComparison, setShowNaiveComparison] =
    useState<boolean>(false);
  const [showWhitened, setShowWhitened] = useState<boolean>(true);
  const [highPass80Hz, setHighPass80Hz] = useState<boolean>(false);
  const [bandBoostDb, setBandBoostDb] = useState<number>(0.0);
  const [injectedDriftPpm, setInjectedDriftPpm] = useState<number>(0.0);

  // Media Playback & Decoded Audio State
  const [rawPcmBuffer, setRawPcmBuffer] = useState<Float32Array | null>(null);
  const [sampleRate, setSampleRate] = useState<number>(8000);
  const [tracedRidges, setTracedRidges] = useState<TracedRidge[]>([]);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0.0);
  const [duration, setDuration] = useState<number>(0.0);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  const [selectedBasin, setSelectedBasin] = useState<BasinSummary | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const flightImportRef = useRef<HTMLInputElement | null>(null);

  // Setup HTMLAudioElement
  useEffect(() => {
    if (!audioElRef.current) {
      audioElRef.current = new Audio();
    }
    const audio = audioElRef.current;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => {
      if (audio.duration && Number.isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  // Compute STFT Spectrogram from uploaded media
  const spectrogram: SpectrogramData | null = useMemo(() => {
    if (!rawPcmBuffer) return null;
    let processed: Float32Array = rawPcmBuffer;
    if (highPass80Hz) {
      processed = applyHighPassFilter(processed, sampleRate, 80.0);
    }
    if (bandBoostDb > 0.5) {
      processed = applyAircraftBandBoost(processed, sampleRate, bandBoostDb);
    }
    if (Math.abs(injectedDriftPpm) > 0.1) {
      processed = applyClockDrift(processed, injectedDriftPpm);
    }
    return computeSpectrogram(processed, sampleRate, 512, 128, 500.0);
  }, [rawPcmBuffer, sampleRate, highPass80Hz, bandBoostDb, injectedDriftPpm]);

  // Trace Viterbi ridges when spectrogram updates
  useEffect(() => {
    if (!spectrogram) {
      setTracedRidges([]);
      return;
    }
    const r1 = traceDopplerRidge(spectrogram, 55.0, 140.0);
    const r2 = traceDopplerRidge(spectrogram, 140.0, 260.0);
    const ridges: TracedRidge[] = [];
    if (r1) ridges.push(r1);
    if (r2) ridges.push(r2);
    setTracedRidges(ridges);
  }, [spectrogram]);

  // Run Geolocation Solver on user's events and flights
  const inversionResult: InversionResult = useMemo(() => {
    return runGeolocationInversion({
      bounds,
      atmos: {
        temp_c: tempC,
        wind_e_ms: windEMs,
        wind_n_ms: windNMs,
      },
      flights,
      events,
      injectedClockDriftPpm: injectedDriftPpm,
    });
  }, [bounds, tempC, windEMs, windNMs, flights, events, injectedDriftPpm]);

  const handleTogglePlay = useCallback(() => {
    const audio = audioElRef.current;
    if (!audio || (!rawPcmBuffer && !uploadedVideoUrl)) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.muted = isMuted;
      audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(true));
    }
  }, [isPlaying, isMuted, rawPcmBuffer, uploadedVideoUrl]);

  useEffect(() => {
    if (!isPlaying || duration <= 0) return;
    const timer = setInterval(() => {
      if (audioElRef.current && !audioElRef.current.paused) return;
      setCurrentTime((t) => (t + 0.25 >= duration ? 0 : t + 0.25));
    }, 250);
    return () => clearInterval(timer);
  }, [isPlaying, duration]);

  const handleSeek = useCallback(
    (timeS: number) => {
      const clamped = Math.max(0, Math.min(Math.max(1, duration), timeS));
      setCurrentTime(clamped);
      if (audioElRef.current) {
        audioElRef.current.currentTime = clamped;
      }
    },
    [duration]
  );

  const handleToggleMute = useCallback(() => {
    setIsMuted((m) => {
      const next = !m;
      if (audioElRef.current) audioElRef.current.muted = next;
      return next;
    });
  }, []);

  const handleClearMedia = useCallback(() => {
    if (audioElRef.current) {
      audioElRef.current.pause();
      audioElRef.current.src = '';
    }
    setUploadedVideoUrl(null);
    setUploadedFileName(null);
    setRawPcmBuffer(null);
    setTracedRidges([]);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    setUploadedFileName(file.name);
    const url = URL.createObjectURL(file);
    if (
      file.type.startsWith('video/') ||
      /\.(mp4|webm|mov)$/i.test(file.name)
    ) {
      setUploadedVideoUrl(url);
    } else {
      setUploadedVideoUrl(null);
    }
    if (audioElRef.current) {
      audioElRef.current.src = url;
      audioElRef.current.load();
    }

    try {
      const arrayBuf = await file.arrayBuffer();
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: 8000 });
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const chan = decoded.getChannelData(0);
      const maxLen = Math.min(chan.length, 8000 * 120);
      const slice = new Float32Array(maxLen);
      slice.set(chan.subarray(0, maxLen));
      setRawPcmBuffer(slice);
      setSampleRate(decoded.sampleRate);
      const dur = maxLen / decoded.sampleRate;
      setDuration(dur);
      await ctx.close();
    } catch {
      setRawPcmBuffer(null);
    }
  }, []);

  const handleAddEvent = () => {
    const nextIdx = events.length + 1;
    const defaultCpa = duration > 0 ? Number((duration * 0.5).toFixed(1)) : 20.0;
    const newEv: FittedDopplerEvent = {
      id: `EV0${nextIdx}`,
      f0_hz: 85.0,
      speed_ms: 100.0,
      slant_m: 1500.0,
      t_cpa_s: defaultCpa,
      snr_db: 35.0,
      orders: [1, 2, 3],
      curves: [],
    };
    setEvents((prev) => [...prev, newEv]);
    setActiveEventIndex(events.length);
  };

  const handleAutoFitFromSpectrogram = () => {
    if (tracedRidges.length === 0) return;
    const fitted = fitDopplerEventFromRidge(
      tracedRidges[0],
      `EV0${events.length + 1}`,
      0.0
    );
    if (!fitted) return;
    if (events.length === 0) {
      setEvents([fitted]);
      setActiveEventIndex(0);
    } else {
      setEvents((prev) =>
        prev.map((ev, i) =>
          i === activeEventIndex
            ? {
                ...ev,
                f0_hz: fitted.f0_hz,
                speed_ms: fitted.speed_ms,
                slant_m: fitted.slant_m,
                t_cpa_s: fitted.t_cpa_s,
              }
            : ev
        )
      );
    }
  };

  const updateActiveEvent = (patch: Partial<FittedDopplerEvent>) => {
    setEvents((prev) =>
      prev.map((ev, idx) =>
        idx === activeEventIndex ? { ...ev, ...patch } : ev
      )
    );
  };

  const handleRemoveEvent = (idxToRemove: number) => {
    setEvents((prev) => prev.filter((_, idx) => idx !== idxToRemove));
    setActiveEventIndex(0);
  };

  const [flightSortField, setFlightSortField] =
    useState<FlightSortField | null>(null);
  const [flightSortDir, setFlightSortDir] =
    useState<FlightSortDirection>('asc');

  const handleToggleFlightSort = (field: FlightSortField) => {
    if (flightSortField === field) {
      setFlightSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setFlightSortField(field);
      setFlightSortDir('asc');
    }
  };

  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [overlay3dTrajectory, setOverlay3dTrajectory] =
    useState<boolean>(false);

  const sortedFlights = useMemo(
    () => sortFlights(flights, flightSortField, flightSortDir),
    [flights, flightSortField, flightSortDir]
  );

  const handleAddFlight = () => {
    const nextNum = flights.length + 1;
    const centerLat = Number(((bounds.south + bounds.north) * 0.5).toFixed(4));
    const centerLon = Number(((bounds.west + bounds.east) * 0.5).toFixed(4));
    const speedMs = 85 + (((nextNum - 1) * 18) % 65);
    const altM = 1100 + (((nextNum - 1) * 450) % 2200);
    const newFlight: ScenarioFlight = {
      id: `FL-${Date.now()}-${nextNum}`,
      callsign: `N10${nextNum}SP`,
      aircraft: nextNum % 2 === 1 ? 'C172' : 'SR22',
      aircraft_name: nextNum % 2 === 1 ? 'Cessna 172' : 'Cirrus SR22',
      heading_deg: nextNum % 2 === 1 ? 90 : 210,
      speed_ms: speedMs,
      speed_kts: Math.round(speedMs * 1.94384),
      alt_m: altM,
      bpf_nominal_hz: 85,
      is_target: true,
      cpa: {
        lat: centerLat,
        lon: centerLon,
        time_s: 20,
        slant_m: altM,
      },
      waypoints: [],
      notes: '',
    };
    setFlights((prev) => [...prev, newFlight]);
    setSelectedFlightId((prev) => prev || newFlight.id);
  };

  const handleUpdateFlight = (
    flightId: string,
    patch: Partial<ScenarioFlight>,
    cpaPatch?: Partial<NonNullable<ScenarioFlight['cpa']>>
  ) => {
    const resetsKinematics =
      cpaPatch !== undefined ||
      patch.heading_deg !== undefined ||
      patch.speed_ms !== undefined ||
      patch.alt_m !== undefined;

    setFlights((prev) =>
      prev.map((fl) => {
        if (fl.id !== flightId) return fl;
        const nextCpa = fl.cpa
          ? { ...fl.cpa, ...(cpaPatch || {}) }
          : {
              lat: (bounds.south + bounds.north) * 0.5,
              lon: (bounds.west + bounds.east) * 0.5,
              time_s: 20,
              slant_m: fl.alt_m,
              ...(cpaPatch || {}),
            };
        return {
          ...fl,
          ...patch,
          cpa: nextCpa,
          waypoints: resetsKinematics ? [] : fl.waypoints,
        };
      })
    );
  };

  const [isFetchingAdsb, setIsFetchingAdsb] = useState<boolean>(false);
  const [adsbStatusMsg, setAdsbStatusMsg] = useState<string | null>(null);

  const handleFetchLiveOpenSky = async () => {
    setIsFetchingAdsb(true);
    setAdsbStatusMsg(null);
    try {
      const url = `https://opensky-network.org/api/states/all?lamin=${bounds.south}&lomin=${bounds.west}&lamax=${bounds.north}&lomax=${bounds.east}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      const states: unknown[][] = Array.isArray(data?.states) ? data.states : [];
      const airborne = states.filter(
        (s) =>
          s[5] !== null &&
          s[6] !== null &&
          !s[8] && // on_ground === false
          typeof s[9] === 'number' &&
          s[9] > 25 &&
          s[9] < 165 // prop speed range (25..165 m/s)
      );

      if (airborne.length === 0) {
        setAdsbStatusMsg('No airborne propeller-speed aircraft in bounds right now.');
      } else {
        const imported: ScenarioFlight[] = airborne.slice(0, 12).map((s, idx) => {
          const icao24 = String(s[0] || `OS${idx}`);
          const callsign = String(s[1] || icao24).trim() || icao24;
          const lon = Number(s[5]);
          const lat = Number(s[6]);
          const baroAlt = typeof s[7] === 'number' ? s[7] : 1500;
          const velocityMs = typeof s[9] === 'number' ? s[9] : 95;
          const headingDeg = typeof s[10] === 'number' ? s[10] : 90;

          return {
            id: `${icao24}-${idx}`,
            callsign,
            aircraft: velocityMs < 105 ? 'C172' : 'SR22',
            aircraft_name: 'OpenSky Live ADS-B',
            heading_deg: Number(headingDeg.toFixed(1)),
            speed_ms: Number(velocityMs.toFixed(1)),
            speed_kts: Number((velocityMs * 1.94384).toFixed(1)),
            alt_m: Number(baroAlt.toFixed(0)),
            bpf_nominal_hz: velocityMs < 105 ? 85 : 115,
            is_target: idx < 2,
            cpa: {
              lat: Number(lat.toFixed(4)),
              lon: Number(lon.toFixed(4)),
              time_s: 20 + idx * 15,
              slant_m: Number(Math.max(400, baroAlt).toFixed(0)),
            },
            waypoints: [],
          };
        });
        setFlights(imported);
        setAdsbStatusMsg(`Loaded ${imported.length} live aircraft from OpenSky.`);
      }
    } catch {
      setAdsbStatusMsg('OpenSky rate-limited or unreachable; import JSON/CSV or add manually.');
    } finally {
      setIsFetchingAdsb(false);
    }
  };

  const handleImportFlightsJson = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.csv')) {
        // Parse CSV (e.g. callsign,aircraft,heading,speed_ms,alt_m,lat,lon,time_s)
        const lines = text
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter(Boolean);
        if (lines.length > 1) {
          const headers = lines[0].toLowerCase().split(',');
          const parsedFlights: ScenarioFlight[] = lines.slice(1).map((line, idx) => {
            const cols = line.split(',');
            const getCol = (name: string, fallback: string) => {
              const i = headers.findIndex((h) => h.includes(name));
              return i >= 0 && cols[i] !== undefined ? cols[i].trim() : fallback;
            };
            const lat = parseFloat(getCol('lat', String((bounds.south + bounds.north) * 0.5))) || 0;
            const lon = parseFloat(getCol('lon', String((bounds.west + bounds.east) * 0.5))) || 0;
            const alt = parseFloat(getCol('alt', '1500')) || 1500;
            const spd = parseFloat(getCol('speed', '100')) || 100;
            const hdg = parseFloat(getCol('head', '90')) || 90;
            const timeS = parseFloat(getCol('time', '20')) || 20;
            return {
              id: `CSV-${idx + 1}`,
              callsign: getCol('callsign', `FL0${idx + 1}`),
              aircraft: getCol('aircraft', 'C172'),
              aircraft_name: 'CSV Import',
              heading_deg: hdg,
              speed_ms: spd,
              speed_kts: Number((spd * 1.94384).toFixed(1)),
              alt_m: alt,
              bpf_nominal_hz: 85,
              is_target: true,
              cpa: { lat, lon, time_s: timeS, slant_m: alt },
              waypoints: [],
            };
          });
          if (parsedFlights.length > 0) setFlights(parsedFlights);
        }
        return;
      }

      const parsed = JSON.parse(text);
      const list = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed.flights)
        ? parsed.flights
        : [];
      if (list.length > 0) {
        setFlights(list);
      }
      if (parsed.bounds) {
        setBounds(parsed.bounds);
      }
      if (Array.isArray(parsed.events) && parsed.events.length > 0) {
        setEvents(parsed.events);
      }
    } catch {
      // Ignore malformed file
    }
  };

  const handleExportGeoJson = () => {
    const geojson = {
      type: 'FeatureCollection',
      metadata: {
        status: inversionResult.status,
        score: inversionResult.score,
        radius_m: inversionResult.radius_m,
      },
      features:
        inversionResult.basins.length > 0
          ? [
              {
                type: 'Feature',
                properties: {
                  role: 'estimated_fix',
                  status: inversionResult.status,
                  score: inversionResult.score,
                  radius_m: inversionResult.radius_m,
                },
                geometry: {
                  type: 'Point',
                  coordinates: [
                    inversionResult.fix.lon,
                    inversionResult.fix.lat,
                  ],
                },
              },
            ]
          : [],
    };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], {
      type: 'application/geo+json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'skyprint_fix.geojson';
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeEv =
    events[Math.min(activeEventIndex, Math.max(0, events.length - 1))];
  const effectiveSoundSpeed = pathSoundSpeed(0, 1800, tempC);

  return (
    <div className="flex flex-col h-screen w-screen bg-[#07090e] text-slate-100 select-none overflow-hidden">
      {/* Clean Single Top Bar */}
      <header className="h-12 px-5 bg-[#0b101b] border-b border-slate-800/90 flex items-center justify-between gap-8 shrink-0">
        <a
          href="#spectrogram"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('spectrogram');
          }}
          className="text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          SkyPrint
        </a>

        <nav
          aria-label="Workspace Tabs"
          className="flex items-center gap-6 text-xs font-medium text-slate-400"
        >
          <a
            href="#spectrogram"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('spectrogram');
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'spectrogram'
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'hover:text-slate-100'
            }`}
          >
            Spectrogram
          </a>
          <a
            href="#geolocation"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('geolocation');
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'geolocation'
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'hover:text-slate-100'
            }`}
          >
            Geolocation
          </a>
          <a
            href="#flights"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('flights');
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'flights'
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'hover:text-slate-100'
            }`}
          >
            Flight Tracks ({flights.length})
          </a>
        </nav>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleExportGeoJson}
            disabled={inversionResult.basins.length === 0}
            className="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-100 rounded border border-slate-700 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Export GeoJSON
          </button>
        </div>
      </header>

      {/* TAB 1: SPECTROGRAM & MEDIA */}
      {activeTab === 'spectrogram' && (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr] min-h-0 overflow-hidden">
          <aside className="bg-[#0b101b] border-r border-slate-800/80 flex flex-col min-h-0 overflow-y-auto p-4 gap-4">
            <ForensicVideoMonitor
              isPlaying={isPlaying}
              currentTime={currentTime}
              duration={duration}
              onTogglePlay={handleTogglePlay}
              onSeek={handleSeek}
              onFileUpload={handleFileUpload}
              onClearMedia={handleClearMedia}
              uploadedVideoUrl={uploadedVideoUrl}
              uploadedFileName={uploadedFileName}
              rawPcmBuffer={rawPcmBuffer}
              sampleRate={sampleRate}
              isMuted={isMuted}
              onToggleMute={handleToggleMute}
            />

            {/* Doppler Events Section */}
            <div className="border-b border-slate-800/80 pb-4 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">
                  Doppler Curves
                </span>
                <div className="flex items-center gap-1.5">
                  {tracedRidges.length > 0 && (
                    <button
                      type="button"
                      onClick={handleAutoFitFromSpectrogram}
                      className="px-2 py-1 text-[11px] font-mono bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-700 rounded flex items-center gap-1 cursor-pointer"
                    >
                      <Sparkles className="w-3 h-3" />
                      Auto-Fit
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleAddEvent}
                    className="px-2 py-1 text-[11px] font-medium bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded flex items-center gap-1 cursor-pointer"
                  >
                    <Plus className="w-3 h-3" />
                    Add Curve
                  </button>
                </div>
              </div>

              {events.length === 0 ? (
                <div className="text-xs text-slate-500 py-2">
                  No Doppler curves added yet.
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {events.map((ev, idx) => (
                      <div key={ev.id} className="flex items-center">
                        <button
                          type="button"
                          onClick={() => setActiveEventIndex(idx)}
                          className={`px-2 py-1 text-[11px] font-mono rounded-l border cursor-pointer ${
                            activeEventIndex === idx
                              ? 'bg-cyan-950/70 text-cyan-300 border-cyan-500/50'
                              : 'bg-slate-900 text-slate-400 border-slate-800'
                          }`}
                        >
                          {ev.id}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleRemoveEvent(idx)}
                          aria-label={`Remove ${ev.id}`}
                          className="px-1.5 py-1 text-[11px] bg-slate-900 hover:text-rose-400 text-slate-500 border-y border-r border-slate-800 rounded-r cursor-pointer"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>

                  {activeEv && (
                    <div className="bg-[#0e1422] border border-slate-800/90 rounded p-3 flex flex-col gap-2.5">
                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-slate-400">f0</span>
                          <span className="text-slate-100 tabular-nums">
                            {activeEv.f0_hz.toFixed(1)} Hz
                          </span>
                        </div>
                        <input
                          type="range"
                          min={35}
                          max={160}
                          step={0.5}
                          value={activeEv.f0_hz}
                          onChange={(e) =>
                            updateActiveEvent({
                              f0_hz: parseFloat(e.target.value),
                            })
                          }
                          aria-label="f0 frequency"
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-slate-400">Speed</span>
                          <span className="text-slate-100 tabular-nums">
                            {activeEv.speed_ms.toFixed(0)} m/s
                          </span>
                        </div>
                        <input
                          type="range"
                          min={45}
                          max={160}
                          step={1}
                          value={activeEv.speed_ms}
                          onChange={(e) =>
                            updateActiveEvent({
                              speed_ms: parseFloat(e.target.value),
                            })
                          }
                          aria-label="Speed"
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-slate-400">Slant Range</span>
                          <span className="text-slate-100 tabular-nums">
                            {activeEv.slant_m.toFixed(0)} m
                          </span>
                        </div>
                        <input
                          type="range"
                          min={200}
                          max={4500}
                          step={25}
                          value={activeEv.slant_m}
                          onChange={(e) =>
                            updateActiveEvent({
                              slant_m: parseFloat(e.target.value),
                            })
                          }
                          aria-label="Slant range"
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-slate-400">CPA Time</span>
                          <span className="text-slate-100 tabular-nums">
                            {activeEv.t_cpa_s.toFixed(1)} s
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={Math.max(60, duration || 60)}
                          step={0.5}
                          value={activeEv.t_cpa_s}
                          onChange={(e) =>
                            updateActiveEvent({
                              t_cpa_s: parseFloat(e.target.value),
                            })
                          }
                          aria-label="CPA Time"
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                        />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Audio Filters */}
            <div className="flex flex-col gap-2.5">
              <span className="text-xs font-semibold text-slate-200">
                Audio Filters
              </span>
              <div className="bg-[#0e1422] border border-slate-800/90 rounded p-3 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-300">
                    80 Hz High-Pass
                  </span>
                  <button
                    type="button"
                    onClick={() => setHighPass80Hz((v) => !v)}
                    className={`px-2.5 py-1 text-[11px] font-mono rounded border cursor-pointer ${
                      highPass80Hz
                        ? 'bg-rose-950/70 text-rose-200 border-rose-500/50'
                        : 'bg-slate-900 text-slate-400 border-slate-800'
                    }`}
                  >
                    {highPass80Hz ? 'On' : 'Off'}
                  </button>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] font-mono mb-1">
                    <span className="text-slate-400">Sub-Band Boost</span>
                    <span className="text-slate-100 tabular-nums">
                      +{bandBoostDb.toFixed(0)} dB
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={18}
                    step={2}
                    value={bandBoostDb}
                    onChange={(e) => setBandBoostDb(Number(e.target.value))}
                    aria-label="Sub-band boost"
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] font-mono mb-1">
                    <span className="text-slate-400">Clock Drift</span>
                    <span className="text-slate-100 tabular-nums">
                      {injectedDriftPpm >= 0 ? '+' : ''}
                      {injectedDriftPpm.toFixed(1)} ppm
                    </span>
                  </div>
                  <input
                    type="range"
                    min={-20}
                    max={20}
                    step={1}
                    value={injectedDriftPpm}
                    onChange={(e) =>
                      setInjectedDriftPpm(Number(e.target.value))
                    }
                    aria-label="Clock drift"
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>
              </div>
            </div>
          </aside>

          <main className="flex flex-col min-h-0 overflow-hidden bg-[#05080f]">
            <SpectrogramViewport
              spectrogram={spectrogram}
              events={events}
              tracedRidges={tracedRidges}
              currentTime={currentTime}
              duration={duration}
              onSeek={handleSeek}
              showNaiveComparison={showNaiveComparison}
              onToggleNaiveComparison={() => setShowNaiveComparison((v) => !v)}
              showWhitened={showWhitened}
              onToggleWhitened={() => setShowWhitened((v) => !v)}
              activeEventIndex={activeEventIndex}
              onSelectEventIndex={setActiveEventIndex}
              highPass80HzActive={highPass80Hz}
              bandBoostDb={bandBoostDb}
            />
          </main>
        </div>
      )}

      {/* TAB 2: GEOLOCATION MAP */}
      {activeTab === 'geolocation' && (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr] min-h-0 overflow-hidden">
          <aside className="bg-[#0b101b] border-r border-slate-800/80 flex flex-col min-h-0 overflow-y-auto p-4 gap-4">
            {/* Result Summary */}
            <div className="bg-[#0e1422] border border-slate-800/90 rounded p-3.5 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">
                  Estimated Fix
                </span>
                <span
                  data-testid="status-indicator"
                  className={`text-[11px] font-mono font-semibold ${
                    inversionResult.status === 'located'
                      ? 'text-emerald-400'
                      : inversionResult.status === 'ambiguous'
                      ? 'text-amber-400'
                      : 'text-slate-400'
                  }`}
                >
                  {inversionResult.status.toUpperCase()}
                </span>
              </div>

              {inversionResult.basins.length === 0 ? (
                <p className="text-xs text-slate-500 py-1">
                  Add Doppler curves and flight tracks to compute a fix.
                </p>
              ) : (
                <div className="flex flex-col gap-1.5 text-xs font-mono tabular-nums pt-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Coordinates</span>
                    <span className="text-slate-100">
                      {inversionResult.fix.lat.toFixed(4)}°,{' '}
                      {inversionResult.fix.lon.toFixed(4)}°
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">95% Radius</span>
                    <span className="text-slate-100">
                      ±{inversionResult.radius_m.toFixed(0)} m
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Score</span>
                    <span className="text-cyan-400">
                      {inversionResult.score.toFixed(2)}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Search Bounds */}
            <div className="border-b border-slate-800/80 pb-4 flex flex-col gap-2">
              <span className="text-xs font-semibold text-slate-200">
                Search Bounds
              </span>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <label className="flex flex-col gap-1 text-slate-400">
                  <span>North Lat</span>
                  <input
                    type="number"
                    step="0.05"
                    value={bounds.north}
                    onChange={(e) =>
                      setBounds((b) => ({
                        ...b,
                        north: parseFloat(e.target.value) || 0,
                      }))
                    }
                    className="bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                  />
                </label>
                <label className="flex flex-col gap-1 text-slate-400">
                  <span>South Lat</span>
                  <input
                    type="number"
                    step="0.05"
                    value={bounds.south}
                    onChange={(e) =>
                      setBounds((b) => ({
                        ...b,
                        south: parseFloat(e.target.value) || 0,
                      }))
                    }
                    className="bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                  />
                </label>
                <label className="flex flex-col gap-1 text-slate-400">
                  <span>West Lon</span>
                  <input
                    type="number"
                    step="0.05"
                    value={bounds.west}
                    onChange={(e) =>
                      setBounds((b) => ({
                        ...b,
                        west: parseFloat(e.target.value) || 0,
                      }))
                    }
                    className="bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                  />
                </label>
                <label className="flex flex-col gap-1 text-slate-400">
                  <span>East Lon</span>
                  <input
                    type="number"
                    step="0.05"
                    value={bounds.east}
                    onChange={(e) =>
                      setBounds((b) => ({
                        ...b,
                        east: parseFloat(e.target.value) || 0,
                      }))
                    }
                    className="bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                  />
                </label>
              </div>
            </div>

            {/* Atmosphere */}
            <div className="border-b border-slate-800/80 pb-4 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">
                  Atmosphere
                </span>
                <span className="text-[11px] font-mono text-cyan-400">
                  c={effectiveSoundSpeed.toFixed(1)} m/s
                </span>
              </div>
              <div className="bg-[#0e1422] border border-slate-800/90 rounded p-3 flex flex-col gap-2.5">
                <div>
                  <div className="flex justify-between text-[11px] font-mono mb-1">
                    <span className="text-slate-400">Temp</span>
                    <span className="text-slate-100">{tempC.toFixed(1)} °C</span>
                  </div>
                  <input
                    type="range"
                    min={-25}
                    max={40}
                    step={1}
                    value={tempC}
                    onChange={(e) => setTempC(parseFloat(e.target.value))}
                    aria-label="Temperature"
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="flex justify-between text-[11px] font-mono mb-1">
                      <span className="text-slate-400">Wind E</span>
                      <span className="text-slate-100">
                        {windEMs.toFixed(1)} m/s
                      </span>
                    </div>
                    <input
                      type="range"
                      min={-15}
                      max={15}
                      step={0.5}
                      value={windEMs}
                      onChange={(e) => setWindEMs(parseFloat(e.target.value))}
                      aria-label="Wind East"
                      className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] font-mono mb-1">
                      <span className="text-slate-400">Wind N</span>
                      <span className="text-slate-100">
                        {windNMs.toFixed(1)} m/s
                      </span>
                    </div>
                    <input
                      type="range"
                      min={-15}
                      max={15}
                      step={0.5}
                      value={windNMs}
                      onChange={(e) => setWindNMs(parseFloat(e.target.value))}
                      aria-label="Wind North"
                      className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Candidate Basins */}
            {inversionResult.basins.length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-xs font-semibold text-slate-200">
                  Candidate Basins
                </span>
                <div className="bg-[#0e1422] border border-slate-800/90 rounded overflow-hidden">
                  <table className="w-full text-left font-mono text-[11px] tabular-nums">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 bg-slate-900/50">
                        <th className="py-1.5 px-2">#</th>
                        <th className="py-1.5 px-2">Coordinates</th>
                        <th className="py-1.5 px-2 text-right">Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {inversionResult.basins.slice(0, 6).map((b) => (
                        <tr
                          key={b.rank}
                          onClick={() => setSelectedBasin(b)}
                          className={`cursor-pointer transition-colors ${
                            selectedBasin?.rank === b.rank
                              ? 'bg-cyan-950/40 text-cyan-200'
                              : 'hover:bg-slate-800/40 text-slate-300'
                          }`}
                        >
                          <td className="py-1.5 px-2 font-semibold">
                            {b.rank}
                          </td>
                          <td className="py-1.5 px-2">
                            {b.lat.toFixed(4)}°, {b.lon.toFixed(4)}°
                          </td>
                          <td className="py-1.5 px-2 text-right">
                            {b.score.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </aside>

          <main className="flex flex-col min-h-0 overflow-hidden bg-[#05080f]">
            <GeodeticMapViewport
              bounds={bounds}
              onChangeBounds={setBounds}
              fix={inversionResult.fix}
              radiusM={inversionResult.radius_m}
              status={inversionResult.status}
              flights={flights}
              candidates={inversionResult.candidates}
              basins={inversionResult.basins}
              selectedBasin={selectedBasin}
              onSelectBasin={setSelectedBasin}
              currentTime={currentTime}
              duration={duration}
              selectedFlightId={selectedFlightId}
              onSelectFlight={setSelectedFlightId}
              show3dTrajectory={overlay3dTrajectory}
              onToggle3dTrajectory={() => setOverlay3dTrajectory((v) => !v)}
            />
          </main>
        </div>
      )}

      {/* TAB 3: FLIGHT TRACKS */}
      {activeTab === 'flights' && (
        <main className="flex-1 overflow-y-auto p-6 bg-[#07090e]">
          <div className="max-w-7xl mx-auto flex flex-col gap-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-3">
                <h1 className="text-sm font-semibold text-slate-100">
                  ADS-B Flight Tracks
                </h1>
                {adsbStatusMsg && (
                  <span className="text-xs font-mono text-cyan-400">
                    {adsbStatusMsg}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button
                  type="button"
                  data-testid="toggle-3d-trajectory-overlay"
                  aria-pressed={overlay3dTrajectory}
                  onClick={() => setOverlay3dTrajectory((v) => !v)}
                  className={`px-3 py-1.5 text-xs font-medium rounded border flex items-center gap-1.5 transition-colors cursor-pointer ${
                    overlay3dTrajectory
                      ? 'bg-purple-950/70 text-purple-200 border-purple-500/60'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                  }`}
                >
                  <Box className="w-3.5 h-3.5 text-purple-400" />
                  <span>
                    {overlay3dTrajectory
                      ? '3D Trajectory Overlay: ON'
                      : 'Overlay 3D Trajectory'}
                  </span>
                </button>
                {overlay3dTrajectory && flights.length > 0 && (
                  <button
                    type="button"
                    data-testid="view-3d-on-map-btn"
                    onClick={() => setActiveTab('geolocation')}
                    className="px-3 py-1.5 text-xs font-medium bg-purple-600/20 hover:bg-purple-600/30 text-purple-200 rounded border border-purple-500/40 flex items-center gap-1.5 cursor-pointer"
                  >
                    <MapPin className="w-3.5 h-3.5 text-purple-400" />
                    <span>View on Map</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleFetchLiveOpenSky}
                  disabled={isFetchingAdsb}
                  className="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-cyan-300 rounded border border-slate-700 flex items-center gap-1.5 cursor-pointer"
                >
                  {isFetchingAdsb ? 'Fetching...' : 'Fetch Live OpenSky'}
                </button>
                <button
                  type="button"
                  onClick={() => flightImportRef.current?.click()}
                  className="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded border border-slate-700 flex items-center gap-1.5 cursor-pointer"
                >
                  <Upload className="w-3.5 h-3.5 text-cyan-400" />
                  Import JSON / CSV
                </button>
                <input
                  ref={flightImportRef}
                  type="file"
                  accept=".json,.csv,application/json,text/csv"
                  onChange={handleImportFlightsJson}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={handleAddFlight}
                  className="px-3 py-1.5 text-xs font-medium bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded flex items-center gap-1.5 cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add Flight
                </button>
              </div>
            </div>

            {flights.length === 0 ? (
              <div className="bg-[#0b101b] border border-slate-800 rounded p-12 text-center flex flex-col items-center gap-3">
                <p className="text-xs text-slate-400">
                  No flight tracks loaded. Fetch live OpenSky ADS-B for your map
                  bounds, import a JSON/CSV file, or add a flight manually.
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleFetchLiveOpenSky}
                    disabled={isFetchingAdsb}
                    className="px-3.5 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded border border-slate-700 cursor-pointer"
                  >
                    {isFetchingAdsb ? 'Fetching...' : 'Fetch Live OpenSky'}
                  </button>
                  <button
                    type="button"
                    onClick={handleAddFlight}
                    className="px-3.5 py-1.5 text-xs font-medium bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded flex items-center gap-1.5 cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add Flight
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-[#0b101b] border border-slate-800 rounded overflow-x-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 bg-slate-900/60">
                      <th className="py-2.5 px-2.5 whitespace-nowrap">3D Path</th>
                      <th
                        className="py-2 px-2.5 whitespace-nowrap"
                        aria-sort={
                          flightSortField === 'callsign'
                            ? flightSortDir === 'asc'
                              ? 'ascending'
                              : 'descending'
                            : 'none'
                        }
                      >
                        <button
                          type="button"
                          data-testid="sort-flights-callsign"
                          onClick={() => handleToggleFlightSort('callsign')}
                          className={`inline-flex items-center gap-1.5 font-semibold transition-colors cursor-pointer ${
                            flightSortField === 'callsign'
                              ? 'text-cyan-300'
                              : 'text-slate-300 hover:text-slate-100'
                          }`}
                        >
                          <span>Callsign</span>
                          {flightSortField === 'callsign' ? (
                            flightSortDir === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                            ) : (
                              <ArrowDown className="w-3.5 h-3.5 text-cyan-400" />
                            )
                          ) : (
                            <ArrowUpDown className="w-3.5 h-3.5 text-slate-500" />
                          )}
                        </button>
                      </th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">Aircraft</th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">Heading (°)</th>
                      <th
                        className="py-2 px-2.5 whitespace-nowrap"
                        aria-sort={
                          flightSortField === 'speed'
                            ? flightSortDir === 'asc'
                              ? 'ascending'
                              : 'descending'
                            : 'none'
                        }
                      >
                        <button
                          type="button"
                          data-testid="sort-flights-speed"
                          onClick={() => handleToggleFlightSort('speed')}
                          className={`inline-flex items-center gap-1.5 font-semibold transition-colors cursor-pointer ${
                            flightSortField === 'speed'
                              ? 'text-cyan-300'
                              : 'text-slate-300 hover:text-slate-100'
                          }`}
                        >
                          <span>Speed (m/s)</span>
                          {flightSortField === 'speed' ? (
                            flightSortDir === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                            ) : (
                              <ArrowDown className="w-3.5 h-3.5 text-cyan-400" />
                            )
                          ) : (
                            <ArrowUpDown className="w-3.5 h-3.5 text-slate-500" />
                          )}
                        </button>
                      </th>
                      <th
                        className="py-2 px-2.5 whitespace-nowrap"
                        aria-sort={
                          flightSortField === 'alt'
                            ? flightSortDir === 'asc'
                              ? 'ascending'
                              : 'descending'
                            : 'none'
                        }
                      >
                        <button
                          type="button"
                          data-testid="sort-flights-alt"
                          onClick={() => handleToggleFlightSort('alt')}
                          className={`inline-flex items-center gap-1.5 font-semibold transition-colors cursor-pointer ${
                            flightSortField === 'alt'
                              ? 'text-cyan-300'
                              : 'text-slate-300 hover:text-slate-100'
                          }`}
                        >
                          <span>Alt (m)</span>
                          {flightSortField === 'alt' ? (
                            flightSortDir === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                            ) : (
                              <ArrowDown className="w-3.5 h-3.5 text-cyan-400" />
                            )
                          ) : (
                            <ArrowUpDown className="w-3.5 h-3.5 text-slate-500" />
                          )}
                        </button>
                      </th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">CPA Lat</th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">CPA Lon</th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">CPA Time (s)</th>
                      <th className="py-2.5 px-2.5 whitespace-nowrap">Pilot Notes / Comments</th>
                      <th className="py-2.5 px-2.5 text-right whitespace-nowrap">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {sortedFlights.map((fl) => {
                      const isSelectedFor3d =
                        (selectedFlightId || flights[0]?.id) === fl.id;
                      return (
                        <tr
                          key={fl.id}
                          onClick={() => setSelectedFlightId(fl.id)}
                          className={`transition-colors ${
                            isSelectedFor3d && overlay3dTrajectory
                              ? 'bg-purple-950/20 hover:bg-purple-950/30'
                              : isSelectedFor3d
                              ? 'bg-slate-900/50 hover:bg-slate-900/70'
                              : 'hover:bg-slate-900/30'
                          }`}
                        >
                          <td className="py-2 px-2.5">
                            <button
                              type="button"
                              data-testid="select-flight-3d-btn"
                              aria-pressed={
                                isSelectedFor3d && overlay3dTrajectory
                              }
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isSelectedFor3d && overlay3dTrajectory) {
                                  setOverlay3dTrajectory(false);
                                } else {
                                  setSelectedFlightId(fl.id);
                                  setOverlay3dTrajectory(true);
                                }
                              }}
                              className={`px-2 py-1 text-[11px] font-mono rounded border flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer ${
                                isSelectedFor3d && overlay3dTrajectory
                                  ? 'bg-purple-950/80 text-purple-200 border-purple-500/60'
                                  : isSelectedFor3d
                                  ? 'bg-slate-800 text-cyan-300 border-slate-700'
                                  : 'bg-[#0e1422] text-slate-400 border-slate-800 hover:text-slate-200'
                              }`}
                            >
                              <Box className="w-3 h-3 shrink-0" />
                              <span>
                                {isSelectedFor3d && overlay3dTrajectory
                                  ? '3D Active'
                                  : '3D Path'}
                              </span>
                            </button>
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="text"
                              aria-label="Flight callsign"
                              value={fl.callsign}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  callsign: e.target.value,
                                })
                              }
                              className="w-20 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <select
                              aria-label="Aircraft type"
                              value={fl.aircraft}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  aircraft: e.target.value,
                                })
                              }
                              className="bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            >
                              <option value="C172">C172</option>
                              <option value="C182">C182</option>
                              <option value="SR22">SR22</option>
                              <option value="DA40">DA40</option>
                              <option value="PA34">PA34</option>
                              <option value="C208">C208</option>
                            </select>
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              aria-label="Heading degrees"
                              value={fl.heading_deg}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  heading_deg: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="w-16 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              aria-label="Flight speed m/s"
                              value={fl.speed_ms}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  speed_ms: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="w-16 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              aria-label="Flight altitude meters"
                              value={fl.alt_m}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  alt_m: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="w-16 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              step="0.01"
                              aria-label="CPA latitude"
                              value={fl.cpa?.lat ?? 0}
                              onChange={(e) =>
                                handleUpdateFlight(
                                  fl.id,
                                  {},
                                  { lat: parseFloat(e.target.value) || 0 }
                                )
                              }
                              className="w-20 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              step="0.01"
                              aria-label="CPA longitude"
                              value={fl.cpa?.lon ?? 0}
                              onChange={(e) =>
                                handleUpdateFlight(
                                  fl.id,
                                  {},
                                  { lon: parseFloat(e.target.value) || 0 }
                                )
                              }
                              className="w-20 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5">
                            <input
                              type="number"
                              aria-label="CPA time seconds"
                              value={fl.cpa?.time_s ?? 0}
                              onChange={(e) =>
                                handleUpdateFlight(
                                  fl.id,
                                  {},
                                  { time_s: parseFloat(e.target.value) || 0 }
                                )
                              }
                              className="w-16 bg-[#0e1422] border border-slate-800 rounded px-2 py-1 text-slate-100"
                            />
                          </td>
                          <td className="py-2 px-2.5 min-w-[200px]">
                            <textarea
                              rows={2}
                              aria-label="Pilot notes or mission comments"
                              placeholder="Add pilot notes or mission comments..."
                              value={fl.notes ?? ''}
                              onChange={(e) =>
                                handleUpdateFlight(fl.id, {
                                  notes: e.target.value,
                                })
                              }
                              className="w-full min-w-[180px] bg-[#0e1422] border border-slate-800 focus:border-cyan-500/60 focus:outline-none rounded px-2.5 py-1.5 text-slate-100 placeholder:text-slate-600 font-sans text-xs resize-y leading-relaxed"
                            />
                          </td>
                          <td className="py-2 px-2.5 text-right">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setFlights((prev) =>
                                  prev.filter((item) => item.id !== fl.id)
                                );
                              }}
                              className="p-1.5 text-slate-400 hover:text-rose-400 cursor-pointer"
                              aria-label="Delete flight"
                            >
                              <Trash2 className="w-4 h-4 inline" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </main>
      )}
    </div>
  );
}
export default App;
