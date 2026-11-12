import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SpectrogramData,
  TracedRidge,
  FittedDopplerEvent,
  applyAircraftBandBoost,
  applyClockDrift,
  applyHighPassFilter,
  applyNotchFilter,
  applyTrafficTransientSuppression,
  computeEventAcousticMetrics,
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

interface PrecisionFaderProps {
  label: string;
  ariaLabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  onChange: (nextVal: number) => void;
}

const PrecisionFaderControl: React.FC<PrecisionFaderProps> = ({
  label,
  ariaLabel,
  value,
  min,
  max,
  step,
  displayValue,
  onChange,
}) => {
  const ratio = Math.max(0, Math.min(1, (value - min) / Math.max(1e-6, max - min)));
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="relative h-7 bg-[#111111] border border-[#222222] px-2.5 flex items-center justify-between overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-[#1F1F22] pointer-events-none"
          style={{ width: `${(ratio * 100).toFixed(1)}%` }}
        />
        <span
          className="relative z-10 text-[10px] uppercase tracking-[0.08em] text-[#A1A1AA] font-mono truncate mr-2 min-w-0"
          title={label}
        >
          {label}
        </span>
        <span className="relative z-10 text-[11px] font-mono tabular-nums text-[#FAFAFA] shrink-0 whitespace-nowrap">
          {displayValue}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={ariaLabel}
        className="w-full block cursor-ew-resize focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white"
      />
    </div>
  );
};

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

  // Audio DSP & Spectrogram Editing Controls
  const [fftSize, setFftSize] = useState<number>(1024);
  const [contrastGateZ, setContrastGateZ] = useState<number>(0.0);
  const [notchFreqHz, setNotchFreqHz] = useState<number>(0.0);
  const [rejectGroundTraffic, setRejectGroundTraffic] = useState<boolean>(true);
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
  const activeObjectUrlRef = useRef<string | null>(null);

  // Revoke active object URL on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current);
      }
    };
  }, []);

  const [selectedBasin, setSelectedBasin] = useState<BasinSummary | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const flightImportRef = useRef<HTMLInputElement | null>(null);

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

  // Compute STFT Spectrogram from uploaded media with full DSP chain (HPF, Notch, Band Boost, Clock Drift, Traffic Transient Suppression)
  const spectrogram: SpectrogramData | null = useMemo(() => {
    if (!rawPcmBuffer) return null;
    let processed: Float32Array = rawPcmBuffer;
    if (highPass80Hz) {
      processed = applyHighPassFilter(processed, sampleRate, 80.0);
    }
    if (notchFreqHz > 10.0) {
      processed = applyNotchFilter(processed, sampleRate, notchFreqHz, 12.0);
    }
    if (bandBoostDb > 0.5) {
      processed = applyAircraftBandBoost(processed, sampleRate, bandBoostDb);
    }
    if (Math.abs(injectedDriftPpm) > 0.1) {
      processed = applyClockDrift(processed, injectedDriftPpm);
    }
    const baseSpec = computeSpectrogram(
      processed,
      sampleRate,
      fftSize,
      fftSize >> 2,
      500.0
    );
    return rejectGroundTraffic
      ? applyTrafficTransientSuppression(baseSpec)
      : baseSpec;
  }, [
    rawPcmBuffer,
    sampleRate,
    fftSize,
    highPass80Hz,
    notchFreqHz,
    bandBoostDb,
    injectedDriftPpm,
    rejectGroundTraffic,
  ]);

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

  // Run Geolocation Solver on user's events and flights (automatically rejecting ground traffic / car curves when enabled)
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
      rejectGroundTraffic,
    });
  }, [
    bounds,
    tempC,
    windEMs,
    windNMs,
    flights,
    events,
    injectedDriftPpm,
    rejectGroundTraffic,
  ]);

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
    if (activeObjectUrlRef.current) {
      URL.revokeObjectURL(activeObjectUrlRef.current);
      activeObjectUrlRef.current = null;
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
    if (activeObjectUrlRef.current) {
      URL.revokeObjectURL(activeObjectUrlRef.current);
      activeObjectUrlRef.current = null;
    }
    setUploadedFileName(file.name);
    const url = URL.createObjectURL(file);
    activeObjectUrlRef.current = url;
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

  const handleAddCarPassByEvent = () => {
    const nextIdx = events.length + 1;
    const defaultCpa = duration > 0 ? Number((duration * 0.35).toFixed(1)) : 14.0;
    const carEv: FittedDopplerEvent = {
      id: `CAR0${nextIdx}`,
      f0_hz: 68.0,
      speed_ms: 26.0, // ~93 km/h road traffic speed
      slant_m: 35.0, // 35m distance from road
      t_cpa_s: defaultCpa,
      snr_db: 38.0,
      orders: [1, 2],
      curves: [],
      rejectedAsTraffic: true,
    };
    setEvents((prev) => [...prev, carEv]);
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

  const handleLoadSampleScenarioFlights = () => {
    const centerLat = (bounds.south + bounds.north) * 0.5;
    const centerLon = (bounds.west + bounds.east) * 0.5;
    const presets: ScenarioFlight[] = [
      {
        id: 'DEMO-C172',
        callsign: 'N739UL',
        aircraft: 'C172',
        aircraft_name: 'Cessna 172 Skyhawk',
        heading_deg: 320.0,
        speed_ms: 58.0,
        speed_kts: 112.7,
        alt_m: 1250,
        bpf_nominal_hz: 85.0,
        is_target: true,
        cpa: {
          lat: Number((centerLat - 0.012).toFixed(4)),
          lon: Number((centerLon + 0.018).toFixed(4)),
          time_s: 24.5,
          slant_m: 1350,
        },
        waypoints: [],
      },
      {
        id: 'DEMO-SR22',
        callsign: 'N882SR',
        aircraft: 'SR22',
        aircraft_name: 'Cirrus SR22 G6',
        heading_deg: 115.0,
        speed_ms: 86.0,
        speed_kts: 167.2,
        alt_m: 1850,
        bpf_nominal_hz: 115.0,
        is_target: true,
        cpa: {
          lat: Number((centerLat + 0.021).toFixed(4)),
          lon: Number((centerLon - 0.014).toFixed(4)),
          time_s: 38.0,
          slant_m: 1920,
        },
        waypoints: [],
      },
      {
        id: 'DEMO-DA40',
        callsign: 'N420DA',
        aircraft: 'DA40',
        aircraft_name: 'Diamond DA40 Star',
        heading_deg: 205.0,
        speed_ms: 68.0,
        speed_kts: 132.2,
        alt_m: 1400,
        bpf_nominal_hz: 90.0,
        is_target: false,
        cpa: {
          lat: Number((centerLat + 0.035).toFixed(4)),
          lon: Number((centerLon + 0.028).toFixed(4)),
          time_s: 55.0,
          slant_m: 1550,
        },
        waypoints: [],
      },
      {
        id: 'DEMO-C182',
        callsign: 'N5182B',
        aircraft: 'C182',
        aircraft_name: 'Cessna 182 Skylane',
        heading_deg: 45.0,
        speed_ms: 74.0,
        speed_kts: 143.8,
        alt_m: 1600,
        bpf_nominal_hz: 88.0,
        is_target: false,
        cpa: {
          lat: Number((centerLat - 0.028).toFixed(4)),
          lon: Number((centerLon - 0.032).toFixed(4)),
          time_s: 72.0,
          slant_m: 1720,
        },
        waypoints: [],
      },
    ];
    setFlights(presets);
    setAdsbStatusMsg(`Loaded ${presets.length} benchmark scenario aircraft.`);
  };

  const handleFetchLiveOpenSky = async () => {
    setIsFetchingAdsb(true);
    setAdsbStatusMsg(null);
    const centerLat = (bounds.south + bounds.north) * 0.5;
    const centerLon = (bounds.west + bounds.east) * 0.5;
    const radiusNm = Math.min(
      50,
      Math.max(
        15,
        Math.ceil(
          Math.max(bounds.north - bounds.south, bounds.east - bounds.west) * 60 * 0.75
        )
      )
    );

    // 1. Try adsb.lol (community open network, free, high limit)
    try {
      const lolUrls = [
        `/api/adsb-lol/v2/point/${centerLat.toFixed(4)}/${centerLon.toFixed(4)}/${radiusNm}`,
        `https://api.adsb.lol/v2/point/${centerLat.toFixed(4)}/${centerLon.toFixed(4)}/${radiusNm}`,
      ];

      for (const u of lolUrls) {
        try {
          const res = await fetch(u);
          if (!res.ok) continue;
          const data = await res.json();
          interface AdsbLolAircraft {
            hex?: string;
            flight?: string;
            r?: string;
            t?: string;
            lat?: number;
            lon?: number;
            alt_baro?: number | string;
            gs?: number;
            track?: number;
          }
          const acList: AdsbLolAircraft[] = Array.isArray(data?.ac) ? data.ac : [];
          const airborne = acList.filter(
            (a) =>
              typeof a.lat === 'number' &&
              typeof a.lon === 'number' &&
              typeof a.alt_baro === 'number' &&
              typeof a.gs === 'number' &&
              a.gs > 25 &&
              a.gs < 185
          );

          if (airborne.length > 0) {
            const imported: ScenarioFlight[] = airborne.slice(0, 16).map((a, idx) => {
              const hex = String(a.hex || `AL${idx}`).toUpperCase();
              const callsign = String(a.flight || a.r || hex).trim() || hex;
              const lat = Number(a.lat);
              const lon = Number(a.lon);
              const baroAltM = typeof a.alt_baro === 'number' ? a.alt_baro * 0.3048 : 1500;
              const velocityMs = typeof a.gs === 'number' ? a.gs * 0.514444 : 85;
              const headingDeg = typeof a.track === 'number' ? a.track : 90;
              const typeCode = String(a.t || (velocityMs < 95 ? 'C172' : 'SR22')).toUpperCase();

              return {
                id: `${hex}-${idx}`,
                callsign,
                aircraft: typeCode,
                aircraft_name: `${callsign} (${typeCode})`,
                heading_deg: Number(headingDeg.toFixed(1)),
                speed_ms: Number(velocityMs.toFixed(1)),
                speed_kts: Number((velocityMs * 1.94384).toFixed(1)),
                alt_m: Number(baroAltM.toFixed(0)),
                bpf_nominal_hz: velocityMs < 95 ? 85 : 115,
                is_target: idx < 2,
                cpa: {
                  lat: Number(lat.toFixed(4)),
                  lon: Number(lon.toFixed(4)),
                  time_s: 20 + idx * 12,
                  slant_m: Number(Math.max(400, baroAltM).toFixed(0)),
                },
                waypoints: [],
              };
            });
            setFlights(imported);
            setAdsbStatusMsg(`Loaded ${imported.length} live aircraft from ADSB.lol.`);
            setIsFetchingAdsb(false);
            return;
          }
        } catch {
          // Continue to next endpoint or fallback
        }
      }
    } catch {
      // Continue to OpenSky fallback
    }

    // 2. Fallback to OpenSky Network via proxy / direct
    try {
      const openSkyUrls = [
        `/api/opensky/states/all?lamin=${bounds.south}&lomin=${bounds.west}&lamax=${bounds.north}&lomax=${bounds.east}`,
        `https://opensky-network.org/api/states/all?lamin=${bounds.south}&lomin=${bounds.west}&lamax=${bounds.north}&lomax=${bounds.east}`,
      ];

      for (const u of openSkyUrls) {
        try {
          const res = await fetch(u);
          if (!res.ok) continue;
          const data = await res.json();
          const states: unknown[][] = Array.isArray(data?.states) ? data.states : [];
          const airborne = states.filter(
            (s) =>
              s[5] !== null &&
              s[6] !== null &&
              !s[8] &&
              typeof s[9] === 'number' &&
              s[9] > 25 &&
              s[9] < 165
          );

          if (airborne.length > 0) {
            const imported: ScenarioFlight[] = airborne.slice(0, 12).map((s, idx) => {
              const icao24 = String(s[0] || `OS${idx}`).toUpperCase();
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
            setIsFetchingAdsb(false);
            return;
          }
        } catch {
          // Continue to next URL
        }
      }
      setAdsbStatusMsg('No airborne propeller-speed aircraft found in bounds. Use "Sample Scenario" or try a wider region.');
    } catch {
      setAdsbStatusMsg('Live feeds unavailable (CORS/Rate-limit). Use "Sample Scenario" or import JSON/CSV.');
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

  const activeEvMetrics = useMemo(() => {
    if (!activeEv) return null;
    return computeEventAcousticMetrics(activeEv, effectiveSoundSpeed);
  }, [activeEv, effectiveSoundSpeed]);

  return (
    <div className="flex flex-col h-screen w-screen bg-[#000000] text-[#FAFAFA] select-none overflow-hidden">
      {/* Top Instrument Header Bar */}
      <header className="h-11 px-4 bg-[#0A0A0A] border-b border-[#1F1F1F] flex items-center justify-between gap-4 md:gap-8 shrink-0 overflow-x-auto min-w-0">
        <button
          type="button"
          onClick={() => setActiveTab('spectrogram')}
          className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-[#FAFAFA] flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer bg-transparent border-none p-0"
        >
          <span className="w-2 h-2 bg-[#10B981] inline-block shrink-0" />
          <span>SKYPRINT</span>
        </button>

        <nav
          role="tablist"
          aria-label="Workspace Tabs"
          className="flex items-center h-full gap-4 md:gap-6 text-xs font-medium text-[#A1A1AA] overflow-x-auto min-w-0"
        >
          <button
            type="button"
            role="tab"
            id="tab-spectrogram"
            aria-selected={activeTab === 'spectrogram'}
            aria-controls="tabpanel-spectrogram"
            onClick={() => setActiveTab('spectrogram')}
            className={`h-full flex items-center transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'spectrogram'
                ? 'text-[#FAFAFA] border-[#FFFFFF]'
                : 'border-transparent hover:text-[#FAFAFA]'
            }`}
          >
            Spectrogram
          </button>
          <button
            type="button"
            role="tab"
            id="tab-geolocation"
            aria-selected={activeTab === 'geolocation'}
            aria-controls="tabpanel-geolocation"
            onClick={() => setActiveTab('geolocation')}
            className={`h-full flex items-center transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'geolocation'
                ? 'text-[#FAFAFA] border-[#FFFFFF]'
                : 'border-transparent hover:text-[#FAFAFA]'
            }`}
          >
            Geolocation
          </button>
          <button
            type="button"
            role="tab"
            id="tab-flights"
            aria-selected={activeTab === 'flights'}
            aria-controls="tabpanel-flights"
            onClick={() => setActiveTab('flights')}
            className={`h-full flex items-center transition-colors whitespace-nowrap shrink-0 border-b-2 font-mono tabular-nums cursor-pointer ${
              activeTab === 'flights'
                ? 'text-[#FAFAFA] border-[#FFFFFF]'
                : 'border-transparent hover:text-[#FAFAFA]'
            }`}
          >
            Flight Tracks ({flights.length})
          </button>
        </nav>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleExportGeoJson}
            disabled={inversionResult.basins.length === 0}
            className="h-7 px-2.5 text-[11px] font-mono bg-[#111111] hover:bg-[#161616] disabled:opacity-40 text-[#FAFAFA] rounded-none border border-[#262626] flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Download className="w-3 h-3" />
            Export GeoJSON
          </button>
        </div>
      </header>

      {/* TAB 1: SPECTROGRAM & MEDIA */}
      {activeTab === 'spectrogram' && (
        <div
          role="tabpanel"
          id="tabpanel-spectrogram"
          aria-labelledby="tab-spectrogram"
          className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr] min-h-0 overflow-hidden"
        >
          {/* Unboxed Full-Width Hairline Sidebar */}
          <aside className="bg-[#0A0A0A] border-r border-[#1F1F1F] flex flex-col min-h-0 overflow-y-auto p-0">
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

            {/* DOPPLER CURVES & WAVELENGTH TELEMETRY SECTION */}
            <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-3">
              <div className="flex items-center justify-between gap-1.5 flex-wrap">
                <span className="text-[10px] uppercase tracking-[0.08em] text-[#737373] font-semibold whitespace-nowrap">
                  DOPPLER CURVES
                </span>
                <div className="flex items-center gap-1 shrink-0">
                  {tracedRidges.length > 0 && (
                    <button
                      type="button"
                      onClick={handleAutoFitFromSpectrogram}
                      className="h-6 px-1.5 text-[10px] font-mono bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] border border-[#262626] rounded-none flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                    >
                      <Sparkles className="w-3 h-3" />
                      Auto-Fit
                    </button>
                  )}
                  <button
                    type="button"
                    data-testid="add-car-curve-btn"
                    onClick={handleAddCarPassByEvent}
                    className="h-6 px-1.5 text-[10px] font-mono bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA] border border-[#262626] rounded-none flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                    title="Simulate a passing road car Doppler whip (v=26 m/s, r0=35 m) to verify Traffic Rejection"
                  >
                    + Car Noise
                  </button>
                  <button
                    type="button"
                    onClick={handleAddEvent}
                    className="h-6 px-2 text-[10.5px] font-medium bg-[#FFFFFF] hover:bg-[#E5E5E5] text-[#000000] rounded-none flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                  >
                    <Plus className="w-3 h-3" />
                    Add Curve
                  </button>
                </div>
              </div>

              {events.length === 0 ? (
                <div className="text-[11px] font-mono text-[#737373] py-1">
                  NO ACTIVE DOPPLER CURVES
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-1 flex-wrap">
                    {events.map((ev, idx) => {
                      const m = computeEventAcousticMetrics(
                        ev,
                        effectiveSoundSpeed
                      );
                      const isCar =
                        ev.rejectedAsTraffic === true ||
                        m.sourceClass !== 'aircraft';
                      return (
                        <div
                          key={ev.id}
                          className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]"
                        >
                          <button
                            type="button"
                            onClick={() => setActiveEventIndex(idx)}
                            className={`h-6 px-2 text-[11px] font-mono rounded-none flex items-center gap-1.5 cursor-pointer ${
                              activeEventIndex === idx
                                ? 'bg-[#18181B] text-[#FAFAFA]'
                                : 'bg-[#111111] text-[#A1A1AA]'
                            }`}
                          >
                            <span
                              className="w-1.5 h-1.5"
                              style={{
                                backgroundColor: isCar
                                  ? '#f43f5e'
                                  : activeEventIndex === idx
                                  ? '#10b981'
                                  : '#3f3f46',
                              }}
                            />
                            <span>{ev.id}</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleRemoveEvent(idx)}
                            aria-label={`Remove ${ev.id}`}
                            className="h-6 px-1.5 text-[11px] font-mono bg-[#111111] hover:bg-[#161616] hover:text-[#F43F5E] text-[#737373] rounded-none cursor-pointer"
                          >
                            ×
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  {activeEv && activeEvMetrics && (
                    <div className="flex flex-col gap-2.5 pt-1">
                      <PrecisionFaderControl
                        label="f0"
                        ariaLabel="f0 frequency"
                        value={activeEv.f0_hz}
                        min={25}
                        max={180}
                        step={0.5}
                        displayValue={`${activeEv.f0_hz.toFixed(1)} Hz`}
                        onChange={(v) => updateActiveEvent({ f0_hz: v })}
                      />
                      <PrecisionFaderControl
                        label="Speed"
                        ariaLabel="Speed"
                        value={activeEv.speed_ms}
                        min={10}
                        max={180}
                        step={1}
                        displayValue={`${activeEv.speed_ms.toFixed(0)} m/s`}
                        onChange={(v) =>
                          updateActiveEvent({
                            speed_ms: v,
                            rejectedAsTraffic: undefined,
                          })
                        }
                      />
                      <PrecisionFaderControl
                        label="Slant Range"
                        ariaLabel="Slant range"
                        value={activeEv.slant_m}
                        min={15}
                        max={4500}
                        step={15}
                        displayValue={`${activeEv.slant_m.toFixed(0)} m`}
                        onChange={(v) =>
                          updateActiveEvent({
                            slant_m: v,
                            rejectedAsTraffic: undefined,
                          })
                        }
                      />
                      <PrecisionFaderControl
                        label="CPA Time"
                        ariaLabel="CPA Time"
                        value={activeEv.t_cpa_s}
                        min={0}
                        max={Math.max(60, duration || 60)}
                        step={0.5}
                        displayValue={`${activeEv.t_cpa_s.toFixed(1)} s`}
                        onChange={(v) => updateActiveEvent({ t_cpa_s: v })}
                      />

                      {/* Wavelength & Kinematic Telemetry Readout Box */}
                      <div
                        data-testid="curve-wavelength-telemetry"
                        className="border border-[#222222] bg-[#111111] divide-y divide-[#1F1F1F] text-[10.5px] font-mono tabular-nums min-w-0"
                      >
                        <div className="flex items-center justify-between gap-2 px-2.5 py-1">
                          <span className="text-[#A1A1AA] shrink-0">WAVELENGTH λ₀</span>
                          <span className="text-[#10B981] font-mono tabular-nums text-right truncate">
                            {activeEvMetrics.wavelength0_m.toFixed(3)} m
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2 px-2.5 py-1">
                          <span className="text-[#A1A1AA] shrink-0">
                            λ_APP / λ_REC
                          </span>
                          <span className="text-[#FAFAFA] font-mono tabular-nums text-right truncate">
                            {activeEvMetrics.wavelengthApp_m.toFixed(2)} m /{' '}
                            {activeEvMetrics.wavelengthRec_m.toFixed(2)} m
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2 px-2.5 py-1">
                          <span className="text-[#A1A1AA] shrink-0">
                            MAX SLOPE |df/dt|
                          </span>
                          <span className="text-[#FAFAFA] font-mono tabular-nums text-right truncate">
                            {activeEvMetrics.maxInflectionSlope_hz_s.toFixed(2)}{' '}
                            Hz/s
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2 px-2.5 py-1">
                          <span className="text-[#A1A1AA] shrink-0">SOURCE FILTER</span>
                          <span
                            className={`text-right truncate font-medium ${
                              activeEvMetrics.sourceClass === 'aircraft' &&
                              !activeEv.rejectedAsTraffic
                                ? 'text-[#10B981]'
                                : 'text-[#F43F5E]'
                            }`}
                            title={
                              activeEvMetrics.sourceClass === 'aircraft' &&
                              !activeEv.rejectedAsTraffic
                                ? 'AIRCRAFT (KEEP)'
                                : 'CAR / TRAFFIC (REJECTED)'
                            }
                          >
                            {activeEvMetrics.sourceClass === 'aircraft' &&
                            !activeEv.rejectedAsTraffic
                              ? 'AIRCRAFT (KEEP)'
                              : 'CAR / TRAFFIC (REJECTED)'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* AUDIO & TRAFFIC FILTERS SECTION */}
            <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-2.5">
              <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
                AUDIO & TRAFFIC FILTERS
              </span>

              {/* Reject Ground Traffic / Passing Cars Toggle */}
              <div className="flex items-center justify-between h-7 bg-[#111111] border border-[#222222] px-2.5">
                <span className="text-[10px] uppercase tracking-[0.08em] text-[#A1A1AA] font-mono">
                  REJECT CAR / TRAFFIC
                </span>
                <button
                  type="button"
                  data-testid="toggle-traffic-filter"
                  aria-pressed={rejectGroundTraffic}
                  onClick={() => setRejectGroundTraffic((v) => !v)}
                  className="h-5 px-2 text-[10px] font-mono bg-[#161616] border border-[#2E2E33] text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer"
                >
                  <span
                    className="w-1.5 h-1.5"
                    style={{
                      backgroundColor: rejectGroundTraffic
                        ? '#10b981'
                        : '#3f3f46',
                    }}
                  />
                  <span>{rejectGroundTraffic ? 'ON' : 'OFF'}</span>
                </button>
              </div>

              <div className="flex items-center justify-between h-7 bg-[#111111] border border-[#222222] px-2.5">
                <span className="text-[10px] uppercase tracking-[0.08em] text-[#A1A1AA] font-mono">
                  80 HZ HIGH-PASS
                </span>
                <button
                  type="button"
                  onClick={() => setHighPass80Hz((v) => !v)}
                  className="h-5 px-2 text-[10px] font-mono bg-[#161616] border border-[#2E2E33] text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer"
                >
                  <span
                    className="w-1.5 h-1.5"
                    style={{
                      backgroundColor: highPass80Hz ? '#f43f5e' : '#3f3f46',
                    }}
                  />
                  <span>{highPass80Hz ? 'ON' : 'OFF'}</span>
                </button>
              </div>

              {/* STFT Window Resolution Selector */}
              <div className="flex items-center justify-between h-7 bg-[#111111] border border-[#222222] px-2.5">
                <span className="text-[10px] uppercase tracking-[0.08em] text-[#A1A1AA] font-mono">
                  STFT FFT WINDOW
                </span>
                <div className="inline-flex items-center border border-[#2E2E33] divide-x divide-[#2E2E33]">
                  {[512, 1024, 2048].map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setFftSize(n)}
                      className={`h-5 px-1.5 text-[10px] font-mono cursor-pointer ${
                        fftSize === n
                          ? 'bg-[#FFFFFF] text-[#000000] font-medium'
                          : 'bg-[#161616] text-[#A1A1AA]'
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <PrecisionFaderControl
                label="Noise Gate (Z)"
                ariaLabel="Noise gate threshold"
                value={contrastGateZ}
                min={0}
                max={4}
                step={0.2}
                displayValue={`${contrastGateZ.toFixed(1)} σ`}
                onChange={(v) => setContrastGateZ(v)}
              />

              <PrecisionFaderControl
                label="Notch Filter"
                ariaLabel="Notch filter frequency"
                value={notchFreqHz}
                min={0}
                max={300}
                step={5}
                displayValue={
                  notchFreqHz > 10 ? `${notchFreqHz.toFixed(0)} Hz` : 'OFF'
                }
                onChange={(v) => setNotchFreqHz(v)}
              />

              <PrecisionFaderControl
                label="Sub-Band Boost"
                ariaLabel="Sub-band boost"
                value={bandBoostDb}
                min={0}
                max={18}
                step={2}
                displayValue={`+${bandBoostDb.toFixed(0)} dB`}
                onChange={(v) => setBandBoostDb(v)}
              />

              <PrecisionFaderControl
                label="Clock Drift"
                ariaLabel="Clock drift"
                value={injectedDriftPpm}
                min={-20}
                max={20}
                step={1}
                displayValue={`${
                  injectedDriftPpm >= 0 ? '+' : ''
                }${injectedDriftPpm.toFixed(1)} ppm`}
                onChange={(v) => setInjectedDriftPpm(v)}
              />
            </div>
          </aside>

          <main className="flex flex-col min-h-0 overflow-hidden bg-[#000000]">
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
              soundSpeedMs={effectiveSoundSpeed}
              contrastGateZ={contrastGateZ}
              notchFreqHz={notchFreqHz}
              onChangeNotchFreqHz={setNotchFreqHz}
              rejectGroundTraffic={rejectGroundTraffic}
              onFitCustomEvent={(fitted) => {
                setEvents((prev) => [...prev, fitted]);
                setActiveEventIndex(events.length);
              }}
              onUpdateActiveF0={(nextF0) => {
                if (events.length > 0) {
                  updateActiveEvent({ f0_hz: nextF0 });
                }
              }}
            />
          </main>
        </div>
      )}

      {/* TAB 2: GEOLOCATION MAP */}
      {activeTab === 'geolocation' && (
        <div
          role="tabpanel"
          id="tabpanel-geolocation"
          aria-labelledby="tab-geolocation"
          className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr] min-h-0 overflow-hidden"
        >
          {/* Unboxed Full-Width Hairline Sidebar */}
          <aside className="bg-[#0A0A0A] border-r border-[#1F1F1F] flex flex-col min-h-0 overflow-y-auto p-0">
            {/* ESTIMATED FIX SECTION */}
            <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
                  ESTIMATED FIX
                </span>
                <span
                  data-testid="status-indicator"
                  className={`text-[11px] font-mono font-semibold flex items-center gap-1.5 ${
                    inversionResult.status === 'located'
                      ? 'text-[#10B981]'
                      : inversionResult.status === 'ambiguous'
                      ? 'text-[#F59E0B]'
                      : 'text-[#737373]'
                  }`}
                >
                  <span
                    className="w-1.5 h-1.5"
                    style={{
                      backgroundColor:
                        inversionResult.status === 'located'
                          ? '#10b981'
                          : inversionResult.status === 'ambiguous'
                          ? '#f59e0b'
                          : '#52525b',
                    }}
                  />
                  {inversionResult.status.toUpperCase()}
                </span>
              </div>

              {inversionResult.rejectedTrafficEventIds.length > 0 && (
                <div
                  data-testid="traffic-rejection-notice"
                  className="px-2.5 py-1.5 bg-[#111111] border border-[#262626] text-[10px] font-mono text-[#F43F5E]"
                >
                  FILTERED CAR/TRAFFIC CURVES:{' '}
                  {inversionResult.rejectedTrafficEventIds.join(', ')}
                </div>
              )}

              {inversionResult.basins.length === 0 ? (
                <p className="text-[11px] font-mono text-[#737373] py-0.5">
                  ADD DOPPLER CURVES + ADS-B TRACKS TO SOLVE FIX
                </p>
              ) : (
                <div className="flex flex-col divide-y divide-[#1F1F1F] border border-[#222222] bg-[#111111] text-[11px] font-mono tabular-nums min-w-0">
                  <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
                    <span className="text-[#A1A1AA] shrink-0">COORDINATES</span>
                    <span className="text-[#FAFAFA] text-right truncate">
                      {inversionResult.fix.lat.toFixed(4)}°,{' '}
                      {inversionResult.fix.lon.toFixed(4)}°
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
                    <span className="text-[#A1A1AA] shrink-0">95% RADIUS</span>
                    <span className="text-[#FAFAFA] text-right truncate">
                      ±{inversionResult.radius_m.toFixed(0)} m
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
                    <span className="text-[#A1A1AA] shrink-0">100M HEATMAP GRID</span>
                    <span className="text-[#EF4444] text-right truncate">
                      {inversionResult.heatmapGrid.length} DOTS (100m)
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
                    <span className="text-[#A1A1AA] shrink-0">SCORE</span>
                    <span className="text-[#10B981] text-right truncate">
                      {inversionResult.score.toFixed(2)}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* SEARCH BOUNDS SECTION */}
            <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-2.5">
              <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
                SEARCH BOUNDS
              </span>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono tabular-nums">
                <label className="flex flex-col gap-1 text-[#A1A1AA]">
                  <span className="text-[10px] uppercase tracking-wider text-[#737373]">
                    NORTH LAT
                  </span>
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
                    className="h-7 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                  />
                </label>
                <label className="flex flex-col gap-1 text-[#A1A1AA]">
                  <span className="text-[10px] uppercase tracking-wider text-[#737373]">
                    SOUTH LAT
                  </span>
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
                    className="h-7 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                  />
                </label>
                <label className="flex flex-col gap-1 text-[#A1A1AA]">
                  <span className="text-[10px] uppercase tracking-wider text-[#737373]">
                    WEST LON
                  </span>
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
                    className="h-7 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                  />
                </label>
                <label className="flex flex-col gap-1 text-[#A1A1AA]">
                  <span className="text-[10px] uppercase tracking-wider text-[#737373]">
                    EAST LON
                  </span>
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
                    className="h-7 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                  />
                </label>
              </div>
            </div>

            {/* ATMOSPHERE SECTION */}
            <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
                  ATMOSPHERE
                </span>
                <span className="text-[11px] font-mono tabular-nums text-[#FAFAFA]">
                  c={effectiveSoundSpeed.toFixed(1)} m/s
                </span>
              </div>

              <div className="flex flex-col gap-2.5">
                <PrecisionFaderControl
                  label="Temp"
                  ariaLabel="Temperature"
                  value={tempC}
                  min={-25}
                  max={40}
                  step={1}
                  displayValue={`${tempC.toFixed(1)} °C`}
                  onChange={(v) => setTempC(v)}
                />

                <div className="grid grid-cols-2 gap-2">
                  <PrecisionFaderControl
                    label="Wind E"
                    ariaLabel="Wind East"
                    value={windEMs}
                    min={-15}
                    max={15}
                    step={0.5}
                    displayValue={`${windEMs.toFixed(1)} m/s`}
                    onChange={(v) => setWindEMs(v)}
                  />
                  <PrecisionFaderControl
                    label="Wind N"
                    ariaLabel="Wind North"
                    value={windNMs}
                    min={-15}
                    max={15}
                    step={0.5}
                    displayValue={`${windNMs.toFixed(1)} m/s`}
                    onChange={(v) => setWindNMs(v)}
                  />
                </div>
              </div>
            </div>

            {/* CANDIDATE BASINS SECTION */}
            {inversionResult.basins.length > 0 && (
              <div className="border-b border-[#1F1F1F] px-4 py-3.5 flex flex-col gap-2">
                <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
                  CANDIDATE BASINS
                </span>
                <div className="border border-[#262626] bg-[#111111]">
                  <table className="w-full text-left font-mono text-[11px] tabular-nums border-collapse">
                    <thead>
                      <tr className="border-b border-[#262626] text-[10px] uppercase tracking-wider text-[#737373] bg-[#0A0A0A]">
                        <th className="py-1.5 px-2">#</th>
                        <th className="py-1.5 px-2">COORDINATES</th>
                        <th className="py-1.5 px-2 text-right">SCORE</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F1F1F]">
                      {inversionResult.basins.slice(0, 6).map((b) => (
                        <tr
                          key={b.rank}
                          onClick={() => setSelectedBasin(b)}
                          className={`cursor-pointer transition-colors ${
                            selectedBasin?.rank === b.rank
                              ? 'bg-[#1F1F22] text-[#FAFAFA]'
                              : 'hover:bg-[#161616] text-[#A1A1AA]'
                          }`}
                        >
                          <td className="py-1.5 px-2 font-semibold text-[#FAFAFA]">
                            {b.rank}
                          </td>
                          <td className="py-1.5 px-2">
                            {b.lat.toFixed(4)}°, {b.lon.toFixed(4)}°
                          </td>
                          <td className="py-1.5 px-2 text-right text-[#10B981]">
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

          <main className="flex flex-col min-h-0 overflow-hidden bg-[#000000]">
            <GeodeticMapViewport
              bounds={bounds}
              onChangeBounds={setBounds}
              fix={inversionResult.fix}
              radiusM={inversionResult.radius_m}
              status={inversionResult.status}
              flights={flights}
              candidates={inversionResult.candidates}
              heatmapGrid={inversionResult.heatmapGrid}
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

      {/* TAB 3: FULL-BLEED ADS-B FLIGHT TRACKS WORKSTATION TABLE */}
      {activeTab === 'flights' && (
        <main
          role="tabpanel"
          id="tabpanel-flights"
          aria-labelledby="tab-flights"
          className="flex-1 flex flex-col min-h-0 bg-[#000000] overflow-hidden"
        >
          <div className="h-10 px-4 bg-[#0A0A0A] border-b border-[#1F1F1F] flex items-center justify-between gap-3 shrink-0 overflow-x-auto min-w-0">
            <div className="flex items-center gap-3 min-w-0 flex-1 mr-2">
              <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold whitespace-nowrap shrink-0">
                ADS-B TELEMETRY LOG
              </span>
              {adsbStatusMsg && (
                <span className="text-[11px] font-mono text-[#A1A1AA] truncate min-w-0" title={adsbStatusMsg}>
                  {adsbStatusMsg}
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]">
                <button
                  type="button"
                  data-testid="toggle-3d-trajectory-overlay"
                  aria-pressed={overlay3dTrajectory}
                  onClick={() => setOverlay3dTrajectory((v) => !v)}
                  className={`h-7 px-2.5 text-[11px] font-mono rounded-none flex items-center gap-2 transition-colors cursor-pointer ${
                    overlay3dTrajectory
                      ? 'bg-[#161616] text-[#FAFAFA]'
                      : 'bg-[#111111] hover:bg-[#161616] text-[#A1A1AA]'
                  }`}
                >
                  <span
                    className="w-1.5 h-1.5 shrink-0"
                    style={{
                      backgroundColor: overlay3dTrajectory
                        ? '#ff4f00'
                        : '#3f3f46',
                    }}
                  />
                  <Box className="w-3 h-3" />
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
                    className="h-7 px-2.5 text-[11px] font-mono bg-[#161616] hover:bg-[#1F1F22] text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer"
                  >
                    <MapPin className="w-3 h-3 text-[#FF4F00]" />
                    <span>View on Map</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={handleFetchLiveOpenSky}
                  disabled={isFetchingAdsb}
                  title="Query live airborne flights via ADSB.lol with OpenSky fallback"
                  className="h-7 px-2.5 text-[11px] font-mono bg-[#111111] hover:bg-[#161616] disabled:opacity-50 text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer transition-colors"
                >
                  {isFetchingAdsb ? 'Fetching...' : 'Fetch Live ADS-B'}
                </button>

                <button
                  type="button"
                  onClick={handleLoadSampleScenarioFlights}
                  title="Load 4 realistic benchmark propeller aircraft (C172, SR22, DA40, C182) in bounds"
                  className="h-7 px-2.5 text-[11px] font-mono bg-[#111111] hover:bg-[#161616] text-[#A1A1AA] hover:text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer transition-colors"
                >
                  <Sparkles className="w-3 h-3 text-[#10B981]" />
                  Sample Scenario
                </button>

                <button
                  type="button"
                  onClick={() => flightImportRef.current?.click()}
                  className="h-7 px-2.5 text-[11px] font-mono bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] rounded-none flex items-center gap-1.5 cursor-pointer transition-colors"
                >
                  <Upload className="w-3 h-3" />
                  Import JSON / CSV
                </button>
              </div>

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
                className="h-7 px-3 text-[11px] font-medium bg-[#FFFFFF] hover:bg-[#E5E5E5] text-[#000000] rounded-none flex items-center gap-1.5 cursor-pointer transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Flight
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-auto bg-[#000000]">
            <table className="w-full text-left font-mono text-xs tabular-nums border-collapse">
              <thead className="sticky top-0 z-10 bg-[#0A0A0A] border-b border-[#1F1F1F]">
                <tr className="text-[10px] uppercase tracking-[0.1em] text-[#737373]">
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    3D PATH
                  </th>
                  <th
                    className="py-2.5 px-3 whitespace-nowrap font-semibold"
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
                      className={`inline-flex items-center gap-1.5 uppercase tracking-[0.1em] font-semibold transition-colors cursor-pointer ${
                        flightSortField === 'callsign'
                          ? 'text-[#FAFAFA]'
                          : 'text-[#737373] hover:text-[#FAFAFA]'
                      }`}
                    >
                      <span>CALLSIGN</span>
                      {flightSortField === 'callsign' ? (
                        flightSortDir === 'asc' ? (
                          <ArrowUp className="w-3 h-3 text-[#FAFAFA]" />
                        ) : (
                          <ArrowDown className="w-3 h-3 text-[#FAFAFA]" />
                        )
                      ) : (
                        <ArrowUpDown className="w-3 h-3 text-[#52525B]" />
                      )}
                    </button>
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    AIRCRAFT
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    TRACK (°)
                  </th>
                  <th
                    className="py-2.5 px-3 whitespace-nowrap font-semibold"
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
                      className={`inline-flex items-center gap-1.5 uppercase tracking-[0.1em] font-semibold transition-colors cursor-pointer ${
                        flightSortField === 'speed'
                          ? 'text-[#FAFAFA]'
                          : 'text-[#737373] hover:text-[#FAFAFA]'
                      }`}
                    >
                      <span>VELOCITY (M/S)</span>
                      {flightSortField === 'speed' ? (
                        flightSortDir === 'asc' ? (
                          <ArrowUp className="w-3 h-3 text-[#FAFAFA]" />
                        ) : (
                          <ArrowDown className="w-3 h-3 text-[#FAFAFA]" />
                        )
                      ) : (
                        <ArrowUpDown className="w-3 h-3 text-[#52525B]" />
                      )}
                    </button>
                  </th>
                  <th
                    className="py-2.5 px-3 whitespace-nowrap font-semibold"
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
                      className={`inline-flex items-center gap-1.5 uppercase tracking-[0.1em] font-semibold transition-colors cursor-pointer ${
                        flightSortField === 'alt'
                          ? 'text-[#FAFAFA]'
                          : 'text-[#737373] hover:text-[#FAFAFA]'
                      }`}
                    >
                      <span>ALT (M)</span>
                      {flightSortField === 'alt' ? (
                        flightSortDir === 'asc' ? (
                          <ArrowUp className="w-3 h-3 text-[#FAFAFA]" />
                        ) : (
                          <ArrowDown className="w-3 h-3 text-[#FAFAFA]" />
                        )
                      ) : (
                        <ArrowUpDown className="w-3 h-3 text-[#52525B]" />
                      )}
                    </button>
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    LAT
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    LON
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    CPA TIME (S)
                  </th>
                  <th className="py-2.5 px-3 whitespace-nowrap font-semibold">
                    PILOT NOTES / COMMENTS
                  </th>
                  <th className="py-2.5 px-3 text-right whitespace-nowrap font-semibold">
                    ACTION
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#141414]">
                {flights.length === 0 ? (
                  <tr className="border-b border-[#141414]">
                    <td
                      colSpan={11}
                      className="py-16 px-4 text-center text-[11px] font-mono tracking-[0.12em] text-[#737373]"
                    >
                      NO ACTIVE ADS-B TRACKS IN BOUNDS
                    </td>
                  </tr>
                ) : (
                  sortedFlights.map((fl) => {
                    const isSelectedFor3d =
                      (selectedFlightId || flights[0]?.id) === fl.id;
                    return (
                      <tr
                        key={fl.id}
                        onClick={() => setSelectedFlightId(fl.id)}
                        className={`transition-colors border-b border-[#141414] ${
                          isSelectedFor3d && overlay3dTrajectory
                            ? 'bg-[#111111]'
                            : isSelectedFor3d
                            ? 'bg-[#0A0A0A]'
                            : 'hover:bg-[#0A0A0A]'
                        }`}
                      >
                        <td className="py-2 px-3">
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
                            className={`h-7 px-2.5 text-[11px] font-mono rounded-none border flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer ${
                              isSelectedFor3d && overlay3dTrajectory
                                ? 'bg-[#161616] text-[#FAFAFA] border-[#3F3F46]'
                                : isSelectedFor3d
                                ? 'bg-[#111111] text-[#FAFAFA] border-[#2E2E33]'
                                : 'bg-[#111111] text-[#A1A1AA] border-[#262626] hover:text-[#FAFAFA]'
                            }`}
                          >
                            <span
                              className="w-1.5 h-1.5 shrink-0"
                              style={{
                                backgroundColor:
                                  isSelectedFor3d && overlay3dTrajectory
                                    ? '#ff4f00'
                                    : '#3f3f46',
                              }}
                            />
                            <Box className="w-3 h-3 shrink-0" />
                            <span>
                              {isSelectedFor3d && overlay3dTrajectory
                                ? '3D Active'
                                : '3D Path'}
                            </span>
                          </button>
                        </td>
                        <td className="py-2 px-3">
                          <input
                            type="text"
                            aria-label="Flight callsign"
                            value={fl.callsign}
                            onChange={(e) =>
                              handleUpdateFlight(fl.id, {
                                callsign: e.target.value,
                              })
                            }
                            className="h-7 w-24 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
                          <select
                            aria-label="Aircraft type"
                            value={fl.aircraft}
                            onChange={(e) =>
                              handleUpdateFlight(fl.id, {
                                aircraft: e.target.value,
                              })
                            }
                            className="h-7 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          >
                            <option value="C172">C172</option>
                            <option value="C182">C182</option>
                            <option value="SR22">SR22</option>
                            <option value="DA40">DA40</option>
                            <option value="PA34">PA34</option>
                            <option value="C208">C208</option>
                          </select>
                        </td>
                        <td className="py-2 px-3">
                          <input
                            type="number"
                            aria-label="Heading degrees"
                            value={fl.heading_deg}
                            onChange={(e) =>
                              handleUpdateFlight(fl.id, {
                                heading_deg: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-7 w-16 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
                          <input
                            type="number"
                            aria-label="Flight speed m/s"
                            value={fl.speed_ms}
                            onChange={(e) =>
                              handleUpdateFlight(fl.id, {
                                speed_ms: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-7 w-16 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
                          <input
                            type="number"
                            aria-label="Flight altitude meters"
                            value={fl.alt_m}
                            onChange={(e) =>
                              handleUpdateFlight(fl.id, {
                                alt_m: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-7 w-20 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
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
                            className="h-7 w-20 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
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
                            className="h-7 w-20 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3">
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
                            className="h-7 w-16 bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] outline-none rounded-none px-2 text-[#FAFAFA]"
                          />
                        </td>
                        <td className="py-2 px-3 min-w-[220px]">
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
                            className="w-full min-w-[200px] bg-[#111111] border border-[#262626] focus:border-[#FFFFFF] focus:outline-none rounded-none px-2.5 py-1 text-[#FAFAFA] placeholder:text-[#52525B] font-sans text-xs resize-y leading-relaxed"
                          />
                        </td>
                        <td className="py-2 px-3 text-right">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setFlights((prev) =>
                                prev.filter((item) => item.id !== fl.id)
                              );
                            }}
                            className="h-7 w-7 bg-[#111111] hover:bg-[#161616] border border-[#262626] text-[#A1A1AA] hover:text-[#F43F5E] inline-flex items-center justify-center rounded-none cursor-pointer transition-colors"
                            aria-label="Delete flight"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </main>
      )}
    </div>
  );
}
export default App;
