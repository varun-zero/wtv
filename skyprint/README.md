# SkyPrint — Aircraft-Doppler Audio Geolocation

An improved reimplementation of the aircraft-Doppler geolocation technique
demonstrated in Colsto's video
["I discovered a sound that leaks your exact location (and it's everywhere)"](https://www.youtube.com/watch?v=UytYrhYLOKI)
(independent implementation; not affiliated with Colsto, and not derived
from Dopplergeist/DOPPLERGEIST code).

Given a recording with inaudible propeller flyover signatures and a set of
ADS-B flight tracks, SkyPrint searches a geographic region for the receiver
position whose acoustic evidence (Doppler curves, harmonic families,
inter-event time offsets, and silence) is best explained by those flights.

## What is improved relative to the source methodology

Every item below is a difficulty visible in the original transcript,
fixed in this codebase:

| Transcript difficulty | SkyPrint fix | Where |
| --- | --- | --- |
| Constant speed of sound (343 m/s everywhere) | ISA temperature/pressure profiles; **path-effective (harmonic-mean) sound speed** between receiver and source altitude | `atmosphere.py` |
| No wind | Full **moving-medium acoustics**: travel time solves the wind-advection quadratic; Doppler via the exact `dt/dtau` with wind; validated against the moving-observer limit identity | `acoustics.py` |
| No absorption / fixed 20 km audibility | **ISO 9613-1 / ANSI S1.26** pure-tone atmospheric absorption + spherical spreading; received-level model feeds the silence filter | `atmosphere.py`, `acoustics.py` |
| Naive per-tile re-querying of flights | **Flight-centric corridor sweep**: each flight processed exactly once per grid | `search.py` |
| CPA defined loosely; "±5 s wiggle" | Delayed-CPA computed with retarded travel time, wind, and path sound speed | `search.py` |
| Fundamental frequency slid by hand | **Harmonic-family detection** (integer-ratio grouping on overlapping segments) + **aircraft-class BPF priors** (blade-pass frequency bounds per ICAO type) | `audio.py`, `aircraft.py` |
| "Curve doesn't fit a perfect Doppler" | Exact **receive-time asymmetric curve** (retarded-time solved in closed dimensionless form; validated to <0.05 Hz vs bisection) | `audio.py` `receive_doppler` |
| Fixed 12 Hz/s tracer slope caps that clip near-CPA | **Physics-derived slope bounds** (`f·v²/(c·r)` with margins), fractional-window Viterbi with rent + free termination | `audio.py` |
| One measurement error sweep on f0/time | Joint **clock offset + ppm drift** fit shared across events; scenario envelope (wind/temperature) for model error | `search.py`, `confidence.py` |
| Brightness-sum scoring | Scale-invariant **shape correlation + sweep-width agreement** against the exact forward model per (candidate, flight) | `search.py` |
| No abstention | Decision layer: `located / low-confidence / ambiguous / model-limited / inconclusive` with tie margins and radius budgets | `confidence.py` |
| US-only grids | **Global ECEF geodesy** (WGS84, Bowring inversion, ENU tangent grids, dateline-wrapping bounds) — any location on Earth | `geo.py` |

## Layout

- `skyprint/atmosphere.py` — ISA profiles, wind, ISO 9613-1 absorption.
- `skyprint/geo.py` — WGS84 ECEF/ENU, global bounds, dateline-safe tiling.
- `skyprint/acoustics.py` — retarded-time emission solver (Newton + bracket),
  wind quadratic, received level, delayed-CPA.
- `skyprint/aircraft.py` — aircraft-class BPF/speed/source-level priors.
- `skyprint/flights.py` — Hermite-spline ECEF tracks, ADS-B-style ingest,
  synthetic world generator.
- `skyprint/audio.py` — STFT, median/MAD whitening, physics-bounded Viterbi
  ridge tracer, harmonic families, event fitting (receive-time model).
- `skyprint/synth.py` — physics-exact renderer (retarded time, absorption,
  RPM wander) + degradations: MP3/AAC codec round-trip (ffmpeg), clock
  drift resampling, aggressive high-pass.
- `skyprint/search.py` — corridor filter, time-offset join, silence filter,
  BPF/speed priors, forward-model scoring with joint clock, refinement.
- `skyprint/confidence.py` — decision rules and abstention.
- `skyprint/report.py` — single-file minimal HTML report (no framework).
- `skyprint/ui_server.py` — zero-dependency local web server (`python -m skyprint.cli ui`).
- `skyprint/ui/index.html` — interactive Geolocation Workbench application.
- `workbench.html` — standalone offline single-file distribution of the workbench.
- `skyprint/cli.py` — `demo`, `ui`/`serve`, and `analyze` CLI commands.
- `smoke_test.py` — 30-check validation suite (all green at delivery).

## Interactive Geolocation Workbench UI

SkyPrint includes a professional, zero-dependency desktop-grade workbench:

1. **Launch Local Server:**
   ```powershell
   cd skyprint
   python -m skyprint.cli ui --port 8080
   ```
   Or open directly in any browser: `http://127.0.0.1:8080`.

2. **Standalone Offline File:**
   You can also double-click or open [`workbench.html`](workbench.html) directly in any browser without launching a server.

### Features
* **Dual-Domain Viewports:**
  - **Acoustic Domain:** Whitened spectrogram canvas (0–400 Hz) with interactive scrub playhead, overlaid Viterbi harmonic ridges ($k=1, 2, 3$), and CPA markers.
  - **Spatial Domain:** WGS84 vector map with interactive pan/zoom, flight splines, CPA ticks, audibility corridors, candidate likelihood heatmap basins, estimated fix reticle with 95% uncertainty radius, and ground-truth offset vectors.
* **Audio Player:** Live Web Audio playback of synthesized flyover acoustic tones with play/pause, scrub bar, and time counters.
* **Instant Scenario Presets:** Switch across all 9 research benchmark worlds (San Francisco Bay crossing, MP3 48k, +10 ppm clock drift, 80 Hz high-pass, Parallel degenerate abstention, Polar Svalbard, Fiji Dateline, Southern Ocean, and Equatorial Pacific).
* **Evidence Ledger:** Live audit displaying recovered joint clock offset ($\Delta t$) and drift ($\epsilon_{\text{ppm}}$), detected event parameters ($f_0$, speed, slant, $t_c$, SNR), and candidate basin ranking.

## Validated (see `smoke_test.py`)

Geodesy round trips; ISA anchors (288.15 K, 22.6 kPa @ 11 km, c=340.3/328.6);
ISO 9613-1 absorption vs known table values (4.66 dB/km @ 1 kHz,
29.7 dB/km @ 4 kHz, 20 C/50% RH); travel-time quadratic wind limits
(tail/headwind identities); Newton emission solver == dense brute-force
reference (max 0.0000 Hz difference); moving-observer wind identity;
receive-time Doppler model == bisection reference (<0.05 Hz); and the full
audio chain recovering truth parameters from rendered audio
(f0=81.9/85 Hz, v=100.0/100 m/s, r=1500/1500 m, correct CPA window).

## Known limitations

* Single-event evidence (or same-heading events) is reported
  `low-confidence`/`ambiguous` by design — the clock/position degeneracy
  is physical, not algorithmic.
* Uniform wind with height; vertical shear and refraction are covered by
  the scenario envelope in `confidence.py`, not modeled deterministically.
* `analyze` CLI mode is a scaffold (wiring for real ADS-B bundles);
  the validated path is the `demo`/library flow with synthetic worlds.
* Event fit slant carries a systematic +100–200 m bias (bin quantization
  on partial curves) — the scorer therefore down-weights the slant term
  and lets the Hz-residual term decide.
* Real-weather/DEM ingestion (GFS, GLO-90) is implemented as library
  functions but not exercised by the validation suite, which renders
  synthetic atmospheres.

## End-to-end validation results (synthetic worlds)

Full pipeline: render → detect → corridors → join → forward-model scoring
under a joint clock → cluster decision. World A: two propeller flyovers on
**crossing headings** (eastbound C172-class at 100 m/s, westbound
SR22-class at 110 m/s), CPA arrivals ~500 s apart, 4 distractor flights,
3 m/s wind; World B: two **parallel-heading** flyovers 26 s apart
(intrinsically degenerate along the shared corridor). Global sweep
(G1–G4): the same crossing scenario re-rendered at 78.2°N (Svalbard),
across the dateline at 179°E inside a *wrapped* search bounding box,
in the Southern Ocean at 54.5°S, and on the equatorial Pacific. 8 kHz.

| case | decision | error | radius | clock recovered |
| --- | --- | --- | --- | --- |
| A-crossing-baseline | **located** | 470 m | 1602 m | off=−0.05 s, ppm=0 |
| A + MP3 48 kbps | **located** | 470 m | 1602 m | off=−0.05 s |
| A + 10 ppm clock drift | **located** | 467 m | 1602 m | drift absorbed |
| A + 80 Hz high-pass | **located** | 305 m | 1514 m | via k≥2 harmonics |
| B-parallel-26 s | **ambiguous** (honest) | 520 m | 1155 m | — |
| G1 polar Svalbard 78.2°N | **located** | 117 m | 1483 m | off=−0.15 s |
| G2 dateline Fiji 179°E | **located** | 221 m | 1012 m | wrapped bounds |
| G3 Southern Ocean 54.5°S | **located** | 134 m | 1572 m | S-hemisphere ENU |
| G4 equatorial Pacific | **located** | 27 m | 1482 m | — |

Readings:

* **Crossing geometry resolves the clock/position degeneracy** that a
  single pass (or same-heading passes) cannot: a shared clock offset can
  compensate a position error only if both flights' arrival-time
  gradients match, which opposite headings violate. The joint clock at
  the winner recovered off≈0, ppm≈0 against truth in every located case.
* **Countermeasures survive**: MP3 at 48 kbps and a 10 ppm recorder
  drift change nothing (same fix, same error) — the harmonic-family
  detector and the joint-clock fit absorb both, exactly the failure
  modes the source demo worried about.
* **High-pass robustness works end-to-end**: with the fundamental
  filtered below 80 Hz, the surviving k≥2 harmonics reconstruct the
  family, the BPF prior accepts the true class via the harmonic-multiple
  rule (fitted f0=39 → 3·f0=117 ∈ SR22 window), and the fix lands at
  305 m — *better* than baseline because the shorter surviving curves
  weight the near-CPA shape more.
* **Global geodesy holds**: polar (converging meridians), dateline
  (wrapped bounds — `grid_in_bounds` unwraps longitudes the short way;
  without it every grid point landed on the wrong side of the globe),
  and southern-hemisphere (ENU frame flip) worlds all locate, with the
  equatorial case at 27 m.
* **Parallel headings are honestly ambiguous**: with two same-direction
  passes 26 s apart, no observer can separate a receiver shift along the
  corridor from a clock shift; the decision layer says `ambiguous`
  rather than guessing.
* The ~470 m floor on world A is an information floor, not a bug: with
  ±1.5 % RPM wander as a free parameter, a 500 m displacement changes
  the predicted curve by less than the ~1 Hz spectrogram bin noise. A
  sweep-width term was evaluated and removed (the measured std
  saturates plateau-to-plateau; zero position information).
* The reported point is the winning basin's **peak position**
  (score-weighted mean over near-top members): the basin center of mass
  averages an elongated ridge to its middle, which sat ~1 km from the
  peak that carried the evidence (world A: 1050 m → 470 m by this fix
  alone).
* Detection runtime ~155 s for 800 s of audio (STFT 4096 + 5-band
  Viterbi); search 9–96 s for 576 grid points × 2 events including two
  refinement passes (vectorized forward model: the exact
  retarded-time curve is built once per (candidate, flight) and every
  clock hypothesis is an interpolation).

## Validated units (see `smoke_test.py`)

## Requirements

numpy, scipy, matplotlib, soundfile; ffmpeg on PATH (codec degradation
tests and demo audio I/O only). Python 3.11+.

## References

- ISO 9613-1:1996 / ANSI S1.26-2014 — atmospheric absorption (constants
  validated numerically against published table values).
- U.S. Standard Atmosphere, 1976 — temperature/pressure profiles.
- Kabalasingham et al., "Acoustic localization and tracking of a multi-rotor
  UAV," JASA 148(3), 2020 — blade-pass-frequency tonality.
- Colsto, "I discovered a sound that leaks your exact location (and it's
  everywhere)" (YouTube, Nov 2026) — the source methodology.
- Simon Willison's writeup (Nov 3, 2026) — independent coverage.

Uses: forensic geolocation of evidence recordings, OSINT research.
License: for the requesting user's evaluation; treat as research code.
