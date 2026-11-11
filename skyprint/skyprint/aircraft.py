"""Aircraft-class priors: blade-pass frequency bounds, harmonic counts,
cruise-speed bounds and source levels, keyed by ICAO type designator.

These priors collapse the fundamental-frequency degeneracy that drove most
ambiguous abstentions in the source methodology: an observed harmonic
family must have a fundamental inside the class's BPF range, which ties
harmonic index k to physical blade counts. Values are engineering ranges
compiled from manufacturer data (2-blade fixed props ~2100-2700 RPM,
3-4 blade constant-speed props, turboprop prop RPM 1200-1800, helicopter
rotor RPM 240-400); generous bounds are deliberate -- the prior rejects
the clearly impossible, not the unusual.

Blade-pass frequency physics: BPF = blades x RPM / 60. See e.g.
Kabalasingham et al., "Acoustic localization and tracking of a multi-rotor
UAV" (JASA 148(3), 2020) for BPF tonality in rotor noise.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftClass:
    name: str
    tonal: bool            # produces stable tonal Doppler tracks?
    bpf_lo_hz: float        # fundamental blade-pass frequency lower bound
    bpf_hi_hz: float        # upper bound
    max_harmonic: int       # highest harmonic order with usable power
    speed_lo_ms: float      # cruise speed lower bound
    speed_hi_ms: float
    source_level_db: float  # broadband-ish level @ 1 m (priors only)

    def f0_bounds(self, allow_missing_fundamental: bool = True):
        if allow_missing_fundamental:
            # observed family may start at harmonic 2 (codec/HPF loss)
            return self.bpf_lo_hz, self.bpf_hi_hz
        return self.bpf_lo_hz, self.bpf_hi_hz


PISTON2 = AircraftClass("piston2", True, 60, 100, 6, 30, 75, 82)
PISTON3 = AircraftClass("piston3", True, 90, 140, 5, 40, 95, 84)
TWIN_PISTON = AircraftClass("twin_piston", True, 70, 120, 6, 50, 100, 88)
TURBOPROP = AircraftClass("turboprop", True, 55, 170, 6, 90, 180, 95)
HELICOPTER = AircraftClass("helicopter", True, 8, 32, 12, 45, 90, 92)
JET = AircraftClass("jet", False, 0, 0, 0, 120, 260, 105)
UNKNOWN_TONAL = AircraftClass("unknown_tonal", True, 55, 400, 5, 40, 180, 90)

# Type designator -> class. Unmatched types resolve to UNKNOWN_TONAL unless
# they match a jet pattern (jets are broadband; excluded from tonal matching
# but retained for silence/masking analysis).
_TYPE_MAP = {
    # 2-blade piston singles
    "C150": PISTON2, "C152": PISTON2, "C172": PISTON2, "C177": PISTON2,
    "C182": PISTON2, "C185": PISTON2, "PA28": PISTON2, "PA24": PISTON2,
    "AA5": PISTON2, "G73": PISTON2, "SR2": PISTON2, "M20": PISTON2,
    "BE36": PISTON2, "RV1": PISTON2, "RV7": PISTON2, "RV9": PISTON2,
    "CJ1": PISTON2, "CJ2": PISTON2,
    # 3-blade / constant-speed piston singles and twins
    "DA40": PISTON3, "DA42": PISTON3, "SR22": PISTON3, "C210": PISTON3,
    "F33": PISTON3, "V35": PISTON3, "LANC": PISTON3, "TRIN": PISTON3,
    "PA46": PISTON3, "PA32": PISTON3, "PA34": TWIN_PISTON, "PA44": TWIN_PISTON,
    "BE58": TWIN_PISTON, "BE76": TWIN_PISTON, "BE95": TWIN_PISTON,
    "TNav": TWIN_PISTON, "P28A": PISTON2, "P28B": PISTON2,
    # turboprops
    "C208": TURBOPROP, "C208B": TURBOPROP, "PC12": TURBOPROP, "TBM9": TURBOPROP,
    "TBM7": TURBOPROP, "BE20": TURBOPROP, "BE9L": TURBOPROP, "B190": TURBOPROP,
    "B1900": TURBOPROP, "DH8A": TURBOPROP, "DH8B": TURBOPROP, "DH8C": TURBOPROP,
    "DH8D": TURBOPROP, "AT72": TURBOPROP, "AT75": TURBOPROP, "AT76": TURBOPROP,
    "D228": TURBOPROP, "DHC6": TURBOPROP, "DHC7": TURBOPROP, "E50P": TURBOPROP,
    "EA50": TURBOPROP, "P18": TURBOPROP, "C130": TURBOPROP, "C30J": TURBOPROP,
    "SW4": TURBOPROP, "METRO": TURBOPROP, "F50": TURBOPROP, "AT43": TURBOPROP,
    "SF34": TURBOPROP, "DO228": TURBOPROP,
    # helicopters
    "R22": HELICOPTER, "R44": HELICOPTER, "R66": HELICOPTER,
    "H60": HELICOPTER, "H64": HELICOPTER, "H53": HELICOPTER, "H1": HELICOPTER,
    "H130": HELICOPTER, "H135": HELICOPTER, "H145": HELICOPTER, "H155": HELICOPTER,
    "EC35": HELICOPTER, "EC30": HELICOPTER, "EC45": HELICOPTER, "EC55": HELICOPTER,
    "A109": HELICOPTER, "A139": HELICOPTER, "A169": HELICOPTER, "A189": HELICOPTER,
    "B06": HELICOPTER, "B06T": HELICOPTER, "B212": HELICOPTER, "B412": HELICOPTER,
    "B222": HELICOPTER, "B230": HELICOPTER, "B407": HELICOPTER, "B429": HELICOPTER,
    "B06": HELICOPTER, "S76": HELICOPTER, "S92": HELICOPTER, "EN28": HELICOPTER,
    "SA31": HELICOPTER, "SA32": HELICOPTER, "SA34": HELICOPTER, "SA35": HELICOPTER,
    "AS50": HELICOPTER, "AS55": HELICOPTER, "AS65": HELICOPTER, "L4": HELICOPTER,
}

# jet designator patterns (prefix match, 2+ chars)
_JET_PREFIXES = (
    "B7", "B1", "B2", "B3", "B4", "B5", "B6", "B75", "B76", "B77", "B78",
    "A2", "A3", "A30", "A31", "A32", "A33", "A34", "A35", "A38",
    "E1", "E7", "E5", "E3", "E9", "CRJ", "CL6", "CL3", "C5", "C17", "C25",
    "C56", "C68", "C75", "C76", "FA7", "FA5", "FA6", "F2", "GL5", "GL6",
    "GL7", "MD8", "MD9", "MD1", "MD8", "P8", "K3", "T1", "T2", "IL9",
    "IL7", "IL8", "SU9", "SSJ", "MC2", "DC9", "MD1", "Galaxy", "B7",
)


def classify(type_designator: str | None):
    """Resolve an ICAO type designator to an AircraftClass."""
    if not type_designator:
        return UNKNOWN_TONAL
    t = str(type_designator).strip().upper()
    if t in _TYPE_MAP:
        return _TYPE_MAP[t]
    for p in _JET_PREFIXES:
        if t.startswith(p) and len(t) >= 3:
            return JET
    # known heli-ish patterns not in the map
    if t[:1] in ("R",) and len(t) == 3 and t[1:].isdigit():
        return HELICOPTER
    if t[:2] in ("EC", "AS", "SA") and len(t) == 4:
        return HELICOPTER
    return UNKNOWN_TONAL


def tonal_classes():
    return {PISTON2, PISTON3, TWIN_PISTON, TURBOPROP, HELICOPTER, UNKNOWN_TONAL}
