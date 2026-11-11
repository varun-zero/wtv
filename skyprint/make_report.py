"""Generate the validation HTML report from real suite results."""
import json
import sys

RESULTS = "/tmp/skyprint_cache/results.json"
OUT = "/home/z/my-project/download/skyprint/validation_report.html"


def main():
    with open(RESULTS) as f:
        results = json.load(f)
    rows = ""
    for r in results:
        if r.get("status") == "skipped":
            rows += (f"<tr><td>{r['name']}</td><td>skipped</td>"
                     f"<td colspan=6>{r.get('reason','')}</td></tr>")
            continue
        err = r.get("error_m")
        err_s = f"{err:.0f}" if isinstance(err, (int, float)) else "—"
        top = r.get("top") or {}
        rows += (
            f"<tr><td>{r['name']}</td>"
            f"<td><b>{r['status']}</b></td>"
            f"<td>{err_s}</td>"
            f"<td>{r.get('radius_m','—')}</td>"
            f"<td>{top.get('score','—')}</td>"
            f"<td>{', '.join(top.get('flights', []))}</td>"
            f"<td>{top.get('clock_off','—')}s / {top.get('clock_ppm','—')}ppm</td>"
            f"<td>{r.get('n_events','—')} (f0 {r.get('event_f0')})</td>"
            f"<td>{r.get('detect_s','—')}+{r.get('search_s','—')}s</td></tr>"
        )
    located = [r for r in results if r.get("status") == "located"]
    errs = [r["error_m"] for r in located if isinstance(r.get("error_m"), (int, float))]
    summary = (
        f"{len(located)}/{len(results)} cases located"
        + (f", median error {sorted(errs)[len(errs)//2]:.0f} m" if errs else "")
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SkyPrint validation</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:2rem auto;max-width:62rem;
color:#111;background:#fafafa}} h1{{font-size:1.35rem}}
table{{border-collapse:collapse;width:100%;font-size:.82rem}}
td,th{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}}
th{{background:#eee}} .sum{{font-size:1.05rem;margin:.6rem 0 1rem}}
.note{{color:#555;font-size:.85rem}}
</style></head><body>
<h1>SkyPrint — end-to-end validation (synthetic worlds)</h1>
<p class="sum"><b>{summary}</b></p>
<table><tr><th>case</th><th>decision</th><th>error (m)</th><th>radius (m)</th>
<th>score</th><th>flights</th><th>clock (off/drift)</th><th>events</th>
<th>time</th></tr>
{rows}</table>
<p class="note">World A: two propeller flyovers on crossing headings
(eastbound C172-class at 100 m/s, westbound SR22-class at 110 m/s),
CPA arrivals ~500 s apart, distractor flights, 3 m/s wind; 800 s of
8 kHz audio rendered with retarded-time acoustics, ISO 9613-1
absorption, and RPM wander. Degradations applied to the rendered audio
only. World B: parallel headings 26 s apart — intrinsically degenerate
along the shared corridor; an honest system must report that ambiguity
rather than guess. Global sweep (G1-G4): the same crossing-geometry
scenario re-rendered at 78.2 N (Svalbard), across the dateline at 179 E
inside a wrapped search bounding box, in the Southern Ocean at 54.5 S,
and on the equatorial Pacific — exercising polar/dateline geodesy and
hemisphere-flipped ENU frames. Truth per case: A/B at 37.4000 N
122.3000 W; G1-G4 at the coordinates named in each case label.</p>
</body></html>"""
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
