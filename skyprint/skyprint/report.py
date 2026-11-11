"""Single-file HTML report (minimal, no framework): detected events,
ranked candidates with map-free coordinates, spectrogram of the best
event's band, and the decision + reasons."""
from __future__ import annotations

import base64
import io
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os

for _font_path in [
    '/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]:
    if os.path.exists(_font_path):
        fm.fontManager.addfont(_font_path)

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from .audio import spectrogram, whiten


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_report(path, events, cands, decision, truth, x, fs):
    # spectrogram overview
    spec, freqs, times = spectrogram(x, fs, n_fft=4096)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    m = freqs <= 400
    ax.imshow(20 * np.log10(spec[m] + 1e-9), aspect="auto", origin="lower",
              extent=[0, times[-1], 0, 400], cmap="magma")
    ax.set_xlabel("window time (s)")
    ax.set_ylabel("Hz")
    ax.set_title("recording spectrogram (0-400 Hz)")
    spec_b64 = _fig_b64(fig)

    # best event family curve if present
    ev_b64 = ""
    if events:
        ev = events[0]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(np.asarray(ev.freq_samples) - ev.freq_samples.min(),
                ev.freq_values, ".", ms=2, alpha=0.5)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("fundamental-implied Hz")
        ax.set_title(f"best event fit: f0={ev.f0_hz:.1f} Hz v={ev.speed_ms:.0f} m/s "
                     f"r={ev.slant_m:.0f} m")
        ev_b64 = _fig_b64(fig)

    d = decision.as_dict()
    rows = ""
    for i, c in enumerate(cands[:10]):
        rows += (
            f"<tr><td>{i+1}</td><td>{c.lat:.5f}</td><td>{c.lon:.5f}</td>"
            f"<td>{c.score:.2f}</td><td>{', '.join(c.flight_ids)}</td>"
            f"<td>{c.clock_offset_s:+.1f}s / {c.clock_drift_ppm:+.0f}ppm</td></tr>"
        )
    err = ""
    if truth and decision.top:
        from .geo import great_circle_distance
        e = great_circle_distance(decision.top.lat, decision.top.lon,
                                  truth[0], truth[1])
        err = f"<p>error vs truth: <b>{e:.0f} m</b></p>"
    reasons = "".join(f"<li>{r}</li>" for r in d.get("reasons", []))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SkyPrint report</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:2rem auto;max-width:60rem;
color:#111;background:#fafafa}} h1{{font-size:1.4rem}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}}
img{{max-width:100%;background:#fff;border:1px solid #ddd}}
.status{{font-size:1.1rem;font-weight:600}}
</style></head><body>
<h1>SkyPrint — aircraft-Doppler geolocation report</h1>
<p class="status">decision: {d['status']} (radius {d.get('radius_m',0):.0f} m)</p>
{err}<ul>{reasons}</ul>
<h2>Recording</h2><img src="data:image/png;base64,{spec_b64}">
<h2>Best event</h2><img src="data:image/png;base64,{ev_b64}">
<h2>Ranked candidates</h2>
<table><tr><th>#</th><th>lat</th><th>lon</th><th>score</th><th>flights</th>
<th>clock (offset/drift)</th></tr>{rows}</table>
<h2>JSON</h2>
<pre>{json.dumps({k: v for k, v in d.items()}, indent=2, default=str)[:1200]}</pre>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    return path
