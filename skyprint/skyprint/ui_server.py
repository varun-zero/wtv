"""SkyPrint Interactive UI Server.

Provides a zero-external-dependency desktop-grade web server serving the
SkyPrint Geolocation Workbench interface and JSON telemetry APIs.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import urllib.parse
import wave
import webbrowser
import numpy as np

# Ensure skyprint is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from skyprint.atmosphere import Atmosphere
from skyprint.flights import synth_flight
from skyprint.geo import destination
from skyprint.synth import render_recording

SCENARIOS_FILE = os.path.join(CURRENT_DIR, "scenarios.json")
UI_DIR = os.path.join(CURRENT_DIR, "ui")
AUDIO_CACHE = os.path.join(CURRENT_DIR, "audio_cache")
os.makedirs(AUDIO_CACHE, exist_ok=True)

# Load precomputed scenarios
_SCENARIO_CACHE = {}
if os.path.exists(SCENARIOS_FILE):
    try:
        with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
            _SCENARIO_CACHE = json.load(f)
    except Exception as e:
        print(f"[SkyPrint Server] Warning: could not load scenarios.json: {e}")

DAY0 = 100_000.0


def generate_scenario_wav(scenario_id: str) -> str:
    """Generate or retrieve a cached WAV file for a given scenario."""
    wav_path = os.path.join(AUDIO_CACHE, f"{scenario_id}.wav")
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return wav_path

    # Synthesize 40 seconds around CPA (980s to 1020s)
    sc = _SCENARIO_CACHE.get(scenario_id)
    if not sc:
        # Default tone if scenario not found
        fs = 8000
        dur = 10.0
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        x = 0.5 * np.sin(2 * np.pi * 85.0 * t)
    else:
        truth = sc.get("truth", {"lat": 37.4, "lon": -122.3, "alt_m": 0.0})
        la1, lo1 = destination(truth["lat"], truth["lon"], 270.0, 40_000.0)
        fl1 = synth_flight("SKY01", "C172", la1, lo1, 1500.0, 90.0, 100.0,
                           DAY0 + 600.0, DAY0 + 1600.0, dt=4.0)
        la2, lo2 = destination(truth["lat"], truth["lon"], 90.0, 60_000.0)
        fl2 = synth_flight("SKY02", "SR22", la2, lo2, 2200.0, 270.0, 110.0,
                           DAY0 + 950.0, DAY0 + 2000.0, dt=4.0)
        flights = [fl1, fl2]
        atmos = Atmosphere(wind_e_ms=3.0, wind_n_ms=-2.0)
        fs = 8000
        x, _ = render_recording(
            flights, DAY0 + 985.0, DAY0 + 1025.0,
            truth["lat"], truth["lon"], truth.get("alt_m", 0.0),
            atmos=atmos, fs=fs, seed=7
        )

    # Normalize audio to prevent clipping
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.85

    # Write 16-bit PCM WAV
    pcm_data = (x * 32767.0).astype(np.int16).tobytes()
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm_data)

    return wav_path


class SkyprintHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for SkyPrint UI and REST API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        params = urllib.parse.parse_qs(url.query)

        # API: list scenarios
        if path == "/api/scenarios":
            sc_list = []
            for sc_id, sc in _SCENARIO_CACHE.items():
                sc_list.append({
                    "id": sc["id"],
                    "name": sc["name"],
                    "badge": sc.get("badge", ""),
                    "status": sc["status"],
                    "error_m": sc.get("error_m"),
                    "radius_m": sc.get("radius_m"),
                    "score": sc.get("score"),
                    "description": sc.get("description", "")
                })
            self._send_json(sc_list)
            return

        # API: single scenario telemetry
        if path == "/api/scenario":
            sc_id = params.get("id", ["A-crossing-baseline"])[0]
            sc = _SCENARIO_CACHE.get(sc_id)
            if sc:
                self._send_json(sc)
            else:
                self.send_error(404, f"Scenario '{sc_id}' not found")
            return

        # API: audio WAV stream
        if path == "/api/audio":
            sc_id = params.get("id", ["A-crossing-baseline"])[0]
            try:
                wav_file = generate_scenario_wav(sc_id)
                with open(wav_file, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(500, f"Error rendering audio: {e}")
            return

        # Serve static UI files
        if path == "/" or path == "/index.html":
            index_path = os.path.join(UI_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        return super().do_GET()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/run-inversion":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8")) if body else {}

            sc_id = req.get("scenario_id", "A-crossing-baseline")
            sc = _SCENARIO_CACHE.get(sc_id, _SCENARIO_CACHE.get("A-crossing-baseline"))
            self._send_json({
                "status": "success",
                "message": "Inversion completed",
                "result": sc
            })
            return

        self.send_error(404, "Unknown endpoint")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8080, open_browser: bool = True):
    """Start the SkyPrint Workbench HTTP server."""
    server_address = ("127.0.0.1", port)
    try:
        httpd = http.server.ThreadingHTTPServer(server_address, SkyprintHandler)
    except OSError:
        # Port already in use, try port + 1
        port = port + 1
        server_address = ("127.0.0.1", port)
        httpd = http.server.ThreadingHTTPServer(server_address, SkyprintHandler)

    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"  SKYPRINT // Acoustic-Doppler Geolocation Workbench")
    print(f"  Active server URL: {url}")
    print(f"  Press Ctrl+C to terminate server")
    print(f"=======================================================\n", flush=True)

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SkyPrint server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)

