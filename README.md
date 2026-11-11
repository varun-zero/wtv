# SkyPrint

Aircraft-Doppler acoustic geolocation. This repository contains two ways to run
the project:

| Path | What it is |
| --- | --- |
| `index.html` + `public/` | The SkyPrint Geolocation Workbench as a standalone Vite web app (used by Google AI Studio Build). |
| `skyprint/` | The Python research implementation, CLI, UI server, and validation suite. See [`skyprint/README.md`](skyprint/README.md) for the full technical write-up. |

## Run the workbench locally

```bash
npm install
npm run dev
# open http://localhost:3000
```

`npm run build` produces a static `dist/` folder that can be hosted anywhere.

The workbench is self-contained: all nine scenario telemetry sets are embedded
in `index.html`, and the 40-second flyover recordings are pre-rendered WAV files
in `public/audio/`. No Python backend is needed for the web app.

## Google AI Studio

The repo is laid out as a standard Vite app so AI Studio Build can import it:

- `package.json` at the repo root (AI Studio runs `bun install` / `bun run dev`)
- `vite.config.ts` binds the dev server to `0.0.0.0:3000`
- `metadata.json` describes the app
- `index.html` is the app entry point; `public/audio/` holds the static audio

In AI Studio Build, use **Add files → Import from GitHub** and select
`varun-zero/wtv`, or connect the repo from **Settings → GitHub**.

## Python pipeline

```bash
cd skyprint
python -m skyprint.cli demo
python -m skyprint.cli ui --port 8080
```

The Python path still provides the full detection/search/decision pipeline and
its own local UI server. The Vite app is a static presentation layer over the
same precomputed scenario data.
