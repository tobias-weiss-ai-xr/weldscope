# WeldScope

In-process **OCT** (optical coherence tomography) weld-quality monitoring —
a portfolio system demonstrating a real-time signal pipeline in **Rust**,
physics-based feature extraction, an **AI** verdict, a Python test harness,
and an interactive **Three.js + WASM** showcase on GitHub Pages.

## Use case: battery connector welding

Laser welding is the standard joining process for cell connectors and battery
tabs in EV battery-pack production: thin copper and aluminium parts welded at
high volume, where a single bad joint means lost electrical contact, thermal
stress, and a rejected pack. The keyhole is millimetre-scale and the whole
weld takes tens of milliseconds, so quality must be judged *during* the weld,
not after. The defects that matter most —

- **spatter** (droplet ejection, weakening the seam),
- **pore / porosity** (gas bubbles trapped in the weld),
- **incomplete penetration** (the seam does not reach full depth —
  the most safety-critical),
- **humping** (periodic surface spilling, from high welding speed),

— appear directly in the keyhole depth dynamics in the sub-100 µm range.
WeldScope is an in-process OCT monitor for exactly this: it reads the keyhole
depth at high rate, extracts those defect signatures, and flags them
per-weld in real time.

## What it does
- simulates interferometric spectra of a laser keyhole (with seeded defects:
  spatter, pores, incomplete penetration, humping),
- runs the SD-OCT FFT chain (background subtraction → k-resampling → Hann →
  realfft → log-magnitude) per A-scan in real time,
- extracts the keyhole depth trace and derives physical quality features,
- classifies weld quality with a trained linear model (JSON weights),
- ships a documented binary wire format (`WS01`), process-level modules, and
  end-to-end latency p50/p99 instrumentation.

## Repository layout
```
crates/io        # WS01 framing + payload codecs
crates/sim       # keyhole dynamics + interferometric spectrum generator
crates/core      # SD-OCT FFT pipeline + peak extraction (→ WASM)
crates/features  # physical quality features
crates/ai        # softmax classifier (JSON weights)
crates/weldscope # orchestrator (offline / run / kill) + latency stats
crates/webcore   # wasm-bindgen shim (in-browser FFT)
harness/         # pytest: golden, latency, soak, regression
web/             # GitHub Pages: Three.js demo + workflow docs
docs/wire-format.md
```

## Try it
- **Browser demo:** https://tobias-weiss-ai-xr.github.io/weldscope/
- **CLI full chain:**
  `cargo run -p weldscope -- offline && tail -1 data/offline_report.txt`
- **Live multi-process system:**
  `cargo build --release && WELDSCOPE_BIN_DIR=target/release cargo run -p weldscope -- run`
  (Ctrl-C to stop; `weldscope kill` to kill module processes)
- **Tests:** `cargo test --workspace && python3 -m pytest harness -v`

## Notes
Data is synthetic by design: public in-process weld-OCT data does not exist
(verified via GitHub/paper searches, 2026-09); the simulator is
physically consistent (spectrum = IFFT of designed reflectivity) so the FFT
chain is exercised with realistic signals. A real sensor plugs in behind the
`acq` frame producer without touching the core.
