let wasm = null, state = { frame: 0, window: true, pad: true };
const SIM_JSON = JSON.stringify({ seed: 1, noise_amp: 2.0, dc_level: 40.0,
  peak_amp: 120.0, peak_width_bins: 4.0, depth0_bins: 300.0,
  osc_amp_bins: 0.0, osc_hz: 0.0, defects: [] });
const SPEC_BINS = 2048, DEPTH_BINS = 2048;

const canvas = document.getElementById("demo");
const ctx = canvas.getContext("2d");

function drawSpectrumRow() {
  const spec = new Float32Array(wasm.generate_spectra(SIM_JSON, 1));
  // plot raw + windowed + padded versions
  ctx.fillStyle = "#0e1116";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const mid = canvas.height / 2;
  plot(ctx, spec, "#3fb6ff", mid, 0.9);
  if (state.window) {
    const w = new Float32Array(SPEC_BINS);
    for (let i = 0; i < SPEC_BINS; i++)
      w[i] = spec[i] * 0.5 * (1 - Math.cos(2 * Math.PI * i / SPEC_BINS));
    plot(ctx, w, "#e879f9", mid, 0.45);
  }
  const profile = new Float32Array(wasm.process_spectrum(spec));
  // without zero-padding the FFT runs on 2048 points: coarser depth sampling
  const shown = state.pad ? profile : profile.subarray(0, SPEC_BINS);
  plot(ctx, shown, "#4ade80", mid + mid * 0.9, 0.9);
  ctx.fillStyle = "#9fb2cc";
  ctx.fillText("blue: spectrum · magenta: windowed · green: A-scan (|FFT|², log)", 10, 16);
}

function plot(ctx, arr, color, yBase, scale) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i += 8) {
    const x = (i / arr.length) * canvas.width;
    const y = yBase - (arr[i] / 120) * scale * 60;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

document.getElementById("d_play").onclick = () => {
  state.frame += 40;
  drawSpectrumRow();
};
document.getElementById("d_window").onclick = () => {
  state.window = !state.window;
  drawSpectrumRow();
};
document.getElementById("d_pad").onclick = () => {
  state.pad = !state.pad;
  drawSpectrumRow();
};
document.getElementById("d_status").textContent = "loading…";

const wasmMod = await import("./wasm/pkg/webcore.js");
await wasmMod.default();
wasm = wasmMod;
drawSpectrumRow();
document.getElementById("d_status").textContent =
  "WASM core runs the exact realfft pipeline of the native Rust system.";
