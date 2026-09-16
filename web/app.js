import * as THREE from "three";

const state = {
  n: 256,           // A-scans
  spec: null,       // Float32Array n*2048 (spectra)
  depth: null,      // Float32Array n*2048 (profiles)
  profiles: null,   // 2D array for plotting
  wasm: null,
  frame: 0,
};

// Recipe matches config/sim.json's serde schema: defects are
// (start_s, end_s, kind) tuples; 256 frames at 80 kHz = 3.2 ms, so defect
// windows are scaled into that span to stay visible in the demo run.
const SIM_JSON = JSON.stringify({
  seed: 7, noise_amp: 4.0, dc_level: 50.0, peak_amp: 120.0,
  peak_width_bins: 4.0, depth0_bins: 300.0, osc_amp_bins: 14.0,
  osc_hz: 120.0,
  defects: [
    [0.0004, 0.0009, "spatter"],
    [0.0014, 0.0022, "incomplete"],
    [0.0024, 0.0031, "humping"]
  ],
});

const DEPTH_BINS = 2048;
const SPEC_BINS = 2048;

async function init() {
  const wasm = await import("./wasm/webcore.js");
  await wasm.default();
  state.wasm = wasm;
  state.spec = new Float32Array(wasm.generate_spectra(SIM_JSON, state.n));
  state.depth = new Float32Array(wasm.process_run(SIM_JSON, state.n));
  sliceProfiles();
  build3d();
  drawAll();
  document.getElementById("status").textContent = "ready (WASM fft computed " +
    state.n + " ascans)";
}

function drawAll() {
  drawTrace();
  drawAscan();
}

function drawAscan() {
  const c = document.getElementById("ascan");
  const ctx = c.getContext("2d");
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = 200 * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  const off = state.profiles[state.frame] || new Float32Array(DEPTH_BINS);
  ctx.beginPath();
  for (let i = 0; i < DEPTH_BINS; i += 4) {
    const x = (i / DEPTH_BINS) * w;
    const y = h - (off[i] / 40) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#3fb6ff";
  ctx.stroke();
}

function drawTrace() {
  const c = document.getElementById("trace");
  const ctx = c.getContext("2d");
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = 200 * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  for (let i = 0; i < state.n; i++) {
    const profile = state.profiles[i];
    if (!profile) continue;
    let peak = -1, pv = -Infinity;
    for (let b = 40; b < 1000; b++) if (profile[b] > pv) { pv = profile[b]; peak = b; }
    const x = (i / state.n) * w;
    const y = h - ((peak - 40) / 960) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#4ade80";
  ctx.stroke();
}

function build3d() {
  const container = document.getElementById("three");
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / 320, 0.1, 5000);
  camera.position.set(0, -40, 90);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, 320);
  container.appendChild(renderer.domElement);

  const pts = [];
  for (let i = 0; i < state.n; i += 2) {
    const profile = state.profiles[i];
    if (!profile) continue;
    for (let b = 0; b < DEPTH_BINS; b += 4) {
      const v = profile[b];
      if (v < 3) continue;
      pts.push((i / state.n - 0.5) * 30, (b / DEPTH_BINS - 0.5) * 40, -(v / 40) * 6);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.PointsMaterial({ color: 0x3fb6ff, size: 0.15 });
  const cloud = new THREE.Points(geo, mat);
  scene.add(cloud);
  const axes = new THREE.AxesHelper(20);
  scene.add(axes);

  let angle = 0;
  const animate = () => {
    requestAnimationFrame(animate);
    angle += 0.004;
    camera.position.x = Math.sin(angle) * 90;
    camera.position.z = Math.cos(angle) * 90;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  };
  animate();
}

// profile rows from flat depth buffer
function sliceProfiles() {
  const n = state.n;
  state.profiles = [];
  for (let i = 0; i < n; i++) {
    state.profiles.push(state.depth.slice(i * DEPTH_BINS, (i + 1) * DEPTH_BINS));
  }
}

document.getElementById("play").addEventListener("click", async () => {
  if (!state.wasm) return;
  const t0 = performance.now();
  state.depth = new Float32Array(state.wasm.process_run(SIM_JSON, state.n));
  document.getElementById("status").textContent =
    `recomputed ${state.n} FFTs in ${(performance.now() - t0).toFixed(1)} ms`;
  sliceProfiles();
  drawAll();
});
document.getElementById("frame").addEventListener("input", (e) => {
  state.frame = parseInt(e.target.value, 10);
  drawAscan();
});

await init();
