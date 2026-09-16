use core::{SdOct, PAD_LEN, SPEC_BINS};
use sim::{KeyholeModel, SimConfig};
use wasm_bindgen::prelude::*;

/// Generate `n` interferometric spectra in the browser from a sim recipe
/// given as a JSON string (same schema as config/sim.json).
/// Returns one big f32 slice: n * SPEC_BINS values (row-major).
#[wasm_bindgen]
pub fn generate_spectra(sim_json: &str, n: usize) -> Vec<f32> {
    let cfg: SimConfig = serde_json::from_str(sim_json).expect("valid sim json");
    let mut model = KeyholeModel::new(cfg);
    let mut out = Vec::with_capacity(n * SPEC_BINS);
    for _ in 0..n {
        out.extend_from_slice(&model.advance());
    }
    out
}

/// Process one spectrum -> depth profile (`depth_bins` values).
#[wasm_bindgen]
pub fn process_spectrum(spec: &[f32]) -> Vec<f32> {
    // PAD_LEN (not 4096): a padded FFT doubles bin indices, so padding to
    // 2*SPEC_BINS would silently double all depth units vs the trained model.
    let mut oct = SdOct::new(SPEC_BINS, PAD_LEN, vec![50.0; SPEC_BINS]);
    let mut profile = vec![0.0; oct.depth_bins];
    oct.process(spec, &mut profile);
    profile
}

/// Depth trace for a full generated run: returns `depth_bins * n` f32 values.
/// Internally re-runs the same SdOct instance for coherence.
#[wasm_bindgen]
pub fn process_run(sim_json: &str, n: usize) -> Vec<f32> {
    let cfg: SimConfig = serde_json::from_str(sim_json).expect("valid sim json");
    let mut model = KeyholeModel::new(cfg);
    let mut oct = SdOct::new(SPEC_BINS, PAD_LEN, vec![50.0; SPEC_BINS]);
    let mut out = Vec::with_capacity(n * oct.depth_bins);
    for _ in 0..n {
        let spec = model.advance();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        out.extend_from_slice(&profile);
    }
    out
}

/// Simulated keyhole depth in bins at frame `i` (mirrors sim::KeyholeModel).
#[wasm_bindgen]
pub fn sim_keyhole_depth(sim_json: &str, frame: usize) -> f32 {
    let cfg: SimConfig = serde_json::from_str(sim_json).expect("valid sim json");
    let mut m = KeyholeModel::new(cfg);
    for _ in 0..frame {
        m.advance();
    }
    m.depth()
}
