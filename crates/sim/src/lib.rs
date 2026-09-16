#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seed: u64) -> SimConfig {
        SimConfig {
            seed, noise_amp: 0.0, dc_level: 0.0, peak_amp: 100.0,
            peak_width_bins: 4.0, depth0_bins: 300.0, osc_amp_bins: 10.0,
            osc_hz: 100.0, defects: vec![],
        }
    }

    #[test]
    fn depth_stays_in_bounds() {
        let mut m = KeyholeModel::new(cfg(7));
        for _ in 0..1000 { m.advance(); }
        let d = m.depth();
        assert!(d > 1.0 && d < 2048.0 * 0.45, "depth {d}");
    }

    #[test]
    fn incomplete_defect_ramps_depth_down() {
        let mut c = cfg(7);
        // window sized to the 5000-sample walk (5000/80_000 s) so p reaches 1.0
        c.defects = vec![(0.0, 5000.0 / 80_000.0, Defect::Incomplete)];
        let mut m = KeyholeModel::new(c);
        m.advance();
        let mut lo = m.depth();
        for _ in 0..5000 { lo = lo.min(m.depth()); m.advance(); }
        assert!(lo < 200.0, "incomplete should drop depth, min={lo}");
    }

    #[test]
    fn spectrum_is_deterministic_for_seed() {
        let a = { let mut m = KeyholeModel::new(cfg(7)); m.advance() };
        let b = { let mut m = KeyholeModel::new(cfg(7)); m.advance() };
        assert_eq!(a, b);
        assert!(a.iter().any(|&v| v != 0.0));
    }
}

use num_complex::Complex32;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

pub const SPEC_BINS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Defect {
    None,
    Spatter,
    Pore,
    Incomplete,
    Humping,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    pub seed: u64,
    pub noise_amp: f32,        // additive gaussian noise on the spectrum
    pub dc_level: f32,         // DC pedestal
    pub peak_amp: f32,         // keyhole reflection amplitude
    pub peak_width_bins: f32,  // keyhole peak width in depth bins
    pub depth0_bins: f32,      // nominal keyhole depth (bins)
    pub osc_amp_bins: f32,     // keyhole oscillation amplitude
    pub osc_hz: f64,           // keyhole oscillation frequency
    pub defects: Vec<(f64, f64, Defect)>, // (start_s, end_s, kind)
}

pub struct KeyholeModel {
    cfg: SimConfig,
    t: f64,       // simulated time (s)
    dt: f64,      // seconds per A-scan (1/80_000 s)
    rng: StdRng,
    c2c: std::sync::Arc<dyn rustfft::Fft<f32>>,
    tmp: Vec<Complex32>,
    noise: Normal<f32>,
}

impl KeyholeModel {
    pub fn new(cfg: SimConfig) -> Self {
        let dt = 1.0 / 80_000.0;
        let rng = StdRng::seed_from_u64(cfg.seed);
        let noise = Normal::new(0.0, cfg.noise_amp.max(1e-6)).unwrap();
        let mut planner = FftPlanner::<f32>::new();
        let c2c = planner.plan_fft_inverse(SPEC_BINS);
        let tmp = vec![Complex32::new(0.0, 0.0); SPEC_BINS];
        KeyholeModel { cfg, t: 0.0, dt, rng, c2c, tmp, noise }
    }

    fn frame(&self) -> u64 {
        (self.t / self.dt) as u64
    }

    /// Deterministic [0,1) per-frame value (LCG), mirror of train_model.py hash01.
    fn hash01(k: u64) -> f32 {
        let s = k.wrapping_mul(2_654_435_761).wrapping_add(40_503);
        ((s >> 16) & 0x7fff) as f32 / 32_767.0
    }

    /// Current keyhole depth in bins (before advancing).
    pub fn depth(&self) -> f32 {
        let cfg = &self.cfg;
        let mut z = cfg.depth0_bins
            + cfg.osc_amp_bins * (2.0 * std::f64::consts::PI * cfg.osc_hz * self.t).sin() as f32;
        for &(t0, t1, kind) in &cfg.defects {
            if self.t < t0 || self.t > t1 {
                continue;
            }
            let p = ((self.t - t0) / (t1 - t0)).clamp(0.0, 1.0) as f32;
            z = match kind {
                Defect::None => z,
                // steady ±30-bin scattered-reflection jitter: per-sample steps
                // > spike_delta, so spatter_rate (fast-step count) can see it.
                Defect::Spatter => z + (Self::hash01(self.frame()) * 2.0 - 1.0) * 30.0,
                Defect::Pore => z * (1.0 - 0.35 * (0.5 + 0.5 * (p * 12.0).sin())),
                Defect::Incomplete => z * (1.0 - 0.65 * p),            // ramp down
                // fast ±20-bin oscillation that stays above the penetration
                // threshold: high std without pore-dropout crossings.
                Defect::Humping => z + 20.0 * (p * 24.0).sin(),
            };
        }
        z.clamp(1.0, SPEC_BINS as f32 * 0.45)
    }

    /// Generate the next interferometric spectrum and advance simulated time.
    pub fn advance(&mut self) -> Vec<f32> {
        let z = self.depth();
        let cfg = &self.cfg;
        // depth-domain reflectivity: gaussian peak at z (Hermitian-symmetric
        // so the inverse FFT yields a real-valued spectrum).
        self.tmp.fill(Complex32::new(0.0, 0.0));
        let w = cfg.peak_width_bins;
        for b in 0..SPEC_BINS {
            let d1 = (b as f32 - z) / w;
            let d2 = (SPEC_BINS as f32 - b as f32 - z) / w;
            let g = (-0.5 * d1 * d1).exp() + (-0.5 * d2 * d2).exp();
            self.tmp[b] = Complex32::new(cfg.peak_amp * g, 0.0);
        }
        self.c2c.process(&mut self.tmp);
        let mut spec = vec![0.0f32; SPEC_BINS];
        for (i, c) in self.tmp.iter().enumerate() {
            let v = c.re + cfg.dc_level;
            let n: f32 = if cfg.noise_amp > 0.0 { self.noise.sample(&mut self.rng) } else { 0.0 };
            spec[i] = (v + n).max(0.0);
        }
        self.t += self.dt;
        spec
    }
}
