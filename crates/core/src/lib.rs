#[cfg(test)]
mod tests {
    use super::*;

    fn oct() -> SdOct {
        SdOct::new(SPEC_BINS, PAD_LEN, vec![0.0; SPEC_BINS])
    }

    #[test]
    fn recovers_peak_from_clean_spectrum() {
        let n = SPEC_BINS;
        let spec: Vec<f32> = (0..n)
            .map(|i| 100.0 + 80.0 * (2.0 * std::f32::consts::PI * 300.0 / n as f32 * i as f32).cos())
            .collect();
        let mut oct = oct();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        let p = oct.peak_depth(&profile, 250, 350).expect("peak");
        assert!((p - 300.0).abs() < 3.0, "peak at {p}, expected ~300");
    }

    #[test]
    fn peak_respects_gate() {
        let n = SPEC_BINS;
        let spec: Vec<f32> = (0..n)
            .map(|i| 100.0 + 80.0 * (2.0 * std::f32::consts::PI * 800.0 / n as f32 * i as f32).cos())
            .collect();
        let mut oct = oct();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        let p2 = oct.peak_depth(&profile, 750, 850).unwrap();
        assert!((p2 - 800.0).abs() < 3.0, "peak at {p2}, expected ~800");
    }

    #[test]
    fn no_peak_returns_none() {
        let spec = vec![0.0; SPEC_BINS];
        let mut oct = oct();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        assert!(oct.peak_depth(&profile, 100, 400).is_none());
    }
}

use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};

pub const SPEC_BINS: usize = 2048;
// zero-padding (PAD_LEN>SPEC_BINS) would rescale every depth bin by
// PAD_LEN/SPEC_BINS, silently breaking sim↔core consistency; interpolate via
// the CoM instead. ponytail: reintroduce padding only when sub-bin accuracy
// is measured to need more than CoM interpolation.
pub const PAD_LEN: usize = SPEC_BINS;

pub struct SdOct {
    r2c: std::sync::Arc<dyn RealToComplex<f32>>,
    scratch: Vec<f32>,
    fft_out: Vec<Complex32>,
    grid: Vec<f32>,   // k-resampling grid (fractional source indices)
    window: Vec<f32>, // precomputed Hann
    bg: Vec<f32>,
    pub depth_bins: usize,
}

impl SdOct {
    pub fn new(spec_bins: usize, pad_len: usize, bg: Vec<f32>) -> Self {
        // k-grid: linear for v1 (real systems use a calibration curve here)
        let grid: Vec<f32> = (0..spec_bins).map(|i| i as f32).collect();
        let window: Vec<f32> = (0..spec_bins)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / spec_bins as f32).cos()))
            .collect();
        let mut p = RealFftPlanner::<f32>::new();
        let r2c = p.plan_fft_forward(pad_len);
        let scratch = r2c.make_input_vec();
        let fft_out = r2c.make_output_vec();
        SdOct { r2c, scratch, fft_out, grid, window, bg, depth_bins: pad_len / 2 }
    }

    /// Process one spectrum into a log-scaled power depth profile.
    pub fn process(&mut self, spec: &[f32], profile: &mut [f32]) {
        let n = spec.len();
        let mut s: Vec<f32> = spec.to_vec();
        // 1) background subtraction
        for (s, b) in s.iter_mut().zip(self.bg.iter()) { *s -= *b; }
        // 2) k-resampling (linear grid in v1; calibration curve plugs in here)
        let mut resampled = vec![0.0f32; n];
        for (j, &g) in self.grid.iter().enumerate() {
            let b = (g.min(n as f32 - 1.0).max(0.0) as usize).min(n - 2);
            resampled[j] = s[b] + (g - b as f32) * (s[b + 1] - s[b]);
        }
        // 3) spectral shaping (precomputed window)
        for (r, w) in resampled.iter_mut().zip(self.window.iter()) { *r *= *w; }
        // 4) zero-pad
        self.scratch[..n].copy_from_slice(&resampled);
        self.scratch[n..].fill(0.0);
        // 5) R2C FFT
        self.r2c.process(&mut self.scratch, &mut self.fft_out).unwrap();
        // 6) magnitude from power with log compression
        const K: f32 = 10.0 * 0.301_029_995_66; // 10*log10(2)
        for i in 0..profile.len() {
            let p = self.fft_out[i].re * self.fft_out[i].re
                + self.fft_out[i].im * self.fft_out[i].im;
            profile[i] = K * (p + 1.0).log2();
        }
    }

    /// Keyhole depth: centre-of-mass around the max of profile[gmin..gmax].
    pub fn peak_depth(&self, profile: &[f32], gmin: usize, gmax: usize) -> Option<f32> {
        if gmax <= gmin + 1 || gmax > profile.len() { return None; }
        let (mut imax, mut vmax) = (gmin, f32::MIN);
        for (i, &v) in profile[gmin..gmax].iter().enumerate() {
            if v > vmax { vmax = v; imax = gmin + i; }
        }
        if vmax <= 0.0 { return None; }
        let lo = imax.saturating_sub(2);
        let hi = (imax + 3).min(profile.len());
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (i, &v) in profile[lo..hi].iter().enumerate() {
            let w = (v - vmax * 0.5).max(0.0);
            num += w * (lo + i) as f32;
            den += w;
        }
        if den <= 0.0 { Some(imax as f32) } else { Some(num / den) }
    }
}
