#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_fills_then_emits() {
        let mut fe = FeatureExtractor::new(4, 250.0, 30.0);
        assert!(fe.push(Some(300.0)).is_none());
        assert!(fe.push(Some(300.0)).is_none());
        assert!(fe.push(Some(300.0)).is_none());
        let f = fe.push(Some(300.0)).unwrap();
        assert_eq!(f.mean, 300.0);
        assert_eq!(f.std, 0.0);
        assert_eq!(f.pene_ratio, 1.0);
    }

    #[test]
    fn detects_spatter_and_pore() {
        let mut fe = FeatureExtractor::new(9, 250.0, 30.0);
        for _ in 0..4 {
            fe.push(Some(300.0));
        }
        fe.push(Some(360.0)); // spatter spike
        for _ in 0..3 {
            fe.push(Some(300.0));
        }
        let f = fe.push(Some(100.0)).unwrap(); // pore dropout
        assert!(f.spatter_rate > 0.0);
        assert!(f.pore_count > 0.0);
        assert!(f.pene_ratio < 1.0);
    }

    const TRUE_DEPTH: f32 = 300.0;

    #[test]
    fn percentile_beats_mean_under_spatter() {
        // 7 true samples + two spatter spikes + one dropout
        let win = [Some(TRUE_DEPTH); 7]
            .into_iter()
            .chain([Some(850.0), Some(950.0), Some(100.0)])
            .collect::<Vec<_>>();
        let some: Vec<f32> = win.iter().flatten().copied().collect();
        let mean = DepthEstimator::Mean.estimate(&some).unwrap();
        let pct = DepthEstimator::Percentile(0.5).estimate(&some).unwrap();
        assert!((mean - 400.0).abs() < 1e-3, "mean pulled up: {}", mean);
        assert_eq!(pct, TRUE_DEPTH);
        assert!((pct - TRUE_DEPTH).abs() < (mean - TRUE_DEPTH).abs());
    }

    #[test]
    fn percentile_beats_mean_under_dropout() {
        // pore dropouts drag the mean below the true depth
        let some = [
            TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH,
            100.0, 100.0, 100.0,
        ];
        let mean = DepthEstimator::Mean.estimate(&some).unwrap();
        let pct = DepthEstimator::Percentile(0.5).estimate(&some).unwrap();
        assert!((mean - 240.0).abs() < 1e-3, "mean dragged down: {}", mean);
        assert_eq!(pct, TRUE_DEPTH);
    }

    #[test]
    fn density_percentile_drops_isolated_noise_cluster() {
        // tight cluster of spike samples survives a plain percentile at p=0.75
        // (lands inside it) but is dropped as noise by the density filter
        let some = [
            TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, TRUE_DEPTH, 320.0, 900.0, 910.0, 920.0,
        ];
        let mean = DepthEstimator::Mean.estimate(&some).unwrap();
        let pct = DepthEstimator::Percentile(0.75).estimate(&some).unwrap();
        let den = DepthEstimator::DensityPercentile {
            eps: 100.0,
            min_pts: 3,
            p: 0.75,
        }
        .estimate(&some)
        .unwrap();
        assert!((mean - 4550.0 / 9.0).abs() < 1e-3);
        assert_eq!(pct, 900.0);
        assert_eq!(den, TRUE_DEPTH);
        assert!((den - TRUE_DEPTH).abs() < (pct - TRUE_DEPTH).abs());
    }

    #[test]
    fn estimator_changes_values_not_format() {
        let mut a = FeatureExtractor::new(9, 250.0, 30.0); // default stays Mean
        let mut b =
            FeatureExtractor::with_estimator(9, 250.0, 30.0, DepthEstimator::Percentile(0.5));
        for s in [Some(TRUE_DEPTH); 8] {
            a.push(s);
            b.push(s);
        }
        let fa = a.push(Some(600.0)).unwrap();
        let fb = b.push(Some(600.0)).unwrap();
        assert!(fa.mean > fb.mean, "spike must lift mean above percentile");
        assert!((fa.mean - 333.333).abs() < 0.01);
        assert_eq!(fb.mean, TRUE_DEPTH);
        // encoding unchanged: round-trip of both vectors
        assert_eq!(from_vec(&to_vec(&fa)), fa);
        assert_eq!(to_vec(&fb).len(), 8);
    }

    #[test]
    fn empty_and_all_noise_windows_degrade_gracefully() {
        assert_eq!(DepthEstimator::Mean.estimate(&[]), None);
        // min_pts so high that every sample is "noise" -> None
        let den = DepthEstimator::DensityPercentile {
            eps: 1.0,
            min_pts: 99,
            p: 0.5,
        };
        assert_eq!(den.estimate(&[300.0, 300.0]), None);
    }

    #[test]
    fn to_from_vec_consistent() {
        let f = Features {
            mean: 1.0,
            std: 2.0,
            min: 3.0,
            max: 4.0,
            pene_ratio: 0.5,
            spatter_rate: 0.1,
            pore_count: 2.0,
            humping_index: 0.2,
        };
        assert_eq!(from_vec(&to_vec(&f)), f);
    }
}

/// Pluggable keyhole-depth estimator over one window of valid depth samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthEstimator {
    /// Arithmetic mean (baseline, historical default).
    Mean,
    /// Windowed percentile filter over valid depth samples
    /// (Boley et al. 2019, Optics and Lasers in Engineering, doi:10.1016/j.optlaseng.2019.03.014).
    /// `p` is the quantile in 0.0..=1.0.
    Percentile(f32),
    /// Density-based noise removal, then percentile filter
    /// (Xie et al. 2023, Sensors, doi:10.3390/s23115223).
    /// Samples with fewer than `min_pts` neighbours within `eps` depth are
    /// dropped as noise before the `p` percentile is taken.
    DensityPercentile { eps: f32, min_pts: usize, p: f32 },
}

impl DepthEstimator {
    /// Robust depth estimate from the valid samples of one window.
    pub fn estimate(&self, samples: &[f32]) -> Option<f32> {
        if samples.is_empty() {
            return None;
        }
        match *self {
            DepthEstimator::Mean => Some(samples.iter().sum::<f32>() / samples.len() as f32),
            DepthEstimator::Percentile(p) => Some(percentile(samples, p)),
            DepthEstimator::DensityPercentile { eps, min_pts, p } => {
                // ponytail: O(n^2) density check; windows are tens of samples
                let kept: Vec<f32> = samples
                    .iter()
                    .copied()
                    // count includes the sample itself, so `> min_pts` means
                    // at least `min_pts` neighbours besides itself
                    .filter(|&x| {
                        samples.iter().filter(|&&y| (y - x).abs() <= eps).count() > min_pts
                    })
                    .collect();
                if kept.is_empty() {
                    return None;
                }
                Some(percentile(&kept, p))
            }
        }
    }
}

/// Nearest-rank percentile, `p` in 0.0..=1.0.
fn percentile(samples: &[f32], p: f32) -> f32 {
    let mut v = samples.to_vec();
    v.sort_by(f32::total_cmp);
    let idx = (p.clamp(0.0, 1.0) * (v.len() - 1) as f32).round() as usize;
    v[idx.min(v.len() - 1)]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Features {
    /// Central depth estimate per the configured `DepthEstimator`
    /// (arithmetic mean by default).
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub pene_ratio: f32,    // fraction of window above penetration threshold
    pub spatter_rate: f32,  // fraction of adjacent steps exceeding spike delta
    pub pore_count: f32,    // number of above->below threshold crossings
    pub humping_index: f32, // std of 8 sub-window means (periodicity proxy)
}

pub struct FeatureExtractor {
    window: usize,
    thresh: f32,
    spike_delta: f32,
    est: DepthEstimator,
    buf: Vec<Option<f32>>,
}

impl FeatureExtractor {
    pub fn new(window: usize, thresh: f32, spike_delta: f32) -> Self {
        FeatureExtractor::with_estimator(window, thresh, spike_delta, DepthEstimator::Mean)
    }

    /// `new` with an explicit depth estimator (see `DepthEstimator`).
    pub fn with_estimator(
        window: usize,
        thresh: f32,
        spike_delta: f32,
        est: DepthEstimator,
    ) -> Self {
        FeatureExtractor {
            window,
            thresh,
            spike_delta,
            est,
            buf: Vec::with_capacity(window),
        }
    }

    /// Push one depth sample; returns features every `window` samples (sliding).
    pub fn push(&mut self, d: Option<f32>) -> Option<Features> {
        self.buf.push(d);
        if self.buf.len() < self.window {
            return None;
        }
        let win = self.buf.drain(..).collect::<Vec<_>>();
        Some(compute(&win, self.thresh, self.spike_delta, &self.est))
    }
}

fn compute(win: &[Option<f32>], thresh: f32, spike_delta: f32, est: &DepthEstimator) -> Features {
    let w = win.len() as f32;
    let some: Vec<f32> = win.iter().flatten().copied().collect();
    let (mean, std) = if some.is_empty() {
        (0.0, 0.0)
    } else {
        let m = est
            .estimate(&some)
            .unwrap_or_else(|| some.iter().sum::<f32>() / some.len() as f32);
        let v = some.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / some.len() as f32;
        (m, v.sqrt())
    };
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for &v in &some {
        min = min.min(v);
        max = max.max(v);
    }
    if some.is_empty() {
        min = 0.0;
        max = 0.0;
    }
    let pene_ratio = some.iter().filter(|&&v| v > thresh).count() as f32 / w;
    let mut spatter = 0;
    for pair in some.windows(2) {
        if (pair[1] - pair[0]).abs() > spike_delta {
            spatter += 1;
        }
    }
    let spatter_rate = spatter as f32 / (w.max(1.0) - 1.0).max(1.0);
    let mut pore = 0;
    let mut prev_above = false;
    for &v in &some {
        let above = v > thresh;
        if prev_above && !above {
            pore += 1;
        }
        prev_above = above;
    }
    // humping: std of 8 sub-window means
    let g = 8usize.min(some.len().max(1));
    let per = (some.len() / g).max(1);
    let mut sub = Vec::with_capacity(g);
    for k in 0..g {
        let chunk = &some[k * per..((k + 1) * per).min(some.len())];
        if !chunk.is_empty() {
            sub.push(chunk.iter().sum::<f32>() / chunk.len() as f32);
        }
    }
    let humping_index = if sub.len() >= 2 {
        let sm = sub.iter().sum::<f32>() / sub.len() as f32;
        (sub.iter().map(|x| (x - sm) * (x - sm)).sum::<f32>() / sub.len() as f32).sqrt()
    } else {
        0.0
    };
    Features {
        mean,
        std,
        min,
        max,
        pene_ratio,
        spatter_rate,
        pore_count: pore as f32,
        humping_index,
    }
}

pub fn to_vec(f: &Features) -> [f32; 8] {
    [
        f.mean,
        f.std,
        f.min,
        f.max,
        f.pene_ratio,
        f.spatter_rate,
        f.pore_count,
        f.humping_index,
    ]
}

pub fn from_vec(v: &[f32]) -> Features {
    Features {
        mean: v[0],
        std: v[1],
        min: v[2],
        max: v[3],
        pene_ratio: v[4],
        spatter_rate: v[5],
        pore_count: v[6],
        humping_index: v[7],
    }
}

/// Peak bin index of a depth profile (mirrors core::SdOct::peak_depth); used
/// by the features module binary so it does not depend on core's instance.
pub fn peak_of(profile: &[f32]) -> Option<f32> {
    let gmin = 40usize.min(profile.len().saturating_sub(1));
    let gmax = 1000usize.min(profile.len());
    if gmax <= gmin + 1 {
        return None;
    }
    let (mut imax, mut vmax) = (gmin, f32::MIN);
    for (i, &v) in profile[gmin..gmax].iter().enumerate() {
        if v > vmax {
            vmax = v;
            imax = gmin + i;
        }
    }
    if vmax <= 0.0 {
        return None;
    }
    let lo = imax.saturating_sub(2);
    let hi = (imax + 3).min(profile.len());
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (i, &v) in profile[lo..hi].iter().enumerate() {
        let w = (v - vmax * 0.5).max(0.0);
        num += w * (lo + i) as f32;
        den += w;
    }
    if den <= 0.0 {
        Some(imax as f32)
    } else {
        Some(num / den)
    }
}
