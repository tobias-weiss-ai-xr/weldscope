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
        for _ in 0..4 { fe.push(Some(300.0)); }
        fe.push(Some(360.0)); // spatter spike
        for _ in 0..3 { fe.push(Some(300.0)); }
        let f = fe.push(Some(100.0)).unwrap(); // pore dropout
        assert!(f.spatter_rate > 0.0);
        assert!(f.pore_count > 0.0);
        assert!(f.pene_ratio < 1.0);
    }

    #[test]
    fn to_from_vec_consistent() {
        let f = Features {
            mean: 1.0, std: 2.0, min: 3.0, max: 4.0,
            pene_ratio: 0.5, spatter_rate: 0.1, pore_count: 2.0, humping_index: 0.2,
        };
        assert_eq!(from_vec(&to_vec(&f)), f);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Features {
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
    buf: Vec<Option<f32>>,
}

impl FeatureExtractor {
    pub fn new(window: usize, thresh: f32, spike_delta: f32) -> Self {
        FeatureExtractor { window, thresh, spike_delta, buf: Vec::with_capacity(window) }
    }

    /// Push one depth sample; returns features every `window` samples (sliding).
    pub fn push(&mut self, d: Option<f32>) -> Option<Features> {
        self.buf.push(d);
        if self.buf.len() < self.window {
            return None;
        }
        let win = self.buf.drain(..).collect::<Vec<_>>();
        Some(compute(&win, self.thresh, self.spike_delta))
    }
}

fn compute(win: &[Option<f32>], thresh: f32, spike_delta: f32) -> Features {
    let w = win.len() as f32;
    let some: Vec<f32> = win.iter().flatten().copied().collect();
    let (mean, std) = if some.is_empty() {
        (0.0, 0.0)
    } else {
        let m = some.iter().sum::<f32>() / some.len() as f32;
        let v = some.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / some.len() as f32;
        (m, v.sqrt())
    };
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for &v in &some {
        min = min.min(v);
        max = max.max(v);
    }
    if some.is_empty() { min = 0.0; max = 0.0; }
    let pene_ratio = some.iter().filter(|&&v| v > thresh).count() as f32 / w;
    let mut spatter = 0;
    for pair in some.windows(2) {
        if (pair[1] - pair[0]).abs() > spike_delta { spatter += 1; }
    }
    let spatter_rate = spatter as f32 / (w.max(1.0) - 1.0).max(1.0);
    let mut pore = 0;
    let mut prev_above = false;
    for &v in &some {
        let above = v > thresh;
        if prev_above && !above { pore += 1; }
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
    Features { mean, std, min, max, pene_ratio, spatter_rate, pore_count: pore as f32, humping_index }
}

pub fn to_vec(f: &Features) -> [f32; 8] {
    [f.mean, f.std, f.min, f.max, f.pene_ratio, f.spatter_rate, f.pore_count, f.humping_index]
}

pub fn from_vec(v: &[f32]) -> Features {
    Features {
        mean: v[0], std: v[1], min: v[2], max: v[3], pene_ratio: v[4],
        spatter_rate: v[5], pore_count: v[6], humping_index: v[7],
    }
}

/// Peak bin index of a depth profile (mirrors core::SdOct::peak_depth); used
/// by the features module binary so it does not depend on core's instance.
pub fn peak_of(profile: &[f32]) -> Option<f32> {
    let gmin = 40usize.min(profile.len().saturating_sub(1));
    let gmax = 1000usize.min(profile.len());
    if gmax <= gmin + 1 { return None; }
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
