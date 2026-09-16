use core::{SdOct, SPEC_BINS, PAD_LEN};
use sim::{KeyholeModel, SimConfig};

fn main() {
    let cfg: SimConfig = serde_json::from_str(
        &std::fs::read_to_string("config/sim.json").expect("config/sim.json"),
    )
    .expect("valid sim config");
    let mut model = KeyholeModel::new(cfg);
    let bg: Vec<f32> = vec![50.0; SPEC_BINS];
    let mut oct = SdOct::new(SPEC_BINS, PAD_LEN, bg);
    let n = 8000usize; // 8000 A-scans @80 kHz = 100 ms weld
    let mut rows = String::with_capacity(n * 24);
    rows.push_str("frame,depth_bins\n");
    for i in 0..n {
        let spec = model.advance();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        let d = oct.peak_depth(&profile, 40, 1000).unwrap_or(0.0);
        rows.push_str(&format!("{i},{d:.3}\n"));
    }
    std::fs::create_dir_all("data").unwrap();
    std::fs::write("data/depth.csv", rows).unwrap();
    eprintln!("wrote data/depth.csv ({n} A-scans)");
}
