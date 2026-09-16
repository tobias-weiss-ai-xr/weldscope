use core::{SdOct, SPEC_BINS, PAD_LEN};
use features::FeatureExtractor;
use sim::{Defect, KeyholeModel, SimConfig};
use std::io::Write;

const WIN: usize = 80;          // feature window (1 ms @ 80 kHz)
const RATE: f64 = 80_000.0;
const N_FRAMES: usize = 8000;   // 100 ms weld

fn label_at(defects: &[(f64, f64, Defect)], t: f64) -> &'static str {
    for &(t0, t1, d) in defects {
        if t >= t0 && t <= t1 {
            return match d {
                Defect::None => "ok",
                Defect::Spatter => "spatter",
                Defect::Pore => "pore",
                Defect::Incomplete => "incomplete",
                Defect::Humping => "humping",
            };
        }
    }
    "ok"
}

fn main() {
    let cfg: SimConfig =
        serde_json::from_str(&std::fs::read_to_string("config/sim.json").unwrap()).unwrap();
    let weights = std::fs::read_to_string("ai/weights.json").unwrap_or_else(|_| {
        let zero8 = "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]".to_string();
        format!(r#"{{"classes":["ok","spatter","pore","incomplete","humping"],"weights":[{zero8},{zero8},{zero8},{zero8},{zero8}],"bias":[0.0,0.0,0.0,0.0,0.0]}}"#)
    });
    let clf = ai::Classifier::load(&weights).unwrap();

    let mut model = KeyholeModel::new(cfg.clone());
    let mut oct = SdOct::new(SPEC_BINS, PAD_LEN, vec![50.0; SPEC_BINS]);
    let mut fe = FeatureExtractor::new(WIN, 250.0, 15.0);

    let dt = 1.0 / RATE;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut report = String::new();
    for i in 0..N_FRAMES {
        let spec = model.advance();
        let mut profile = vec![0.0; oct.depth_bins];
        oct.process(&spec, &mut profile);
        let d = oct.peak_depth(&profile, 40, 1000);
        if let Some(f) = fe.push(d) {
            let t = (i as f64 - WIN as f64 / 2.0) * dt;
            let expected = label_at(&cfg.defects, t);
            let (pred, conf) = clf.predict(&features::to_vec(&f));
            report.push_str(&format!("[{i:>4}] t={t:.4}s pred={pred:<12} conf={conf:.2} expected={expected}\n"));
            if pred == expected { correct += 1; }
            total += 1;
        }
    }
    report.push_str(&format!("accuracy={correct}/{total}\n"));
    // write report to data/offline_report.txt AND stdout (harness reads file)
    std::fs::create_dir_all("data").unwrap();
    std::fs::write("data/offline_report.txt", &report).unwrap();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(report.as_bytes()).unwrap();
}
