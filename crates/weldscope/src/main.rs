use core::{SdOct, SPEC_BINS, PAD_LEN};
use features::FeatureExtractor;
use sim::{Defect, KeyholeModel, SimConfig};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command};

const WIN: usize = 80;          // feature window (1 ms @ 80 kHz)
const RATE: f64 = 80_000.0;
const N_FRAMES: usize = 8000;   // 100 ms weld

fn pids_dir() -> PathBuf {
    let d = PathBuf::from("data/pids");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_pid(tag: &str, pid: u32) {
    std::fs::write(pids_dir().join(format!("{tag}.pid")), pid.to_string()).unwrap();
}

fn read_pid(tag: &str) -> Option<u32> {
    std::fs::read_to_string(pids_dir().join(format!("{tag}.pid")))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn spawn(tag: &str, bin: &str, args: &[String]) -> Child {
    let child = Command::new(bin)
        .args(args)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {tag}: {e}"));
    write_pid(tag, child.id());
    child
}

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").arg(pid.to_string()).status();
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
}

fn bin_dir() -> String {
    std::env::var("WELDSCOPE_BIN_DIR").unwrap_or_else(|_| "target/debug".into())
}

fn cmd_run(cfg: &serde_json::Value) {
    let p = &cfg["ports"];
    let (a, c, f, s) = (
        p["acq_out"].as_u64().unwrap() as u16,
        p["core_out"].as_u64().unwrap() as u16,
        p["features_out"].as_u64().unwrap() as u16,
        p["ai_out"].as_u64().unwrap() as u16,
    );
    let dir = bin_dir();
    let mut kids = Vec::new();
    kids.push(spawn("core", &format!("{dir}/core"), &[a.to_string(), c.to_string()]));
    std::thread::sleep(std::time::Duration::from_millis(300));
    kids.push(spawn("features", &format!("{dir}/features"), &[c.to_string(), f.to_string()]));
    std::thread::sleep(std::time::Duration::from_millis(300));
    kids.push(spawn("ai", &format!("{dir}/ai"), &[f.to_string(), s.to_string(), "ai/weights.json".into()]));
    std::thread::sleep(std::time::Duration::from_millis(300));
    kids.push(spawn("acq", &format!("{dir}/acq"), &[format!("--out={a}")]));
    println!("weldscope: 4 modules running on ports {a}->{c}->{f}->{s}. Ctrl-C to stop.");
    for k in kids.iter_mut() {
        let _ = k.wait();
    }
}

fn cmd_kill() {
    for tag in ["acq", "core", "features", "ai"] {
        if let Some(pid) = read_pid(tag) {
            kill_pid(pid);
            println!("killed {tag} ({pid})");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("config/weld.json").unwrap()).unwrap();
    match args.get(1).map(|s| s.as_str()).unwrap_or("offline") {
        "offline" => offline(),
        "run" => cmd_run(&cfg),
        "kill" => cmd_kill(),
        other => println!("unknown subcommand: {other} (offline|run|kill)"),
    }
}

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

fn offline() {
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
