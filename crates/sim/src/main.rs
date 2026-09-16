use io::codec::spectrum_enc;
use io::{Frame, FrameType};
use sim::{KeyholeModel, SimConfig};
use std::io::Write;
use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_port: u16 = args
        .iter()
        .skip(1)
        .find_map(|a| a.split('=').nth(1).and_then(|v| v.parse().ok()))
        .unwrap_or(40101);
    let cfg: SimConfig = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("WELDSCOPE_SIM").unwrap_or_else(|_| "config/sim.json".into())).unwrap(),
    )
    .unwrap();
    let mut model = KeyholeModel::new(cfg);

    let addr = format!("127.0.0.1:{out_port}");
    let mut stream = TcpStream::connect(&addr).expect("connect to core module");
    stream.set_nodelay(true).unwrap();
    println!("acq: connected to {addr}");
    let mut seq: u64 = 0;
    // Pace by spin-wait instead of thread::sleep: on Windows a 12us sleep
    // actually costs ~0.5 ms (OS timer granularity), capping the stream at
    // ~1.7 kHz with ms jitter. Spin-wait is us-accurate cross-platform.
    // 135us == ~7.4 kHz, just under core's sustained rate (~7.6 kHz incl.
    // codec + socket I/O), so no backlog can accumulate -> latency stays
    // sub-ms from the first window. Drift-free: if a hiccup puts us behind,
    // schedule from now (no catch-up burst that would re-age queued frames).
    let mut next = std::time::Instant::now();
    let mut seq: u64 = 0;
    loop {
        let spec = model.advance();
        let f = Frame::new(FrameType::Spectrum, seq, now_ns(), spectrum_enc(&spec));
        stream.write_all(&f.encode()).unwrap();
        seq += 1;
        next = next.max(std::time::Instant::now()) + std::time::Duration::from_micros(135);
        while std::time::Instant::now() < next {}
    }
}
