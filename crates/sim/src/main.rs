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
    println!("acq: connected to {addr}");
    let mut seq: u64 = 0;
    loop {
        let spec = model.advance();
        let f = Frame::new(FrameType::Spectrum, seq, now_ns(), spectrum_enc(&spec));
        stream.write_all(&f.encode()).unwrap();
        seq += 1;
        std::thread::sleep(std::time::Duration::from_micros(12)); // ~80 kHz
    }
}
