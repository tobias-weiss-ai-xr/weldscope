use ai::Classifier;
use io::codec::{features_dec, verdict_enc};
use io::{Frame, FrameReader, FrameType};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_port: u16 = args.iter().skip(1).nth(0).and_then(|a| a.parse().ok()).unwrap_or(40103);
    let out_port: u16 = args.iter().skip(1).nth(1).and_then(|a| a.parse().ok()).unwrap_or(40104);
    let model_path = args.iter().skip(1).nth(2).cloned().unwrap_or_else(|| "ai/weights.json".into());

    let ws = std::fs::read_to_string(&model_path).unwrap();
    let clf = Classifier::load(&ws).unwrap();

    let listener = TcpListener::bind(("0.0.0.0", in_port)).unwrap();
    println!("ai: bound {in_port}, waiting for features...");
    let (up, _) = listener.accept().unwrap();
    let tmp = TcpStream::connect(format!("127.0.0.1:{out_port}")).ok(); // no sink yet: stdout is the demo output
    let mut sink = tmp;
    println!("ai: connected at {in_port} (sink {})", if sink.is_some() { "on" } else { "off" });

    let mut reader = FrameReader::new(up);
    while let Ok(f) = reader.read() {
        if f.ty != FrameType::Features {
            continue;
        }
        let feat = features_dec(&f.payload);
        let (cls, conf) = clf.predict(&feat);
        let lat_us = (now_ns() - f.ts_ns) / 1000;
        println!("verdict: {cls:<12} conf={conf:.2} seq={} lat_us={lat_us}", f.seq);
        if let Some(s) = &mut sink {
            let out = Frame::new(FrameType::Verdict, f.seq, f.ts_ns, verdict_enc(clf.class_index(&cls), conf));
            s.write_all(&out.encode()).unwrap();
        }
    }
}
