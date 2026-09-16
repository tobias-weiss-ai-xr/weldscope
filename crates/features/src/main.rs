use features::{peak_of, to_vec, FeatureExtractor};
use io::codec::{depth_dec, features_enc};
use io::{Frame, FrameReader, FrameType};
use std::io::Write;
use std::net::{TcpListener, TcpStream};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_port: u16 = args.iter().skip(1).nth(0).and_then(|a| a.parse().ok()).unwrap_or(40102);
    let out_port: u16 = args.iter().skip(1).nth(1).and_then(|a| a.parse().ok()).unwrap_or(40103);
    let listener = TcpListener::bind(("0.0.0.0", in_port)).unwrap();
    println!("features: bound {in_port}, waiting for core...");
    let (up, _) = listener.accept().unwrap();
    let mut sink = TcpStream::connect(format!("127.0.0.1:{out_port}")).unwrap();
    println!("features: connected to ai at {out_port}");

    let mut reader = FrameReader::new(up);
    let mut fe = FeatureExtractor::new(80, 250.0, 15.0);
    while let Ok(f) = reader.read() {
        if f.ty != FrameType::DepthTrace {
            continue;
        }
        let depth = depth_dec(&f.payload);
        let d = peak_of(&depth);
        if let Some(feat) = fe.push(d) {
            let out = Frame::new(
                FrameType::Features,
                f.seq,
                f.ts_ns,
                features_enc(&to_vec(&feat)),
            );
            sink.write_all(&out.encode()).unwrap();
        }
    }
}
