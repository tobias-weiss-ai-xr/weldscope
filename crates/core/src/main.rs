use core::{SdOct, SPEC_BINS, PAD_LEN};
use io::codec::{depth_enc, spectrum_dec};
use io::{Frame, FrameReader, FrameType};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_port: u16 = args.iter().skip(1).nth(0).and_then(|a| a.parse().ok()).unwrap_or(40101);
    let out_port: u16 = args.iter().skip(1).nth(1).and_then(|a| a.parse().ok()).unwrap_or(40102);

    let listener = TcpListener::bind(("0.0.0.0", in_port)).unwrap();
    println!("core: bound {in_port}, waiting for acq...");
    let (up, _) = listener.accept().unwrap();
    let mut sink = TcpStream::connect(format!("127.0.0.1:{out_port}")).unwrap();
    println!("core: connected to features at {out_port}");

    let mut reader = FrameReader::new(up);
    let mut oct = SdOct::new(SPEC_BINS, PAD_LEN, vec![50.0; SPEC_BINS]);
    let mut profile = vec![0.0f32; oct.depth_bins];
    let mut n: u64 = 0;
    let t0 = Instant::now();
    while let Ok(f) = reader.read() {
        if f.ty != FrameType::Spectrum {
            continue;
        }
        let spec = spectrum_dec(&f.payload);
        oct.process(&spec, &mut profile);
        let out = Frame::new(FrameType::DepthTrace, f.seq, f.ts_ns, depth_enc(&profile));
        sink.write_all(&out.encode()).unwrap();
        n += 1;
        if n % 100_000 == 0 {
            eprintln!(
                "core: {n} ascans in {:.1}s ({:.0} kHz)",
                t0.elapsed().as_secs_f64(),
                n as f64 / t0.elapsed().as_secs_f64() / 1e3
            );
        }
    }
}
