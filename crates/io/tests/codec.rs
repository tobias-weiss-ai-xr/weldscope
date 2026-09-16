use io::codec::{depth_dec, depth_enc, features_dec, features_enc,
                spectrum_dec, spectrum_enc, verdict_dec, verdict_enc};

#[test]
fn spectrum_roundtrip() {
    let s: Vec<f32> = (0..64).map(|i| (i as f32) * 5.0).collect();
    let d = spectrum_dec(&spectrum_enc(&s));
    assert_eq!(s, d);
}

#[test]
fn depth_roundtrip() {
    let d: Vec<f32> = (0..1024).map(|i| 0.001 * i as f32).collect();
    assert_eq!(depth_dec(&depth_enc(&d)), d);
}

#[test]
fn features_roundtrip() {
    let f = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(features_dec(&features_enc(&f)), f);
}

#[test]
fn verdict_roundtrip() {
    assert_eq!(verdict_dec(&verdict_enc(3, 0.87)), (3, 0.87));
}

#[test]
fn frame_roundtrip() {
    let f = io::Frame::new(io::FrameType::Verdict, 42, 123456789, vec![1, 2, 3]);
    let g = io::Frame::decode(&f.encode()).unwrap();
    assert_eq!(g.ty, io::FrameType::Verdict);
    assert_eq!(g.seq, 42);
    assert_eq!(g.ts_ns, 123456789);
    assert_eq!(g.payload, vec![1, 2, 3]);
}

#[test]
fn stream_roundtrip() {
    // FrameReader/FrameWriter over a real TCP pair (this caught the header-
    // only decode bug: any non-empty payload made FrameReader::read panic).
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let t = std::thread::spawn(move || {
        let (s, _) = l.accept().unwrap();
        let mut r = io::FrameReader::new(s);
        r.read().unwrap()
    });
    let c = std::net::TcpStream::connect(addr).unwrap();
    let mut w = io::FrameWriter::new(c);
    let f = io::Frame::new(io::FrameType::Spectrum, 7, 99, vec![42; 4098]);
    w.write(&f).unwrap();
    let g = t.join().unwrap();
    assert_eq!(g.ty, io::FrameType::Spectrum);
    assert_eq!(g.seq, 7);
    assert_eq!(g.ts_ns, 99);
    assert_eq!(g.payload, vec![42; 4098]);
}
