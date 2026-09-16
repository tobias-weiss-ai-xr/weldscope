pub const MAGIC: [u8; 2] = [b'W', b'S'];
pub const VERSION: u8 = 1;
pub const HDR_LEN: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Spectrum = 0,
    DepthTrace = 1,
    Features = 2,
    Verdict = 3,
    Hello = 4,
    Bye = 5,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub ty: FrameType,
    pub seq: u64,
    pub ts_ns: u64,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(ty: FrameType, seq: u64, ts_ns: u64, payload: Vec<u8>) -> Self {
        Frame { ty, seq, ts_ns, payload }
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HDR_LEN + self.payload.len());
        b.extend_from_slice(&MAGIC);
        b.push(VERSION);
        b.push(self.ty as u8);
        b.extend_from_slice(&self.seq.to_le_bytes());
        b.extend_from_slice(&self.ts_ns.to_le_bytes());
        b.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.payload);
        b
    }
    pub fn decode(buf: &[u8]) -> Option<Frame> {
        if buf.len() < HDR_LEN || buf[0] != MAGIC[0] || buf[1] != MAGIC[1] {
            return None;
        }
        let len = u32::from_le_bytes(buf[20..24].try_into().ok()?) as usize;
        if buf.len() < HDR_LEN + len {
            return None;
        }
        let ty = match buf[3] {
            0 => FrameType::Spectrum,
            1 => FrameType::DepthTrace,
            2 => FrameType::Features,
            3 => FrameType::Verdict,
            4 => FrameType::Hello,
            _ => FrameType::Bye,
        };
        Some(Frame {
            ty,
            seq: u64::from_le_bytes(buf[4..12].try_into().ok()?),
            ts_ns: u64::from_le_bytes(buf[12..20].try_into().ok()?),
            payload: buf[HDR_LEN..HDR_LEN + len].to_vec(),
        })
    }
}

/// Blocking frame reader over a TcpStream (header + payload read_exact).
pub struct FrameReader {
    s: std::net::TcpStream,
}

impl FrameReader {
    pub fn new(s: std::net::TcpStream) -> Self {
        FrameReader { s }
    }
    pub fn read(&mut self) -> std::io::Result<Frame> {
        use std::io::Read;
        let mut hdr = [0u8; HDR_LEN];
        self.s.read_exact(&mut hdr)?;
        if hdr[0] != MAGIC[0] || hdr[1] != MAGIC[1] {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic",
            ));
        }
        let len = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; len];
        self.s.read_exact(&mut payload)?;
        // Build directly from the header: Frame::decode needs the full buffer
        // (header + payload) and returns None for a header-only slice.
        let ty = match hdr[3] {
            0 => FrameType::Spectrum,
            1 => FrameType::DepthTrace,
            2 => FrameType::Features,
            3 => FrameType::Verdict,
            4 => FrameType::Hello,
            _ => FrameType::Bye,
        };
        let f = Frame {
            ty,
            seq: u64::from_le_bytes(hdr[4..12].try_into().unwrap()),
            ts_ns: u64::from_le_bytes(hdr[12..20].try_into().unwrap()),
            payload,
        };
        Ok(f)
    }
}

/// Blocking frame writer over a TcpStream.
pub struct FrameWriter {
    s: std::net::TcpStream,
}

impl FrameWriter {
    pub fn new(s: std::net::TcpStream) -> Self {
        FrameWriter { s }
    }
    pub fn write(&mut self, f: &Frame) -> std::io::Result<()> {
        use std::io::Write;
        self.s.write_all(&f.encode())
    }
}

pub mod codec {
    // f32 samples -> u16 quantized (display window viewer)
    pub fn spectrum_enc(s: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(2 + s.len() * 2);
        b.extend_from_slice(&(s.len() as u16).to_le_bytes());
        for &v in s {
            let q = v.clamp(0.0, 4095.0) as u16;
            b.extend_from_slice(&q.to_le_bytes());
        }
        b
    }
    pub fn spectrum_dec(b: &[u8]) -> Vec<f32> {
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        (0..n)
            .map(|i| u16::from_le_bytes([b[2 + i * 2], b[3 + i * 2]]) as f32)
            .collect()
    }
    pub fn depth_enc(d: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(2 + d.len() * 4);
        b.extend_from_slice(&(d.len() as u16).to_le_bytes());
        for v in d {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }
    pub fn depth_dec(b: &[u8]) -> Vec<f32> {
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        (0..n)
            .map(|i| {
                f32::from_le_bytes([b[2 + i * 4], b[3 + i * 4], b[4 + i * 4], b[5 + i * 4]])
            })
            .collect()
    }
    pub fn features_enc(f: &[f32; 8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(32);
        for v in f {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }
    pub fn features_dec(b: &[u8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for (i, o) in out.iter_mut().enumerate() {
            *o = f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        }
        out
    }
    pub fn verdict_enc(class: u8, conf: f32) -> Vec<u8> {
        let mut b = vec![class];
        b.extend_from_slice(&conf.to_le_bytes());
        b
    }
    pub fn verdict_dec(b: &[u8]) -> (u8, f32) {
        (b[0], f32::from_le_bytes([b[1], b[2], b[3], b[4]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG (numerical-recipes constants) — fuzz inputs
    /// without a rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 33) as u8
        }
    }

    fn payload(l: &mut Lcg, n: usize) -> Vec<u8> {
        (0..n).map(|_| l.byte()).collect()
    }

    /// 50 LCG-derived frames across every FrameType: decode recovers every
    /// field and the envelope roundtrips to byte-identical output.
    #[test]
    fn frame_roundtrip_is_byte_exact() {
        let mut l = Lcg(42);
        let types = [
            FrameType::Spectrum,
            FrameType::DepthTrace,
            FrameType::Features,
            FrameType::Verdict,
            FrameType::Hello,
            FrameType::Bye,
        ];
        for i in 0..50u64 {
            let seq = l.next();
            let ts = l.next();
            let pl = payload(&mut l, (i as usize * 7) % 257);
            let f = Frame::new(types[(i % 6) as usize], seq, ts, pl);
            let bytes = f.encode();
            let g = Frame::decode(&bytes).expect("valid frame must decode");
            assert_eq!(g.ty, f.ty);
            assert_eq!(g.seq, f.seq);
            assert_eq!(g.ts_ns, f.ts_ns);
            assert_eq!(g.payload, f.payload);
            assert_eq!(g.encode(), bytes, "re-encode must be byte-identical");
        }
    }

    /// Any prefix shorter than the full envelope decodes to None — never a
    /// panic (decode is Option-based; prose's "Err" maps to None here).
    #[test]
    fn truncated_input_is_never_a_panic() {
        let mut l = Lcg(7);
        for _ in 0..50 {
            let seq = l.next();
            let ts = l.next();
            let f = Frame::new(FrameType::DepthTrace, seq, ts, payload(&mut l, 64));
            let bytes = f.encode();
            for k in 0..bytes.len() {
                assert!(
                    Frame::decode(&bytes[..k]).is_none(),
                    "truncation to {k} bytes must be rejected"
                );
            }
        }
    }

    /// Corrupt 4 distinct bytes of 50 LCG packets: decode must reject (None)
    /// or yield a different packet — never an identical one, never a panic.
    /// No AcqPacket type exists in this crate; Frame is the wire envelope, so
    /// this is the closest equivalent invariant.
    #[test]
    fn corrupted_packet_is_rejected_or_changed() {
        let mut l = Lcg(99);
        for _ in 0..50 {
            let seq = l.next();
            let ts = l.next();
            let f = Frame::new(FrameType::Spectrum, seq, ts, payload(&mut l, 128));
            let bytes = f.encode();
            let mut cor = bytes.clone();
            // Skip offset 2: the version byte is unchecked by decode, so a
            // flip there is legitimately a valid, identical packet.
            let mut offs: Vec<usize> = Vec::with_capacity(4);
            while offs.len() < 4 {
                let p = (l.next() as usize) % bytes.len();
                if p != 2 && !offs.contains(&p) {
                    offs.push(p);
                    cor[p] ^= l.byte() | 1; // nonzero flip
                }
            }
            if let Some(g) = Frame::decode(&cor) {
                let same = g.ty == f.ty
                    && g.seq == f.seq
                    && g.ts_ns == f.ts_ns
                    && g.payload == f.payload;
                assert!(!same, "4-byte corruption decoded to an identical packet");
            }
        }
    }

    /// Codec fuzz: 50 LCG payloads per codec roundtrip byte-exactly.
    /// depth/features/verdict carry raw f32 bit patterns (incl. NaN/inf), so
    /// equality is checked on re-encoded bytes, not f32 comparison.
    #[test]
    fn codec_roundtrips_are_byte_exact() {
        let mut l = Lcg(1234);
        for _ in 0..50 {
            // spectrum: quantized to u16, lossless only within [0, 4095]
            let spec: Vec<f32> = (0..64).map(|_| (l.next() % 4096) as f32).collect();
            let b = codec::spectrum_enc(&spec);
            assert_eq!(codec::spectrum_dec(&b), spec);
            assert_eq!(codec::spectrum_enc(&codec::spectrum_dec(&b)), b);

            let d: Vec<f32> = (0..16).map(|_| f32::from_bits((l.next() >> 32) as u32)).collect();
            let db = codec::depth_enc(&d);
            assert_eq!(codec::depth_enc(&codec::depth_dec(&db)), db);

            let fv: [f32; 8] = [0.0; 8].map(|_| f32::from_bits((l.next() >> 32) as u32));
            let fb = codec::features_enc(&fv);
            assert_eq!(codec::features_enc(&codec::features_dec(&fb)), fb);

            let cls = l.next() as u8;
            let conf = f32::from_bits((l.next() >> 32) as u32);
            let vb = codec::verdict_enc(cls, conf);
            let (c2, f2) = codec::verdict_dec(&vb);
            assert_eq!(codec::verdict_enc(c2, f2), vb);
        }
    }
}
