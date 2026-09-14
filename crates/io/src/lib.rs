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
        let mut f = Frame::decode(&hdr).unwrap(); // header validated above
        f.payload = payload;
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
