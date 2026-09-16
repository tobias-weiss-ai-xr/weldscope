# WeldScope Wire Format (WS01)

Binary, length-prefixed frames over TCP. Little-endian.

## Header (24 bytes)
| offset | size | field |
|--------|------|-------|
| 0      | 2    | magic `WS` (0x57 0x53) |
| 2      | 1    | version (1) |
| 3      | 1    | frame_type |
| 4      | 8    | sequence number (u64) |
| 12     | 8    | origin timestamp ns since UNIX_EPOCH (u64) |
| 20     | 4    | payload length (u32) |
| 24     | n    | payload |

## Frame types
| value | type        | payload |
|-------|-------------|---------|
| 0     | SPECTRUM    | u16 count + count × u16 samples (0..4095) |
| 1     | DEPTH_TRACE | u16 count + count × f32 depth bins |
| 2     | FEATURES    | 8 × f32 |
| 3     | VERDICT     | u8 class + f32 confidence |
| 4     | HELLO       | u16 strlen + utf8 string |
| 5     | BYE         | (empty) |

## Sequencing & latency
- seq follows the originating spectrum through the whole pipeline
  (depth trace, features, verdict reuse the same seq).
- ts_ns is set once by acq (spectrum production time) and preserved on
  forwarding. End-to-end latency = now − ts_ns at the sink.

## Latency metric
End-to-end latency = (local now − frame.ts_ns) at the VERDICT consumer,
aggregated by the ai module as mean/p50/p99 over rolling 2-second windows.
ts_ns is the acq-side spectrum production timestamp and is preserved by every
module on forwarding.

## Latency metric
End-to-end latency = (local now − frame.ts_ns) at the VERDICT consumer,
aggregated by the ai module as mean/p50/p99 over rolling 2-second windows.
ts_ns is the acq-side spectrum production timestamp and is preserved by every
module on forwarding.
