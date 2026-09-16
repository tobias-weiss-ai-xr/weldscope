import re
import time

import pytest


@pytest.mark.timeout(60)
def test_live_latency_below_bound(live_ports):
    """Steady-state latency must meet real-time bounds within 8 s of startup.

    Startup necessarily has a cold-start transient (module warmup + TCP
    slow-start at every hop), so we scan for the first *settled* latency
    window rather than asserting on the first report line.
    """
    ports, proc = live_ports
    lines_accum = []
    deadline = time.time() + 8
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            continue
        lines_accum.append(line)
        if "latency us:" in line:
            m = re.search(r"p50=(\d+) p99=(\d+)", line)
            assert m, line
            p50, p99 = int(m.group(1)), int(m.group(2))
            if p50 < 5_000 and p99 < 20_000:
                return
    pytest.fail(
        "no settled latency window (p50<5ms, p99<20ms) within 8 s; got:\n"
        + "".join(
            l for l in lines_accum if "latency us:" in l
        )[-10:]
    )
