import os
import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _collect(proc, seconds: float):
    lines = []
    deadline = time.time() + seconds
    while time.time() < deadline and proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            lines.append(line)
    return lines


def test_soak_no_crash_no_frame_loss(live_ports):
    ports, proc = live_ports
    lines = _collect(proc, 6.0)
    joined = "".join(lines)
    assert proc.poll() is None, "weldscope exited during soak"
    n_verdicts = len(re.findall(r"verdict:", joined))
    assert n_verdicts > 5, f"expected verdicts during soak, got {n_verdicts}"
    for bad in ("panic", "test FAILED", "unwrap failed"):
        assert bad not in joined, f"soak saw {bad!r}"


def test_restartable_after_kill(live_ports):
    """killing the modules must leave the orchestrator re-spawnable."""
    ports, proc = live_ports
    subprocess.run(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "kill"],
        cwd=ROOT, capture_output=True, timeout=30,
    )
    time.sleep(2)
    env = dict(os.environ)
    env["WELDSCOPE_BIN_DIR"] = str(ROOT / "target" / "release")
    proc2 = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "run"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    assert proc2.poll() is None, "restart failed"
    proc2.terminate()
    try:
        proc2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc2.kill()
    subprocess.run(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "kill"],
        cwd=ROOT, capture_output=True, timeout=30,
    )
