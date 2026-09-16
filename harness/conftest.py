import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def offline_report():
    """Build, run the offline chain, return its full stdout."""
    subprocess.run(["cargo", "build", "-p", "weldscope"], cwd=ROOT, check=True)
    r = subprocess.run(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "offline"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def live_ports():
    """Start the full multi-process system in release (latency bounds are
    only meaningful on optimized code); yield ports + proc; teardown kills."""
    subprocess.run(["cargo", "build", "--release", "--workspace"], cwd=ROOT, check=True)
    env = dict(os.environ)
    env["WELDSCOPE_BIN_DIR"] = str(ROOT / "target" / "release")
    proc = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "run"],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(3)
    ports = json.loads((ROOT / "config" / "weld.json").read_text())["ports"]
    yield ports, proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    # kill any lingering module processes
    subprocess.run(
        ["cargo", "run", "-q", "-p", "weldscope", "--", "kill"],
        cwd=ROOT, env=env, capture_output=True,
    )


def _os_env_with_bin_dir():
    env = dict(os.environ)
    env["WELDSCOPE_BIN_DIR"] = str(ROOT / "target" / "release")
    return env
