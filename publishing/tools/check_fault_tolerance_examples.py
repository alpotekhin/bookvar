"""Execute 44h's CPU example and a bounded, isolated torchrun restart test."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md"


def chapter_blocks():
    blocks = re.findall(
        r"^(?P<fence>\x60{3}|~{3})python\n(?P<code>.*?)^(?P=fence)\s*$",
        CHAPTER.read_text(), re.M | re.S,
    )
    assert len(blocks) == 2, "all published Python examples must be exercised"
    return [code for _, code in blocks]


def checkpoint_worker(path):
    namespace = {"__name__": "bookvar_recovery_example"}
    exec(compile(chapter_blocks()[1], str(CHAPTER), "exec"), namespace)
    namespace["run_recovery"](path)


def restart_test(workspace):
    command = [sys.executable, "-m", "torch.distributed.run",
               "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--rdzv-id=bookvar-test",
               "--local-addr=127.0.0.1",
               "--nproc-per-node=2", "--max-restarts=1", "--monitor-interval=0.1",
               str(Path(__file__).resolve()), "--checkpoint-worker",
               str(Path(workspace) / "checkpoint-step-2")]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "LC_ALL": "C",
           "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
           "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"}
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, start_new_session=True)
    try:
        out, err = child.communicate(timeout=55)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        out, err = child.communicate()
        raise AssertionError("isolated restart test exceeded 55 seconds: " + err[-2500:])
    assert child.returncode == 0, out[-3000:] + err[-5000:]
    workers = [tuple(map(int, v)) for v in re.findall(r"WORKER (\d+) (\d+) (\d+)", out)]
    loaded = [tuple(map(int, v)) for v in re.findall(r"LOADED (\d+) (\d+) (\d+)", out)]
    verified = [tuple(map(int, v)) for v in re.findall(r"VERIFIED (\d+) (\d+) (\d+)", out)]
    assert {(a, b) for a, b, _ in workers} == {(0, 0), (0, 1), (1, 0), (1, 1)}, out
    assert len({pid for _, _, pid in workers}) == 4, workers
    assert set(loaded) == {(1, 0, 2), (1, 1, 2)}, out
    assert set(verified) == {(1, 0, 3), (1, 1, 3)}, out
    assert "INJECTED_EXIT 23" in out, out
    print(json.dumps({"workers": workers, "loaded": loaded, "verified": verified}))
    print("PASS published torchrun/DDP/DCP example: exit 23, full group restart, state resume")


def main():
    text = CHAPTER.read_text()
    blocks = chapter_blocks()
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(blocks[0], str(CHAPTER), "exec"), namespace)
    assert output.getvalue().splitlines() == ["complete: True", "no optimizer: False",
                                             "no RNG: False", "no cursor: False"], output.getvalue()
    ref, resume = namespace["reference"], namespace["resume"]
    assert namespace["same_step"](ref, resume())
    no_opt = resume(optimizer=False)
    assert torch.equal(ref[1], no_opt[1]) and torch.equal(ref[2], no_opt[2])
    assert not torch.equal(ref[3], no_opt[3])
    no_rng = resume(rng=False)
    assert torch.equal(ref[0], no_rng[0]) and not torch.equal(ref[2], no_rng[2])
    no_cursor = resume(data_cursor=False)
    assert not torch.equal(ref[0], no_cursor[0])
    assert all(torch.isfinite(v).all() for v in ref[:4])
    print("PASS actual chapter code: exact next CPU step; three omission counterexamples")

    m = 5 * 365 * 24 / 1024
    assert round(m, 1) == 42.8
    assert round(168 / m, 2) == 3.93
    assert 0.019 < math.exp(-168 / m) < 0.021
    c, m = 120, 43200
    young = math.sqrt(2 * c * m)
    daly = young * (1 + math.sqrt(c / (2 * m)) / 3 + c / (2 * m) / 9) - c
    assert round(young / 60, 1) == 53.7
    assert round((young - c) / 60, 1) == 51.7
    assert round(daly / 60, 1) == 52.3 and round((daly + c) / 60, 1) == 54.3
    costs = lambda t: c / t + t / (2 * m)
    assert round(100 * costs(600), 2) == 20.69
    assert round(100 * costs(young), 2) == 7.45
    assert round(100 * costs(10800), 2) == 13.61
    assert costs(young) < costs(young * 0.99) and costs(young) < costs(young * 1.01)
    assert 8 * 2 * 16 == 4 * 2 * 32 == 256
    assert math.isclose(20 * 1.18 - 22, 1.6)
    assert math.isclose(math.sqrt(2 * 2.5 * 120), 24.49489742783178)
    assert sum([3, 2, 1, 2, 3, 2, 3, 3]) == 19
    assert max([3, 2, 1, 2, 3, 2, 3, 3]) < 8
    assert math.isclose(1000 * 1.2 - (180 + 1000), 20)
    assert 100 * 1.2 < 180 + 100
    assert math.isclose(180 / (1.2 - 1), 900)
    assert math.isclose(20 / 1.6 + 10 / 60, 12 + 40 / 60)
    assert math.isclose(10 / 1.6 + 10, 16.25)
    print("PASS arithmetic: failure rate, Young/Daly conventions, overheads, batch, placement")
    with tempfile.TemporaryDirectory(prefix="bookvar-restart-") as ready_dir:
        restart_test(ready_dir)
    print("markdown_sha256=" + hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
    print(f"PASS torch={torch.__version__}; real CPU/Gloo/DDP/DCP restart; no GPU, scheduler, multi-host or storage fault claim")


if __name__ == "__main__":
    checkpoint_worker(sys.argv[-1]) if "--checkpoint-worker" in sys.argv else main()
