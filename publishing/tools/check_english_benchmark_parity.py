"""Execute the English benchmark chapter; no serving or power benchmark."""
from hashlib import sha256
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "en/00 Textbook/14 Inference and optimization/58b Benchmarking SLOs and inference operations.md"
RU = ROOT / "00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md"
english, canonical = EN.read_text(), RU.read_text()
blocks = lambda text: re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
assert blocks(english) == blocks(canonical)
assert len(blocks(english)) == 8
ns = {}
for index, code in enumerate(blocks(english)):
    exec(compile(code, f"{EN.name}:block-{index + 1}", "exec"), ns)
energy = re.compile(r"<!-- serving-operations-completion: energy start -->(.*?)<!-- serving-operations-completion: energy end -->", re.S)
assert energy.search(english)[1] == energy.search(canonical)[1]
figures = lambda text: set(re.findall(r"!\[\[([^\]|]+)", text))
assert figures(english) == figures(canonical)
assert len(figures(english)) == 6
for figure in figures(english):
    assert (ROOT / figure.removeprefix("02 Areas/ML & DL/")).is_file()

rng = np.random.default_rng(20261029)
for _ in range(200):
    sent = rng.uniform(0, 10)
    times = sent + np.cumsum(rng.uniform(1, 100, rng.integers(1, 30)))
    result = ns["request_metrics"](sent, times)
    assert np.isclose(result["ttft"] + sum(result["itl"]), result["e2e"])
    if len(times) == 1:
        assert result["tpot"] is None
    else:
        assert np.isclose(result["tpot"], (times[-1] - times[0]) / (len(times) - 1))
    prompt, output = int(rng.integers(0, 10000)), int(rng.integers(1, 500))
    assert ns["decode_key_positions"](prompt, output) == sum(range(prompt + 1, prompt + output))
    durations = rng.uniform(.1, 60, 5)
    watts = rng.uniform(0, 1000, 5)
    boundaries = np.r_[0, durations.cumsum()].tolist()
    count = int(rng.integers(1, 1000))
    actual = ns["interval_energy"](boundaries, watts.tolist(), count)
    expected_joules = float(np.dot(durations, watts))
    np.testing.assert_allclose(actual, [expected_joules, expected_joules / durations.sum(), expected_joules / count])

assert ns["nearest_rank"]([10.] * 1000 + [1000.] * 10, .99) == 10.
assert ns["summary"]["attainment_all"] == 2 / 6
assert ns["summary"]["goodput_rps"] == .2
assert ns["a_positions"] == 450672 and ns["b_positions"] == 4321392
assert not ns["two_window_page"](20, 0) and ns["two_window_page"](20, 20)
assert not re.search(r"^\$$", re.sub(r"```[\s\S]*?```", "", english), re.M)
print("PASS: all 8 actual Python blocks; 600 independent timestamp/context/energy cases; six original figures.")
print("Complete English energy section retained byte-for-byte. No measured GPU, serving, or power result.")
print("en_sha256", sha256(EN.read_bytes()).hexdigest())
print("canonical_sha256", sha256(RU.read_bytes()).hexdigest())
