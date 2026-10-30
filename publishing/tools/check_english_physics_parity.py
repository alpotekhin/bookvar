"""Check the published English physics examples, not a hardware benchmark."""
import contextlib
from fractions import Fraction
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "en/00 Textbook/14 Inference and Optimization/55a The physics of LLM inference — prefill, decode, and roofline.md"
RU = ROOT / "00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md"
english, canonical = EN.read_text(), RU.read_text()
pattern = r"<!-- example: ([^>]+) -->\n```python\n(.*?)\n```"
blocks = dict(re.findall(pattern, english, re.S))
assert blocks == dict(re.findall(pattern, canonical, re.S))
assert len(blocks) == 5
namespaces, outputs = {}, {}
for name, code in blocks.items():
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(code, f"{EN.name}:{name}", "exec"), ns)
    namespaces[name], outputs[name] = ns, out.getvalue().splitlines()

assert outputs["inference-mbu-batch"] == ["1 71.43 50.00%", "4 200.00 35.00%"]
assert outputs["inference-mbu-moe"] == ["(0, 0) 3 GiB", "(0, 1) 4 GiB"]
assert outputs["inference-mbu-ranks"] == ["12 ms 58.33%"]
for batch, seconds, expected in [(1, Fraction(14, 1000), Fraction(1, 2)),
                                  (4, Fraction(20, 1000), Fraction(7, 20))]:
    rate = batch / seconds
    assert rate * 14 * 10**9 / (batch * 2 * 10**12) == expected
ns = namespaces["inference-mbu-moe"]
for a in range(4):
    for b in range(4):
        traffic = ns["shared_bytes"] + sum(ns["expert_bytes"][i] for i in {a, b})
        assert traffic == (3 if a == b else 4) * 2**30
rank = namespaces["inference-mbu-ranks"]
for amount, peak, expected in zip(rank["rank_bytes"], rank["rank_peaks"],
                                  [Fraction(1), Fraction(1, 6)]):
    assert Fraction(amount, 1) / (Fraction(12, 1000) * peak) == expected

figures = lambda text: set(re.findall(r"!\[\[([^\]|]+)", text))
assert figures(canonical) <= figures(english)
assert len(figures(english)) == 8
for figure in figures(english):
    assert (ROOT / figure.removeprefix("02 Areas/ML & DL/")).is_file(), figure
assert not re.search(r"^\$$", re.sub(r"```[\s\S]*?```", "", english), re.M)
assert len(re.findall(r"^\$\$$", english, re.M)) % 2 == 0
excerpt = re.compile(r"<!-- BEGIN SOURCE: JAX inference Appendix A -->(.*?)<!-- END SOURCE: JAX inference Appendix A -->", re.S)
assert excerpt.search(english)[1] == excerpt.search(canonical)[1]

# Re-run the independent read/write ledger, attention counts, and latency
# identities against this English file, not against the canonical examples.
spec = importlib.util.spec_from_file_location("physics", Path(__file__).with_name("check_inference_physics_examples.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.CHAPTER = EN
module.main()
print("PASS: 5 identical executable examples, 16 MoE routes, uneven ranks, all canonical figures.")
print("No GPU kernel, measured traffic, throughput, or serving benchmark was run.")
print("en_sha256", sha256(EN.read_bytes()).hexdigest())
print("canonical_sha256", sha256(RU.read_bytes()).hexdigest())
