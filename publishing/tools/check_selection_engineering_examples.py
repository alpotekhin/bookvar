"""Execute the nine reader programs, then check independent decision/accounting cases.

Standard-library CPU arithmetic only. No training, downloads, timing benchmark,
filesystem writes, or measurements of cloud/hardware performance.
"""
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
from heapq import nlargest
from io import StringIO
from itertools import product
import json
from math import ceil, isclose
from pathlib import Path
import random
import re
import struct
import zlib

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества.md"
text = CHAPTER.read_text()
programs = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
assert len(programs) == 9
scopes = []
output = StringIO()
for index, program in enumerate(programs):
    scope = {}
    with redirect_stdout(output):
        exec(compile(program, f"{CHAPTER}:program-{index+1}", "exec"), scope)
    scopes.append(scope)

# Exhaustive small candidate sets: a cheap but ineligible run must never win.
# Fractions make the reference comparisons exact, including quality boundaries.
choose_candidate = scopes[0]["choose_candidate"]
choice_cases = 0
for qualities in product([Fraction(79, 100), Fraction(80, 100), Fraction(85, 100)], repeat=2):
    for readiness in product([False, True], repeat=2):
        for costs in [(0, 10), (10, 0), (10, 10)]:
            candidates = [dict(name=str(i), prerequisites_met=readiness[i],
                               overall=Fraction(9, 10), rare=qualities[i],
                               preparation=costs[i], training=20-i)
                          for i in range(2)]
            admitted = [r for r in candidates if r["prerequisites_met"]
                        and r["overall"] >= Fraction(9, 10)
                        and r["rare"] >= Fraction(4, 5)]
            expected = sorted(admitted, key=lambda r: r["preparation"] + r["training"])
            actual = choose_candidate(candidates, Fraction(9, 10), Fraction(4, 5))
            assert actual == (expected[0] if expected else None)
            choice_cases += 1
assert choose_candidate([], .9, .8) is None
assert choose_candidate([dict(name="low-overall", prerequisites_met=True,
                             overall=.89, rare=1., preparation=0., training=0.)],
                        .9, .8) is None

# Independent serial refresh accounting, including partial final intervals.
refresh_cases = 0
for epochs in range(1, 31):
    for interval in range(1, 16):
        explicit_events = list(range(0, epochs, interval))
        assert len(explicit_events) == ceil(epochs / interval)
        cost = sum(Fraction(7, 10) for _ in explicit_events)
        cost += sum(Fraction(2, 5) for _ in range(epochs))
        expected = len(explicit_events) * Fraction(7, 10) + epochs * Fraction(2, 5)
        assert cost == expected
        refresh_cases += 1

# Profitability is strict; test exact equality and non-integral thresholds.
break_even_cases = 0
for investment in [0, 1, 20, 55_000]:
    for saving in [1, 3, 10_000]:
        first = investment // saving + 1
        assert first * saving > investment
        assert (first - 1) * saving <= investment
        break_even_cases += 1

# Prove the local-top-k implementation against a full sort on random shardings.
# Empty shards, unequal lengths, equal scores, and k larger than a shard occur.
rng = random.Random(29)
key = scopes[4]["key"]
topk_cases = 0
for count in [1, 2, 9, 31]:
    for worker_count in [1, 2, 8]:
        for k in [1, 3, 7]:
            for seed in range(10):
                shards = [[] for _ in range(worker_count)]
                for identifier in range(count):
                    shards[rng.randrange(worker_count)].append(
                        (identifier, rng.randrange(5)))
                records = [r for shard in shards for r in shard]
                candidates = [r for shard in shards
                              for r in nlargest(k, shard, key=key)]
                actual = nlargest(k, candidates, key=key)
                expected = sorted(records, key=key, reverse=True)[:k]
                assert actual == expected
                assert len({r[0] for r in candidates}) == len(candidates)
                topk_cases += 1

# A finite two-stage event schedule independently checks the steady-state model.
echo_cases = 0
for upstream, downstream, echo, groups in product(
        [Fraction(1, 300), Fraction(1, 1000)],
        [Fraction(1, 800), Fraction(1, 200)], [1, 2, 3, 4], [1, 3, 10]):
    downstream_done = Fraction(0)
    for item in range(1, groups + 1):
        upstream_ready = item * upstream
        downstream_done = max(upstream_ready, downstream_done) + echo * downstream
    group_time = max(upstream, echo * downstream)
    closed = upstream + echo * downstream + (groups - 1) * group_time
    assert downstream_done == closed
    assert isclose(float(echo / group_time),
                   min(float(echo / upstream), float(1 / downstream)))
    echo_cases += 1

# Validate actual chapter metric functions against exact rational area arithmetic.
metric_cases = 0
area = scopes[7]["area"]
first_budget = scopes[7]["first_observed_budget"]
for increments in product([1, 3], repeat=3):
    x = [0]
    for increment in increments:
        x.append(x[-1] + increment)
    for numerators in [(0, 2, 5, 7), (5, 5, 5, 5), (9, 7, 6, 3)]:
        y = [Fraction(n, 10) for n in numerators]
        exact = sum((x[i+1]-x[i])*(y[i+1]+y[i])/2 for i in range(3))
        assert area(x, y) == exact
        assert isclose(area(x, [.5]*4)/x[-1], .5)
        expected = next((n for n, value in zip(x, y) if value >= Fraction(7, 10)), None)
        assert first_budget(x, y, Fraction(7, 10)) == expected
        metric_cases += 1

# Source payloads and file integrity. Presence alone is not visual acceptance.
assets = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/selection-engineering-2026-10-28/assets.json").read_text())
assert len(assets) == 4
extra = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/harvard-data-selection-2026-10-28/assets.json").read_text())
assets += [row for row in extra if CHAPTER.relative_to(ROOT).as_posix() in row["used_in"]]
assert len(assets) == 5
for row in assets:
    data = (ROOT / row["asset"]).read_bytes()
    original = (ROOT / row["source_asset"]).read_bytes()
    assert data == original
    assert sha256(data).hexdigest() == row["sha256"] == row["source_asset_sha256"]
    html = (ROOT / row["source_page_snapshot"]).read_bytes()
    assert sha256(html).hexdigest() == row["source_page_sha256"]
    assert 'https://creativecommons.org/licenses/by-nc-nd/4.0/' in html.decode()
    assert row["source_image_url"].rsplit("/", 1)[-1] in html.decode()
    if row["asset"].endswith(".png"):
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert list(struct.unpack(">II", data[16:24])) == row["dimensions_px"]
        position, payload = 8, bytearray()
        while position < len(data):
            length = struct.unpack(">I", data[position:position+4])[0]
            chunk = data[position+4:position+8]
            content = data[position+8:position+8+length]
            crc = struct.unpack(">I", data[position+8+length:position+12+length])[0]
            assert zlib.crc32(chunk + content) & 0xffffffff == crc
            if chunk == b"IDAT":
                payload.extend(content)
            position += length + 12
            if chunk == b"IEND":
                break
        assert position == len(data)
        assert zlib.decompress(payload)
    else:
        assert b"<svg" in data and b"</svg>" in data
        assert not re.search(rb'(?:href|src)="https?://', data)

print(output.getvalue())
print(json.dumps({"chapter_sha256": sha256(CHAPTER.read_bytes()).hexdigest(),
                  "actual_programs": len(programs),
                  "quality_and_prerequisite_choices": choice_cases,
                  "refresh_cases": refresh_cases,
                  "break_even_cases": break_even_cases,
                  "topk_cases": topk_cases,
                  "finite_echo_cases": echo_cases,
                  "metric_cases": metric_cases,
                  "unchanged_original_figures": len(assets)}, indent=2))
