"""Execute the actual introductory-system examples; CPU arithmetic, not timing."""
from collections import defaultdict, deque
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import math
import random
import re

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "00 Учебник/10 ML Systems/01 Модель как часть системы.md"
text = PATH.read_text()
blocks = re.findall(r"^~~~python\n(.*?)^~~~\s*$", text, re.M | re.S)
assert len(blocks) == 3
ns = {}
for i, code in enumerate(blocks, 1):
    out = StringIO()
    with redirect_stdout(out):
        exec(compile(code, f"{PATH}:block{i}", "exec"), ns)
    print(f"PASS actual block{i}: {out.getvalue().strip()}")
timeline, admit = ns["stage_timeline"], ns["admit_serial"]


def tick_oracle(arrivals, services):
    """Independent integer-time queues and running jobs, not the max recurrence."""
    if not arrivals:
        return []
    p = len(services)
    queues = [deque() for _ in services]
    active = [None] * p
    results = [[None] * p for _ in arrivals]
    tick = arrived = done = 0
    while done < len(arrivals):
        for j in reversed(range(p)):
            job = active[j]
            if job is not None and job[1] == tick:
                i, _ = job
                active[j] = None
                if j == p - 1:
                    done += 1
                else:
                    queues[j + 1].append(i)
        while arrived < len(arrivals) and arrivals[arrived] == tick:
            queues[0].append(arrived)
            arrived += 1
        for j in range(p):
            if active[j] is None and queues[j]:
                i = queues[j].popleft()
                finish = tick + services[j]
                active[j] = (i, finish)
                results[i][j] = finish
        tick += 1
        assert tick < 10000
    return results


def population(arrivals, completions, window):
    assert all(0 <= a <= c <= window for a, c in zip(arrivals, completions))
    events = defaultdict(int)
    for a, c in zip(arrivals, completions):
        events[a] += 1
        events[c] -= 1
    area = previous = count = peak = 0
    for time, change in sorted(events.items()):
        area += count * (time - previous)
        count += change
        assert count >= 0
        peak = max(peak, count)
        previous = time
    assert count == 0
    return area / window, peak


rng = random.Random(335)
cases = admission_cases = 0
for stages in range(1, 6):
    for n in range(0, 8):
        for repeat in range(20):
            arrivals = sorted(rng.randrange(25) for _ in range(n))
            services = [rng.randrange(1, 7) for _ in range(stages)]
            expected = tick_oracle(arrivals, services)
            actual = timeline(iter(arrivals), iter(services))
            assert actual == expected
            # No overtaking or resource overlap; future arrivals cannot change a prefix.
            for cut in range(n + 1):
                assert timeline(arrivals[:cut], services) == actual[:cut]
            if n:
                finish = [row[-1] for row in actual]
                window = finish[-1] + 5
                mean_n, _ = population(arrivals, finish, window)
                mean_w = sum(c - a for a, c in zip(arrivals, finish)) / n
                assert math.isclose(mean_n, n / window * mean_w, abs_tol=1e-12)
            cases += 1

# Periodic fixture: eight bursts leave each period empty at both boundaries.
arrivals = [120 * group + offset for group in range(8) for offset in [0, 1, 2]]
for services, expected_n, expected_w in [([35], 1.725, 69), ([8, 4, 20, 3], 1.35, 54)]:
    actual = timeline(arrivals, services)
    assert actual == tick_oracle(arrivals, services)
    finishes = [row[-1] for row in actual]
    mean_n, peak = population(arrivals, finishes, 960)
    assert math.isclose(mean_n, expected_n, abs_tol=1e-12)
    assert peak == 3
    assert sum(c - a for c, a in zip(finishes, arrivals)) / len(arrivals) == expected_w
    assert sum(c - a > 50 for c, a in zip(finishes, arrivals)) == 16

# Admission oracle reschedules only the accepted set plus the proposed new frame.
for n in range(8):
    for repeat in range(40):
        arrivals = sorted(rng.randrange(25) for _ in range(n))
        service, budget = rng.randrange(1, 9), rng.randrange(0, 25)
        accepted, rejected, accepted_arrivals = [], [], []
        for i, arrival in enumerate(arrivals):
            proposed = tick_oracle(accepted_arrivals + [arrival], [service])[-1][-1]
            if proposed <= arrival + budget:
                accepted.append((i, proposed))
                accepted_arrivals.append(arrival)
            else:
                rejected.append(i)
        assert admit(iter(arrivals), service, budget) == (accepted, rejected)
        assert all(finish - arrivals[i] <= budget for i, finish in accepted)
        assert sorted([i for i, _ in accepted] + rejected) == list(range(n))
        admission_cases += 1

assert admit([0], 10, 10) == ([(0, 10)], [])
assert admit([0], 10, 9) == ([], [0])
assert admit([0, 1, 2], 35, 50) == ([(0, 35)], [1, 2])
# Wrong full-input/accepted-only denominator: 0.875 is not the accepted population.
mean_n, _ = population([0], [35], 120)
assert math.isclose(mean_n, 35 / 120)
assert not math.isclose(mean_n, 3 / 120 * 35)

invalid = [
    ([1, 0], [2]), ([-1], [2]), ([float("nan")], [2]),
    ([float("inf")], [2]), ([0], []), ([0], [0]),
    ([0], [-1]), ([0], [float("nan")]), ([0], [float("inf")])
]
for arrivals, services in invalid:
    try:
        timeline(arrivals, services)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid arrival/service inputs accepted")
for budget in [-1, float("nan"), float("inf")]:
    try:
        admit([0], 1, budget)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid deadline budget accepted")

print(f"PASS {cases} independent tick-queue schedules and prefix controls; "
      f"{admission_cases} admission schedules; exact periodic populations/peaks, "
      "deadline boundary and wrong-denominator controls; CPU arithmetic only")
