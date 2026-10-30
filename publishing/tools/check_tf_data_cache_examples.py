"""Run the actual S6 tf.data fences on CPU; no throughput/GPU claim."""
from hashlib import sha256
from pathlib import Path
import ast
import re


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md"
PATTERN = (r"<!-- BEGIN REVIEW: tf.data cache 2026-10-28 -->\n(.*?)"
           r"<!-- END REVIEW: tf.data cache 2026-10-28 -->\n\n")
BASELINE = "4ec7b2a6d0665e0c0c6a43337e508f4d7cb41c30009d1b850850c6559d1a25aa"
WRAPPED_PAIRS = r"""\begin{aligned}
&128^2+140^2+160^2+1024^2\\
&=1\,110\,160,
\end{aligned}"""


def main():
    text = CHAPTER.read_text()
    matches = list(re.finditer(PATTERN, text, re.S))
    assert len(matches) == 1
    original = re.sub(PATTERN, "", text, flags=re.S)
    for key in ("last_updated", "last_verified"):
        original = original.replace(f"{key}: 2026-10-28", f"{key}: 2026-10-23", 1)
    assert text.count(WRAPPED_PAIRS) == 1
    original = original.replace(
        WRAPPED_PAIRS, r"128^2+140^2+160^2+1024^2=1\,110\,160,", 1)
    assert sha256(original.encode()).hexdigest() == BASELINE

    blocks = re.findall(r"```python\n(.*?)\n```", matches[0].group(1), re.S)
    assert len(blocks) == 5
    namespace = {}
    for index, block in enumerate(blocks, 1):
        ast.parse(block)
        exec(compile(block, f"{CHAPTER}#cache-example-{index}", "exec"), namespace)
        print(f"PASS: actual cache fence {index}")

    tf = namespace["tf"]
    assert tf.__version__ == "2.21.0"
    assert tf.config.get_visible_devices("GPU") == []
    collect, numbers = namespace["collect"], namespace["numbers"]
    counter, prepare = namespace["decode_calls"], namespace["prepare"]
    cases = 0
    for size in range(8):
        counter.assign(0)
        cached = numbers(size).map(prepare).cache()
        expected = [x*x for x in range(size)]
        assert collect(cached) == expected
        assert int(counter.numpy()) == size
        assert collect(cached) == expected
        assert int(counter.numpy()) == size
        counter.assign(0)
        downstream = numbers(size).cache().map(prepare)
        assert collect(downstream) == expected
        assert collect(downstream) == expected
        assert int(counter.numpy()) == 2*size
        cases += 1

    # Deliberately finite prefixes: before-cache is complete, after-cache is not.
    for size in range(2, 7):
        for prefix in range(1, size):
            counter.assign(0)
            incomplete = numbers(size).map(prepare).cache().take(prefix)
            expected = [x*x for x in range(prefix)]
            assert collect(incomplete) == expected
            before = int(counter.numpy())
            assert collect(incomplete) == expected
            assert int(counter.numpy()) > before
            counter.assign(0)
            complete = numbers(size).take(prefix).map(prepare).cache()
            assert collect(complete) == collect(complete) == expected
            assert int(counter.numpy()) == prefix
            cases += 1

    # Counts and outputs for empty and nonempty epoch-dependent cached datasets.
    epoch, transform = namespace["epoch"], namespace["epoch_dependent"]
    for size in range(5):
        epoch.assign(0)
        frozen = numbers(size).map(transform).cache()
        live = numbers(size).cache().map(transform)
        assert collect(frozen) == collect(live) == list(range(size))
        for value in (1, 3):
            epoch.assign(value)
            assert collect(frozen) == list(range(size))
            assert collect(live) == [x + 100*value for x in range(size)]
            cases += 1

    # This fixed seed's observed permutations, not a universal uniqueness rule.
    assert namespace["fresh_epochs"][0] != namespace["fresh_epochs"][1]
    assert sorted(namespace["fixed_epochs"][0]) == list(range(8))
    assert 1000*224*224*3 == 150_528_000
    assert 1000*224*224*3*4 == 602_112_000
    assert f"{150_528_000/2**20:.2f}" == "143.55"
    assert f"{602_112_000/2**20:.2f}" == "574.22"
    print(f"PASS: {cases} independent reuse/prefix/epoch cases; shuffle and bytes")
    print("chapter_sha256=" + sha256(CHAPTER.read_bytes()).hexdigest())
    print(f"TensorFlow {tf.__version__}; CPU semantics only; no timing benchmark")


if __name__ == "__main__":
    main()
