"""Check the actual saved CAI data examples; no model, network or training call."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 2
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    assert [r["completion"] for r in ns["sft_records"]] == [
        r["revision"] for r in ns["trace"]
    ]
    assert all(set(r) == {"prompt", "completion"} for r in ns["sft_records"])
    prob, loss = ns["probability_of_a"], ns["pair_loss"]
    comparisons = 0
    for a, b in [(math.log(.02), math.log(.08)), (-1001., -1000.),
                 (-100000000., -100000000.), (0., -1000.), (-1000., 0.)]:
        q = prob(a, b)
        assert 0 <= q <= 1
        assert math.isclose(q + prob(b, a), 1, abs_tol=1e-14)
        assert math.isclose(prob(a+317, b+317), q, abs_tol=1e-13)
        comparisons += 1
    gradient_checks = 0
    for delta in (-10., -.5, 0., .5, 10.):
        for q in (0., .01, .2, .4, .5, .6, .8, .99, 1.):
            h = 1e-5
            numerical = (loss(delta+h, q)-loss(delta-h, q))/(2*h)
            expected = 1/(1+math.exp(-delta))-q
            assert math.isclose(numerical, expected, abs_tol=1e-9)
            assert math.isclose(loss(delta,q),loss(-delta,1-q),abs_tol=1e-12)
            gradient_checks += 1
    for q in (.01,.2,.4,.6,.8,.99):
        optimum = math.log(q/(1-q))
        assert loss(optimum,q) < loss(optimum-.1,q)
        assert loss(optimum,q) < loss(optimum+.1,q)
    for delta in (-1000.,1000.):
        for q in (0.,.01,.5,.99,1.):
            assert math.isfinite(loss(delta,q))
            assert loss(delta,q) >= 0
    for q in (0.,.01,.2,.4,.5,.6,.8,.99,1.):
        clamped = min(.6,max(.4,q))
        assert (.5-q)*(.5-clamped) >= 0
        assert math.isclose(min(.6,max(.4,1-q)),1-clamped,abs_tol=1e-15)
    for fn,args in [(prob,(float("nan"),0.)),(prob,(-float("inf"),0.)),
                    (loss,(float("inf"),.5)),(loss,(0.,float("nan"))),
                    (loss,(0.,-.01)),(loss,(0.,1.01))]:
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input accepted: {args}")
    print(json.dumps({"chapter_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "published_python_fences":len(blocks),"sft_records":len(ns["sft_records"]),
        "probability_shift_swap_cases":comparisons,"gradient_checks":gradient_checks,
        "q_a":ns["q"],"gradient":ns["gradient"],"optimum":ns["optimum"],
        "hard_raw_clamped_at_zero":[.5,.49,.1],"device":"cpu",
        "language_model_calls":0},indent=2))
    print("PASS saved CAI examples; data construction and numerical loss only")


if __name__ == "__main__":
    main()
