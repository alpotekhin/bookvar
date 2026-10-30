#!/usr/bin/env python3
"""Execute chapter 70's Lean/Coq examples; this is not a prover benchmark."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md"
ARCHIVE = ROOT / "05 Источники/Originals/formal-math-2026-10-21"
COQ_IMAGE = "coqorg/coq@sha256:18ebf3da56e60e3ddfd7d4e51f4c53d10241a129f34e93dacbc71562dd43c57a"


def run(command, expected=0):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != expected:
        raise AssertionError((command, result.returncode, result.stdout, result.stderr))
    return result.stdout + result.stderr


def blocks(text, language):
    return re.findall(r"^```" + language + r"\n(.*?)^```\s*$", text, re.M | re.S)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--lean", default=shutil.which("lean"))
    args = parser.parse_args()
    if not args.lean:
        parser.error("--lean must point to the reviewed Lean 4.19.0 executable")
    lean_version = run([args.lean, "--version"])
    assert "4.19.0" in lean_version, lean_version
    assert shutil.which("docker"), "Local Docker is required for the pinned Coq checker"
    context = run(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"])
    assert context.strip().startswith("unix://"), "Refuse a remote Docker daemon"

    text = CHAPTER.read_text()
    lean = blocks(text, "lean")
    positive = [block for block in lean if "theorem unfinished" not in block]
    negative = [block for block in lean if "theorem unfinished" in block]
    assert len(positive) == 5 and len(negative) == 1
    coq = blocks(text, "coq")
    assert len(coq) == 7, len(coq)
    with tempfile.TemporaryDirectory(prefix="bookvar-formal-") as tmp:
        tmp = Path(tmp)
        all_positive = "\n".join(positive)
        names = re.findall(r"^theorem (\w+)", all_positive, re.M)
        positive_file = tmp / "Positive.lean"
        positive_file.write_text(all_positive + "\n" + "\n".join(
            "#print axioms " + name for name in names) + "\n")
        output = run([args.lean, str(positive_file)])
        for name in names:
            line = next(line for line in output.splitlines() if line.startswith(f"'{name}' "))
            if "does not depend on any axioms" not in line:
                dependencies = re.search(r"\[(.*?)\]", line).group(1).split(", ")
                assert set(dependencies) <= {"propext", "Classical.choice", "Quot.sound"}, line

        for index, bad in enumerate([
            "theorem bad (P Q : Prop) (h : P ∧ Q) : Q := by exact h.left",
            "theorem bad : ∃ m : Nat, ∀ n : Nat, m > n := by\n"
            "  exact ⟨0, fun n => Nat.lt_succ_self n⟩",
        ]):
            path = tmp / f"Rejected{index}.lean"
            path.write_text(bad + "\n")
            assert "error:" in run([args.lean, str(path)], expected=1)

        negative_file = tmp / "Untrusted.lean"
        negative_file.write_text(negative[0] + "\n"
            "axiom SMT_VERIF (P : Prop) : P\n"
            "theorem smt_false : False := SMT_VERIF False\n"
            "#print axioms smt_false\n")
        output = run([args.lean, str(negative_file)])
        for name, dependency in [("unfinished", "sorryAx"),
                                  ("forged", "invented"),
                                  ("smt_false", "SMT_VERIF")]:
            line = next(line for line in output.splitlines() if f"'{name}' depends on" in line)
            assert dependency in line, line
        # Successful elaboration is explicitly NOT acceptance under this policy.
        assert all(name not in {"propext", "Classical.choice", "Quot.sound"}
                   for name in ("sorryAx", "invented", "SMT_VERIF"))

        coq_file = tmp / "StackMachine.v"
        coq_file.write_text("\n".join(coq))
        command = ["docker", "run", "--rm", "--network", "none", "--platform", "linux/amd64",
                   "--memory", "1g", "--cpus", "1", "--read-only", "--cap-drop", "ALL",
                   "--tmpfs", "/tmp:rw,exec,size=128m", "--mount",
                   f"type=bind,source={tmp},target=/work,readonly", COQ_IMAGE,
                   "/bin/sh", "-c", "cp /work/StackMachine.v /tmp/StackMachine.v && "
                   "cd /tmp && coqc --version && coqc StackMachine.v"]
        output = run(command)
        assert "version 8.20.1" in output, output
        assert output.count("Closed under the global context") == 2, output

    # Arithmetic underlying the fully explained examples, not source benchmark reproduction.
    a, b, c = F(1, 3), F(2, 3), F(2, 3)
    assert a*a+b*b+c*c == 1
    assert 3*a*a+3*b*b+c*c == 3-2*c*c == F(19, 9)
    assert 1/(3-2*c*c) == F(9, 19)
    for n in range(100):
        assert 3*(14*n+3)-2*(21*n+4) == 1
    for a in [F(n, 7) for n in range(-100, 101)]:
        square = (a-F(5, 28))**2
        assert 28*square == 28*a*a-10*a+F(25, 28)
        assert square >= 0 and 28*a*a+1-10*a >= F(3, 28)
    for x in range(-10, 11):
        assert x*x <= x*x+1 and x*x <= x*x+2
        assert not (x*x+2 <= x*x+1)
    for alpha, x, y in [(F(2, 5), F(4, 5), F(1, 5)),
                         (F(1, 4), F(7, 8), F(1, 8))]:
        # Angles as fractions of pi in the omitted Euclidean configuration.
        assert alpha+y < 1 and 2-alpha-x < 1
        assert x > 1-alpha > y
    assert F(463-434, 10) == F(29, 10)  # percentage points, not relative percent

    refs_path = ARCHIVE / "references.json"
    if refs_path.exists():
        refs = json.loads(refs_path.read_text())
        assert digest(CHAPTER) == refs["chapter_sha256"]
        for row in refs["downloaded"] + refs["existing_sources"] + refs["figures"] + refs["derived"]:
            assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
        for row in refs["figures"]:
            assert digest(ROOT / row["license_evidence"]) == row["license_evidence_sha256"]
            if row.get("crop_box_px"):
                page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
                crop = page.crop(row["crop_box_px"])
                asset = Image.open(ROOT / row["asset"]).convert("RGB")
                assert list(page.size) == row["render_page_size_px"]
                assert crop.size == asset.size
                assert ImageChops.difference(crop, asset).getbbox() is None
        for row in refs.get("additional_course_figures", []):
            assert digest(ROOT / row["asset"]) == row["sha256"]
            assert digest(ROOT / row["source_asset"]) == row["source_asset_sha256"]
            assert Image.open(ROOT / row["asset"]).size == (2000, 1500)
    print("PASS: 5 Lean theorems within the standard axiom allowlist; 2 rejected proofs; 3 compiled but "
          "axiom-rejected artifacts; Coq definitions, general lemma, corollary, "
          "4 computations and 2 closed assumption reports; exact arithmetic; source hashes. "
          "No LLM training/search benchmark or LeanEuclid SMT run.")


if __name__ == "__main__":
    main()
