#!/usr/bin/env python3
"""Compare the chapter's attributed passages with pinned upstream originals.

Only presentation markup, link destinations and image embeds are normalized.
This checks reproduction, not the correctness of upstream technical claims.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "05 Источники/Originals/serving-engines-2026-10-14"


def words(text):
    text = re.sub(r"<!-- source-caption:start -->.*?<!-- source-caption:end -->", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<img\b[^>]*>(?:</img>)?", "", text)
    text = re.sub(r"</?figure[^>]*>", "", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"^\.\. image::[^\n]*\n(?:  :[^\n]*\n)*", "", text, flags=re.M)
    text = re.sub(r"^\.\. _[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"^\.\. (?:note::|code:: python)$", "", text, flags=re.M)
    text = re.sub(r"^```[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"^[=~\-]{3,}$", "", text, flags=re.M)
    text = re.sub(r"^#{1,6} +", "", text, flags=re.M)
    text = re.sub(r"^> ?", "", text, flags=re.M)
    text = re.sub(r"`([^`<>]+) <([^>]+)>`_", r"\1", text)
    text = re.sub(r"(?<!!)\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r":(?:math|class|meth):`([^`]+)`", r"\1", text)
    text = text.replace("`", "").replace("$", "").replace("*", "")
    return text.split()


def main():
    manifest = json.loads((BASE / "excerpts.json").read_text())
    chapter = (ROOT / manifest["chapter"]).read_text()
    total = 0
    for item in manifest["excerpts"]:
        source = (BASE / item["source"]).read_text()
        assert item["raw"] in source, f"Source passage changed: {item['id']}"
        start = f"<!-- source-excerpt:{item['id']}:start -->"
        end = f"<!-- source-excerpt:{item['id']}:end -->"
        assert chapter.count(start) == chapter.count(end) == 1
        actual = chapter.split(start)[1].split(end)[0]
        raw = item["raw"]
        if item["id"] == "tensorrt-architecture":
            raw = raw.removeprefix("# Architecture Overview\n")
        if item["id"] == "vllm-weight-loading":
            raw = raw.removeprefix("3\\. **Sharding and Quantization at Initialization**: ")
        expected_words, actual_words = words(raw), words(actual)
        # Prose normalization must not hide changes inside original code.
        upstream_code = re.findall(r"^```python\n(.*?)^```$", raw, re.M | re.S)
        if item["source"].endswith(".rst"):
            upstream_code = [
                "\n".join(line[2:] for line in block.splitlines()).rstrip()
                for block in re.findall(r"^\.\. code:: python\n\n((?:  [^\n]*\n?)+)", raw, re.M)
            ]
        chapter_code = re.findall(r"^```python\n(.*?)^```$", actual, re.M | re.S)
        assert [c.rstrip() for c in upstream_code] == [c.rstrip() for c in chapter_code], f"Code changed: {item['id']}"
        if expected_words != actual_words:
            first = next((i for i, pair in enumerate(zip(expected_words, actual_words))
                          if pair[0] != pair[1]), min(len(expected_words), len(actual_words)))
            raise AssertionError(
                f"{item['id']} differs at token {first}: "
                f"{expected_words[first:first+12]} != {actual_words[first:first+12]}"
            )
        total += len(expected_words)
        print(f"PASS {item['id']}: {len(expected_words)} unchanged text/code tokens")
    assert len(manifest["excerpts"]) == 10
    assert chapter.count("![[") == 7
    assert chapter.count("<!-- source-caption:start -->") == chapter.count("<!-- source-caption:end -->") == 6
    print(f"PASS 10 source excerpts, {total} unchanged text/code tokens, 7 original figures")
    print("Only markup normalized; no upstream engine/GPU benchmark executed.")


if __name__ == "__main__":
    main()
