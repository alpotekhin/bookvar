#!/usr/bin/env python3
"""Check full original image bytes and decode every animation frame."""
from pathlib import Path
import hashlib
import json
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
specs = json.loads((Path(__file__).with_name("full-transformer-figures.json")).read_text())
for s in specs:
    asset, source = ROOT/s["asset"], ROOT/s["source_asset"]
    assert asset.read_bytes() == source.read_bytes(), s["name"]
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == s["sha256"]
    page = ROOT/s["source_page_snapshot"]
    assert hashlib.sha256(page.read_bytes()).hexdigest() == s["source_page_sha256"]
    assert s["source_image_url"].split("jalammar.github.io")[-1] in page.read_text()
    assert "creativecommons.org/licenses/by-nc-sa/4.0/" in page.read_text()
    with Image.open(asset) as im:
        assert im.size == (s["width"], s["height"])
        assert im.n_frames == s["frames"]
        for frame in range(im.n_frames):
            im.seek(frame)
            im.load()
    print("PASS exact original and all frames:", s["name"], s["width"], s["height"], s["frames"])
print("PASS seven original figures; no crops or authored replacement diagrams")
