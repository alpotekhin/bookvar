#!/usr/bin/env python3
"""Check original bytes, licensing snapshot, dimensions and all animation frames."""
from pathlib import Path
import hashlib
import json
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
specs=json.loads(Path(__file__).with_name("gpt2-figures.json").read_text())
assert len(specs)==6
for s in specs:
    for path,digest in [(s["asset"],s["sha256"]),(s["source_asset"],s["source_asset_sha256"]),
        (s["source_page_snapshot"],s["source_page_sha256"]),(s["license_evidence"],s["license_evidence_sha256"])]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert (ROOT/s["asset"]).read_bytes()==(ROOT/s["source_asset"]).read_bytes()
    page=(ROOT/s["source_page_snapshot"]).read_text()
    assert s["source_image_url"].split("jalammar.github.io")[-1] in page
    assert "creativecommons.org/licenses/by-nc-sa/4.0/" in page
    with Image.open(ROOT/s["asset"]) as im:
        assert im.size==(s["width"],s["height"])
        assert getattr(im,"n_frames",1)==s["frames"]
        for i in range(s["frames"]): im.seek(i); im.load()
    print("PASS original bytes/source/license/dimensions/all frames",s["name"],s["frames"])
