#!/usr/bin/env python3
"""Re-render exact paper crops and verify copied originals and provenance."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
manifest = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("architecture-figures.json")
specs = json.loads(manifest.read_text())
for s in specs:
    for path, digest in [(s["asset"], s["sha256"]),
                         (s["source_asset"], s["source_asset_sha256"]),
                         (s["source_page_snapshot"], s["source_page_sha256"]),
                         (s.get("license_evidence",s["source_page_snapshot"]),
                          s.get("license_evidence_sha256",s["source_page_sha256"]))]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    with Image.open(ROOT/s["asset"]) as image:
        image.load()
        assert image.size == (s["width"], s["height"])
        actual = image.convert("RGB")
    if s["derivation"] == "pdf-crop":
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)/"page"
            subprocess.run(["pdftoppm", "-f", str(s["source_page"]),
                            "-l", str(s["source_page"]), "-r", str(s["render_dpi"]),
                            "-singlefile", "-png", str(ROOT/s["source_asset"]), str(prefix)],
                           check=True)
            with Image.open(str(prefix)+".png") as full:
                assert list(full.size) == s["render_page_size_px"]
                expected = full.crop(s["crop_box_px"]).convert("RGB")
            assert ImageChops.difference(actual, expected).getbbox() is None, s["name"]
        print("PASS zero pixel difference, full original figure and caption:", s["name"])
    else:
        assert (ROOT/s["asset"]).read_bytes() == (ROOT/s["source_asset"]).read_bytes()
        page = (ROOT/s["source_page_snapshot"]).read_text()
        assert s["source_image_url"].split("jalammar.github.io")[-1] in page
        assert "creativecommons.org/licenses/by-nc-sa/4.0/" in page
        print("PASS byte-for-byte original:", s["name"])
pdf_count=sum(s["derivation"]=="pdf-crop" for s in specs)
print("PASS",len(specs),"figures;",pdf_count,"exact PDF crops and",len(specs)-pdf_count,"unchanged original PNGs")
