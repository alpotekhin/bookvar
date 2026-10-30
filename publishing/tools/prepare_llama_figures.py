#!/usr/bin/env python3
"""Extract or independently re-render original LLaMA/CS336 diagrams."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
parser.add_argument("--only", help="Process one named new figure without rewriting others.")
args = parser.parse_args()
specs = json.loads(Path(__file__).with_name("llama-figures.json").read_text())
if args.only:
    specs = [s for s in specs if s['name'] == args.only]
    assert len(specs) == 1, args.only
for spec in specs:
    for path, digest in [("source_asset", "source_asset_sha256"),
                         ("source_page_snapshot", "source_page_sha256"),
                         ("license_evidence", "license_evidence_sha256")]:
        if path in spec:
            assert hashlib.sha256((ROOT/spec[path]).read_bytes()).hexdigest() == spec[digest]
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp)/"page"
        subprocess.run(["pdftoppm", "-f", str(spec["source_page"]), "-l",
                        str(spec["source_page"]), "-r", str(spec["render_dpi"]),
                        "-singlefile", "-png", str(ROOT/spec["source_asset"]),
                        str(prefix)], check=True)
        with Image.open(str(prefix)+".png") as page:
            assert list(page.size) == spec["render_page_size_px"]
            result = page.crop(spec["crop_box_px"])
            assert result.size == (spec["width"], spec["height"])
            dest = ROOT/spec["asset"]
            if args.check:
                with Image.open(dest) as saved:
                    assert ImageChops.difference(result.convert("RGB"), saved.convert("RGB")).getbbox() is None
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == spec["sha256"]
                print("PASS exact source pixels", spec["name"])
            else:
                result.save(dest)
                print(spec["name"], result.size, hashlib.sha256(dest.read_bytes()).hexdigest())

# These existing images are reused, not regenerated in normal extraction mode.
if args.check:
    reused = [
        dict(name="roformer-figure1.png",
             source="05 Источники/Originals/positional-information-2026-10-21/roformer-v5.pdf",
             source_sha="e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79",
             asset="00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png",
             asset_sha="b1a65eb5e30f1510dce571cedb548b98c1db5c5e57ca04112b856f931a0d2591",
             page=5,dpi=360,box=[705,2010,2375,3075],page_size=[3060,3960]),
        dict(name="cs336-silu-vs-relu.png",
             source="05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/cs336_assignment1_basics.pdf",
             source_sha="cf402d1c6d3c66da035bdf33b3e2c1bbc4bb92a78b006ee36577e925a265e328",
             asset="00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png",
             asset_sha="59a942959793b641878bf65c6e83c6941e5153dc09a9893c6a5cb44f279f70e0",
             page=21,dpi=240,box=[633,293,1420,1084],page_size=[2040,2640]),
    ]
    for spec in reused:
        assert hashlib.sha256((ROOT/spec["source"]).read_bytes()).hexdigest()==spec["source_sha"]
        assert hashlib.sha256((ROOT/spec["asset"]).read_bytes()).hexdigest()==spec["asset_sha"]
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/"page"
            subprocess.run(["pdftoppm","-f",str(spec["page"]),"-l",str(spec["page"]),
                            "-r",str(spec["dpi"]),"-singlefile","-png",
                            str(ROOT/spec["source"]),str(prefix)],check=True)
            with Image.open(str(prefix)+".png") as page,Image.open(ROOT/spec["asset"]) as saved:
                assert list(page.size)==spec["page_size"]
                assert ImageChops.difference(page.crop(spec["box"]).convert("RGB"),
                                             saved.convert("RGB")).getbbox() is None
        print("PASS reused original pixels",spec["name"])
    gqa=ROOT/"00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png"
    source=ROOT/"05 Источники/Originals/llama-2026-10-22/gqa-source.png"
    assert gqa.read_bytes()==source.read_bytes()
    assert hashlib.sha256(gqa.read_bytes()).hexdigest()=="02c3d00ec9f339211daae34392aeed850b013457b64f8088f1c5c054a5e94ab6"
    evidence=ROOT/"05 Источники/Originals/llama-2026-10-22/gqa-arxiv.html"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest()=="05bae25916ae7ac838b2ee1b2113789c45a76e83b60f814fa899b35a0ce405ea"
    assert 'href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article"' in evidence.read_text()
    print("PASS byte-identical GQA original and explicit CC BY 4.0 evidence")
