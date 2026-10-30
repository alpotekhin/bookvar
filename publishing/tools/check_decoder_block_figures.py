#!/usr/bin/env python3
"""Check all six decoder chapter images against actual pinned source pixels."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[2]
specs=json.loads(Path(__file__).with_name("decoder-block-figures.json").read_text())
block=next(s for s in json.loads(Path(__file__).with_name("llama-figures.json").read_text())
           if s["name"]=="cs336-prenorm-block.png")
for s in [block,*specs]:
    assert hashlib.sha256((ROOT/s["source_asset"]).read_bytes()).hexdigest()==s["source_asset_sha256"]
    assert hashlib.sha256((ROOT/s["asset"]).read_bytes()).hexdigest()==s["sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        prefix=Path(tmp)/"page"
        subprocess.run(["pdftoppm","-f",str(s["source_page"]),"-l",str(s["source_page"]),
                        "-r",str(s["render_dpi"]),"-singlefile","-png",
                        str(ROOT/s["source_asset"]),str(prefix)],check=True)
        with Image.open(str(prefix)+".png") as source,Image.open(ROOT/s["asset"]) as saved:
            assert list(source.size)==s["render_page_size_px"]
            expected=source.crop(s["crop_box_px"]) if "crop_box_px" in s else source
            assert saved.size==expected.size
            assert ImageChops.difference(saved.convert("RGB"),expected.convert("RGB")).getbbox() is None
    print("PASS source pixels",s["name"])

byte_copies=[
 ("00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png",
  "05 Источники/Originals/llama-2026-10-22/gqa-source.png",
  "02c3d00ec9f339211daae34392aeed850b013457b64f8088f1c5c054a5e94ab6"),
 ("00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png",
  "05 Источники/Originals/gpt2-2026-10-22/gpt2-transformer-block-vectors-2.png",
  "94d7c51658e7feffd01c66e61acd8897cb17e93989725f0a523a95ab8b771047")
]
for asset,source,sha in byte_copies:
    assert (ROOT/asset).read_bytes()==(ROOT/source).read_bytes()
    assert hashlib.sha256((ROOT/asset).read_bytes()).hexdigest()==sha
    print("PASS byte-identical source",Path(asset).name)
print("Six source checks passed. CS336 lecture-slide reuse rights remain independently unconfirmed; pixel proof is not a license.")
