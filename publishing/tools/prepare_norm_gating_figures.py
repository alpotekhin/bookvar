"""Render/check original Xiong figures; no label edits or diagram redraws."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[2]
specs=json.loads(Path(__file__).with_name("norm-gating-figures.json").read_text())
for spec in specs:
    source=ROOT/spec["source_asset"]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==spec["source_asset_sha256"]
    with tempfile.TemporaryDirectory(prefix="bookvar-norm-gating-") as folder:
        prefix=Path(folder)/"page"
        subprocess.run(["pdftoppm","-f",str(spec["source_page"]),"-l",str(spec["source_page"]),
                        "-r",str(spec["render_dpi"]),"-singlefile","-png",str(source),str(prefix)],check=True)
        with Image.open(prefix.with_suffix(".png")) as page:
            assert list(page.size)==spec["render_page_size_px"]
            expected=page.crop(spec["crop_box_px"]).convert("RGB")
            target=ROOT/spec["asset"]
            if "--check" not in sys.argv:
                target.parent.mkdir(parents=True,exist_ok=True)
                expected.save(target)
            with Image.open(target) as actual:
                assert actual.size==expected.size
                assert ImageChops.difference(actual.convert("RGB"),expected).getbbox() is None
    print("PASS source pixels",spec["name"],hashlib.sha256(target.read_bytes()).hexdigest())
    assert hashlib.sha256(target.read_bytes()).hexdigest()==spec["sha256"]

# Reused, already licensed originals retain their source-page pixels.
rows=json.loads((ROOT/"00 Учебник/Assets/Figures/curated/chameleon-2026-10/assets.json").read_text())
for row in rows:
    if not row["asset"].endswith(("chameleon-norm-qk-dropout.png","chameleon-training-ablations.png")):
        continue
    for path,digest in [(row["asset"],row["sha256"]),(row["source_asset"],row["source_asset_sha256"]),
                        (row["source_page_snapshot"],row["source_page_sha256"]),
                        (row["license_evidence"],row["license_evidence_sha256"])]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    with Image.open(ROOT/row["source_page_snapshot"]) as page,Image.open(ROOT/row["asset"]) as actual:
        expected=page.crop(row["crop_box_px"])
        assert actual.size==expected.size
        assert ImageChops.difference(actual.convert("RGB"),expected.convert("RGB")).getbbox() is None
    print("PASS saved original-page crop",row["label"])

source=ROOT/"05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/cs336_assignment1_basics.pdf"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="cf402d1c6d3c66da035bdf33b3e2c1bbc4bb92a78b006ee36577e925a265e328"
with tempfile.TemporaryDirectory() as folder:
    prefix=Path(folder)/"page"
    subprocess.run(["pdftoppm","-f","21","-l","21","-r","240","-singlefile","-png",str(source),str(prefix)],check=True)
    with Image.open(prefix.with_suffix(".png")) as page,Image.open(ROOT/"00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png") as actual:
        expected=page.crop([633,293,1420,1084])
        assert ImageChops.difference(actual.convert("RGB"),expected.convert("RGB")).getbbox() is None
print("PASS source pixels CS336 SiLU; five original figures checked")
