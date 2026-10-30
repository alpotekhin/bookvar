"""Extract the existing source figure without rescaling or discarding its ICC profile."""
from hashlib import sha256
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[2]
folder = root / "00 Учебник/Assets/Figures/curated/arithmetic-review-2026-09"
source = folder / "nemo-checkpointing.jpg"
assert sha256(source.read_bytes()).hexdigest() == "08f34fd0293f8d7077bf081539350a26fa066b15a27ae0c09816ee989dcbae56"
with Image.open(source) as original:
    rgb = original.convert("RGB")
    options = {"icc_profile": original.info["icc_profile"]} if "icc_profile" in original.info else {}
    outputs = ((folder / "nemo-checkpointing-source-exact.png", rgb),
               (folder / "nemo-checkpointing-detail.png", rgb.crop((0, 0, 728, 604))))
    for path, image in outputs:
        image.save(path, **options)
        with Image.open(path) as result:
            assert result.convert("RGB").tobytes() == image.tobytes()
            assert result.info.get("icc_profile") == original.info.get("icc_profile")
        print(path.name, sha256(path.read_bytes()).hexdigest())
