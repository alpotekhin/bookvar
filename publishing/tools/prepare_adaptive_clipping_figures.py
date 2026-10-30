"""Export one complete NFNet figure and copy the original AutoClip animation."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/gradient-clipping-2026-10-29"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/gradient-clipping-2026-10-29"
CHAPTER = "00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md"
CROP = (1215, 265, 2185, 775)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    pdf = SOURCE / "brock21a.pdf"
    with tempfile.TemporaryDirectory(prefix="bookvar-agc-") as directory:
        prefix = str(Path(directory) / "page4")
        subprocess.run(["pdftoppm", "-f", "4", "-l", "4", "-r", "288",
                        "-singlefile", "-png", str(pdf), prefix], check=True)
        with Image.open(prefix + ".png") as page:
            assert page.size == (2448, 3168)
            page.crop(CROP).save(TARGET / "nfnet-agc-ablation.png")
    gif = SOURCE / "autoclip-mi.gif"
    shutil.copyfile(gif, TARGET / gif.name)
    license_source = SOURCE / "autoclip-MIT.txt"
    public_license = ROOT / "site/public/licenses/autoclip-dfb622c3-MIT.txt"
    shutil.copyfile(license_source, public_license)
    pmlr_license = ROOT / "05 Источники/Originals/peft-2026-10-29/pmlr-license-agreement.pdf"
    common = {
        "last_verified": "2026-10-29", "metadata_status": "verified",
        "provenance_confirmation": "source-license-verified", "used_in": [CHAPTER],
    }
    rows = []
    for name, extra in [
        ("nfnet-agc-ablation.png", {
            "author": "Andy Brock, Soham De, Samuel L. Smith, Karen Simonyan",
            "title": "NFNet Figure 2: batch size and AGC threshold ablations",
            "source_url": "https://proceedings.mlr.press/v139/brock21a.html",
            "source_image_url": "https://proceedings.mlr.press/v139/brock21a/brock21a.pdf#page=4",
            "source_asset": relative(pdf), "source_asset_sha256": digest(pdf),
            "source_page": 4, "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_evidence": relative(pmlr_license),
            "license_evidence_sha256": digest(pmlr_license),
            "copyright": "Copyright 2021 the authors; ICML 2021, PMLR 139:1059-1071",
            "derivation": "pdf-page-render-crop", "render_dpi": 288,
            "render_page_size_px": [2448, 3168], "crop_box_px": list(CROP),
            "modifications": "Both complete panels and original caption cropped from PDF page 4. No redraw, relabeling, translation or resizing.",
        }),
        ("autoclip-mi.gif", {
            "author": "Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux",
            "title": "Original AutoClip mask-inference training animation",
            "source_url": "https://github.com/pseeth/autoclip/blob/dfb622c3caca9dc0ecc8bf0afeb773e190256d6e/README.md#training-dynamics",
            "source_image_url": "https://raw.githubusercontent.com/pseeth/autoclip/dfb622c3caca9dc0ecc8bf0afeb773e190256d6e/images/mi.gif",
            "source_asset": relative(gif), "source_asset_sha256": digest(gif),
            "source_commit": "dfb622c3caca9dc0ecc8bf0afeb773e190256d6e",
            "source_page_snapshot": relative(SOURCE / "autoclip-README.md"),
            "source_page_sha256": digest(SOURCE / "autoclip-README.md"),
            "license": "MIT",
            "license_url": "https://github.com/pseeth/autoclip/blob/dfb622c3caca9dc0ecc8bf0afeb773e190256d6e/LICENSE",
            "license_evidence": relative(license_source),
            "license_evidence_sha256": digest(license_source),
            "public_license_copy": relative(public_license),
            "copyright": "Copyright (c) 2020 pseeth",
            "derivation": "source-raster-copy",
            "modifications": "None. Complete original GIF copied byte-for-byte, including all forty frames and their timing.",
        }),
    ]:
        asset = TARGET / name
        with Image.open(asset) as image:
            row = dict(common, **extra, asset=relative(asset),
                       dimensions_px=list(image.size), width=image.width,
                       height=image.height, bytes=asset.stat().st_size,
                       sha256=digest(asset))
            if image.format == "GIF":
                assert image.n_frames == 40
                row["frames"] = image.n_frames
        row["id"] = "curated-adaptive-clipping-" + row["sha256"][:12]
        rows.append(row)
    (TARGET / "assets.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
