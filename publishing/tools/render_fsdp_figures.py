"""Rasterize complete pinned source SVGs; never redraw or crop their contents.

Run with uv run --with cairosvg==2.9.1 python publishing/tools/render_fsdp_figures.py.
On Homebrew macOS, set DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
Original SVGs must already be retained in the source directory.
"""
from pathlib import Path
import hashlib
import json
import cairosvg
import struct

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/fsdp-2026-10-23"
OUTPUT = ROOT / "00 Учебник/Assets/Figures/curated/fsdp-2026-10-23"
CASES = [
    ("zero_memory.svg", "hf-zero-memory.png", "c27c9cec9cac888ad998e07b3d2c5d6fd2fd8a7bc977ff255ccd101371dd022c"),
    ("dp_zero3_fwd.svg", "hf-fsdp-forward.png", "df0793b7d70e31ce289bf135288dc7a8c08f5888ef91cc4f9b1224c8248d9b3b"),
    ("dp_zero3_bwd.svg", "hf-fsdp-backward.png", "ed98197af27915a3b34401939e9dc8d4fb99a218350dc255354ad89e10d6918a"),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert cairosvg.__version__ == "2.9.1", "Use the recorded renderer version"
    results = []
    for source_name, target_name, digest in CASES:
        data = (SOURCE / source_name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, source_name
        destination = OUTPUT / target_name
        cairosvg.svg2png(bytestring=data, write_to=str(destination),
                        output_width=2400, background_color="white")
        rendered = destination.read_bytes()
        width, height = struct.unpack(">II", rendered[16:24])
        results.append({"file": target_name, "width": width, "height": height,
                        "sha256": hashlib.sha256(rendered).hexdigest(),
                        "source": source_name, "source_sha256": digest,
                        "renderer": "CairoSVG " + cairosvg.__version__,
                        "crop": None})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
