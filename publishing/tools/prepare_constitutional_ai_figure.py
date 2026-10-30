"""Render the complete original Figure 5, including its caption; no redraw."""
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "05 Источники/Originals/constitutional-ai-2026-10-29/bai-v1.pdf"
OUTPUT = ROOT / "00 Учебник/Assets/Figures/curated/constitutional-ai-2026-10-29/revisions-tradeoff.png"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "pdftoppm", "-f", "9", "-l", "9", "-r", "288", "-singlefile",
        "-x", "340", "-y", "280", "-W", "1780", "-H", "750", "-png",
        str(PDF), str(OUTPUT.with_suffix("")),
    ], check=True)
    data = OUTPUT.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (1780, 750)
    print(f"Rendered original Figure 5: {OUTPUT}")


if __name__ == "__main__":
    main()
