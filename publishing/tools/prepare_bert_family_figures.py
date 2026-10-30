#!/usr/bin/env python3
"""Extract the original BERT input representation and archive original PNGs."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"05 Источники/Originals/bert-family-2026-10-22"
PDF = ROOT/"05 Источники/Originals/architecture-patterns-2026-10-21/bert-naacl.pdf"
OUT = ROOT/"00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure2.png"
with tempfile.TemporaryDirectory() as tmp:
    prefix = Path(tmp)/"page"
    subprocess.run(["pdftoppm","-f","5","-l","5","-r","360","-singlefile","-png",str(PDF),str(prefix)],check=True)
    with Image.open(str(prefix)+".png") as full:
        box = (350,300,2630,1040)
        assert full.size == (2977,4210)
        full.crop(box).save(OUT)
        print("PDF crop",box,Image.open(OUT).size,hashlib.sha256(OUT.read_bytes()).hexdigest())
for name in ["bert-base-bert-large-encoders.png","bert-next-sentence-prediction.png"]:
    target = BASE/name
    with urllib.request.urlopen("https://jalammar.github.io/images/"+name) as response:
        source = response.read()
    target.write_bytes(source)
    curated = ROOT/"00 Учебник/Assets/Figures/curated/bert"/name
    assert curated.read_bytes() == source, name
    with Image.open(target) as im:
        print(name,im.size,hashlib.sha256(source).hexdigest())
