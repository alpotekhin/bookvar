#!/usr/bin/env python3
"""Extract unchanged paper figures, with the original captions, at 360 dpi."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21"
T5 = "05 Источники/Originals/positional-information-2026-10-21/t5-jmlr.pdf"
BASE = "05 Источники/Originals/architecture-patterns-2026-10-21/"
specs = [
 {"name":"t5-figure1.png","source":T5,"page":3,"box":[425,430,2630,1680]},
 {"name":"t5-figure2.png","source":T5,"page":13,"box":[425,435,2630,1565]},
 {"name":"t5-figure3.png","source":T5,"page":15,"box":[425,415,2630,1870]},
 {"name":"t5-figure4.png","source":T5,"page":16,"box":[425,450,2630,2130]},
 {"name":"bert-figure4.png","source":BASE+"bert-naacl.pdf","page":15,"box":[530,300,2480,2340]},
]
OUT.mkdir(parents=True,exist_ok=True)
for s in specs:
 with tempfile.TemporaryDirectory() as tmp:
  prefix=Path(tmp)/"page"
  subprocess.run(["pdftoppm","-f",str(s["page"]),"-l",str(s["page"]),
                  "-r","360","-singlefile","-png",str(ROOT/s["source"]),str(prefix)],check=True)
  with Image.open(str(prefix)+".png") as im:
   assert 0<=s["box"][0]<s["box"][2]<=im.width
   assert 0<=s["box"][1]<s["box"][3]<=im.height
   crop=im.crop(s["box"])
   crop.save(OUT/s["name"])
   print(s["name"],im.size,crop.size,hashlib.sha256((OUT/s["name"]).read_bytes()).hexdigest())
