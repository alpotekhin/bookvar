#!/usr/bin/env python3
"""Extract complete source figures; no redrawing or content edits."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"05 Источники/Originals/positional-information-2026-10-21"
OUT=ROOT/"00 Учебник/Assets/Figures/curated/positions-2026-10-21"
OUT.mkdir(parents=True,exist_ok=True)
SPECS=[
 ("shaw-N18-2074.pdf",3,(345,300,1475,1140),"shaw-relative-figure1.png"),
 ("roformer-v5.pdf",5,(705,2010,2375,3075),"roformer-figure1.png"),
]
with tempfile.TemporaryDirectory() as tmp:
 for source,page,box,name in SPECS:
  prefix=Path(tmp)/name
  subprocess.run(["pdftoppm","-f",str(page),"-l",str(page),"-r","360",
                  "-singlefile","-png",str(SRC/source),str(prefix)],check=True)
  with Image.open(str(prefix)+".png") as rendered:
   image=rendered.convert("RGB").crop(box)
   image.save(OUT/name)
   print(name,"page",page,"page_size",rendered.size,"crop",box,"size",image.size)
  print("sha256",hashlib.sha256((OUT/name).read_bytes()).hexdigest())
shutil.copyfile(SRC/"alibi-original.jpeg",OUT/"alibi-original.jpeg")
shutil.copyfile(SRC/"alibi-MIT.txt",ROOT/"site/public/licenses/alibi-4b92f28a-MIT.txt")
print("ALiBi source JPEG and MIT copyright/license copied without changes")
