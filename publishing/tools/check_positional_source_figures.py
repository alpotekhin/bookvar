#!/usr/bin/env python3
"""Verify exact copies and pixels against newly rendered authoritative PDF pages."""
from pathlib import Path
import hashlib,json
import subprocess,tempfile
from PIL import Image,ImageChops
ROOT=Path(__file__).resolve().parents[2]
SPECS=json.loads(r'''[{"asset":"00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_positional_encoding_vectors.png","source":"05 Источники/Originals/positional-information-2026-10-21/alammar-positional-vectors.png","sha":"1447f2a6561e5aef3ca9ab3792490d744b81c34c873d19b1396ed770ca19b669","source_sha":"1447f2a6561e5aef3ca9ab3792490d744b81c34c873d19b1396ed770ca19b669","size":[1438,793]},{"asset":"00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-sinusoidal-curves.svg","source":"05 Источники/Originals/positional-information-2026-10-21/d2l-output-51.svg","sha":"f4d1aeebd58a4444284fd4d8e87a95e8dcb53efe3ad3d7d7bb41cd5fbb08b40a","source_sha":"f4d1aeebd58a4444284fd4d8e87a95e8dcb53efe3ad3d7d7bb41cd5fbb08b40a"},{"asset":"00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-positional-heatmap.svg","source":"05 Источники/Originals/positional-information-2026-10-21/d2l-output-81.svg","sha":"c10d8e03aab02bbe6f24f9851e64674231c2211f34df154705e3f882031ea26d","source_sha":"c10d8e03aab02bbe6f24f9851e64674231c2211f34df154705e3f882031ea26d"},{"asset":"00 Учебник/Assets/Figures/curated/positions-2026-10-21/shaw-relative-figure1.png","source":"05 Источники/Originals/positional-information-2026-10-21/shaw-N18-2074.pdf","sha":"5c73df1712d9b4c44e017c8de539ba0726720c3692c2c70047dd43e69e3dd16c","source_sha":"f10b5d840c34d69e3d145dc20de8f668535dca23a95e0025200797956f268587","page":3,"box":[345,300,1475,1140],"page_size":[2977,4210],"size":[1130,840]},{"asset":"00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png","source":"05 Источники/Originals/positional-information-2026-10-21/roformer-v5.pdf","sha":"b1a65eb5e30f1510dce571cedb548b98c1db5c5e57ca04112b856f931a0d2591","source_sha":"e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79","page":5,"box":[705,2010,2375,3075],"page_size":[3060,3960],"size":[1670,1065]},{"asset":"00 Учебник/Assets/Figures/curated/positions-2026-10-21/alibi-original.jpeg","source":"05 Источники/Originals/positional-information-2026-10-21/alibi-original.jpeg","sha":"ade378177ac2b32c4a7319c3bee16ee61da4ec9aaf7d1f109bd0933d900b6cc7","source_sha":"ade378177ac2b32c4a7319c3bee16ee61da4ec9aaf7d1f109bd0933d900b6cc7","size":[872,354]}]''')
for spec in SPECS:
 asset,source=ROOT/spec["asset"],ROOT/spec["source"]
 assert hashlib.sha256(asset.read_bytes()).hexdigest()==spec["sha"]
 assert hashlib.sha256(source.read_bytes()).hexdigest()==spec["source_sha"]
 if "page" in spec:
  with tempfile.TemporaryDirectory() as tmp:
   prefix=Path(tmp)/"source"
   subprocess.run(["pdftoppm","-f",str(spec["page"]),"-l",str(spec["page"]),
                   "-r","360","-singlefile","-png",str(source),str(prefix)],check=True)
   with Image.open(str(prefix)+".png") as full,Image.open(asset) as published:
    assert list(full.size)==spec["page_size"]
    expected=full.convert("RGB").crop(spec["box"])
    actual=published.convert("RGB")
    assert expected.size==actual.size
    assert ImageChops.difference(expected,actual).getbbox() is None
 else:
  assert asset.read_bytes()==source.read_bytes()
 if "size" in spec:
  with Image.open(asset) as im:
   im.load()
   assert list(im.size)==spec["size"]
 print("PASS source pixels/bytes",asset.name)
print("PASS six original positional figures; no authored replacement diagrams")
