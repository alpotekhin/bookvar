"""Execute additions, pin original timm code, and verify mask arithmetic."""
from pathlib import Path
from itertools import product
import ast
import hashlib
import json
import re
import torch

ROOT = Path(__file__).resolve().parents[2]
DROP = ROOT / "00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md"
RU = ROOT / "00 Учебник/05 Attention и Transformer/03 Полный Transformer.md"
EN = ROOT / "en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md"
SOURCE = ROOT / "05 Источники/Originals/structured-dropout-2026-10-29"

def addition(path, name):
    return path.read_text().split(f"<!-- BEGIN REVIEW: {name} 2026-10-29 -->\n")[1].split(
        f"<!-- END REVIEW: {name} 2026-10-29 -->")[0]

drop = addition(DROP, "structured dropout")
attention = addition(RU, "Transformer dropout")
assert attention == addition(EN, "Transformer dropout")
scopes = []
for label, text, count in [("structured", drop, 3), ("attention", attention, 1)]:
    blocks = re.findall(r"\x60{3}python\n(.*?)\x60{3}", text, re.S)
    assert len(blocks) == count
    for number, block in enumerate(blocks, 1):
        scope = {}
        exec(compile(block, f"{label}-{number}", "exec"), scope)
        scopes.append(scope)

original = ast.parse((SOURCE / "timm-drop-beabdc3a.py").read_text())
source_function = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                       and n.name == "drop_path")
our_function = next(n for n in ast.parse(re.findall(r"\x60{3}python\n(.*?)\x60{3}",
                    drop, re.S)[2]).body if isinstance(n, ast.FunctionDef)
                    and n.name == "drop_path")
assert ast.dump(source_function, include_attributes=False) == ast.dump(
    our_function, include_attributes=False)
source_scope = {}
exec(compile(ast.Module(body=[source_function], type_ignores=[]), "original-timm", "exec"),
     source_scope)
cases = 0
for shape in [(8, 3), (8, 3, 4), (8, 2, 3, 4)]:
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape) + 1
    for seed in range(12):
        for p in [0., .1, .5, .9, 1.]:
            for training in [False, True]:
                for scaling in [False, True]:
                    torch.manual_seed(seed)
                    expected = source_scope["drop_path"](x, p, training, scaling)
                    torch.manual_seed(seed)
                    actual = scopes[2]["drop_path"](x, p, training, scaling)
                    torch.testing.assert_close(actual, expected)
                    cases += 1

W = torch.tensor([[1., 2., -.5], [3., -4., 2.]], dtype=torch.float64)
x = torch.tensor([1., -2., .5], dtype=W.dtype)
masks = torch.tensor(list(product([0., 1.], repeat=6)), dtype=W.dtype).reshape(-1, 2, 3)
for q in [.2, .5, .8]:
    probabilities = torch.where(masks.bool(), torch.full_like(masks, q),
                                torch.full_like(masks, 1 - q)).flatten(1).prod(1)
    y = (masks * W / q) @ x
    mean = probabilities @ y
    centered = y - mean
    covariance = centered.T @ (probabilities[:, None] * centered)
    torch.testing.assert_close(mean, W @ x)
    torch.testing.assert_close(covariance,
        torch.diag((1 - q) / q * (W * x).square().sum(1)),
        rtol=1e-12, atol=1e-12)

mha = RU.read_text().split("class MHA(")[1].split("class EncoderBlock(")[0]
assert mha.index(".softmax(-1)") < mha.index("F.dropout(") < mha.index("weights @ v")
assert "F.dropout(weights, self.dropout, self.training)" in mha
print(json.dumps({
    "standalone_new_blocks": 4, "timm_source_comparisons": cases,
    "nonuniform_connection_mask_cases": 3, "attention_locale_parity": True,
    "cpu_only": True, "gpu_tested": False,
    "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [DROP, RU, EN]},
}))
