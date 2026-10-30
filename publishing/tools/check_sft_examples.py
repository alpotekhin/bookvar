"""Execute all SFT chapter examples with a tiny real HF Llama, CPU only.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_sft_examples.py
No pretrained weights or network requests are required.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 3, "execute every published Python fence"
    ns = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{index+1}", "exec"), ns)
    assert output.getvalue().strip() == "loss=4.840772, padded=4.840772", output.getvalue()

    model, ids, am, attn = [ns[k] for k in ("model", "input_ids", "assistant_mask", "attention_mask")]
    loss_fn = ns["assistant_only_loss"]
    assert sum(p.numel() for p in model.parameters()) == 6704
    assert ns["optimizer"].state, "the published optimizer step must have executed"
    assert all(state["step"].item() == 1 for state in ns["optimizer"].state.values())
    target_ids = ids[:, 1:][am[:, 1:].bool() & attn[:, 1:].bool()]
    assert target_ids.tolist() == [41, 72, 51, 88, 104]
    assert (ns["padded_assistant"][:, 1:].bool() &
            ns["padded_attention"][:, 1:].bool()).sum().item() == 5

    try:
        loss_fn(model, ids, torch.zeros_like(am), attn)
    except ValueError as error:
        assert "нет целевых токенов" in str(error)
    else:
        raise AssertionError("empty response must fail before backward")

    # Capture per-position embedding ACTIVATIONS, not tied vocabulary weights.
    saved = {}
    def embeddings_hook(module, args, out):
        out.retain_grad()
        saved["embeddings"] = out

    def logits_hook(module, args, out):
        out.logits.retain_grad()
        saved["logits"] = out.logits

    first = model.get_input_embeddings().register_forward_hook(embeddings_hook)
    second = model.register_forward_hook(logits_hook)
    model.zero_grad(set_to_none=True)
    loss_fn(model, ids, am, attn).backward()
    first.remove()
    second.remove()
    gradient = saved["logits"].grad
    assert gradient[:, :4].abs().max().item() == 0
    assert gradient[:, 4:9].abs().sum().item() > 0
    assert gradient[:, 9:].abs().max().item() == 0
    prompt_gradient = saved["embeddings"].grad[:, :5].norm().item()
    assert prompt_gradient > 0

    with torch.no_grad():
        current = loss_fn(model, ids, am, attn)
        without_eos = am.clone()
        without_eos[ids == 104] = 0
        assert without_eos.sum().item() == 4
        assert not torch.isclose(current, loss_fn(model, ids, without_eos, attn), atol=1e-7)
        all_tokens = loss_fn(model, ids, torch.ones_like(am), attn)
        assert not torch.isclose(current, all_tokens, atol=1e-7)

    # The chapter has positive isolation/standalone checks and a leaking control.
    assert ns["allowed"].sum().item() == 8 * 9 // 2 + 5 * 6 // 2
    assert not ns["allowed"][8:, :8].any()
    assert not ns["allowed"][:8, 8:].any()
    packing_error = (ns["isolated"] - ns["standalone"]).abs().max().item()
    leak = (ns["ordinary"] - ns["ordinary_changed"]).abs().max().item()
    assert packing_error < 1e-6 and leak > 1e-5
    assert torch.equal(ns["isolated"], ns["after_change"])

    manual = -sum(math.log(v) for v in (0.5, 0.2, 0.8)) / 3
    assert math.isclose(manual, 0.8419095481, abs_tol=1e-10)
    assert math.isclose(math.exp(manual), 2.3207944168, abs_tol=1e-10)
    assert math.isclose((2 * 0.5 + 8 * 1.5) / 10, 1.3)
    assert math.isclose((0.5 + 1.5) / 2, 1.0)
    assert 9000 * 100 == 1000 * 900 == 900000
    def token_share(q):
        total = q[0] * 100 + q[1] * 900
        return q[1] * 900 / total
    assert math.isclose(token_share([0.5, 0.5]), 0.9)
    assert math.isclose(token_share([0.9, 0.1]), 0.5)
    assert 10000 * 0.5 / 1000 == 5
    assert 8 * 1 * 16 == 128 and 4096 - 4000 == 96

    def majority(attempts):
        counts = Counter(v for v in attempts if v is not None)
        value, count = counts.most_common(1)[0]
        return value if 2 * count > len(attempts) else None
    assert majority([42, 42, 42, 37, None]) == 42
    assert majority([42, 42, 37, 37, 11]) is None

    print(json.dumps({
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "published_python_fences": len(blocks), "parameters": 6704,
        "prompt_embedding_gradient_norm": prompt_gradient,
        "packing_max_error": packing_error, "ordinary_attention_change": leak,
        "nll": manual, "perplexity": math.exp(manual),
    }, indent=2))
    print("PASS published SFT loss/update/packing; EOS, padding, empty targets, prompt gradients, mixture arithmetic")


if __name__ == "__main__":
    main()
