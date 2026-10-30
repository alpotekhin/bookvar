"""Execute all six published DPO fences, with a tiny random CPU Llama.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_dpo_examples.py
No pretrained weights or Hub downloads; not a benchmark or quality evaluation.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/05 DPO.md"


def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 6
    ns, output, before = {}, io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        for i, code in enumerate(blocks):
            if i == 3:
                before = {k: v.detach().clone() for k, v in ns["dpo_policy"].state_dict().items()}
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    seq, loss, forward = [ns[k] for k in ("sequence_logp", "dpo_loss", "model_response_logp")]
    policy, reference = ns["dpo_policy"], ns["dpo_reference"]
    assert all(torch.equal(v, before[k]) for k, v in reference.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    for module in (policy.model, policy.lm_head):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    assert all(s["step"].item() == 1 for s in ns["dpo_optimizer"].state.values())
    assert ns["updated_loss"] < ns["batch_dpo_loss"]
    assert not ns["reference_pair_logp"].requires_grad
    ids, am, labels = [ns[k] for k in ("pair_ids", "pair_attention", "pair_labels")]
    assert labels[:, 1:].ne(-100).sum(-1).tolist() == [3, 2, 2, 4]
    max_padding_error = 0.
    with torch.no_grad():
        full = forward(policy, ids, am, labels)
        for i in range(4):
            valid = am[i].bool()
            one_ids, one_labels = ids[i, valid][None], labels[i, valid][None]
            one_am = torch.ones_like(one_ids)
            expected = forward(policy, one_ids, one_am, one_labels)
            torch.testing.assert_close(expected, full[i:i+1], atol=3e-6, rtol=1e-6)
            for n in (1, 3, 5):
                pad = torch.full((1, n), 2)
                empty_am = torch.zeros_like(pad)
                empty_labels = torch.full_like(pad, -100)
                for parts in [
                    (torch.cat([pad, one_ids], 1), torch.cat([empty_am, one_am], 1), torch.cat([empty_labels, one_labels], 1)),
                    (torch.cat([one_ids, pad], 1), torch.cat([one_am, empty_am], 1), torch.cat([one_labels, empty_labels], 1)),
                ]:
                    actual = forward(policy, *parts)
                    max_padding_error = max(max_padding_error, (actual-expected).abs().item())
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)
            # Removing the genuine EOS removes one negative log probability.
            no_eos = one_labels.clone()
            no_eos[:, -1] = -100
            assert forward(policy, one_ids, one_am, no_eos) > expected
        changed = ids.clone()
        changed[am == 0] = 17
        torch.testing.assert_close(forward(policy, changed, am, labels), full)
        changed[0, 1] = 20
        assert not torch.allclose(forward(policy, changed, am, labels)[0], full[0])
    # Independent shifted gather and gradient: prompt/PAD targets contribute zero.
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-100, -100, 3, 4, 2], [-100, 6, 2, -100, -100]])
    actual = seq(logits, target)
    manual = torch.stack([sum(logits[b, t-1].log_softmax(-1)[target[b, t]]
                             for t in range(1, 5) if target[b, t] != -100) for b in range(2)])
    torch.testing.assert_close(actual, manual)
    torch.autograd.gradcheck(lambda x: seq(x, target), (logits,))
    grad = torch.autograd.grad(actual.sum(), logits)[0]
    for b in range(2):
        for t in range(4):
            if target[b, t+1] == -100:
                assert torch.equal(grad[b, t], torch.zeros(7, dtype=torch.float64))
    assert not grad[:, -1].any()
    rejects(seq, logits, target.float())
    rejects(seq, logits, torch.full_like(target, -100))
    rejects(seq, logits[:, :1], target[:, :1])
    rejects(seq, logits[:0], target[:0])
    rejects(seq, logits, target.masked_fill(target == 3, 7))
    for beta in (0, -1, float("nan"), float("inf")):
        rejects(loss, *[torch.zeros(2)]*4, beta=beta)
    rejects(loss, torch.zeros(2), torch.zeros(2, 1), torch.zeros(2), torch.zeros(2))
    rejects(loss, *[torch.zeros(0)]*4)
    a, b, c, d = [torch.tensor([v], dtype=torch.float64, requires_grad=True) for v in (-3., -4., -3.4, -3.6)]
    value = loss(a, b, c, d)
    da, db, dc, dd = torch.autograd.grad(value, (a, b, c, d), allow_unused=True)
    assert dc is dd is None
    torch.testing.assert_close(da, torch.tensor([-.5/(1+math.exp(.4))], dtype=torch.float64))
    torch.testing.assert_close(db, -da)
    torch.autograd.gradcheck(lambda a, b: loss(a, b, c, d), (a, b))
    torch.testing.assert_close(loss(a+5, b+5, c, d), value)
    torch.testing.assert_close(loss(c, d, c, d), torch.tensor(math.log(2), dtype=torch.float64))
    assert math.isclose(ns["loss"].item(), .5130152524, abs_tol=1e-6)
    # Verify the variational identity on a grid, not just its claimed optimizer.
    q, r, beta = ns["reference_probability"], ns["explicit_reward"], ns["reward_beta"]
    optimum = ns["optimal_probability"]
    trial_p = torch.linspace(.001, .999, 999, dtype=torch.float64)
    trial = torch.stack([trial_p, 1-trial_p], -1)
    objective = (trial*r).sum(-1) - beta*(trial*(trial.log()-q.log())).sum(-1)
    identity = beta*ns["log_partition"] - beta*(trial*(trial.log()-optimum.log())).sum(-1)
    torch.testing.assert_close(objective, identity)
    assert objective.max() <= beta*ns["log_partition"]
    for rb in (.5, 1., 2.):
        torch.testing.assert_close(torch.softmax(q.log()+r/rb, -1)[0],
                                   torch.tensor(1/rb, dtype=torch.float64).sigmoid())
    for rb in (.5, 2.):
        m = torch.tensor(math.log(3)/rb, dtype=torch.float64, requires_grad=True)
        noisy = -.75*F.logsigmoid(rb*m) - .25*F.logsigmoid(-rb*m)
        assert torch.autograd.grad(noisy, m)[0].abs() < 1e-12
    assert 2/100-1/10 == -.08
    pairs = ns["reasoning_pairs"]
    candidates = ns["reasoning_candidates"]
    assert pairs(candidates, 4, 5)[-1] == (candidates[0], candidates[2])
    assert pairs(candidates[:2], 4, 2) == []
    rejects(pairs, candidates, 4, -1)
    assert math.isclose(ns["irpo_pair_loss"].item(), .7130152524, abs_tol=1e-6)
    orpo = -math.log(.2) - .1*math.log(9/13)
    assert math.isclose(orpo, 1.6462103906, abs_tol=1e-9)
    assert math.isclose(F.softplus(torch.tensor(.1)).item(), .74439666, abs_tol=1e-7)
    assert math.isclose(1-torch.tensor(.3).sigmoid().item(), .42555748, abs_tol=1e-7)
    print(json.dumps({"python_fences": len(blocks), "device": "cpu", "torch": torch.__version__,
        "transformers": transformers.__version__, "parameters": sum(p.numel() for p in policy.parameters()),
        "loss_before": ns["batch_dpo_loss"].item(), "loss_after": ns["updated_loss"].item(),
        "padding_comparisons": 24, "max_padding_error": max_padding_error,
        "example_stdout": output.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print("PASS actual DPO fences; CPU only, no downloads or language-quality claim")


if __name__ == "__main__":
    main()
