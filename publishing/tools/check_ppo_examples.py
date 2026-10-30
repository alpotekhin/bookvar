"""Execute the six actual chapter fences, then check independent CPU cases.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_ppo_examples.py
No pretrained weights, downloads, performance or language-quality claims.
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
import transformers

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM.md"


def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 6, "execute every actual published fence"
    ns, output = {}, io.StringIO()
    before = {}
    with contextlib.redirect_stdout(output):
        for i, code in enumerate(blocks):
            if i == 5:
                before = {k: v.detach().clone() for k, v in ns["policy"].state_dict().items()}
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    target, loss = ns["ppo_targets"], ns["ppo_clip_loss"]
    dtype = torch.float64
    for b, variance in [(0., .0384), (.6, .0096), (.4, 0.)]:
        mean, actual = ns["baseline_results"][b]
        assert math.isclose(mean, .24, abs_tol=1e-12)
        assert math.isclose(actual, variance, abs_tol=1e-12)
    z = torch.tensor(math.log(.6/.4), dtype=dtype, requires_grad=True)
    assert math.isclose(torch.autograd.grad(z.sigmoid(), z)[0].item(), .24)
    lp, ref, value, scores, mask = [ns[k] for k in
        ("old_lp", "ref_lp", "old_v", "outcome_scores", "response_mask")]
    expected_r = torch.tensor([[-.02, -.01, .97], [-.01, .49, 0]], dtype=dtype)
    expected_a = torch.tensor([[.499425, .4415, .37], [.3655, .29, 0]], dtype=dtype)
    torch.testing.assert_close(ns["toy_rewards"], expected_r)
    torch.testing.assert_close(ns["toy_advantages"], expected_a)
    torch.testing.assert_close(ns["toy_returns"][1], torch.tensor([.4655, .49, 0], dtype=dtype))
    # Closed-form TD sum, independently of the helper's backward recurrence.
    for gamma in (0., .9, 1.):
        for lam in (0., .5, .95, 1.):
            r, a, ret = target(lp, ref, scores, value, mask, gamma=gamma, lam=lam)
            for row, length in enumerate(mask.sum(-1).tolist()):
                delta = [r[row, t].item() + gamma *
                         (value[row, t+1].item() if t+1 < length else 0) - value[row, t].item()
                         for t in range(length)]
                for t in range(length):
                    expected = sum((gamma*lam)**(k-t)*delta[k] for k in range(t, length))
                    assert math.isclose(a[row, t].item(), expected, abs_tol=1e-12)
                    if lam == 1:
                        mc = sum(gamma**(k-t)*r[row, k].item() for k in range(t, length))
                        assert math.isclose(ret[row, t].item(), mc, abs_tol=1e-12)
    nan_padding = [x.masked_fill(~mask, float("nan")) for x in (lp, ref, value)]
    padded = target(nan_padding[0], nan_padding[1], scores, nan_padding[2], mask)
    for actual, expected in zip(padded, (ns["toy_rewards"], ns["toy_advantages"], ns["toy_returns"])):
        torch.testing.assert_close(actual, expected)
    grad_inputs = [x.detach().clone().requires_grad_() for x in (lp, ref, scores, value)]
    assert all(not x.requires_grad for x in target(*grad_inputs, mask))
    for bad in (torch.zeros_like(mask), torch.tensor([[True, False, True], [True, True, False]])):
        rejects(target, lp, ref, scores, value, bad)
    rejects(target, lp, ref, scores, value, mask.float())
    rejects(target, lp[:, :0], ref[:, :0], scores, value[:, :0], mask[:, :0])
    rejects(target, lp, ref, scores[:, None], value, mask)
    for kwargs in ({"gamma": -1}, {"lam": 1.1}, {"beta": -1}, {"beta": float("nan")}):
        rejects(target, lp, ref, scores, value, mask, **kwargs)

    new = ns["clip_new_logp"].detach().clone().requires_grad_()
    old = ns["clip_old_logp"].detach().clone().requires_grad_()
    adv = ns["clip_advantages"].detach().clone().requires_grad_()
    cm = ns["clip_mask"]
    torch.autograd.gradcheck(lambda x: loss(x, old, adv, cm), (new,))
    grads = torch.autograd.grad(loss(new, old, adv, cm), (new, old, adv), allow_unused=True)
    torch.testing.assert_close(grads[0], ns["expected_gradient"])
    assert grads[1:] == (None, None)
    # Masked NaN/huge logits must be excluded before subtraction/exp.
    padded_loss = loss(torch.cat([new, torch.tensor([[float("nan"), 1e30]], dtype=dtype)], -1),
                       torch.cat([old, torch.zeros((1, 2), dtype=dtype)], -1),
                       torch.cat([adv, torch.zeros((1, 2), dtype=dtype)], -1),
                       torch.cat([cm, torch.zeros((1, 2), dtype=torch.bool)], -1))
    torch.testing.assert_close(padded_loss, ns["clip_loss"])
    rejects(loss, new, old, adv, torch.zeros_like(cm))
    rejects(loss, new, old, adv, cm, eps=1)

    policy, reference, critic = [ns[k] for k in ("policy", "reference", "critic")]
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    assert all(torch.equal(v, before[k]) for k, v in reference.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    for module in (policy.model, policy.lm_head, critic):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    optimizer = ns["ppo_optimizer"]
    assert optimizer.state and all(s["step"].item() == 1 for s in optimizer.state.values())
    for key in ("old_logp", "old_value", "ref_logp", "rewards", "advantages", "returns"):
        assert not ns[key].requires_grad, key
    torch.testing.assert_close(ns["old_logp"], ns["ref_logp"])
    # Each causal prefix reproduces its pre-action hidden state and selected logp.
    ids, P = ns["ids"], ns["prompt_length"]
    max_alignment_error = 0.
    with torch.no_grad():
        full_lp, full_h = ns["response_forward"](policy, ids, P)
        for t in range(3):
            prefix = policy(input_ids=ids[:, :P+t], output_hidden_states=True, use_cache=False)
            h = prefix.hidden_states[-1][:, -1]
            local_lp = prefix.logits[:, -1].log_softmax(-1).gather(-1, ids[:, P+t:P+t+1]).squeeze(-1)
            torch.testing.assert_close(h, full_h[:, t])
            torch.testing.assert_close(local_lp, full_lp[:, t])
            max_alignment_error = max(max_alignment_error, (local_lp-full_lp[:, t]).abs().max().item())
        changed_ids = ids.clone()
        changed_ids[:, P:] = (changed_ids[:, P:] + 1) % 32
        _, changed_h = ns["response_forward"](policy, changed_ids, P)
        torch.testing.assert_close(changed_h[:, 0], full_h[:, 0])
    q = torch.tensor([.6, .4], dtype=dtype)
    kl_terms = (q/torch.tensor([.5, .5], dtype=dtype)).log()
    assert kl_terms[0] > 0 and kl_terms[1] < 0
    assert math.isclose((q*kl_terms).sum().item(), .020135513550688863)
    print(json.dumps({"published_python_fences": len(blocks), "device": "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "parameters": sum(p.numel() for p in policy.parameters()) + sum(p.numel() for p in critic.parameters()),
        "actor_loss": ns["actor_loss"].item(), "value_loss": ns["value_loss"].item(),
        "max_causal_alignment_error": max_alignment_error,
        "example_stdout": output.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print("PASS actual PPO chapter examples; CPU only, no downloads or quality claim")


if __name__ == "__main__":
    main()
