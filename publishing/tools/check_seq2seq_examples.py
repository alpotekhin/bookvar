#!/usr/bin/env python3
"""Execute published seq2seq blocks; independently check masks, state and search."""
from pathlib import Path
import hashlib
import math
import re
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, re.S)
assert len(blocks) == 3, len(blocks)
namespaces = []
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    namespaces.append(namespace)
    print("PASS standalone published block", number)

n = namespaces[0]
model, src, lengths = n["model"], n["src"], n["lengths"]
inputs, targets, PAD = n["dec_input"], n["targets"], n["PAD"]
assert n["after"] < .01 and n["after"] < n["before"] / 100
assert targets.ne(PAD).sum().item() == 14
for i, (_, answer) in enumerate(n["pairs"]):
    ids, ended = model.greedy(src[i:i+1], lengths[i:i+1])
    assert ended and ids == [n["tgt_vocab"][w] for w in answer.split()]
assert model.greedy(src[:1], lengths[:1], max_steps=0) == ([], False)
hidden, context = model.encode(src, lengths)
assert hidden.shape == (2, 4, 16) and context.shape == (4, 16)
logits = model(src, lengths, inputs)
assert logits.shape == (4, 4, 7)
manual = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
expected = manual[targets.ne(PAD)].mean()
torch.testing.assert_close(expected, n["objective"]())
logits.retain_grad()
F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD).backward()
assert logits.grad[targets.eq(PAD)].abs().sum() == 0
assert logits.grad[targets.eq(n["EOS"])].abs().sum() > 0
# Padding targets are irrelevant even if their corresponding scores change.
changed = logits.detach().clone()
changed[targets.eq(PAD)] = torch.arange(7.) * 100
torch.testing.assert_close(
    F.cross_entropy(changed.flatten(0, 1), targets.flatten(), ignore_index=PAD),
    expected.detach(),
)
# One-step teacher forcing must reproduce the complete decoder call.
step_hidden, fixed = model.encode(src, lengths)
chunks = []
for t in range(inputs.size(1)):
    out, step_hidden = model.decode(inputs[:, t:t+1], step_hidden, fixed)
    chunks.append(out)
torch.testing.assert_close(torch.cat(chunks, 1), logits)
torch.testing.assert_close(fixed, context)
# Packing has the same final state as separate unpadded forwards.
for i, length in enumerate(lengths.tolist()):
    single, _ = model.encode(src[i:i+1, :length], torch.tensor([length]))
    torch.testing.assert_close(single, hidden[:, i:i+1])
# Appending arbitrary right-padding does not alter the packed state.
padded = F.pad(src, (0, 5), value=PAD)
torch.testing.assert_close(model.encode(padded, lengths)[0], hidden)

# Test gradient channels on a fresh model, not on the near-converged one.
torch.manual_seed(37)
fresh = n["Seq2Seq"]().double()
def encoder_gradient(detach_initial, detach_context):
    fresh.zero_grad(set_to_none=True)
    initial, fixed = fresh.encode(src, lengths)
    if detach_initial:
        initial = initial.detach()
    if detach_context:
        fixed = fixed.detach()
    out, _ = fresh.decode(inputs, initial, fixed)
    F.cross_entropy(out.flatten(0, 1), targets.flatten(), ignore_index=PAD).backward()
    return sum(
        p.grad.abs().sum().item()
        for p in fresh.encoder.parameters() if p.grad is not None
    )
norms = {(a,b): encoder_gradient(a,b) for a in (False,True) for b in (False,True)}
assert norms[False,False] > 0
assert norms[True,False] > 0 and norms[False,True] > 0
assert norms[True,True] == 0
# A zero PAD embedding does not make an ordinary GRU ignore padding.
_, ordinary = fresh.encoder(fresh.src_emb(src))
packed = fresh.encode(src, lengths)[0]
assert (ordinary[:, 1] - packed[:, 1]).abs().max() > 1e-4
print("PASS EOS/PAD, weighted loss, state shapes, packing, step equivalence, both encoder gradient paths", norms)

b = namespaces[1]
answers = b["all_answers"]
assert len(answers) == 6
expected_probs = {
    ("A", "<eos>"): .24, ("A", "A", "<eos>"): .18,
    ("A", "B", "<eos>"): .18, ("B", "<eos>"): .36,
    ("B", "A", "<eos>"): .02, ("B", "B", "<eos>"): .02,
}
for prefix, score in answers:
    assert math.isclose(math.exp(score), expected_probs[prefix])
assert math.isclose(math.exp(b["greedy"][1]), .24)
assert math.isclose(math.exp(b["beam"][1]), .36)
assert b["beam_search"](2, max_steps=0) is None
assert .5*.3*.6*.6 > .5*.4*.4*.6
assert math.isclose(.5*.4*.4*.6, .048)
assert math.isclose(.5*.3*.6*.6, .054)
assert math.log(.25)/4 > math.log(.30)/2
assert math.isclose(-math.log(.3024), 1.1960046346767592)
# Reversal changes the minimum, not the mean, under monotonic correspondence.
normal_distances = [3, 3, 3]
reversed_distances = [1, 3, 5]
assert sum(normal_distances) == sum(reversed_distances)
print("PASS exhaustive distribution, greedy/beam, EOS truncation, source examples, length normalization")

a = namespaces[2]
torch.testing.assert_close(a["weights"], torch.tensor([.05,.85,.10,0.], dtype=torch.float64))
torch.testing.assert_close(a["context"], torch.tensor([-.05,1.8], dtype=torch.float64))
manual_grad = 2*a["weights"].detach()*(
    (a["states"]-a["context"].detach()) @
    (a["context"].detach()-torch.tensor([0.,2.]))
)
torch.testing.assert_close(a["scores"].grad, manual_grad)
second = torch.tensor([.05,.1,.85,0.], dtype=torch.float64) @ a["states"]
torch.testing.assert_close(second, torch.tensor([-.8,1.05], dtype=torch.float64))
print("PASS attention weighted contexts, padding and independent softmax backward")
print("chapter_sha256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

