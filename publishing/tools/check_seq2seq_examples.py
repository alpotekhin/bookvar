#!/usr/bin/env python3
"""Execute published seq2seq blocks; independently check masks, state and search."""
from pathlib import Path
import hashlib
import math
import re
import torch
import torch.nn.functional as F
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, re.S)
assert len(blocks) == 5, len(blocks)
namespaces = []
new_namespaces = {}
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    if "state_bridge" in namespace:
        new_namespaces["bridge"] = namespace
    elif "selected_parents" in namespace:
        new_namespaces["parents"] = namespace
    else:
        namespaces.append(namespace)
    print("PASS standalone published block", number)

baseline = before_topic_consolidation(text, str(CHAPTER.relative_to(ROOT)))
assert hashlib.sha256(baseline.encode()).hexdigest() == "f8f44caa2a641441cedbc99c9419de2bbb2ce2567585e2ccde65e48c5f02535b"
old_blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", baseline, re.S)
assert [b for b in blocks if not any(k in b for k in ("state_bridge", "selected_parents"))] == old_blocks
assert len(namespaces) == 3 and len(new_namespaces) == 2

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
# A bridged packed batch must match separate unpadded encoder/decoder passes.
torch.manual_seed(1104)
dtype=torch.float64
bridge_cases=0
for B in (1,2,3):
  for layers in (1,2):
    for m in (1,3):
      for decoder_width in (1,4):
        for T in (2,4):
          encoder=torch.nn.GRU(2,m,num_layers=layers,bidirectional=True,
                              batch_first=True,dtype=dtype)
          bridge=torch.nn.Linear(2*m,decoder_width,dtype=dtype)
          decoder=torch.nn.GRU(2+2*m,decoder_width,batch_first=True,dtype=dtype)
          lengths=torch.tensor([T-(row%T) for row in range(B)])
          valid=torch.arange(T)[None,:]<lengths[:,None]
          x=torch.randn(B,T,2,dtype=dtype)
          x[~valid]=1000.  # These features must never enter the packed encoder.
          x.requires_grad_()
          target=torch.randn(B,1+bridge_cases%3,2,dtype=dtype,requires_grad=True)
          packed=torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
          _,hidden=encoder(packed)
          top=hidden.reshape(layers,2,B,m)[-1]
          context=torch.cat((top[0],top[1]),-1)
          initial=bridge(context).tanh()[None]
          inputs=torch.cat((target,context[:,None].expand(-1,target.shape[1],-1)),-1)
          output,final=decoder(inputs,initial)
          independent_outputs=[];independent_finals=[]
          for row,length in enumerate(lengths.tolist()):
            _,single=encoder(x[row:row+1,:length])
            c=torch.cat((single[-2,0],single[-1,0]))[None]
            h=F.linear(c,bridge.weight,bridge.bias).tanh()[None]
            decoder_input=torch.cat((target[row:row+1],c[:,None].expand(-1,target.shape[1],-1)),-1)
            o,h=decoder(decoder_input,h)
            independent_outputs.append(o)
            independent_finals.append(h)
          reference=torch.cat(independent_outputs,0)
          reference_final=torch.cat(independent_finals,1)
          torch.testing.assert_close(output,reference)
          torch.testing.assert_close(final,reference_final)
          leaves=(x,target,*encoder.parameters(),*bridge.parameters(),*decoder.parameters())
          loss=output.square().sum()+.3*final.square().sum()
          control=reference.square().sum()+.3*reference_final.square().sum()
          gradients=torch.autograd.grad(loss,leaves)
          expected=torch.autograd.grad(control,leaves)
          for actual,truth in zip(gradients,expected):
            torch.testing.assert_close(actual,truth,atol=1e-11,rtol=1e-11)
          assert gradients[0][~valid].count_nonzero()==0
          assert sum(p.numel() for p in bridge.parameters())==decoder_width*(2*m+1)
          tags=torch.arange(2*B*m).reshape(2,B,m)
          if B>1:
            assert not torch.equal(tags.reshape(B,2*m),torch.cat((tags[0],tags[1]),-1))
          bridge_cases+=1
assert bridge_cases==48
print("PASS 48 state-bridge batch/independent-forward/all-gradient, padding and axis cases")

# A selected child inherits its chosen parent, including BOTH LSTM states.
beam_cases=0
for factory in (torch.nn.GRU,torch.nn.LSTM):
  for layers in (1,2):
    for old_width in (1,2,3):
      for children in (1,2,4):
        for m in (1,3):
          network=factory(3,m,num_layers=layers,batch_first=True,dtype=dtype)
          parent_h=torch.randn(layers,old_width,m,dtype=dtype)
          parent_c=torch.randn_like(parent_h) if factory==torch.nn.LSTM else None
          parents=(torch.arange(children).square()+1)%old_width
          inputs=torch.randn(children,2,3,dtype=dtype)
          inherited_h=parent_h.index_select(1,parents)
          inherited_c=parent_c.index_select(1,parents) if parent_c is not None else None
          with torch.no_grad():
            output,state=network(inputs,(inherited_h,inherited_c) if parent_c is not None else inherited_h)
            for row,parent in enumerate(parents.tolist()):
              original_h=parent_h[:,parent:parent+1]
              original_c=parent_c[:,parent:parent+1] if parent_c is not None else None
              control,control_state=network(inputs[row:row+1],
                (original_h,original_c) if parent_c is not None else original_h)
              torch.testing.assert_close(output[row:row+1],control)
              if parent_c is None:
                torch.testing.assert_close(state[:,row:row+1],control_state)
              else:
                for current,truth in zip(state,control_state):
                  torch.testing.assert_close(current[:,row:row+1],truth)
          beam_cases+=1
assert beam_cases==72
local_parents=torch.tensor([[1,1],[0,1]])
global_parents=(torch.arange(2)[:,None]*2+local_parents).flatten()
assert global_parents.tolist()==[1,1,2,3]
assert (global_parents//2).tolist()==[0,0,1,1]
print("PASS 72 GRU/LSTM beam-parent gather cases and source-row identity example")

asset=ROOT/"00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg"
assert hashlib.sha256(asset.read_bytes()).hexdigest()=="bf5083530e10aff826bfc6be26b4b32ccd8ade3e7ad86cc7b7ee66dc5601ddaf"
assert asset.read_bytes()==(ROOT/"05 Источники/Originals/seq2seq-2026-10-21/encoder-decoder.svg").read_bytes()
print("PASS complete original encoder-decoder SVG")
print("chapter_sha256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

