#!/usr/bin/env python3
"""Run the chapter examples and check mathematical/source-code contracts."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import re
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT/"00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md"
body = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",body,re.S)
assert len(blocks)==8
spaces=[]
torch.set_num_threads(1)
for i,block in enumerate(blocks,1):
    ns={}
    exec(compile(block,f"{CHAPTER}:block{i}","exec"),ns)
    spaces.append(ns)
    print("PASS published standalone block",i)

n=spaces[0]
model=n["model"]
args=[n[x] for x in ("ids","types","valid","chosen")]
model.eval()
with torch.no_grad():
    before=model(*args)
    changed=args[0].clone()
    changed[~n["valid"]]=5
    after=model(changed,*args[1:])
    torch.testing.assert_close(before[0][n["valid"]],after[0][n["valid"]])
    torch.testing.assert_close(before[1],after[1])
    torch.testing.assert_close(before[2],after[2])
    changed=args[0].clone()
    changed[0,8]=4
    right_changed=model(changed,*args[1:])
    assert (before[0][0,2]-right_changed[0][0,2]).abs().max()>1e-8
assert model.encoder.layers[0].self_attn.in_proj_weight is not model.encoder.layers[1].self_attn.in_proj_weight
assert model.word.weight.grad is not None
assert model.encoder.layers[0].self_attn.in_proj_weight.grad.norm()>0
assert n["mlm_input_grad"][0,2].norm()>0
assert torch.count_nonzero(n["mlm_input_grad"][~n["valid"]])==0
torch.testing.assert_close(n["mlm_input_grad"][0,2].norm(),
    torch.tensor(.11868223670664312,dtype=torch.float64))
captured=[]
def keep_gradient(_module,_args,output):
    output.retain_grad()
    captured.append(output)
hook=model.word.register_forward_hook(keep_gradient)
model.zero_grad()
_,mlm,_=model(*args)
F.cross_entropy(mlm,n["targets"]).backward()
hook.remove()
assert captured[0].grad[0,2].norm()>0  # context path, not output tying
assert n["targets"].tolist()==[7] and n["ids"][n["chosen"]].tolist()==[3]
print("PASS full TinyBert forward/backward/update, PAD isolation, right context and indirect MLM gradient")

n=spaces[1]
assert (n["left"],n["right"])==(3,3)
assert n["start"].argmax()==3 and n["end"].argmax()==0
for x in ("start","end"):
    logits=n[x].detach()
    expected=logits.softmax(0)
    expected[3]-=1
    torch.testing.assert_close(n[x].grad,expected)
assert n["scores"][3,3]==5 and torch.isneginf(n["scores"][3,0])
print("PASS constrained span choice, invalid independent argmax and both CE gradients")

n=next(ns for ns in spaces if "disentangled" in ns)
fn=n["disentangled"]
q,k,v,kr,qr=[n[x].clone().requires_grad_() for x in ("q","k","v","kr","qr")]
actual,weights,parts=fn(q,k,v,kr,qr)
loop=[]
for i in range(len(q)):
    row=[]
    for j in range(len(k)):
        f=max(0,min(3,i-j+2))
        rev=max(0,min(3,j-i+2))
        row.append((q[i]@k[j]+q[i]@kr[f]+k[j]@qr[rev])/(6**.5))
    loop.append(torch.stack(row))
loop=torch.stack(loop).softmax(-1)@v
torch.testing.assert_close(actual,loop)
probe=torch.arange(6.,dtype=q.dtype).reshape(3,2)
ag=torch.autograd.grad((actual*probe).sum(),(q,k,v,kr,qr),retain_graph=True)
lg=torch.autograd.grad((loop*probe).sum(),(q,k,v,kr,qr))
for a,b in zip(ag,lg):
    torch.testing.assert_close(a,b)
assert torch.autograd.gradcheck(lambda q,k,v,kr,qr:fn(q,k,v,kr,qr)[0],(q,k,v,kr,qr))
torch.testing.assert_close(weights.sum(-1),torch.ones(3,dtype=q.dtype))
assert torch.count_nonzero(weights)==9
print("PASS DeBERTa three terms, reversed relative index, normalized output and gradients/gradcheck")

n=next(ns for ns in spaces if "attention_weights" in ns and "history" in ns)
memory=n["memory"]
initial=memory+n["positions"]
first_weight=1/(1+torch.exp(torch.tensor(-1/(2**.5),dtype=memory.dtype)))
second_weight=1/(1+torch.exp(-(2*first_weight)/(2**.5)))
torch.testing.assert_close(n["attention_weights"][0][0,0],first_weight)
torch.testing.assert_close(n["attention_weights"][1][0,0],second_weight)
torch.testing.assert_close(n["history"][1][0],torch.stack([
    1+first_weight+second_weight,2-first_weight-second_weight]))
torch.testing.assert_close(memory,torch.eye(2,dtype=memory.dtype))
torch.testing.assert_close(n["history"][1][1],torch.tensor([2.,2.],dtype=memory.dtype))
# Changing K/V between reads is a different graph, not the fixed-memory example.
new_memory=n["history"][0]
wrong_weights=(new_memory@new_memory.T/(2**.5)).softmax(-1)
wrong=new_memory+wrong_weights@new_memory
assert not torch.allclose(wrong,n["history"][1])
print("PASS numerical two-read example, independent logistic formulas and changed-memory negative control")

src=ROOT/"05 Источники/Originals/bert-family-2026-10-22/masked_language_model.py"
assert hashlib.sha256(src.read_bytes()).hexdigest()=="fd37316d3955845b88929fe18d26fc67a6c38a80d4d62057774f9997303d75f4"
excerpt=re.search(r'\x60\x60\x60python title="Original implementation — MIT"\n(.*?)\n\x60\x60\x60',body,re.S)[1]
normalize=lambda s:"\n".join(line.rstrip() for line in s.strip("\n").splitlines())
assert normalize(excerpt) in normalize(src.read_text())
fn=next(x for x in ast.walk(ast.parse(src.read_text())) if isinstance(x,ast.FunctionDef) and x.name=="emd_context_layer")
ns={"torch":torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(src),"exec"),ns)
records=[]
def layer(hidden,mask,return_att,query_states,relative_pos,rel_embeddings):
    records.append((hidden,mask.clone(),query_states.clone()))
    return query_states+hidden*.25
memory=torch.arange(1.,9.).reshape(1,4,2)
states=[memory*10,memory,memory*20]
encoder=SimpleNamespace(layer=[layer],get_rel_embedding=lambda:torch.zeros(4,2))
labels=torch.tensor([[0,1,0,2]])
valid=torch.tensor([[1,1,0,1]])
position=torch.arange(8.).reshape(1,4,2)/10
initial=position.clone()
out=ns["emd_context_layer"](SimpleNamespace(position_biased_input=False),
    states,position,valid,encoder,labels,labels,valid)
assert len(records)==2
for h,m,_ in records:
    assert h is memory
    assert not m[...,2,:].any() and not m[...,:,2].any()
torch.testing.assert_close(records[0][2],initial+memory)
torch.testing.assert_close(records[1][2],initial+memory*1.25)
selected=labels.view(-1)>0
torch.testing.assert_close(out[1],(initial+memory*1.5).reshape(-1,2)[selected])
assert len(out)==2 and out[-1].shape==(2,2)
records.clear()
plain=ns["emd_context_layer"](SimpleNamespace(position_biased_input=True),
    states,initial.clone(),valid,encoder,labels,labels,valid)
assert not records
torch.testing.assert_close(plain[0],states[-1].reshape(-1,2)[selected])
license_source=ROOT/"05 Источники/Originals/bert-family-2026-10-22/deberta-LICENSE"
assert (ROOT/"site/public/licenses/microsoft-deberta-MIT.txt").read_bytes()==license_source.read_bytes()
assert hashlib.sha256(license_source.read_bytes()).hexdigest()=="c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
print("PASS unchanged MIT excerpt and real EMD routing: two calls/shared layer/fixed H/query update/selected rows")
deberta=(ROOT/"Papers/DeBERTa.md").read_text()
roberta=(ROOT/"Papers/RoBERTa.md").read_text()
for old in ["attention вычисляется диагонально по четырём компонентам",
            "добавляющий абсолютные позиции как дополнительный bias непосредственно перед финальным softmax"]:
    assert old not in deberta
for row in ["| -EMD | 86.1/86.1 | 91.8/85.8 | 81.3/78.0 | 70.3 |",
            "| -C2P (no content-to-position) | 85.9/85.7 | 91.6/85.8 | 81.3/78.3 | 69.3 |",
            "| -P2C (no position-to-content) | 86.0/85.8 | 91.7/85.7 | 80.8/77.6 | 69.6 |"]:
    assert row in deberta
glue_row=next(line for line in deberta.splitlines() if line.startswith("| DeBERTa_L |") and len(line.split("|"))==12)
cells=[cell.strip() for cell in glue_row.split("|")[1:-1]]
values=[float(cell.split("/")[0]) for cell in cells[1:9]]
assert abs(sum(values)/8-89.875)<1e-12
assert float(cells[9])==90.00 and abs(sum(values)/8-float(cells[9]))>.1
assert "719/8=89.875" in body and "89.875" in deberta
assert abs((90.7-89.4)-1.3)<1e-12
for fact in ["78.3/78.7","84.3/84.0","92.5/92.9","WordPiece примерно на 30K",
             "к начальному опыту на 100K шагов"]:
    assert fact in roberta
for note in (deberta,roberta):
    assert "03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa и DeBERTa" in note
print("PASS legacy notes: DeBERTa v6 ablations/reported Avg discrepancy and RoBERTa v1 masking/protocol")
print("PASS chapter SHA256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

# Execute the licensed original WWM function without importing its TensorFlow CLI.
from collections import namedtuple
import json

w = next(ns for ns in spaces if "choose_word_targets" in ns)
source = ROOT/"05 Источники/Originals/bert-objectives-2026-10-29/bert-create_pretraining_data.py"
original_function = next(node for node in ast.walk(ast.parse(source.read_text()))
    if isinstance(node, ast.FunctionDef) and node.name == "create_masked_lm_predictions")
original_ns = {
    "FLAGS": SimpleNamespace(do_whole_word_mask=True),
    "MaskedLmInstance": namedtuple("MaskedLmInstance", "index label"),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_function],
    type_ignores=[])), str(source), "exec"), original_ns)

class SpecifiedDraws:
    def __init__(self, draws):
        self.draws = iter(draws)

    def shuffle(self, groups):
        groups[:] = [groups[i] for i in w["order"]]

    def random(self):
        return next(self.draws)

pieces = w["pieces"]
original_wwm = original_ns["create_masked_lm_predictions"]
out, positions, labels = original_wwm(pieces, 2/len(pieces), 2,
    ["unused"], SpecifiedDraws([.1, .9, .1]))
assert positions == [4, 5] == w["targets"]
assert labels == ["play", "##ing"]
assert out[4] == "[MASK]" and out[5] == "##ing"
_, positions_one, _ = original_wwm(pieces, 1/len(pieces), 1,
    ["unused"], SpecifiedDraws([.1]))
assert positions_one == w["choose_word_targets"](pieces, w["order"], 1) == [2]

s = next(ns for ns in spaces if "direct_sbo_gradient" in ns)
gradient = s["direct_sbo_gradient"]
assert [i for i in range(len(gradient)) if gradient[i].norm() > 0] == [3, 8]
assert torch.count_nonzero(gradient[4:8]) == 0
assert s["final_states"].grad[4:8].norm() > 0
assert not torch.allclose(s["sbo_logits"][0], s["sbo_logits"][2])
torch.testing.assert_close(s["same_offset_logits"],
    s["same_offset_logits"][0].expand_as(s["same_offset_logits"]))

distribution = next(ns for ns in spaces if "mean_words" in ns)
probabilities = distribution["length_probabilities"]
assert abs(probabilities[0] - .2/(1-.8**10)) < 1e-14
assert abs(distribution["mean_words"] - 3.797097503983286) < 1e-12
assert abs(distribution["clamped_geometric_mean"] - 4.463129088) < 1e-12

order = next(ns for ns in spaces if "sop_logits" in ns)
assert order["Counter"](order["ordered"]) == order["Counter"](order["swapped"])
assert order["sentence_order_labels"].tolist() == [0, 1]
expected_grad = order["sop_logits"].detach().softmax(-1)
expected_grad[torch.arange(2), order["sentence_order_labels"]] -= 1
torch.testing.assert_close(order["sop_logits"].grad, expected_grad/2)
sop_source = (ROOT/"05 Источники/Originals/bert-objectives-2026-10-29/albert-create_pretraining_data.py").read_text()
sop_excerpt = re.search(r'```python title="Original ALBERT pair swap — Apache-2.0"\n(.*?)\n```',
    body, re.S)[1]
assert sop_excerpt in sop_source
assert "sentence_order_label = 1 if instance.is_random_next else 0" in sop_source
for published, archived in [
    ("google-bert-eedf5716-Apache-2.0.txt", "bert-LICENSE"),
    ("google-albert-b772393d-Apache-2.0.txt", "albert-LICENSE"),
]:
    assert (ROOT/"site/public/licenses"/published).read_bytes() == (
        ROOT/"05 Источники/Originals/bert-objectives-2026-10-29"/archived).read_bytes()
print(json.dumps({
    "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
    "torch": torch.__version__, "published_standalone_python_blocks": len(blocks),
    "original_wwm_target_positions": positions, "selected_unchanged_piece": out[5],
    "sbo_direct_boundary_rows": [3, 8],
    "sbo_final_internal_state_gradient_nonzero": False,
    "combined_mlm_sbo_internal_gradient_nonzero": True,
    "truncated_geometric_mean": distribution["mean_words"],
    "clamped_geometric_mean": distribution["clamped_geometric_mean"],
    "sop_mean_loss": order["sop_loss"].item(),
    "negative_controls": ["partial word selected despite group budget",
      "all selected pieces assumed literal MASK", "SBO head reads internal final states",
      "same boundary input without relative offset", "clamped versus truncated geometric",
      "topic-only bag features distinguish swapped pair"],
    "scope": "CPU computation and original preprocessing function, not BERT/SpanBERT/ALBERT pretraining",
}, indent=2))
