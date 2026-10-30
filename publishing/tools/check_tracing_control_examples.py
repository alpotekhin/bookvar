"""Execute S8's printed branch/trace/compile/export cases on CPU.

No optimizing compiler, CUDA, accelerator deployment or performance is tested.
"""
import ast
import hashlib
import re
import warnings
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/'00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md'
text=CHAPTER.read_text()
text=re.sub(r'<!-- BEGIN REVIEW: JAX runtime 2026-10-27 -->\n.*?<!-- END REVIEW: JAX runtime 2026-10-27 -->\n\n', '', text, flags=re.S)
start='<!-- BEGIN REVIEW: tracing control flow 2026-10-27 -->'
end='<!-- END REVIEW: tracing control flow 2026-10-27 -->'
section=text.split(start)[1].split(end)[0]
previous=text.replace(start+section+end+'\n\n','')
assert hashlib.sha256(previous.encode()).hexdigest()==\
    '61b8e6b36d4484c9d3ba2d920942701af321638e77556b7215fde7dbaa39922b'
blocks=re.findall(r'```python\n(.*?)```',section,re.S)
assert len(blocks)==7
scope={'__name__':'__main__'}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for i,code in enumerate(blocks,1):
        ast.parse(code)
        exec(compile(code,str(CHAPTER)+f':tracing-{i}','exec'),scope)
assert any(isinstance(w.message,torch.jit.TracerWarning) for w in caught)

cases=0
for values in ([a] for a in range(-12,13)):
    value=torch.tensor(values,dtype=torch.float32)
    expected=value*(2 if value.sum().item()>0 else 3)
    torch.testing.assert_close(scope['conditional_forward'](value),expected)
    torch.testing.assert_close(scope['compiled_branch'](value),expected)
    torch.testing.assert_close(scope['traced'](value),value*2)
    actual=scope['exported_module'](value)
    reference=value.sin() if value.sum().item()>0 else value.cos()
    torch.testing.assert_close(actual,reference)
    cases+=1
assert scope['traced'](torch.tensor([-1.])).item()==-2
assert scope['conditional_forward'](torch.tensor([-1.])).item()==-3
assert scope['traced'](torch.tensor([1.,-1.])).tolist()==[2.,-2.]
assert scope['conditional_forward'](torch.tensor([1.,-1.])).tolist()==[3.,-3.]

# Non-strict export still cannot resolve an ordinary Python condition on unknown data.
try:
    torch.export.export(scope['M_old'](),(torch.ones(1),),strict=False)
except Exception as error:
    assert type(error).__name__=='GuardOnDataDependentSymNode',type(error).__name__
    assert 'data-dependent' in str(error)
else:
    raise AssertionError('non-strict export unexpectedly accepted the unknown Python if')

# Data can change; the exported shape is still specialized to length one.
try:
    scope['exported_module'](torch.ones(2))
except AssertionError as error:
    assert 'Guard failed: x.size()[0] == 1' in str(error)
else:
    raise AssertionError('shape-specialized export unexpectedly accepted length two')
code=scope['exported_branch'].graph_module.code
assert 'higher_order.cond' in code
assert hasattr(scope['exported_branch'].graph_module,'true_graph_0')
assert hasattr(scope['exported_branch'].graph_module,'false_graph_0')

print('PASS: seven actual printed fences; 25 additional scalar cases; TracerWarning observed;')
print('negative/zero/balanced branch distinction; check_inputs rejects the bad trace;')
print('eager-backend compile correct; fullgraph rejects; original M_old/M_new export;')
print('non-strict export rejects implicit data branch; shape contract rejects length two.')
print('Chapter SHA-256:',hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
print('PyTorch:',torch.__version__)
print('CPU correctness only; no Inductor codegen, CUDA, deployment or speedup claim.')
