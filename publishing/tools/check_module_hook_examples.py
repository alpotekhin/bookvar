"""Execute the printed module-hook observations, without a GPU/timing claim."""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md'
text = chapter.read_text()
pattern = r'<!-- BEGIN REVIEW: Module hooks 2026-10-27 -->\n(.*?)<!-- END REVIEW: Module hooks 2026-10-27 -->\n\n'
addition = re.search(pattern, text, re.S)
assert addition is not None
baseline = text[:addition.start()] + text[addition.end():]
assert sha256(baseline.encode()).hexdigest() == '5e7aa6cd8ffc7af8ff995943582f13cbcc59c784ae2399801c16b0ee4f7547e0'
blocks = re.findall(r'```python\n(.*?)\n```', addition.group(1), re.S)
assert len(blocks) == 4
original = (ROOT / '05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
original = original.split('::: {#lst-module_hooks ', 1)[1].split('```{.python}\n', 1)[1].split('```', 1)[0].strip()
assert blocks[0].strip() == original
scope = {}
for index, block in enumerate(blocks, 1):
    with redirect_stdout(StringIO()) as output:
        exec(compile(block, f'{chapter}:hook-block-{index}', 'exec'), scope)
    if output.getvalue():
        print(output.getvalue(), end='')
    print(f'actual hook block {index}: PASS')
assert torch.__version__ == '2.14.1'

for input_grad in (False, True):
    for parameter_grad in (False, True):
        for evaluation in (False, True):
            layer, x, records, loss = scope['inspect_linear'](input_grad, parameter_grad, evaluation)
            assert layer.training == (not evaluation)
            assert not layer._forward_hooks and not layer._backward_hooks
            torch.testing.assert_close(loss, x.new_tensor(13.5))
            torch.testing.assert_close(records[0]['output'], x.new_tensor([[.5, 6.5]]))
            assert len(records) == 1 + int(input_grad or parameter_grad)
            if input_grad:
                torch.testing.assert_close(x.grad, x.new_tensor([[4., 5.]]))
                torch.testing.assert_close(records[1]['input_gradient'], x.grad)
            else:
                assert x.grad is None
                if parameter_grad:
                    assert records[1]['input_gradient'] is None
            if parameter_grad:
                torch.testing.assert_close(layer.weight.grad, x.new_tensor([[1., 2.], [2., 4.]]))
                torch.testing.assert_close(layer.bias.grad, x.new_tensor([1., 2.]))
                # Reject the mistaken interpretation of the printed source norm.
                assert not torch.isclose(layer.weight.grad.norm(), records[1]['output_gradient'].norm())
            else:
                assert layer.weight.grad is layer.bias.grad is None

# Inspect the original Sequential example after its printed handles were removed.
# Its first-layer output derivative includes the later layer and the ReLU mask.
model, x = scope['model'], scope['x']
model.zero_grad(set_to_none=True)
seen = []
handle = model[0].register_full_backward_hook(lambda m, gi, go: seen.append((gi, go[0].detach().clone())))
try:
    model(x).sum().backward()
finally:
    handle.remove()
assert len(seen) == 1 and seen[0][0] == (None,)
z = model[0](x).detach()
expected = (torch.ones(32, 5) @ model[2].weight.detach()) * (z > 0)
torch.testing.assert_close(seen[0][1], expected)
torch.testing.assert_close(model[0].weight.grad, expected.T @ x)
assert seen[0][1].shape == (32, 20) and model[0].weight.grad.shape == (20, 10)

# Additional dimensions and derivatives are checked against matrix algebra,
# not against another copy of the same hook implementation.
for batch in (1, 2, 7):
    for din, dout in ((1, 1), (2, 3), (5, 2), (3, 8)):
        generator = torch.Generator().manual_seed(100*batch + 10*din + dout)
        layer = torch.nn.Linear(din, dout).double()
        with torch.no_grad():
            layer.weight.copy_(torch.randn(dout, din, generator=generator, dtype=torch.float64))
            layer.bias.copy_(torch.randn(dout, generator=generator, dtype=torch.float64))
        x = torch.randn(batch, din, generator=generator, dtype=torch.float64, requires_grad=True)
        g = torch.randn(batch, dout, generator=generator, dtype=torch.float64)
        observations = []
        handle = layer.register_full_backward_hook(lambda m, gi, go: observations.append((gi[0].clone(), go[0].clone())))
        try:
            (layer(x)*g).sum().backward()
        finally:
            handle.remove()
        assert len(observations) == 1
        torch.testing.assert_close(observations[0][0], g @ layer.weight.detach())
        torch.testing.assert_close(observations[0][1], g)
        torch.testing.assert_close(layer.weight.grad, g.T @ x.detach())
        torch.testing.assert_close(layer.bias.grad, g.sum(0))

bad_layer = torch.nn.Linear(2, 2).double()
handle = bad_layer.register_full_backward_hook(lambda *args: None)
try:
    bad_output = bad_layer(torch.ones(1, 2, dtype=torch.float64))
    try:
        bad_output.relu_()
    except RuntimeError as error:
        assert 'view' in str(error) and 'inplace' in str(error).replace('in-place', 'inplace')
    else:
        raise AssertionError('Expected in-place mutation of the backward-hook view to fail')
finally:
    handle.remove()

print('PASS: four actual printed blocks; exact Harvard source; inverse 838-line baseline; '
      'eight input/parameter/mode cases; original Sequential downstream gradient; '
      '12 independent matrix cases; direct-forward bypass and removed handles; '
      'no-grad and invalid in-place controls. CPU only, no performance claim.')
print('chapter_sha256=' + sha256(chapter.read_bytes()).hexdigest())
