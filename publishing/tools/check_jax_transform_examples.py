"""Execute the exact printed JAX/PyTorch transformation example on CPU."""
from pathlib import Path
from hashlib import sha256
import ast
import re

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md'
text = chapter.read_text()
pattern = r'<!-- BEGIN REVIEW: JAX transforms 2026-10-27 -->\n(.*?)<!-- END REVIEW: JAX transforms 2026-10-27 -->\n\n'
addition = re.search(pattern, text, re.S)
assert addition is not None
baseline = text[:addition.start()] + text[addition.end():]
assert sha256(baseline.encode()).hexdigest() == 'f4a436da9642cc59dfec871c9e2134119871e81e7993b0cc2dcd68af68d31b82'
blocks = re.findall(r'```python\n(.*?)\n```', addition.group(1), re.S)
assert len(blocks) == 5
source = (ROOT / '05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
original = source.split('{#lst-jax-transformations', 1)[1].split('```{.python}\n', 1)[1].split('```', 1)[0].strip()
assert blocks[0].strip() == original
scope = {'__name__': '__jax_chapter_example__'}
for index, block in enumerate(blocks, 1):
    ast.parse(block)
    exec(compile(block, f'{chapter}:JAX-block-{index}', 'exec'), scope)
    print(f'actual JAX/PyTorch block {index}: PASS')
assert jax.__version__ == jaxlib.__version__ == '0.12.0.1'
assert torch.__version__ == '2.14.1'
assert all(device.platform == 'cpu' for device in jax.devices())
assert jax.config.jax_enable_x64
assert scope['params']['w'].dtype == jnp.float64

# Check numerical values independently of the two AD implementations.
point = scope['params']
xx, yy = scope['jax_x'], scope['jax_y']
np.testing.assert_allclose(np.asarray(xx) @ np.asarray(point['w']) + float(point['b']), [-.5, -.5])
np.testing.assert_allclose([2.25, .25], np.square(np.asarray([-.5, -.5]) - np.asarray(yy)))
np.testing.assert_allclose(scope['mean_gradient']['w'], [0., -1.], atol=1e-14)
np.testing.assert_allclose(scope['wrong_gradient']['w'], [-2., -3.])
assert scope['per_example']['w'].shape == (2, 2)
assert scope['per_example']['b'].shape == (2,)
assert scope['mean_gradient']['w'].shape == (2,)
assert scope['mean_gradient']['b'].shape == ()
wrong_update = jax.tree.map(lambda p, g: p - .05*g, point, scope['per_example'])
assert wrong_update['w'].shape == (2, 2) and wrong_update['b'].shape == (2,)

loss_fn = scope['loss_fn']
grad_fn = scope['grad_fn']
batched_grad = scope['batched_grad']
compiled = scope['fast_batched_grad']
torch_grad_fn = scope['torch_grad_fn']
for batch in (1, 2, 7):
    for features in (1, 2, 5, 8):
        rng = np.random.default_rng(100*batch + features)
        w = rng.normal(size=features)
        b = rng.normal()
        x = rng.normal(size=(batch, features))
        y = rng.normal(size=batch)
        params = {'w': jnp.array(w), 'b': jnp.array(b)}
        jx, jy = jnp.array(x), jnp.array(y)
        individual = batched_grad(params, jx, jy)
        aggregate = grad_fn(params, jx, jy)
        compiled_individual = jax.block_until_ready(compiled(params, jx, jy))
        errors = x @ w + b - y
        expected = {'w': 2*errors[:, None]*x, 'b': 2*errors}
        # NumPy arrays retain float64 in torch.tensor, while a Python float
        # otherwise takes PyTorch's default float32. Keep every leaf in the
        # independently generated case at the precision printed in the chapter.
        tp = {'w': torch.tensor(w, dtype=torch.float64),
              'b': torch.tensor(b, dtype=torch.float64)}
        tx, ty = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        assert all(value.dtype == torch.float64 for value in (*tp.values(), tx, ty))
        torch_individual = torch.func.vmap(torch_grad_fn, in_dims=(None, 0, 0))(tp, tx, ty)
        torch_aggregate = torch_grad_fn(tp, tx, ty)
        for name in ('w', 'b'):
            np.testing.assert_allclose(individual[name], expected[name], atol=2e-13, rtol=1e-12)
            np.testing.assert_allclose(compiled_individual[name], expected[name], atol=2e-13, rtol=1e-12)
            np.testing.assert_allclose(aggregate[name], expected[name].mean(0), atol=2e-13, rtol=1e-12)
            np.testing.assert_allclose(torch_individual[name].numpy(), expected[name], atol=2e-13, rtol=1e-12)
            np.testing.assert_allclose(torch_aggregate[name].numpy(), expected[name].mean(0), atol=2e-13, rtol=1e-12)
            assert tp[name].grad is None

print('PASS: exact Harvard source fence; five actual printed blocks; dictionary/leaf shapes; '
      'individual and aggregate gradients; compiled CPU agreement; functional update and '
      'unchanged input parameters; matching-value/wrong-shape negative control; '
      '12 independent NumPy/JAX/PyTorch cases. No speedup or multiple-device claim.')
print('versions:', jax.__version__, jaxlib.__version__, torch.__version__, 'devices:', jax.devices())
print('chapter_sha256=' + sha256(chapter.read_bytes()).hexdigest())
