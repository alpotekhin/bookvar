"""Execute the exact JAX runtime fences, including a real four-CPU psum.

Run in a fresh process with JAX/JAXlib 0.12.0.1 and NumPy.
This is not a GPU, multi-host, training-quality or performance check.
"""
import ast
import hashlib
import os
from pathlib import Path
import re

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md'
text = chapter.read_text()
pattern = r'<!-- BEGIN REVIEW: JAX runtime 2026-10-27 -->\n(.*?)<!-- END REVIEW: JAX runtime 2026-10-27 -->\n\n'
match = re.search(pattern, text, re.S)
assert match
baseline = text[:match.start()] + text[match.end():]
assert hashlib.sha256(baseline.encode()).hexdigest() == \
    '4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23'
blocks = re.findall(r'```python\n(.*?)\n```', match.group(1), re.S)
scope = {'__name__': '__jax_runtime_chapter__'}
for index, code in enumerate(blocks, 1):
    ast.parse(code)
    exec(compile(code, f'{chapter}:JAX-runtime-{index}', 'exec'), scope)
    print(f'actual JAX runtime fence {index}: PASS')

jax, jnp, np = (scope[key] for key in ('jax', 'jnp', 'np'))
import jaxlib
assert jax.__version__ == jaxlib.__version__ == '0.12.0.1'
assert len(jax.devices()) == 4
assert all(d.platform == 'cpu' for d in jax.devices())
assert scope['broken_values'] == [1, 1, 1]
assert scope['threaded_values'] == [1, 2, 3]
assert len(scope['trace_signatures']) == 4

cases = 0
for n in (1, 2, 3, 7, 16):
    xs = jnp.arange(1, n + 1, dtype=jnp.float32)
    result, cumulative = jax.lax.scan(scope['cumsum_step'], 0., xs)
    np.testing.assert_allclose(result, n * (n + 1) / 2)
    np.testing.assert_allclose(cumulative, np.cumsum(np.arange(1, n + 1)))
    np.testing.assert_allclose(jax.grad(scope['sum_cumulative'])(xs), np.arange(n, 0, -1))
    cases += 1

for number in range(-25, 26):
    expected = number if number > 0 else 2 * number
    np.testing.assert_allclose(scope['compiled_branch'](jnp.float32(number)), expected)
    cases += 1

# A dynamic compiled conditional cannot change its output shape.
def wrong_branch_shape(x):
    return jax.lax.cond(x > 0, lambda z: z, lambda z: z[None], x)

try:
    jax.jit(wrong_branch_shape)(jnp.float32(1.))
except TypeError as error:
    assert 'shapes' in str(error) or 'types' in str(error)
else:
    raise AssertionError('cond accepted unequal branch shapes')

try:
    jax.lax.scan(lambda carry, x: (carry[None], x), jnp.float32(0.), jnp.ones(2))
except TypeError as error:
    assert 'carry' in str(error)
else:
    raise AssertionError('scan accepted a changed carry shape')

@jax.jit
def masked_dynamic_length(x):
    return x[x > 0]

try:
    masked_dynamic_length(jnp.array([-1., 2., 3.]))
except jax.errors.NonConcreteBooleanIndexError:
    pass
else:
    raise AssertionError('jit accepted a data-dependent Boolean-index result size')

# Refs are intercepted state, but several vmap instances cannot write one shared ref.
shared_ref = jax.new_ref(jnp.float32(0.))
def write_shared(x):
    shared_ref[...] = x

try:
    jax.vmap(write_shared)(jnp.arange(3, dtype=jnp.float32))
except Exception as error:
    assert 'unbatched' in str(error) and 'ref' in str(error).lower()
else:
    raise AssertionError('vmap accepted ambiguous writes to a shared ref')

# The same primitive works on new values without replacing the printed function.
rng = np.random.default_rng(27)
for _ in range(40):
    values = rng.integers(-100, 101, size=16, dtype=np.int32)
    result = jax.block_until_ready(scope['f1'](jnp.array(values)))
    expected = values.reshape(4, 4).sum(axis=0)
    np.testing.assert_array_equal(result, expected)
    for shard in result.addressable_shards:
        np.testing.assert_array_equal(shard.data, expected)
    assert len(result.addressable_shards) == 4
    cases += 1

print(f'PASS {len(blocks)} exact printed fences; {cases} independent scan/branch/collective cases;')
print('negative controls: abstract Python if, unequal cond shapes, changing scan carry,')
print('dynamic Boolean-index length, shared-ref vmap writes, donated ownership and committed placement;')
print('four CPU replicas contain the independently summed shard columns. No GPU/network/timing claim.')
print('versions:', jax.__version__, jaxlib.__version__, 'devices:', jax.devices())
print('chapter_sha256=' + hashlib.sha256(chapter.read_bytes()).hexdigest())
