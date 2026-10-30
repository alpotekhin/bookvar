"""Check printed masked-diffusion programs and independent finite-state controls."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA.md'


def main(output=None):
    text = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    assert len(blocks) == 4
    namespace = {}
    for number, block in enumerate(blocks, 1):
        exec(compile(block, f'masked-diffusion-{number}', 'exec'), namespace)
    reverse = namespace['reverse_token']
    remask = namespace['remask_rates']
    denoise = namespace['toy_denoiser']
    mask = namespace['MASK']

    # Directly enumerate the actual sampler's transition probabilities, not
    # its printed 1/(2T) formula or an empirical frequency over random seeds.
    finite_checks = []
    for steps in [1, 2, 3, 4, 8, 16, 32]:
        states = {(mask, mask): 1.0}
        for remaining in range(steps, 0, -1):
            next_states = {}
            for state, mass in states.items():
                if mask not in state:
                    next_states[state] = next_states.get(state, 0) + mass
                    continue
                predictions = denoise(state)
                transitions = [reverse(z, p, 1 - remaining / steps,
                                       1 - (remaining - 1) / steps)
                               for z, p in zip(state, predictions)]
                for (a, pa), (b, pb) in itertools.product(*(x.items() for x in transitions)):
                    if pa * pb:
                        next_states[a, b] = next_states.get((a, b), 0) + mass * pa * pb
            states = next_states
            assert math.isclose(sum(states.values()), 1.0, abs_tol=1e-12)
        assert all(mask not in state for state in states)
        invalid = sum(p for state, p in states.items() if state not in namespace['DATA'])
        assert math.isclose(invalid, 1 / (2 * steps), abs_tol=1e-12)
        finite_checks.append({'steps': steps, 'invalid_probability': invalid})

    schedule_cases = 0
    for alpha_t in [0, 0.1, 0.25, 0.5, 0.75, 0.9]:
        for fraction in [0, 0.1, 0.5, 0.9, 1]:
            alpha_s = alpha_t + (1 - alpha_t) * fraction
            distribution = reverse(mask, {'a': 0.3, 'b': 0.7}, alpha_t, alpha_s)
            assert math.isclose(sum(distribution.values()), 1)
            assert min(distribution.values()) >= -1e-15
            assert reverse('a', {'b': 1}, alpha_t, alpha_s) == {'a': 1}
            cap = min(1, (1 - alpha_s) / alpha_t) if alpha_t else 1
            for fraction_sigma in [0, 0.25, 0.5, 1]:
                sigma, reveal = remask(alpha_t, alpha_s, cap * fraction_sigma)
                assert -1e-15 <= reveal <= 1 + 1e-15
                visible = alpha_t * (1 - sigma) + (1 - alpha_t) * reveal
                assert math.isclose(visible, alpha_s, abs_tol=1e-12)
                schedule_cases += 1

    # A stream of clean targets cannot read its own answers during BD3-LM
    # denoising. This controls the stated clean/noisy attention boundary.
    n, block = 4, 2
    noisy_access = [[(j // block == i // block) for j in range(n)]
                    + [(j // block < i // block) for j in range(n)]
                    for i in range(n)]
    assert noisy_access[0] == [True, True, False, False, False, False, False, False]
    assert noisy_access[2] == [False, False, True, True, True, True, False, False]
    assert all(not row[n + i] for i, row in enumerate(noisy_access))

    # Nontrivial negative controls: invalid inputs must be rejected, rather
    # than merely making the positive cases continue to pass.
    negatives = [
        lambda: reverse(mask, {'a': 0.4}, 0, 0.5),
        lambda: reverse(mask, {mask: 1}, 0, 0.5),
        lambda: reverse(mask, {'a': 1}, 0.7, 0.2),
        lambda: remask(0.5, 0.75, 0.6),
        lambda: namespace['masked_loss']([0], 0.5, 1),
        lambda: namespace['sample_pair'](0, 1),
    ]
    for check in negatives:
        try:
            check()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    sha = lambda data: hashlib.sha256(data).hexdigest()
    report = {
        'chapter': CHAPTER.relative_to(ROOT).as_posix(),
        'chapter_sha256': sha(CHAPTER.read_bytes()),
        'checker_sha256': sha(Path(__file__).read_bytes()),
        'python_blocks': len(blocks),
        'exact_finite_state_checks': finite_checks,
        'remasking_marginal_cases': schedule_cases,
        'negative_controls': len(negatives),
        'training_stream_leakage_control': True,
        'scope': 'CPU arithmetic and finite-state toy sampler. No neural training, model downloads, GPU timing or language-quality benchmark.'
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if output:
        Path(output).write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    main(parser.parse_args().output)
