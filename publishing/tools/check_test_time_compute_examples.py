"""Run every Python fence in the TTC chapter and independent CPU checks.

uv run --offline --with sympy==1.14.0 python publishing/tools/check_test_time_compute_examples.py
No LLM endpoint, pretrained weights, GPU, full benchmark or performance claim.
The saved ToT functions run with a deterministic replay adapter, not an LLM.
"""
from __future__ import annotations

import ast
from collections import Counter
import contextlib
from fractions import Fraction
from functools import lru_cache, partial
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import sympy

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md'
ARCHIVE = ROOT / '05 Источники/Originals/test-time-compute-2026-10-25'


def rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


@lru_cache(None)
def reachable(values):
    """Independent exhaustive arithmetic, not the paper's LLM value model."""
    if len(values) == 1:
        return values[0] == 24
    for i, j in itertools.combinations(range(len(values)), 2):
        a, b = values[i], values[j]
        rest = [v for k, v in enumerate(values) if k not in (i, j)]
        results = {a+b, a-b, b-a, a*b}
        if b:
            results.add(a/b)
        if a:
            results.add(b/a)
        for value in results:
            if reachable(tuple(sorted([*rest, value]))):
                return True
    return False


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```\s*$', text, re.M | re.S)
    assert len(blocks) == 7
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, block in enumerate(blocks, 1):
            exec(compile(block, f'{CHAPTER}:python-{index}', 'exec'), namespace)
    pass_k = namespace['pass_at_k']
    majority = namespace['strict_majority_probability']
    mode = namespace['select_mode']
    weighted = namespace['weighted_vote']
    expand = namespace['expand_frontier']
    multiplier = namespace['inference_multiplier']

    # Independent controls for the printed Coconut feedback and curriculum.
    latent_loss = namespace['coconut_toy_loss']
    pack_stage = namespace['coconut_stage']
    for index in range(-20, 21):
        a, eps = index / 10, 1e-6
        delta = (a * a + 1) / 2
        expected_gradient = -a / (1 + math.exp(delta))
        observed = (latent_loss(a + eps) - latent_loss(a - eps)) / (2 * eps)
        assert math.isclose(observed, expected_gradient, abs_tol=1e-8)
    steps = [['r1a', 'r1b'], ['r2'], ['r3']]
    for stage, ratio in itertools.product(range(4), range(1, 5)):
        tokens, labels = pack_stage(['Q0', 'Q1'], steps, ['A'], stage, ratio)
        prefix_length = 2 + 2 + stage * ratio
        expected_suffix = sum(steps[stage:], []) + ['A', '<eos>']
        assert tokens.count('<latent>') == stage * ratio
        assert labels[:prefix_length] == [-100] * prefix_length
        assert labels[prefix_length:] == expected_suffix
        assert tokens[prefix_length:] == expected_suffix
    for stage, ratio in [(-1, 2), (4, 2), (True, 2), (1, 0)]:
        rejects(pack_stage, ['Q'], steps, ['A'], stage, ratio)

    # All subsets of small finite candidate sets, then iid unbiasedness.
    subset_cases = 0
    unbiased_cases = 0
    for n in range(1, 11):
        for c in range(n+1):
            for k in range(1, n+1):
                subsets = list(itertools.combinations(range(n), k))
                exact = Fraction(sum(any(index < c for index in s) for s in subsets), len(subsets))
                assert math.isclose(pass_k(n, c, k), float(exact), abs_tol=1e-15)
                subset_cases += 1
        for k in range(1, n+1):
            for p in (Fraction(0), Fraction(1, 10), Fraction(3, 10), Fraction(1, 2), Fraction(1)):
                expected = sum(Fraction(math.comb(n, c))*p**c*(1-p)**(n-c)
                               * Fraction(math.comb(n,k)-(math.comb(n-c,k) if n-c >= k else 0),
                                          math.comb(n,k)) for c in range(n+1))
                assert expected == 1-(1-p)**k
                unbiased_cases += 1
    assert pass_k(10**6, 1, 1) == 1e-6
    assert math.isclose((1-.1**4 + 1-.9**4)/2, .6719)
    assert math.isclose(1-.5**4, .9375)
    assert math.isclose(1-.7**4, .7599)
    for args in [(0,0,1), (10,11,1), (10,-1,1), (10,1,11), (10,1,0),
                 (True,0,1), (10,False,1), (10,1,1.0)]:
        rejects(pass_k, *args)

    majority_cases = 0
    for k in (1,3,5,7,9):
        for p in (Fraction(0), Fraction(1,10), Fraction(3,10), Fraction(1,2), Fraction(3,5), Fraction(9,10), Fraction(1)):
            exact = sum(p**sum(bits)*(1-p)**(k-sum(bits))
                        for bits in itertools.product((0,1), repeat=k) if sum(bits) > k//2)
            assert math.isclose(majority(float(p), k), float(exact), abs_tol=2e-14)
            majority_cases += 1
    assert 0 <= majority(.51,1001) <= 1
    for args in [(float('nan'),3), (float('inf'),3), (-.1,3), (1.1,3),
                 (True,3), (.5,True), (.5,2), (.5,0)]:
        rejects(majority,*args)

    mode_cases = 0
    for n in range(8):
        for answers in itertools.product(('A','B','C'),repeat=n):
            counts = Counter(answers)
            winners = [a for a,c in counts.items() if c == max(counts.values())] if counts else []
            expected = winners[0] if len(winners)==1 else None
            assert mode(answers) == expected
            assert mode(reversed(answers)) == expected
            mode_cases += 1
    rejects(mode,[None])
    assert weighted([]) == (None,{})
    rows = [(4,.7),(8,.9),(4,.75),(5,.2)]
    for permutation in itertools.permutations(rows):
        assert weighted(permutation) == weighted(rows)
    assert weighted([(4,Fraction(1,3)),(8,Fraction(1,6)),(8,Fraction(1,6))])[0] is None
    for bad in (True,'0.5',float('nan'),float('inf'),-.1,1.1):
        rejects(weighted,[(4,bad)])
    rejects(weighted,[(None,.5)])
    assert namespace['frontier'] == ['A\nA1\n','A\nA2\n']
    assert expand([],lambda _:[],lambda _:0,2) == ([],[])
    assert expand(['x'],lambda _:['B','A'],lambda _:1,1)[0] == ['B']
    rejects(expand,['x'],lambda _:['y'],lambda _:float('nan'),1)
    rejects(expand,['x'],lambda _:['y'],lambda _:True,1)
    rejects(expand,[],lambda _:[],lambda _:0,False)

    # Execute the actual pinned upstream search functions without importing tot.models.
    hashes = {'tot-bfs.py':'1fea761421857120e14e003c07411b5943fb5a87bf4b743691cf55f5e2d9a7b9',
              'tot-game24.py':'09d01c6a6c44da339019212a3bc0006e356c96641c3c1b28807b8a3508c0b540'}
    for file, digest in hashes.items():
        assert hashlib.sha256((ARCHIVE/file).read_bytes()).hexdigest() == digest
    calls = []
    branches, scores = namespace['branches'], namespace['branch_scores']

    def replay_gpt(prompt, n, stop=None, **kwargs):
        kind, prefix = prompt
        calls.append((kind,prefix,n))
        if kind == 'propose':
            assert n == 1
            return ['\n'.join(branches[prefix])]
        assert kind == 'value'
        return [str(scores[prefix]/n)]*n

    class ReplayTask:
        steps = 2
        stops = ['\n']*2

        def __init__(self):
            self.value_cache = {}

        def get_input(self,index):
            return 'fixed replay task'

        def propose_prompt_wrap(self,x,y):
            return ('propose',y)

        def value_prompt_wrap(self,x,y):
            return ('value',y)

        def value_outputs_unwrap(self,x,y,outputs):
            return math.fsum(map(float,outputs))

    tree = ast.parse((ARCHIVE/'tot-bfs.py').read_text())
    functions = ast.fix_missing_locations(ast.Module(body=[node for node in tree.body
                                      if isinstance(node,ast.FunctionDef)],type_ignores=[]))
    scope = {'itertools':itertools,'partial':partial,'gpt':replay_gpt}
    exec(compile(functions,str(ARCHIVE/'tot-bfs.py'), 'exec'),scope)
    args = SimpleNamespace(backend='no-network-replay',temperature=0,
                           method_generate='propose',method_evaluate='value',
                           method_select='greedy',n_evaluate_sample=3,n_select_sample=2)
    with contextlib.redirect_stdout(output):
        selected, info = scope['solve'](args,ReplayTask(),0,to_print=False)
    assert selected == namespace['frontier']
    assert [len(s['new_ys']) for s in info['steps']] == [2,4]
    assert sum(kind=='propose' for kind,_,_ in calls) == 3
    assert sum(kind=='value' for kind,_,_ in calls) == 6
    assert all(n==3 for kind,_,n in calls if kind=='value')
    # Exact trajectory repeats get zero; identical value prompts reuse the task cache.
    task, y = ReplayTask(), 'A\n'
    before = len(calls)
    assert scope['get_values'](task,'x',[y,y],3) == [.7,0]
    assert len(calls)==before+1
    assert scope['get_values'](task,'x',[y],3) == [.7]
    assert len(calls)==before+1

    game_tree = ast.parse((ARCHIVE/'tot-game24.py').read_text())
    game_class = next(node for node in game_tree.body if isinstance(node,ast.ClassDef))
    wanted = [node for node in game_class.body if isinstance(node,ast.FunctionDef)
              and node.name in ('test_output','value_outputs_unwrap')]
    for node in wanted:
        node.decorator_list=[]
    game_scope={'re':re,'sympy':sympy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=wanted,type_ignores=[])),
                 str(ARCHIVE/'tot-game24.py'), 'exec'),game_scope)
    task = SimpleNamespace(data=['4 9 10 13'])
    assert game_scope['test_output'](task,0,'Answer: (10-4)*(13-9)=24') == {'r':1}
    # Known, trusted arithmetic string proves the checker does not enforce only + - * /.
    assert game_scope['test_output'](task,0,'Answer: factorial(4)*(13-10)/sqrt(9)=24') == {'r':1}
    assert game_scope['test_output'](task,0,'Answer: 24') == {'r':0}
    assert game_scope['value_outputs_unwrap']('x','step\n',['sure','likely','impossible']) == 21.001
    for values, answer in [((4,9,10,13),True),((4,10,22),False),((10,13,13),False),((4,6),True)]:
        assert reachable(tuple(sorted(map(Fraction,values)))) == answer

    # Equality of total training+inference costs, not equality of model quality.
    for m in range(1,16):
        for r in (Fraction(1,10),Fraction(1),Fraction(10)):
            extra=multiplier(m,r)
            x,y=Fraction(6),2*r
            assert x+extra*y == m*(x+y)
    for args in [(0,1),(4,0),(4,-1),(True,1),(4,True),(4,.1)]:
        rejects(multiplier,*args)
    assert 56+14+30 == 100 and 56/70 == .8
    assert math.isclose(57/82,.6951219512195121)
    assert 4*(400+100)==2000 and 8*(400+100)==4000
    assert math.isclose(1-.99**10,.09561792499119559)
    assert math.isclose(1-.99**100,.6339676587267709)
    assert 16*(3+1)==64 and 8*8==64

    print(json.dumps({'python_fences':len(blocks),'subset_cases':subset_cases,
                      'coconut_feedback_gradient_cases':41, 'coconut_curriculum_cases':16,
                      'coconut_rejected_inputs':4,
                      'unbiased_cases':unbiased_cases,'majority_cases':majority_cases,
                      'mode_cases':mode_cases,'upstream_replay':{'propose_calls':3,'value_calls':6,'value_outputs':18},
                      'stdout':output.getvalue().splitlines(),
                      'chapter_sha256':hashlib.sha256(text.encode()).hexdigest()},indent=2))
    print('PASS chapter fences, probabilities, actual ToT replay and cost identity; not an LLM benchmark')


if __name__=='__main__':
    main()
