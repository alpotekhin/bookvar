"""Execute the complete evaluation chapter and replay pinned sacreBLEU cores.

Only native count/score functions are executed with a metadata-only Score box.
This is not an installed sacreBLEU package, tokenizer/CLI run, BERT inference
or a translation benchmark. Explicit token sequences isolate metric arithmetic.
"""
import ast
from collections import Counter
import contextlib
from fractions import Fraction
import hashlib
import io
import itertools
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md'
PACKET = ROOT / '05 Источники/Originals/bleu-reconciliation-2026-10-30'


class ScoreBox:
    # No scoring behavior: native BLEUScore/CHRFScore only need these fields.
    def __init__(self, name, score):
        self.name, self.score = name, score


def load_native():
    scope = dict(globals(), Score=ScoreBox, Signature=object, Metric=object)
    exec(compile((PACKET / 'sacrebleu-2.5.1-helpers.py').read_text(),
                 'native-helpers', 'exec'), scope)
    tree = ast.parse((PACKET / 'sacrebleu-2.5.1-utils.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'my_log', 'sum_of_lists'}]
    assert len(functions) == 2
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'native-utils', 'exec'), scope)
    for file in ['sacrebleu-2.5.1-bleu.py', 'sacrebleu-2.5.1-chrf.py']:
        nodes = [node for node in ast.parse((PACKET / file).read_text()).body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), scope)
    return scope


def native_bleu(scope, hypotheses, reference_sets, order, smoothing, effective):
    cls = scope['BLEU']
    metric = cls.__new__(cls)
    metric.max_ngram_order = order
    metric.smooth_method = smoothing
    metric.smooth_value = None
    metric.effective_order = effective
    stats = [metric._compute_segment_statistics(
        ' '.join(hyp), metric._extract_reference_info([' '.join(ref) for ref in refs]))
        for hyp, refs in zip(hypotheses, reference_sets)]
    return metric._aggregate_and_compute(stats)


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r'^' + chr(96)*3 + r'python\n(.*?)^' + chr(96)*3 + r'\s*$', text, re.M | re.S)
    assert len(blocks) == 12
    scope, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, block in enumerate(blocks):
            exec(compile(block, f'{CHAPTER}:python-{index+1}', 'exec'), scope)

    toy = scope['bleu_from_tokens']
    native = load_native()
    # Every binary token string up to length four, including the empty string.
    strings = [list(tokens) for size in range(5)
               for tokens in itertools.product(['a', 'b'], repeat=size)]
    cases = 0
    for hyp, ref, order, smoothing, effective in itertools.product(
            strings, strings, [2, 4], ['none', 'exp'], [False, True]):
        ours = toy([hyp], [[ref]], order, smoothing, effective)
        upstream = native_bleu(native, [hyp], [[ref]], order, smoothing, effective)
        assert ours['matches'] == upstream.counts
        assert ours['totals'] == upstream.totals
        assert ours['candidate_length'] == upstream.sys_len
        assert ours['reference_length'] == upstream.ref_len
        assert math.isclose(ours['bp'], upstream.bp, abs_tol=1e-12)
        assert math.isclose(ours['score'], upstream.score, abs_tol=1e-10)
        for a, b in zip(ours['precisions'], upstream.precisions):
            assert math.isclose(100*a, b, abs_tol=1e-10)
        cases += 1

    # Maximum reference count is not a sum across reference sentences.
    repeated = toy([['the']*7], [[['the','cat','is','on','the','mat'],
                                 ['there','is','a','cat','on','the','mat']]])
    assert repeated['matches'][0] == 2
    pooled = toy([['a'], ['b']], [[[ 'b' ]], [[ 'a' ]]], max_order=1)
    assert pooled['matches'] == [0] and pooled['score'] == 0
    six = list('abcdef')
    shorter = [list('abcd')]
    longer = shorter + [list('uvwxyzq')]
    before, after = toy([six], [shorter]), toy([six], [longer])
    assert after['matches'] == before['matches']
    assert after['reference_length'] == 7 and before['reference_length'] == 4
    assert math.isclose(after['score'] / before['score'], math.exp(-1/6))
    native_after = native_bleu(native, [six], [longer], 4, 'none', False)
    assert math.isclose(after['score'], native_after.score, abs_tol=1e-12)
    # Closest-length ties select the shorter reference.
    metric = native['BLEU'].__new__(native['BLEU'])
    assert metric._get_closest_ref_len(6, [8, 4]) == 4

    chrf = native['CHRF'].__new__(native['CHRF'])
    chrf.beta, chrf.char_order, chrf.word_order, chrf.order = 2, 2, 0, 2
    chrf.whitespace, chrf.eps_smoothing = False, False
    ref = chrf._extract_reference_info(['cats'])
    stats = chrf._compute_segment_statistics('cat', ref)
    assert stats == [3,4,3,2,3,2]
    assert math.isclose(chrf._compute_score_from_stats(stats).score, 100*85/113, abs_tol=1e-12)

    weighted = Fraction(2095201, 2781000)
    assert math.isclose(scope['weighted_recall'], float(weighted), abs_tol=1e-12)
    assert math.isclose(scope['f1'], float(Fraction(62238, 78950)), abs_tol=1e-12)
    # One candidate vector can be the best match for more than one reference.
    assert [max(range(4), key=lambda j: row[j]) for row in scope['similarity']] == [0,2,1,2,3]

    fig = json.loads((ROOT / '00 Учебник/Assets/Figures/curated/bleu-reconciliation-2026-10-30/assets.json').read_text())[0]
    original, copy = (ROOT / fig['source_asset']).read_bytes(), (ROOT / fig['asset']).read_bytes()
    assert original == copy
    assert hashlib.sha256(copy).hexdigest() == fig['sha256']
    assert (ROOT / fig['license_evidence']).read_bytes() == (ROOT / fig['public_license_copy']).read_bytes()
    print(json.dumps({'python_fences':len(blocks), 'native_bleu_cases':cases,
                      'chrf_two_order_score':100*85/113,
                      'original_figure_weighted_recall':float(weighted),
                      'candidate_pooling_rejected':True, 'reference_nonmonotonicity_verified':True,
                      'figure_byte_identical':True, 'installed_sacrebleu_package':False,
                      'tokenizer_cli_run':False, 'bert_encoder_run':False,
                      'chapter_sha256':hashlib.sha256(text.encode()).hexdigest(),
                      'stdout':output.getvalue().splitlines()},ensure_ascii=False))


if __name__ == '__main__':
    main()
