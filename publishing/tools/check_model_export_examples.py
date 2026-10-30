"""Run the visible export programs on CPU; no serving or performance claim."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md'
BEGIN = '<!-- BEGIN REVIEW: model export 2026-10-27 -->'
END = '<!-- END REVIEW: model export 2026-10-27 -->'


def main():
    text = CHAPTER.read_text()
    section = text.split(BEGIN, 1)[1].split(END, 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', section, re.M | re.S)
    assert len(blocks) == 2
    ns = {'__name__': '__model_export_examples__'}
    with contextlib.redirect_stdout(io.StringIO()):
        for index, block in enumerate(blocks):
            exec(compile(block, f'{CHAPTER}:export-block-{index+1}', 'exec'), ns)
    np, torch, ort = ns['np'], ns['torch'], ns['ort']
    session, checked, model = ns['session'], ns['checked_scores'], ns['model']
    assert session.get_providers() == ['CPUExecutionProvider']
    rng = np.random.default_rng(20261027)
    max_error = 0.
    cases = 0
    for size in range(2, 9):
        for _ in range(20):
            x = rng.uniform(-20, 20, (size, 2)).astype(np.float32)
            actual = checked(session, x)
            # Independent scalar equations, not a second call to the model.
            expected = np.maximum(np.stack([
                2 * x[:, 0] - x[:, 1] + .5,
                x[:, 0] + 3 * x[:, 1] - .5], axis=1), 0)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
            max_error = max(max_error, float(np.abs(actual-expected).max()))
            cases += 1
    invalid = [np.ones((1, 2), np.float32), np.ones((9, 2), np.float32),
               np.ones((2, 3), np.float32), np.ones((2, 2), np.float64),
               np.ones(2, np.float32), [[1., 2.], [3., 4.]],
               np.full((2, 2), np.nan, np.float32),
               np.full((2, 2), np.inf, np.float32)]
    for x in invalid:
        try:
            checked(session, x)
        except ValueError:
            pass
        else:
            raise AssertionError(f'bad input accepted: {x}')
    # Check runtime-level failures separately from the application wrapper.
    runtime_failures = 0
    for feed in [{'features': np.ones((2, 3), np.float32)},
                 {'features': np.ones((2, 2), np.float64)},
                 {'wrong_name': np.ones((2, 2), np.float32)}]:
        try:
            session.run(None, feed)
        except Exception as error:
            assert 'input' in str(error).lower() or 'dimension' in str(error).lower()
            runtime_failures += 1
        else:
            raise AssertionError(f'bad runtime feed accepted: {feed.keys()}')
    # The captured PyTorch program still enforces the original range.
    try:
        ns['program'].exported_program.module()(torch.ones(9, 2))
    except (RuntimeError, AssertionError) as error:
        assert '8' in str(error) or 'constraint' in str(error).lower()
    else:
        raise AssertionError('ExportedProgram did not reject batch 9')
    assert session.run(None, {'features': np.ones((9, 2), np.float32)})[0].shape == (9, 2)
    old = text.replace(BEGIN+section+END+'\n\n', '')
    old = re.sub(r'<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->.*?'
                 r'<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->\n\n','',old,flags=re.S)
    old = old.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26')
    old = old.replace('last_verified: 2026-10-27', 'last_verified: 2026-10-26')
    assert hashlib.sha256(old.encode()).hexdigest() == '329710d26fc09b696036c217d07f7151e9b0879394411dbc21e54e24d1911869'
    source = (ROOT/'05 Источники/Originals/model-export-2026-10-27/ort-high-level-design.md').read_text()
    original = source.split('## High-level system architecture\n\n')[1].split('\n\n\n### More about partitioning')[0]
    excerpt = section.split('<!-- BEGIN SOURCE: ORT architecture flow -->\n')[1].split('\n<!-- END SOURCE:')[0]
    assert excerpt.replace('simple.\n\n1.', 'simple.\n1.') == original.replace('../performance/graph-optimizations', 'https://onnxruntime.ai/docs/performance/graph-optimizations.html')
    source = (ROOT/'05 Источники/Originals/model-export-2026-10-27/pytorch-onnx.md').read_text()
    original = source.split('newer*\n\n')[1].split('\n\n{doc}')[0]
    excerpt = section.split('<!-- BEGIN SOURCE: PyTorch ONNX capture -->\n')[1].split('\n<!-- END SOURCE:')[0]
    assert excerpt == original.replace('{ref}`torch.export <torch.export>`', '`torch.export`')
    source = (ROOT/'05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
    quote = section.split('<!-- BEGIN SOURCE: Harvard framework constraints -->\n> ')[1].split('\n<!-- END SOURCE:')[0]
    assert quote in source
    print(json.dumps({'chapter_sha256': hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
        'versions': {name: ns[name].__version__ for name in ['torch', 'onnx', 'ort', 'np']},
        'executed_blocks': len(blocks), 'independent_equation_cases': cases,
        'invalid_contract_cases': len(invalid), 'runtime_rejections': runtime_failures,
        'max_independent_error': max_error, 'provider': session.get_providers(),
        'batch9': 'ExportedProgram rejects; ONNX executes; application rejects',
        'custom_operator': 'checker accepts; session rejects without kernel',
        'source_excerpts': 3, 'prior_832_lines': 'byte-preserved except two dates',
        'scope': 'CPU export and finite numerical/input controls; no GPU, task quality, serving or benchmark'}, indent=2))


if __name__ == '__main__':
    main()
