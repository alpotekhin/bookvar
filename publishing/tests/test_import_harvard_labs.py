"""Static Marimo prose must retain Python string semantics, not execute labs."""
import importlib.util
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'publishing/tools'))
SPEC = importlib.util.spec_from_file_location(
    'harvard_importer', ROOT / 'publishing/tools/import_harvard_labs_slides.py')
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


class MarimoExtractionTests(unittest.TestCase):
    def test_literal_latex_is_not_interpreted_as_python_fields(self):
        source = 'mo.md("""\n    **Memory**\n\n    $M_{\\\\text{train}} = M_{\\\\text{weights}}$\n    """)'
        self.assertEqual(IMPORTER.extract_marimo_markdown(source), [
            '**Memory**\n\n$M_{\\text{train}} = M_{\\text{weights}}$'])

    def test_raw_strings_and_escaped_fstring_braces(self):
        source = 'mo.md(r"""$\\theta_{train}$""")\nmo.md(fr"""$\\theta^{{t+1}} = \\sum_k \\frac{{n_k}}{{n}}$""")'
        self.assertEqual(IMPORTER.extract_marimo_markdown(source), [
            '$\\theta_{train}$', '$\\theta^{t+1} = \\sum_k \\frac{n_k}{n}$'])

    def test_nested_format_expressions_are_symbolic_not_truncated(self):
        source = '''mo.md(f"""Memory: {result['memory_mb']:,.1f} MB. Policy: {policy.name}.""")'''
        [text] = IMPORTER.extract_marimo_markdown(source)
        self.assertIn('Memory: ⟨value 1⟩ MB. Policy: ⟨value 2⟩.', text)
        self.assertIn('Runtime fields (not evaluated)', text)
        self.assertIn("result['memory_mb']", text)
        self.assertIn('policy.name', text)
        self.assertNotIn('`…`', text)

    def test_fields_inside_math_use_declared_symbols(self):
        source = 'mo.md(f"""$$C = {cost:.2f} \\\\times 2$$""")'
        [text] = IMPORTER.extract_marimo_markdown(source)
        self.assertIn(r'$$C = \mathrm{value}_{1} \times 2$$', text)
        self.assertIn('cost', text)

    def test_comments_fake_calls_and_runtime_code_are_not_executed(self):
        source = '''
raise RuntimeError("must never execute")
# mo.md("""comment is not a cell""")
fake = 'mo.md("""text inside a string""")'
mo.md("single-line original prose")
mo.md(message_from_runtime)
'''
        self.assertEqual(IMPORTER.extract_marimo_markdown(source), ['single-line original prose'])

    def test_all_34_archived_originals_are_hash_verified_and_parseable(self):
        records = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        labs = [row for row in records if row['kind'] == 'lab']
        self.assertEqual(len(labs), 34)
        for row in labs:
            with self.subTest(source=row['source']):
                source = IMPORTER.archived_lab_source(ROOT, row)
                prose = IMPORTER.extract_marimo_markdown(source)
                self.assertTrue(prose)
                self.assertNotIn('`…`', '\n'.join(prose))

    def test_edge_formula_matches_decoded_original(self):
        records = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        row = next(row for row in records if row['source'] == 'labs/vol2/lab_11_edge_intelligence.py')
        source = IMPORTER.archived_lab_source(ROOT, row)
        prose = '\n'.join(IMPORTER.extract_marimo_markdown(source))
        self.assertIn(r'M_{\text{train}} = M_{\text{weights}} + M_{\text{grad}} + M_{\text{opt}} + M_{\text{act}}', prose)
        self.assertIn(r'\theta^{t+1} = \sum_k \frac{n_k}{n}\theta_k^{t+1}', prose)
        self.assertIn('Recommended Reading', prose)
        self.assertNotIn('    - **Design Constraints**', prose)

    def test_runtime_outcomes_are_separate_from_literal_explanations(self):
        source = 'mo.md("""**Formula**\n\n$M=P$""")\nmo.md(f"OOM: {memory} MB")\nmo.md("Select your prediction")\n'
        page = IMPORTER.render_lab(source, 'labs/vol2/lab_11_edge_intelligence.py')
        narrative = page.split('## Readable lab narrative\n\n', 1)[1].split('<details>', 1)[0]
        self.assertIn('$M=P$', narrative)
        self.assertNotIn('OOM', narrative)
        self.assertNotIn('Select your prediction', narrative)
        templates = page.split('<details>', 1)[1].split('</details>', 1)[0]
        self.assertIn('OOM: ⟨value 1⟩ MB', templates)
        self.assertIn('Select your prediction', templates)
        self.assertIn('## Complete original Marimo source', page)

    def test_text_identifiers_are_typeset_without_editing_archived_source(self):
        source = 'mo.md(r"""**Formula**\n$$\n\\text{size_ok} \\land \\text{hardware_supported}\n$$\n""")\n'
        page = IMPORTER.render_lab(source, 'labs/vol1/lab_10_model_compress.py')
        narrative, original = page.split('## Complete original Marimo source', 1)
        self.assertIn(r'\text{size\_ok} \land \text{hardware\_supported}', narrative)
        self.assertIn(source.rstrip(), original)
        fenced = '```tex\n$$\n\\text{size_ok}\n$$\n```'
        self.assertEqual(IMPORTER.escape_math_text_labels(fenced), fenced)

    def test_math_context_ignores_code_and_escaped_currency(self):
        for prefix in [r'\$100 and ', '```\n$code\n', '`$code` ']:
            self.assertFalse(IMPORTER.inside_markdown_math(prefix))
        self.assertTrue(IMPORTER.inside_markdown_math('$$C = '))
        self.assertFalse(IMPORTER.inside_markdown_math('$$C=1$$ done'))

    def test_edge_quote_and_prior_chapter_are_preserved(self):
        chapter = (ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md').read_text()
        chapter = re.sub(r'<!-- BEGIN REVIEW: Micro arena 2026-10-27 -->\n.*?'
                         r'<!-- END REVIEW: Micro arena 2026-10-27 -->\n\n', '', chapter, flags=re.S)
        chapter = chapter.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26')
        chapter = chapter.replace('last_verified: 2026-10-27', 'last_verified: 2026-10-26')
        start = '<!-- BEGIN REVIEW: Harvard mixed-state memory 2026-10-26 -->'
        end = '<!-- END REVIEW: Harvard mixed-state memory 2026-10-26 -->'
        block = chapter.split(start, 1)[1].split(end, 1)[0]
        baseline = chapter.replace(start + block + end + '\n\n', '').replace('2026-10-26', '2026-10-20')
        def presentation_only(match):
            tex = match[1]
            for token in (r'\begin{aligned}', r'\end{aligned}', r'\begin{gathered}',
                          r'\end{gathered}', r'\\', r'\qquad', r'\quad', '&'):
                tex = tex.replace(token, '')
            return '$$' + re.sub(r'\s+', '', tex) + '$$'
        canonical = re.sub(r'\$\$(.*?)\$\$', presentation_only, baseline, flags=re.S)
        # Fingerprint of the accepted Oct 20 chapter with only display-math
        # whitespace/alignment removed. All other text/code/figures must match.
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                         '5ffa537e1b9d09083d2efa38629e969a5187fb4a20cc6ec16e173ba2b821fa61')
        records = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        row = next(row for row in records if row['source'] == 'labs/vol2/lab_11_edge_intelligence.py')
        source = IMPORTER.archived_lab_source(ROOT, row)
        literals = [node.value for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        original = next(value for value in literals if '**Amplification factor** over inference' in value)
        quote = block.split('<!-- BEGIN SOURCE: Harvard training memory Math Peek -->', 1)[1]
        quote = quote.split('<!-- END SOURCE: Harvard training memory Math Peek -->', 1)[0]
        quote = re.sub(r'^> ?', '', quote, flags=re.M)
        for presentation in (r'\begin{aligned}', r'\end{aligned}', r'\\', r'\quad', '&'):
            quote = quote.replace(presentation, '')
        compact = lambda text: re.sub(r'\s+', '', text)
        self.assertEqual(compact(quote), compact(original))
        self.assertIn('6 > 5', (ROOT / 'publishing/tools/check_edge_examples.py').read_text())

    def test_all_source_notices_resolve_to_the_actual_license_filename(self):
        records = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        self.assertEqual(len(records), 69)
        for row in records:
            page = (ROOT / row['output']).read_text()
            notice = page.split('# VOL', 1)[0]
            self.assertIn('/LICENSE.md)', notice, row['source'])
            self.assertNotIn('/LICENSE)', notice, row['source'])

    def test_original_code_bytes_survive_all_69_notice_repairs(self):
        records = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        for row in records:
            with self.subTest(source=row['source']):
                if row['kind'] == 'lab':
                    IMPORTER.archived_lab_source(ROOT, row)
                else:
                    page = (ROOT / row['output']).read_text()
                    marker = '## Complete original Beamer source\n\n```tex\n'
                    original = page.split(marker, 1)[1].removesuffix('\n```\n') + '\n'
                    self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
