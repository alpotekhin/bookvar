"""Regression cases for the archival Beamer projection; never execute TeX."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'publishing/tools'))
import harvard_beamer as beamer


class BeamerProjectionTests(unittest.TestCase):
    def test_nested_notes_do_not_leak_into_slide_body(self):
        slides = beamer.parse_deck(r'''\begin{document}
\begin{frame}{A \textbf{nested} title}
\note{[2 min] SAY \textbf{keep {these} braces}. 50\%.
% -- WARN: an original comment
}
\textbf{Formula}: $\theta^{t+1}=\sum_k\frac{n_k}{n}\theta_k$.
\end{frame}\end{document}''')
        self.assertEqual(len(slides), 1)
        slide = slides[0]
        self.assertEqual(slide.title, 'A **nested** title')
        self.assertNotIn('SAY', slide.body)
        self.assertEqual(slide.notes, ['[2 min] SAY \\textbf{keep {these} braces}. 50\\%.\n% -- WARN: an original comment\n'])
        self.assertIn(r'$\theta^{t+1}=\sum_k\frac{n_k}{n}\theta_k$', slide.body)

    def test_math_commands_are_untouched_by_prose_conversion(self):
        source = r'\[\underbrace{\dfrac{D}{BW}}_{\text{Data}}+\sqrt{\alpha}\]'
        self.assertEqual(beamer.project_tex(source).strip(),
                         '$$\n' + source[2:-2] + '\n$$')
        self.assertIn(r'$a\times b\neq c$', beamer.project_tex(r'\textbf{Test} $a\times b\neq c$'))

    def test_dollar_currency_does_not_terminate_a_math_expression(self):
        self.assertEqual(beamer.project_tex(r'$S=\$200{,}000$'), r'$S=\textdollar{}200{,}000$')
        self.assertEqual(beamer.project_tex(r'\$20 in fees'), '&#36;20 in fees')
        self.assertIn('$20', beamer.project_tex(r'\texttt{\$20}'))

    def test_adjacent_tex_inline_spans_form_one_markdown_math_span(self):
        self.assertEqual(beamer.project_tex(r'$\sim$$10^{18}$'), r'$\sim 10^{18}$')

    def test_comments_and_escaped_percent_are_distinct(self):
        self.assertEqual(beamer.strip_comments('50\\% visible% hidden\nnext'), '50\\% visible\nnext')
        self.assertIn('50% visible', beamer.project_tex('50\\% visible% hidden\nnext'))
        self.assertNotIn('hidden', beamer.project_tex('50\\% visible% hidden\nnext'))

    def test_typeset_accents_keep_author_names_readable(self):
        self.assertEqual(beamer.project_tex(r'Tram\`er, M\"uller, L\'{o}pez'), 'Tramèr, Müller, López')

    def test_tex_quotes_do_not_create_accidental_markdown_code_spans(self):
        self.assertEqual(beamer.project_tex("``First'' and ``second'' --- 3--5"),
                         '“First” and “second” — 3–5')
        self.assertIn('` --nodes=8 `', beamer.project_tex(r'\texttt{--nodes=8}'))

    def test_known_layout_wrappers_do_not_leave_tex_arguments(self):
        text = beamer.project_tex(r'''\begin{columns}[T]\begin{column}{0.5\textwidth}
\begin{mlsyscard}{crimson}{\scriptsize\textbf{Memory} and \textcolor{red}{data}.}
\end{mlsyscard}\end{column}\end{columns}\vspace{0.3cm}''')
        self.assertIn('**Memory** and data.', text)
        for token in ['textwidth', 'mlsyscard', '0.3cm', '[T]', 'crimson']:
            self.assertNotIn(token, text)

    def test_simple_tables_preserve_math_and_cells(self):
        text = beamer.project_tex(r'''\begin{tabular}{@{}lr@{}}\toprule
\textbf{State} & \textbf{Bytes} \\ \midrule
Adam & $2\times4P$ \\ \bottomrule\end{tabular}''')
        self.assertIn('| **State** | **Bytes** |', text)
        self.assertIn('| Adam | $2\\times4P$ |', text)
        self.assertNotIn('toprule', text)

    def test_complex_table_is_preserved_instead_of_misrepresented(self):
        source = r'\begin{tabular}{ll}A & B \\ \multirow{2}{*}{Total}\end{tabular}'
        warnings = []
        text = beamer.project_tex(source, warnings=warnings)
        self.assertIn('```tex\n' + source, text)
        self.assertTrue(warnings)

    def test_column_spans_keep_their_original_column_positions(self):
        text = beamer.project_tex(r'\begin{tabular}{lrr}A & B & C \\ Ratio & \multicolumn{2}{c}{45,000$\times$}\end{tabular}')
        self.assertIn('| Ratio | 45,000$\\times$ (source columns 2–3) |  |', text)

    def test_multiline_table_cells_and_optional_row_spacing(self):
        text = beamer.project_tex(r'\begin{tabular}{ll}A & B \\ X & line one' + '\nline two' + r' \\[0.04cm]Y & Z\end{tabular}')
        self.assertIn('| X | line one line two |', text)
        self.assertIn('| Y | Z |', text)
        self.assertNotIn('0.04cm', text)

    def test_conditional_probability_bar_is_not_a_table_separator(self):
        text = beamer.project_tex(r'\begin{tabular}{ll}Name & Value \\ Test & $P(Y=1|A=a)$\end{tabular}')
        self.assertIn(r'$P(Y=1\vert A=a)$', text)

    def test_unknown_commands_are_visible_not_deleted(self):
        warnings = []
        text = beamer.project_tex(r'Before \unrecognised{important} after.', warnings=warnings)
        self.assertIn(r'\unrecognised', text)
        self.assertIn('important', text)
        self.assertTrue(warnings)

    def test_named_images_use_a_resolver_and_keep_source_attribution(self):
        calls = []
        def resolve(name):
            calls.append(name)
            return 'Sources/Harvard ML Systems/slides/vol2/11_edge_intelligence/fedavg-protocol.svg'
        text = beamer.project_tex(r'\safeimg[width=\textwidth]{fedavg-protocol.pdf}', image_resolver=resolve)
        self.assertEqual(calls, ['fedavg-protocol.pdf'])
        self.assertIn('![[Sources/Harvard ML Systems/slides/', text)

    def test_focus_and_title_macros_are_not_silently_omitted(self):
        source = r'''\newcommand{\safeimg}[1]{not a frame}
\begin{document}\mlsystitle{Lecture}{Subtitle}{cover.png}
\begin{frame}{First}A\end{frame}
\mlsysfocus{An equation}{$T=\frac{D}{BW}$}
\begin{frame}{Last}B\end{frame}\end{document}'''
        slides = beamer.parse_deck(source)
        self.assertEqual([s.title for s in slides], ['Lecture', 'First', 'An equation', 'Last'])
        self.assertIn(r'\frac{D}{BW}', slides[2].body)

    def test_custom_zero_argument_macros_expand_from_original_definition(self):
        source = r'''\newcommand{\Ccube}{C$^3$}
\begin{document}\begin{frame}{The \Ccube{} Taxonomy}
$T=\Dvol/BW$\end{frame}\end{document}'''
        [slide] = beamer.parse_deck(source)
        self.assertEqual(slide.title, 'The C$^3$ Taxonomy')
        self.assertIn(r'D_{\text{vol}}/BW', slide.body)

    def test_nested_lists_keep_numbering_and_level(self):
        text = beamer.project_tex(r'\begin{enumerate}\item First\begin{itemize}\item Nested\end{itemize}\item Last\end{enumerate}')
        self.assertIn('1. First', text)
        self.assertIn('   - Nested', text)
        self.assertIn('2. Last', text)

    def test_bad_input_fails_explicitly(self):
        with self.assertRaises(ValueError):
            beamer.parse_deck(r'\begin{document}\begin{frame}{Broken}no end')

    def test_commented_frame_commands_do_not_create_or_truncate_slides(self):
        source = r'''\begin{document}
% \begin{frame}{Not a slide}\end{frame}
\begin{frame}{Actual}
% \end{frame}
Visible content.
\end{frame}\end{document}'''
        [slide] = beamer.parse_deck(source)
        self.assertEqual(slide.title, 'Actual')
        self.assertIn('Visible content.', slide.body)

    def test_native_assets_match_the_pinned_git_blobs(self):
        rows = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/slide-assets-manifest.json').read_text())
        local = [r for r in rows if r['state'] == 'local-original']
        self.assertEqual(len(local), 269)
        self.assertEqual(len(rows) - len(local), 14)
        for row in local:
            data = (ROOT / row['local_path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])
            blob = b'blob ' + str(len(data)).encode() + b'\0' + data
            self.assertEqual(hashlib.sha1(blob).hexdigest(), row['git_blob_sha1'])

    def test_source_figure_issue_is_visible_without_rewriting_the_original(self):
        import import_harvard_labs_slides as importer
        page = (ROOT / '05 Источники/Courses/Harvard ML Systems/slides/vol2/11_edge_intelligence.md').read_text()
        source = page.split('## Complete original Beamer source\n\n```tex\n')[1].removesuffix('\n```\n') + '\n'
        projection, report = importer.render_slides(source,
            'slides/vol2/11_edge_intelligence/11_edge_intelligence.tex', ROOT)
        self.assertIn('Source figure limitation', projection)
        self.assertIn('cannot be read as additive memory amounts', projection)
        self.assertIn('Activations dominate.', projection)
        self.assertEqual(report['source_issues'], ['The Training Memory Amplifier'])

    def test_source_flop_comparison_is_checked_against_its_own_formula(self):
        small = 12 * 12 * 768 ** 2 * 1024
        large = 12 * 96 * 12288 ** 2 * 2048
        self.assertEqual(small, 86_973_087_744)
        self.assertEqual(large, 356_241_767_399_424)
        self.assertEqual(large, 4096 * small)
        reviews = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/slide-source-review.json').read_text())
        issue = next(r for r in reviews if r['title'] == 'Transformer Compute Cost')
        self.assertIn(f'{small:,}', issue['text'])
        self.assertIn(f'{large:,}', issue['text'])

    def test_tex_indentation_is_not_accidentally_a_markdown_code_block(self):
        text = beamer.project_tex('\\textbf{Label}\n\n    An explanatory paragraph.\n    Its continuation.')
        self.assertIn('\n\nAn explanatory paragraph.\nIts continuation.', text)
        self.assertNotIn('\n    An explanatory', text)

    def test_all_archived_originals_and_declared_frames_are_retained(self):
        rows = json.loads((ROOT / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
        slides = [row for row in rows if row['kind'] == 'slides']
        self.assertEqual(len(slides), 35)
        explicit = 0
        for row in slides:
            with self.subTest(source=row['source']):
                page = (ROOT / row['output']).read_text()
                source = page.split('## Complete original Beamer source\n\n```tex\n')[1].removesuffix('\n```\n') + '\n'
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), row['sha256'])
                parsed = beamer.parse_deck(source)
                frames = [slide for slide in parsed if slide.kind == 'frame']
                self.assertEqual(len(frames), beamer.strip_comments(source).count(r'\begin{frame}'))
                explicit += len(frames)
                for slide in parsed:
                    self.assertTrue(slide.body or slide.notes)
                    self.assertNotIn(r'\note{', slide.body)
        self.assertEqual(explicit, 1311)


if __name__ == '__main__':
    unittest.main()
