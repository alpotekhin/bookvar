"""Execute the actual corrected Markdown examples, never a second implementation."""
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "04 Вопросы/100 вопросов по NLP — исходные ответы.md"


class QuestionExamples(unittest.TestCase):
    def example(self, name):
        text = PAGE.read_text(encoding="utf-8")
        errata = text.partition("## Редакционные исправления — 15 сентября 2026")[2]
        blocks = [b for b in re.findall(r"```python\n(.*?)\n```", errata, re.S)
                  if f"def {name}(" in b]
        self.assertEqual(len(blocks), 1, f"Missing unique corrected {name} example")
        namespace = {}
        exec(compile(blocks[0], str(PAGE), "exec"), namespace)
        return namespace[name]

    def test_tfidf_shape_sorted_vocabulary_and_document_frequency(self):
        vocab, idf, rows = self.example("tfidf")(["cat cat dog", "dog", ""])
        self.assertEqual(vocab, ["cat", "dog"])
        self.assertAlmostEqual(idf[0], 1.6931471805599454)
        self.assertAlmostEqual(idf[1], 1.2876820724517808)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == 2 for row in rows))
        self.assertAlmostEqual(rows[0][0], 1.1287647870399635)
        self.assertAlmostEqual(rows[0][1], 0.42922735748392693)
        self.assertEqual(rows[1], [0.0, idf[1]])
        self.assertEqual(rows[2], [0.0, 0.0])

    def test_tfidf_ubiquitous_term_and_empty_corpus(self):
        fn = self.example("tfidf")
        self.assertEqual(fn(["dog", "dog dog"]), (["dog"], [1.0], [[1.0], [1.0]]))
        with self.assertRaises(ValueError):
            fn([])
        with self.assertRaises(ValueError):
            fn(["", " "])

    def test_softmax_distribution_temperature_and_shift_invariance(self):
        fn = self.example("temperature_softmax")
        self.assertAlmostEqual(fn([0.0, math.log(3)])[0], 0.25)
        self.assertAlmostEqual(fn([0.0, math.log(3)], 0.5)[0], 0.1)
        self.assertAlmostEqual(fn([1000.0, 1001.0])[0], 0.2689414213699951)
        self.assertEqual(fn([0.0, 1.0]), fn([1000.0, 1001.0]))
        self.assertAlmostEqual(sum(fn([1000.0, 1001.0])), 1.0)
        self.assertEqual(fn([1e308, -1e308], 0.01), [1.0, 0.0])

    def test_softmax_rejects_undefined_inputs(self):
        fn = self.example("temperature_softmax")
        for logits, temp in [([], 1), ([0], 0), ([0], -1), ([math.nan], 1),
                             ([math.inf], 1), ([0], math.nan), ([0], math.inf)]:
            with self.subTest(logits=logits, temperature=temp):
                with self.assertRaises(ValueError):
                    fn(logits, temp)


if __name__ == "__main__":
    unittest.main()
