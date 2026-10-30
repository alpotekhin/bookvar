"""Check the current chapter's executable BPE and worked probability examples."""
from collections import Counter
import ast
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md"
ARCHIVE = ROOT / "05 Источники/Originals/tokenization-2026-10-21"
REVISIT = ROOT / "05 Источники/Originals/tokenization-2026-10-22"


def original_symbols(path, names, expected_sha, namespace=None):
    """Run only the named, hash-pinned pure definitions, without framework imports."""
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected_sha, path
    tree = ast.parse(source)
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in names]
    assert {node.name for node in definitions} == set(names)
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    text = CHAPTER.read_text()
    blocks = [m.group("code") for m in re.finditer(
        r"^(?P<fence>`{3}|~{3})python\n(?P<code>.*?)^(?P=fence)\s*$",
        text, re.M | re.S)]
    assert len(blocks) == 5, len(blocks)
    # Snippets intentionally build on earlier definitions in the chapter.
    extra = '''
toy_merges, toy_vocab = train_bpe("aaabdaaabac", 3)
assert [toy_vocab[i] for i in toy_merges.values()] == [b"aa", b"aaa", b"aaab"]
ids = list(b"aaabdaaabac")
lengths = [len(ids)]
for pair, new_id in toy_merges.items():
    ids = merge(ids, pair, new_id)
    lengths.append(len(ids))
assert lengths == [11, 9, 7, 5]
assert decode(ids, toy_vocab) == "aaabdaaabac"
training_merges, training_vocab = train_bpe("the cat in the hat", 3)
assert list(training_merges.items()) == [
    ((116, 104), 256), ((256, 101), 257), ((257, 32), 258)]
ids = list(b"the cat in the hat")
lengths = [len(ids)]
for pair, new_id in training_merges.items():
    ids = merge(ids, pair, new_id)
    lengths.append(len(ids))
assert lengths == [18, 16, 14, 12]
assert "я".encode("utf-8") == bytes([0xD1, 0x8F])
try:
    bytes([0xD1]).decode("utf-8")
except UnicodeDecodeError:
    pass
else:
    raise AssertionError("An incomplete UTF-8 character was accepted")
# A valid merge ordering can differ from longest-prefix vocabulary matching.
ranked_merges = {(99, 100): 256, (98, 99): 257, (97, 257): 258}
assert encode("abc", ranked_merges) == [258]
assert encode("abcd", ranked_merges) == [97, 98, 256]
'''
    with tempfile.TemporaryDirectory(prefix="bookvar-tokenization-") as tmp:
        script = Path(tmp) / "examples.py"
        script.write_text("\n\n".join(blocks) + "\n" + extra)
        subprocess.run([sys.executable, "-I", str(script)], check=True, timeout=10)

    # Check the worked case with the original GPT-2 rank-based implementation.
    gpt2 = original_symbols(
        ARCHIVE / "gpt2-encoder-pinned.py", {"get_pairs", "Encoder"},
        "84caddd892377b74de8b61f1fdf2437de9dc025c2f3bc73a981edcb0605e4f9a")
    # __new__ deliberately bypasses __init__: this test needs neither regex nor
    # the production vocabulary; bpe uses only its ranks, cache and get_pairs.
    encoder = gpt2["Encoder"].__new__(gpt2["Encoder"])
    encoder.cache = {}
    rules = [("c", "d"), ("b", "c"), ("a", "bc")]
    encoder.bpe_ranks = dict(zip(rules, range(len(rules))))
    assert encoder.bpe("abc") == "abc"
    assert encoder.bpe("abcd") == "a b cd"
    vocabulary = set("abcd") | {"cd", "bc", "abc"}

    def longest_prefix(word):
        output = []
        while word:
            part = max((s for s in vocabulary if word.startswith(s)), key=len)
            output.append(part)
            word = word[len(part):]
        return output

    assert longest_prefix("abcd") == ["abc", "d"]
    # Independent left-to-right passes agree with upstream rank selection on
    # all 5460 nonempty strings of length <= 6 for these rules.
    for length in range(1, 7):
        for chars in product("abcd", repeat=length):
            word = "".join(chars)
            pieces = list(word)
            for left, right in rules:
                result, index = [], 0
                while index < len(pieces):
                    if pieces[index:index + 2] == [left, right]:
                        result.append(left + right)
                        index += 2
                    else:
                        result.append(pieces[index])
                        index += 1
                pieces = result
            assert encoder.bpe(word).split() == pieces, word

    # Original BERT greedy segmentation, not a reimplementation labelled BERT.
    bert = original_symbols(
        REVISIT / "bert-tokenization.py",
        {"convert_to_unicode", "whitespace_tokenize", "WordpieceTokenizer"},
        "576f8e709eaa59c999a8aed3f32c581487d59ab47079fe64b3d3fa457bc7fb34",
        {"six": SimpleNamespace(PY3=True, PY2=False)})
    vocab = {"a", "ab", "##b", "##bc"}
    wordpiece = bert["WordpieceTokenizer"](vocab)
    assert wordpiece.tokenize("abc") == ["[UNK]"]
    assert wordpiece.tokenize("ab") == ["ab"]
    assert bert["WordpieceTokenizer"](vocab - {"ab"}).tokenize("abc") == ["a", "##bc"]
    assert bert["WordpieceTokenizer"](vocab | {"##c"}).tokenize("abc") == ["ab", "##c"]
    assert wordpiece.tokenize("a abc ab") == ["a", "[UNK]", "ab"]

    # Exact byte counts, no enormous arrays allocated and no GPU timing claims.
    width = 4096
    v1, v2, t1, t2 = 32000, 128000, 4096, 3072
    assert (v1*width, v2*width) == (131072000, 524288000)
    assert (2*v1*width // 2**20, 2*v2*width // 2**20) == (250, 1000)
    assert (2*t1*v1 // 2**20, 2*t2*v2 // 2**20) == (250, 750)
    assert (t1**2, t2**2) == (16777216, 9437184)
    assert F(t2*t2, t1*t1) == F(9, 16)
    assert F(t2*width*v2, t1*width*v1) == 3

    # Independent exact calculation of the Unigram example and E/M step.
    p = dict(zip(("a", "b", "c", "ab", "bc"),
                 (F(3, 10), F(2, 10), F(1, 10), F(1, 10), F(3, 10))))
    paths = [("a", "b", "c"), ("ab", "c"), ("a", "bc")]
    weights = []
    for path in paths:
        weight = F(1)
        for token in path:
            weight *= p[token]
        weights.append(weight)
    assert weights == [F(6, 1000), F(1, 100), F(9, 100)]
    z = sum(weights)
    assert z == F(106, 1000)
    posterior = [w / z for w in weights]
    assert posterior == [F(3, 53), F(5, 53), F(45, 53)]
    expected = {token: sum(q*path.count(token) for path, q in zip(paths, posterior))
                for token in p}
    assert list(expected.values()) == [F(48, 53), F(3, 53), F(8, 53), F(5, 53), F(45, 53)]
    length = sum(expected.values())
    assert length == F(109, 53)
    assert [c / length for c in expected.values()] == [
        F(48, 109), F(3, 109), F(8, 109), F(5, 109), F(45, 109)]
    best = [F(1), F(0), F(0), F(0)]
    forward = [F(1), F(0), F(0), F(0)]
    for j in range(1, 4):
        for i in range(j):
            piece = "abc"[i:j]
            if piece in p:
                best[j] = max(best[j], best[i] * p[piece])
                forward[j] += forward[i] * p[piece]
    assert best == [F(1), F(3, 10), F(1, 10), F(9, 100)]
    assert forward[-1] == z

    # Reconstruct the frequency-score example, rather than trust its labels.
    corpus = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
    symbols, pairs = Counter(), Counter()
    for word, count in corpus.items():
        split = [word[0]] + ["##" + c for c in word[1:]]
        for token in split:
            symbols[token] += count
        for pair in zip(split, split[1:]):
            pairs[pair] += count
    score = lambda pair: F(pairs[pair], symbols[pair[0]]*symbols[pair[1]])
    assert score(("##u", "##g")) == F(1, 36)
    assert score(("##g", "##s")) == F(1, 20)
    assert score(("##u", "##g")) < score(("##g", "##s"))

    refs = json.loads((ARCHIVE / "references.json").read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    # Keep the old archive's reviewed revision immutable; the new report records
    # the present chapter hash instead of rewriting the historical source record.
    before = ROOT / "docs/audit/snapshots/2026-10-22-tokenization-revisit-before.md"
    assert sha(before) == refs["chapter_sha256"]
    for row in refs["sources"]:
        assert sha(ROOT / row["file"]) == row["sha256"], row["file"]
    assert sha(REVISIT / "bert-LICENSE") == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    assert sha(REVISIT / "gpt2-LICENSE") == "4682c2fef4e38ad3eaafd0ce6a2058f80892e308caa453923acc5283b559278a"
    assert sha(REVISIT / "gpt2-model.py") == "2ff5065f5cac3dc93065f8a3f36eb013e407972a80abcd4ee6b89a7138a0d5a6"
    print("PASS: 5 sequential chapter snippets; BPE traces, UTF-8 and 5460 upstream rank comparisons; original BERT greedy failure; exact vocabulary/logit/attention arithmetic; WordPiece scores; Unigram Viterbi, forward sum and EM; pinned sources/licenses.")
    print("No SentencePiece binary, full BERT/GPT-2 model, LLM training or tokenizer benchmark executed.")
    print("Chapter SHA256:", sha(CHAPTER))


if __name__ == "__main__":
    main()
