---
title: "BPE: training and encoding"
type: practice
status: active
locale: en
translation_status: current
canonical_revision: "d615d1685748c2331dea1b83dfe8a603dd92f13e229a63348ca36b144a460ebc"
translation_reviewed: "2026-11-10"
translation_of: "06 Практика/01a Обучить BPE на двух строках.md"
last_updated: 2026-11-10
primary_sources:
  - https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py
  - https://aclanthology.org/2023.findings-acl.38/
---

# BPE: training and encoding

## Pair counts and replacements

Start with `aaabdaaabac`. It contains 11 bytes; the ASCII bytes
`a` and `b` have IDs 97 and 98. Pair `97 + 97` occurs four times,
twice in each `aaa`. The first merge assigns ID 256 to `aa`.
Only one pair can be replaced per `aaa` because the two occurrences
share a byte. The result has nine tokens.

Next, `256 + 97` and `97 + 98` each occur twice. The first pair
wins because it appears earlier in a left-to-right scan. The next
two merges assign ID 257 to `aaa` and ID 258 to `aaab`.

| Merges | Training tokens | Encoding `aaabac` |
|---:|---|---|
| 0 | `97 97 97 98 100 97 97 97 98 97 99` | `97 97 97 98 97 99` |
| 1 | `256 97 98 100 256 97 98 97 99` | `256 97 98 97 99` |
| 2 | `257 98 100 257 98 97 99` | `257 98 97 99` |
| 3 | `258 100 258 97 99` | `258 97 99` |

## Unseen text

At step 3, set the test string to `я🦊` without changing the training
string. The six token IDs are `209 143 240 159 166 138`. None of
the three learned rules applies, so each byte remains a separate token.
The decoder joins `D1 8F F0 9F A6 8A` before reading the result as
UTF-8. The first byte `D1` cannot be decoded as `я` on its own.

This lab uses a small byte-level BPE model without normalization,
pre-tokenization, or special tokens. It does not load the GPT-2 vocabulary.
See [minbpe, BasicTokenizer](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py)
and the [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram#bpe-merge-trace|tokenization chapter]].
