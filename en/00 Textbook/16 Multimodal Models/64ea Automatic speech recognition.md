---
title: Automatic speech recognition
description: "From audio frames to transcripts: CTC alignments, RNN-T transitions, Whisper, and the difference between transcription quality and latency."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/16 Multimodal Models/64ea Распознавание речи.md"
translation_status: current
canonical_revision: "61d7736f676c43ae95a160986ba1a35157a2ab6be0dc4a10c13439b39789fdfb"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Speech recognition inputs and outputs"
    anchor: asr-input
  - label: "CTC: blank, repeated labels, and alignment"
    anchor: asr-ctc
  - label: "CTC: forward probabilities and decoding"
    anchor: asr-forward
  - label: "RNN-T and Token-and-Duration Transducer"
    anchor: asr-transducer
  - label: "Speech recognition and translation with Whisper"
    anchor: asr-whisper
  - label: "WER, CER, and streaming speech recognition latency"
    anchor: asr-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/16.pdf
  - https://distill.pub/2017/ctc/
  - https://proceedings.mlr.press/v32/graves14.html
  - https://arxiv.org/abs/2006.11477v3
  - https://arxiv.org/abs/1211.3711v1
  - https://proceedings.mlr.press/v202/xu23g.html
  - https://proceedings.mlr.press/v202/radford23a.html
---

The recording says, “Hello, I hear faint noise.” We can train a
speech recognizer with the recording and that transcript. A
human does not have to mark which samples correspond to each
letter. The model still needs to relate a short string of
letters or subtokens to a much longer series of acoustic frames.

<a id="asr-input"></a>

## From samples to transcripts

At 16 kHz, half a second contains 8,000 samples. With a 25 ms
window and a 10 ms hop, each window contains 400 samples and
advances by 160. Without padding, we get 48 full windows.
For each window we can compute energies in mel bands and take
their logarithms. The result is a sequence of 48 vectors in
place of the 8,000 samples. The
[[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|previous chapter on audio]]
goes through this feature extraction. These counts are for
unpadded windows, not the particular Whisper preprocessor.

A recognizer can consume precomputed features or learn its
first transformation on the waveform. In
[wav2vec 2.0](https://arxiv.org/abs/2006.11477v3), convolutional
layers first turn the waveform into features. During
pretraining, the model distinguishes the correct quantized
representation of a masked region from negative examples.
It is then fine-tuned on audio–transcript pairs with CTC.
Pretraining representations and learning to transcribe use
different objectives.

Depending on the vocabulary and annotations, the output may
consist of characters, phonemes, or subtokens. Use characters
for the alignment example. We will take the part of “hello”
that corresponds to the two written l's. A doubled letter
does not require two distinct sounds. We isolate this region
for the calculation; in training, the transcript belongs to
the entire utterance.

<a id="asr-ctc"></a>

## How CTC preserves two identical letters

At every encoder step, CTC assigns probabilities to the output
labels and to a special blank label, shown here as a dot. Blank
means no character is added at this step. It can appear on a
frame that contains audible speech.

An alignment path has one label per encoder step. To recover
the text, collapse each consecutive run of equal labels to
one and then remove blanks. The order matters: “l l · l”
becomes two l's; “l l l ·” becomes one. In Awni Hannun's
diagram, $\epsilon$ separates the two runs of *l* in *hello*.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-recognition-2026-11-22/hannun-ctc-collapse.svg|Four rows of the Distill example show the original alignment, merged repeats, removed epsilon labels, and the resulting hello with two l's.]]

*Awni Hannun, [Sequence Modeling With CTC](https://distill.pub/2017/ctc/#the-algorithm), [original diagram of label collapse](https://distill.pub/2017/ctc/assets/ctc_alignment_steps.svg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Adjacent repeated letters take extra steps. Two l's need at
least three steps, “l · l”. Having as many frames as letters
is therefore not always enough. Subsampling too aggressively
can leave some transcripts with no valid alignment at all.

<a id="asr-forward"></a>

## Adding the probabilities of the paths

Suppose the encoder produced the following distributions over
a two-label alphabet: *l* and blank.

| Step | $p_t(l)$ | $p_t(\text{blank})$ |
| ---: | ---: | ---: |
| 1 | 0.6 | 0.4 |
| 2 | 0.3 | 0.7 |
| 3 | 0.6 | 0.4 |
| 4 | 0.3 | 0.7 |

For fixed audio $X$, CTC gives a path $\pi=(\pi_1,\ldots,\pi_T)$
the product of its frame probabilities. The independence
assumption is about the labels given the audio; an encoder
state may still depend on neighboring frames or the whole
recording. The training transcript supplies a target $Y$.
We sum the paths whose collapse $B(\pi)$ is $Y$:

$$
P(Y\mid X)=\sum_{\pi:B(\pi)=Y}\prod_{t=1}^{T}p_t(\pi_t).
$$

The path “l · l ·” contributes $0.6\cdot0.7\cdot0.6\cdot0.7=0.1764$.
It is one way to get two l's. Moving or extending a run while
preserving a blank between the runs gives the other paths.

| Path | Probability |
| --- | ---: |
| l · l · | 0.1764 |
| l · · l | 0.0504 |
| · l · l | 0.0144 |
| l l · l | 0.0216 |
| l · l l | 0.0756 |
| Sum | 0.3384 |

When the reference contains two l's, the loss $-\log P(Y\mid X)$
rewards that total probability. It does not specify which
of the five paths the model should prefer.

### Summing without listing every path

A longer target has too many paths to enumerate. CTC inserts
a blank before, between, and after the target labels to
define its states. For two l's these are “· l · l ·”. The
two *l* states are distinct: one refers to the first letter,
the other to the second.

The figure uses a different target, *ab*. The columns are
steps $x_1,\ldots,x_6$; the rows are the states “· a · b ·”.
The arrows into a node show which preceding masses can be
added. A transition can go directly from *a* to *b* because
the labels differ. It cannot skip the blank between the two
*l* states in our example.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-recognition-2026-11-22/hannun-ctc-forward.svg|Forward-state graph for target ab: six input columns, five rows labeled blank, a, blank, b, blank, and two marked final nodes.]]

*Awni Hannun, [Sequence Modeling With CTC](https://distill.pub/2017/ctc/#the-algorithm), [original forward-pass diagram](https://distill.pub/2017/ctc/assets/ctc_cost.svg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

At each step, a path can stay at its current state or move
one state forward. It can skip one state if the arriving
label is not blank and differs from the departing label.
Add the incoming masses and multiply by the probability of
the current label. On step two, the path “l ·” reaches the
separating blank with mass $0.6\cdot0.7=0.42$. On step three,
it can advance to the second *l*, giving that state $0.6\cdot0.42=0.252$.
On step four, the second *l* has 0.162 and the final blank
has 0.1764. These two endings give the same sum, 0.3384.
With $U$ target labels and $T$ frames, the table has order
$TU$ entries.

### The best path need not give the most probable string

Greedy decoding takes the most probable label at each frame
and collapses the path. In the table it chooses “l · l ·”,
which gives two l's. But the paths that give one *l* add up
to 0.5832, against 0.3384 for two. One *l* is more probable
as a string even though the strongest single path gives two.

CTC prefix beam search accumulates the mass of paths that
have produced the same string. For each prefix, it keeps
separate masses for paths ending in blank and in another
label. The split determines whether the next *l* extends
a run or adds a new letter. A language-model score can be
included as well. In forced alignment the text is already
known: we seek times for that transcript, rather than
decoding a new string.

<a id="asr-transducer"></a>

## How RNN-T uses the output so far

An RNN-T state depends both on the audio step $t$ and on the
number $u$ of output tokens already emitted. The acoustic
encoder produces $h_t$. A predictor represents the output
prefix. The joiner combines the two vectors to score tokens
and blank. A new token changes the prefix, so the scores
can change even at the same acoustic step.

Blank takes $(t,u)$ to $(t+1,u)$ without changing the text.
A label takes it to $(t,u+1)$ without advancing the audio.
Repeated labels are not collapsed. At one frame, for example,
the model might emit *l* with probability 0.6, another *l*
under the new prefix with probability 0.7, and then blank
with probability 0.9. That path contributes
$0.6\cdot0.7\cdot0.9=0.378$. As in CTC, training sums over
allowed paths, but here they move on a grid with two
coordinates rather than taking a single step per frame.
A path enters $(t,u)$ either by emitting blank at the previous
frame or by extending the previous prefix at the same frame.
The forward pass adds those contributions. Even after all the
letters have been emitted, the path ends only when a blank
takes the model past the last frame.

In the upper panel below, RNN-T moves through audio frames
along the horizontal axis. On a flat stretch it emits no
new tokens. The bottom panel shows the input spectrogram:
one encoder step in the top plots represents four
spectrogram frames. The middle panel shows a
Token-and-Duration Transducer (TDT), which predicts how
many input steps to advance and can skip frames instead of
visiting every one. Around spectrogram frame 400 (about
$t=100$ above), the conventional RNN-T traces a long horizontal
section across a pause. TDT takes fewer separate steps.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-recognition-2026-11-22/xu-rnnt-tdt-fig1.png|Xu et al.'s three panels show a stepped RNN-T alignment, sparser TDT transitions, and the spectrogram of the same signal. T is encoder time; U is the number of tokens emitted.]]

*Hainan Xu et al., [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://proceedings.mlr.press/v202/xu23g.html), [Figure 1](https://proceedings.mlr.press/v202/xu23g/xu23g.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Full plot from PMLR.*

Despite its name, RNN-T can use an acoustic encoder that
is not recurrent. Streaming depends on how much right
context it needs to produce $h_t$. If the encoder needs
the entire recording, the transducer's transitions alone
do not provide real-time output.

<a id="asr-whisper"></a>

## How Whisper specifies the task

The original Whisper used 30-second windows for training.
Audio is resampled to 16 kHz and represented by an
80-channel log-mel spectrogram with a 10 ms hop. Two
convolutions prepare the features for a Transformer encoder;
the second roughly halves the number of steps. The decoder
uses cross-attention to read the encoder outputs and predicts
one token at a time, conditioned on the output so far.
Unlike CTC, its training targets are the successive output
tokens for the autoregressive decoder.

A task token can request different outputs from the same
audio. Given Russian speech such as “Алло, я слышу слабый шум”,
transcription keeps the Russian and translation produces
English. In the figure, the start token is followed by a
language tag and a task. With timestamps, the decoder
places a start and end time token around the text, in 20 ms
increments within the window. Without timestamps it predicts
the text after a prefix specifying that format. A training
segment without speech has a special no-speech token.
That is a segment-level target, not CTC blank at a frame.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-recognition-2026-11-22/radford-whisper-task-fig1.png|Whisper's lower panel shows the start and language tokens, transcribe and translate branches, text with or without timestamp tokens, and a separate no-speech branch.]]

*Alec Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://proceedings.mlr.press/v202/radford23a.html), [lower panel of Figure 1](https://proceedings.mlr.press/v202/radford23a/radford23a.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Whisper learned from audio with transcripts or translations
collected from the web. Weak supervision refers to the source
and quality of those targets; it is different from pretraining
on audio alone. Timestamp output does not make it a streaming
model. A timestamp gives a position inside an already
processed window. A wrapper that transcribes overlapping
windows still needs to decide when to display a partial
result and which words can be revised later.

<a id="asr-evaluation"></a>

## Transcript errors and time to first text

Word error rate uses edit distance to align a recognized
sequence of words with its reference. Denote the counts of
substitutions, deletions, and insertions by $S$, $D$, and $I$,
and the reference length in words by $N$:

$$
\mathrm{WER}=\frac{S+D+I}{N}.
$$

The reference is “hello i hear faint noise” and the hypothesis
is “hello hear quiet noise again.” This deletes “i,” replaces
“faint” with “quiet,” and inserts “again.” The minimum is
three edits over five reference words: 60% WER. At character
level, the Russian example “ало” instead of “алло” is one
edit in four characters (25% CER), but the one word is wrong
(100% WER). Apply the same punctuation, case, number, and
word-boundary normalization to all systems under comparison.
For a corpus, sum edits and reference words over recordings;
that does not give a short utterance the same weight as a
long one.

Two streaming recognizers with the same WER can take different
amounts of time to show text. Suppose the first result needs
200 ms of audio, 80 ms of right context, and 35 ms of computation.
If these happen in order, the first partial arrives 315 ms
after the start of the audio, before network or queueing delays.
If each 200 ms block takes 35 ms to process, its compute real-time
factor is 0.175. That ratio does not tell us when text first
appears. Separately, measure when a partial stops changing
and when the system commits the final transcript after the
speaker stops.

<details>
<summary>Check the numerical examples in Python</summary>

~~~python
from itertools import groupby, product
from math import isclose, prod

blank = "_"
p_l = [0.6, 0.3, 0.6, 0.3]
rows = [{"L": p, blank: 1 - p} for p in p_l]


def collapse(path):
    return "".join(symbol for symbol, _ in groupby(path) if symbol != blank)


mass = {}
for path in product((blank, "L"), repeat=len(rows)):
    y = collapse(path)
    prob = prod(row[symbol] for row, symbol in zip(rows, path))
    mass[y] = mass.get(y, 0.0) + prob
assert isclose(mass["LL"], 0.3384)
assert isclose(mass["L"], 0.5832)
assert collapse(max(row, key=row.get) for row in rows) == "LL"
assert max(mass, key=mass.get) == "L"


def ctc_forward(rows, target):
    z = [blank] + [s for char in target for s in (char, blank)]
    alpha = [1.0] + [0.0] * (len(z) - 1)
    for row in rows:
        nxt = []
        for s, symbol in enumerate(z):
            incoming = alpha[s] + (alpha[s - 1] if s >= 1 else 0)
            if s >= 2 and symbol != blank and symbol != z[s - 2]:
                incoming += alpha[s - 2]
            nxt.append(incoming * row[symbol])
        alpha = nxt
    return sum(alpha[-2:])


assert isclose(ctc_forward(rows, "LL"), mass["LL"])
assert isclose(0.6 * 0.7 * 0.9, 0.378)  # One RNN-T path


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        nxt = [i]
        for j, b in enumerate(hyp, 1):
            nxt.append(min(prev[j] + 1, nxt[-1] + 1, prev[j - 1] + (a != b)))
        prev = nxt
    return prev[-1]


pairs = [
    ("алло я слышу слабый шум", "алло слышу тихий шум снова"),
    ("hello i hear faint noise", "hello hear quiet noise again"),
]
for ref, hyp in pairs:
    ref, hyp = ref.split(), hyp.split()
    assert edit_distance(ref, hyp) == 3 and len(ref) == 5
assert edit_distance("алло", "ало") == 1
assert 200 + 80 + 35 == 315 and isclose(35 / 200, 0.175)
print("CTC LL 0.3384; CTC L 0.5832; WER 0.60; first partial 315 ms")
~~~

</details>

## Sources

- Daniel Jurafsky and James H. Martin, [Automatic Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/16.pdf), Chapter 16: features, architectures, and WER.
- Awni Hannun, [Sequence Modeling With CTC](https://distill.pub/2017/ctc/); Alex Graves and Navdeep Jaitly, [Towards End-to-End Speech Recognition with Recurrent Neural Networks](https://proceedings.mlr.press/v32/graves14.html): collapse, path sums, and speech recognition.
- Alexei Baevski et al., [wav2vec 2.0](https://arxiv.org/abs/2006.11477v3), §§ 2–3: self-supervised speech representations and CTC fine-tuning.
- Alex Graves, [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711v1), § 2; Hainan Xu et al., [Token-and-Duration Transducer](https://proceedings.mlr.press/v202/xu23g.html), §§ 2–3: RNN-T paths and TDT durations.
- Alec Radford et al., [Whisper](https://proceedings.mlr.press/v202/radford23a.html), §§ 2.1–2.3, 3.2, and A.1: data preparation, task tokens, normalization, and long-form transcription.

**Previous:** [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|How models understand video and answer in speech]].
**Next:** [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64eb Синтез речи|Text-to-speech synthesis]].
