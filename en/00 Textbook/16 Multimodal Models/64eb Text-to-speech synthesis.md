---
title: Text-to-speech synthesis
description: "Text normalization, phoneme alignment, mel frames or codec tokens, and the conversion to a waveform."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/16 Multimodal Models/64eb Синтез речи.md"
translation_status: current
canonical_revision: "26320709949219c813067505986fe48a25fb53ebdc4973cf25fc5498700c2e93"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Text normalization and pronunciation in TTS"
    anchor: tts-text
  - label: "Monotonic alignment and phoneme duration"
    anchor: tts-duration
  - label: "Acoustic models and mel spectrograms"
    anchor: tts-acoustic
  - label: "Neural vocoders and waveform generation"
    anchor: tts-vocoder
  - label: "Speech synthesis with neural codec tokens"
    anchor: tts-codec
  - label: "TTS intelligibility, naturalness, and latency"
    anchor: tts-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/17.pdf
  - https://arxiv.org/abs/1712.05884v2
  - https://openreview.net/forum?id=piLPYqxtWuA
  - https://proceedings.mlr.press/v139/popov21a.html
  - https://arxiv.org/abs/2010.05646v2
  - https://github.com/jik876/hifi-gan/blob/master/config_v1.json
  - https://proceedings.mlr.press/v139/kim21f.html
  - https://arxiv.org/abs/2210.13438v1
  - https://proceedings.mlr.press/v235/yang24x.html
---

“At 5:00” is pronounced “at five” or “at five o'clock.”
Reading each mark in order misses the meaning. After deciding
how to say the text, a synthesizer still has to choose how
long to hold each sound, where to pause, and how pitch should
change. One sentence can have several good readings. Unlike
[[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64ea Распознавание речи|speech recognition]],
there is no recorded waveform to start from.

<a id="tts-text"></a>

## From written text to pronunciation

Normalization turns written forms into something that can be
spoken. In “starts at 5:00,” the colon separates hours from
minutes. In a score such as “5:0,” it separates two numbers.
The reading depends on the surrounding phrase and the language.
If the input does not specify morning, do not add “a.m.”

Normalization can be followed by grapheme-to-phoneme conversion
(G2P). Take *five*, which we can represent as /f aɪ v/. The
silent *e* is not a fourth sound. We will number the three
phonemes 1, 2, and 3 to distinguish sound order from its
realization in a recording. An explicit phoneme input helps
when pronunciation needs to be controlled. A lexicon can supply
common alternatives, and context can resolve the stress of
*record* as a noun or a verb. Some models consume characters or
subword tokens instead. Tacotron 2, for instance, describes
character embeddings at the input. Phonemes themselves still
specify no durations, pauses, or pitch contour.

<a id="tts-duration"></a>

## How many frames a phoneme gets

Suppose the three phonemes of *five* get durations of 4, 12,
and 4 acoustic frames. At a 10 ms hop these are 40, 120, and
40 ms, totaling 200 ms. The encoder produces one vector per
phoneme. Before predicting a spectrogram, repeat the first
vector 4 times, the second 12 times, and the last 4 times.
The decoder now has 20 positions and can emit one acoustic
frame for each.

| Phoneme | Frames | Duration |
| --- | --- | ---: |
| 1 | 1–4 | 40 ms |
| 2 | 5–16 | 120 ms |
| 3 | 17–20 | 40 ms |

Increasing the durations by a factor of 1.5 gives 6, 18, and
6 frames, or 300 ms. The phonemes remain in the same order.
Changing those durations does not on its own specify pitch
or loudness.

### Where duration targets come from

Training data contain a recording and its transcript, but
do not necessarily contain phoneme boundaries. One option is
to align the known phonemes to the recording beforehand.
FastSpeech 2 obtains durations from a separate forced aligner.
Another option is to use scores from the acoustic model.
During Grad-TTS training, monotonic alignment search maps each
frame to a phoneme in order. Each phoneme must get at least
one frame; the path cannot return to a phoneme it has left.

For a small calculation, group the 20 frames into five
groups of four. Each entry below is an illustrative fit score
between a group and a phoneme. No durations were measured
from recorded audio for this table.

| Phoneme | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0 | −2 | 4 | −2 |
| 2 | −2 | 2 | 3 | 2 | −1 |
| 3 | −3 | −2 | −2 | 0 | 3 |

Taking the best entry in each column gives $(1,2,2,1,3)$ with
score 15. At the fourth group it goes back to the first
phoneme. A monotonic path can stay on its current phoneme or
move to the next one. The best valid path $(1,2,2,2,3)$ scores
$3+2+3+2+3=13$. One, three, and one group give the durations
4, 12, and 4 frames. On a long recording dynamic programming
replaces enumeration: add each group's fit score to the
better of the two permitted predecessor scores.

In the previous chapter, CTC added the probabilities of
valid paths. Here we selected one best path to get duration
targets. At synthesis there is no recording to align, so a
duration predictor estimates the timings from text.

<a id="tts-acoustic"></a>

## From phonemes to a mel spectrogram

A text vector repeated 12 times need not give 12 identical
acoustic frames. Energy, fundamental frequency, and spectral
shape can change during a vowel; formants shift as the speaker
moves toward the next phoneme. FastSpeech 2 uses duration
predictions to set the sequence length and separate pitch
and energy predictions to augment the representation. Pitch
targets are extracted from the training recordings. Spelling
alone does not specify them.

Tacotron 2 generates mel frames autoregressively. Its decoder
uses attention to select the relevant text at each step.
If the alignment jumps or stalls, the synthesis can skip or
repeat a word. With explicit durations, the length of each
part is known before generating the frames. Generating them
in parallel does not imply the model can start speaking
before it sees the full input.

In the Grad-TTS figure, the encoder on the left creates
vectors and the Aligner repeats them according to predicted
durations. Positions in the resulting sequence $\mu$
correspond to acoustic frames. On the right, the decoder
turns noise whose distribution depends on $\mu$ into a mel
spectrogram. The pictures show different noise levels of
one sequence, not a cursor moving through the text.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-synthesis-2026-11-22/popov-grad-tts-fig2.png|Grad-TTS: the encoder and Duration Predictor expand the text in time; the decoder uses reverse diffusion to produce a mel spectrogram.]]

*Vadim Popov et al., [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://proceedings.mlr.press/v139/popov21a.html), [Figure 2](https://proceedings.mlr.press/v139/popov21a/popov21a.pdf#page=5), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Complete diagram from the PMLR paper.*

The [Grad-TTS demo](https://grad-tts.github.io/) lets you hear
one sentence after 10 and after 50 reverse-diffusion steps.
It shows the intermediate spectrograms, with audio produced
by the same vocoder.

With 80 mel bands, the example has 20 columns of 80 values.
They describe energy over time and frequency bands. Listening
to the result requires a sequence of audio samples.

<a id="tts-vocoder"></a>

## How the vocoder produces a waveform

A vocoder produces a waveform conditioned on acoustic
features. Converting a waveform to a mel spectrogram loses
phase and compresses frequency information. More than one
waveform can correspond to the resulting spectrogram; there
is no unique inverse. A neural vocoder learns to produce
waveforms that agree with the conditioning features and
sound like speech.

At 24 kHz and a 10 ms hop, one frame step corresponds to
240 waveform samples. At this ratio, without additional
cropping, 20 frames give $20\cdot240=4800$ samples. A
convolutional generator can increase the time dimension
over several layers: $5\cdot4\cdot3\cdot4=240$. Those are
toy rates. In
[HiFi-GAN V1](https://github.com/jik876/hifi-gan/blob/master/config_v1.json),
the factors 8, 8, 2, and 2 give 256 samples per frame at
22,050 Hz. Feeding a spectrogram with a different hop does
not preserve the timing by itself.

Tacotron 2 uses a modified WaveNet that produces samples
one at a time, conditioned on the mel features and previous
samples. HiFi-GAN's convolutional generator produces a
waveform sequence in one pass. During training, discriminators
compare recorded and generated audio; one family of
discriminators looks for periodic structure. Only the
generator is needed for synthesis.

The Single Speaker section of the
[HiFi-GAN demo](https://jik876.github.io/hifi-gan-demo/) pairs
recordings with outputs from several vocoders. Compare them
on the same sentence by listening within one column.

### Synthesis without an explicit mel output

A two-stage model lets us test the acoustic predictor and
vocoder separately, for instance by comparing the output
from a reference mel spectrogram with the output from a
predicted one. VITS supplies its generator with a different
representation. During training on the left, a linear
spectrogram comes from the recording. The Posterior Encoder
produces latent features, and a monotonic alignment links
them to phonemes. On the right there is no recording at
synthesis. The model predicts durations and forms a latent
sequence based on the input text and the sampled variables.
The Decoder turns it into a waveform. No predicted mel
spectrogram sits between those two components.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-synthesis-2026-11-22/kim-vits-fig1.png|VITS: the left training panel uses a linear spectrogram, latent features, and alignment search. The right synthesis panel uses phonemes and durations to construct latents decoded to a waveform.]]

*Jaehyeon Kim, Jungil Kong, and Juhee Son, [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://proceedings.mlr.press/v139/kim21f.html), [Figure 1](https://proceedings.mlr.press/v139/kim21f/kim21f.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Training and synthesis panels from the PMLR paper.*

<a id="tts-codec"></a>

## Predicting audio codes instead

A neural codec learns to compress and reconstruct audio.
Its encoder produces a sequence of vectors, the quantizer
replaces them with indices into finite codebooks, and the
decoder reconstructs a waveform from those quantized
representations. In residual vector quantization (RVQ),
the first codebook gives an approximation for a frame.
Each subsequent codebook describes part of the remaining
error. The [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|chapter on audio]] traces this process in
EnCodec and speech dialogue models.

Suppose the codec runs at 50 frames per second and gives
four indices per frame. The same 200 ms produces 10 codec
frames and 40 indices. If each codebook contains 1,024
entries, the fixed-width indices need 10 bits each: 400 bits
in 200 ms, or 2 kb/s before packing and overhead. These are
illustrative codec settings. A codec frame is a step in
the codec's latent sequence, not a 10 ms mel step.

Rather than predicting a mel spectrogram, a synthesis model
can predict these indices conditioned on the text. In
training, passing a recording through the codec supplies the
target indices. At inference the codec decoder receives the
generated ones. They encode sound, not letters. The same
phoneme can have different codes, and phoneme boundaries
need not line up with codec frame boundaries.

UniAudio's CodecFormer calls the group of indices in one
frame a *patch*. In the figure the lower Patch Predictor reads
the frame history. The Token Predictor uses its output to
predict indices within the next frame in order. This
separates the sequence of audio frames from the sequence
of refinements within each frame. The paper's overview shows
the conditioning inputs, including text for TTS. This diagram
shows the audio prediction stage.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speech-synthesis-2026-11-22/yang-uniaudio-codecformer-fig3.png|UniAudio CodecFormer: the bottom Patch Predictor processes the history of audio frames, and the upper Token Predictor predicts codes within the next frame. Dashed boxes denote predictions.]]

*Dongchao Yang et al., [UniAudio: Towards Universal Audio Generation with Large Language Models](https://proceedings.mlr.press/v235/yang24x.html), [Figure 3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/yang24x/yang24x.pdf#page=4), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Audio predictor diagram from the PMLR paper.*

If consonants already sound wrong when the codec reconstructs
a real recording, test the codec first. If reconstruction
sounds good but synthesized words are wrong, check the text
normalizer and code predictor separately.

<a id="tts-evaluation"></a>

## Intelligibility, naturalness, and latency

The synthesizer might put an odd pause or stress in “at
five” and still get a zero WER when a recognizer transcribes
it correctly. Listening can reveal those errors. A mean
opinion score (MOS) averages listeners' ratings for a stated
question. Ratings of 4, 5, 4, 3, and 4 give a mean of 4.0;
that number is hard to interpret without the scale, raters,
and samples. If resemblance to a particular voice is relevant,
rate it separately. WER on synthetic speech helps detect
some intelligibility problems, but it also depends on the
recognizer and its biases.

For latency, report time to the first playable chunk
separately from the rate of generation after that. Suppose
the acoustic model and the vocoder take 80 ms and 45 ms
in sequence to produce 200 ms of audio. The compute RTF
is $125/200=0.625$. It does not tell listeners when they
will hear sound. The player may wait for a buffer, and
computation may need to wait for more text. In streaming
synthesis, listen to the joins between chunks as well. A bad
phoneme transition and an empty playback buffer can both
produce a break, but they require different fixes.

<details>
<summary>Check the alignment and dimensions in Python</summary>

~~~python
from itertools import product
from math import isclose, prod
from statistics import mean

# Rows: phonemes. Columns: five groups of four frames.
score = [
    [ 3,  0, -2, 4, -2],
    [-2,  2,  3, 2, -1],
    [-3, -2, -2, 0,  3],
]
P, T = len(score), len(score[0])


def path_score(path):
    return sum(score[p][t] for t, p in enumerate(path))


free = tuple(max(range(P), key=lambda p: score[p][t]) for t in range(T))
assert free == (0, 1, 1, 0, 2) and path_score(free) == 15
monotone = (
    path for path in product(range(P), repeat=T)
    if set(path) == set(range(P)) and list(path) == sorted(path)
)
best = max(monotone, key=path_score)
assert best == (0, 1, 1, 1, 2) and path_score(best) == 13

frames_per_group = 4
dur = [frames_per_group * best.count(i) for i in range(P)]
assert dur == [4, 12, 4]
assert [3 * x // 2 for x in dur] == [6, 18, 6]
expanded = [p for p, d in enumerate(dur) for _ in range(d)]
assert expanded[3:5] == [0, 1] and expanded[15:17] == [1, 2]

rate = 24_000
hop = prod([5, 4, 3, 4])
samples = len(expanded) * hop
assert samples == 4800 and isclose(samples / rate, 0.2)
assert prod([8, 8, 2, 2]) == 256  # HiFi-GAN V1

codec_frames = round(50 * samples / rate)
indices = codec_frames * 4
assert (codec_frames, indices) == (10, 40)
assert indices * 10 / (samples / rate) == 2000

assert mean([4, 5, 4, 3, 4]) == 4
assert isclose((80 + 45) / 200, 0.625)
print("alignment 13; mel 20 frames; PCM 4800 samples; RVQ 40 indices")
~~~

</details>

## Sources

- Daniel Jurafsky and James H. Martin, [Text-to-Speech](https://web.stanford.edu/~jurafsky/slp3/17.pdf), Chapter 17: the task, codecs, and token-based synthesis.
- Jonathan Shen et al., [Tacotron 2](https://arxiv.org/abs/1712.05884v2), Section 2: mel spectrograms and WaveNet.
- Yi Ren et al., [FastSpeech 2](https://arxiv.org/abs/2006.04558v3), Sections 2.2–2.3: duration, pitch, and energy.
- Vadim Popov et al., [Grad-TTS](https://proceedings.mlr.press/v139/popov21a.html), Sections 3.1–3.2: alignment and the diffusion acoustic model.
- Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae, [HiFi-GAN](https://arxiv.org/abs/2010.05646v2), Section 2 and the [V1 configuration](https://github.com/jik876/hifi-gan/blob/master/config_v1.json): vocoding and the temporal upsampling ratio.
- Jaehyeon Kim, Jungil Kong, and Juhee Son, [VITS](https://proceedings.mlr.press/v139/kim21f.html), Section 2 and Figure 1: latents, duration, and waveform generation.
- Alexandre Défossez et al., [EnCodec](https://arxiv.org/abs/2210.13438v1), Section 3; Dongchao Yang et al., [UniAudio](https://proceedings.mlr.press/v235/yang24x.html), Sections 2.1–2.3: coding audio and predicting codes.

**Previous:** [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64ea Распознавание речи|Automatic speech recognition]].
**Next:** [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Evaluating VLMs and measuring serving latency]].
