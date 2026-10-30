---
title: "64.5. Video, audio, and omni models"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "cb9d70037a4ead20d49d9883ec0182e7eed038bd4bc1a579f2599cc71bd8bf42"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
source_unit_id:
  - meeting-07-slides-xgen-video
  - meeting-07-slides-gens
primary_sources:
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://arxiv.org/abs/2410.16267v2
  - https://arxiv.org/abs/2503.09146v1
  - https://arxiv.org/abs/2204.14198
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2502.13923
  - https://arxiv.org/abs/2511.21631
  - https://arxiv.org/abs/2503.20215
  - https://arxiv.org/abs/2305.05665
  - https://arxiv.org/abs/2410.00037
  - https://cs336.stanford.edu/
---

# Video, audio, and omni models

Video adds time to images; audio makes time continuous and sampling-rate dependent. Encoding every frame and short audio window quickly exhausts context. An omni model must select evidence, synchronize modalities, and often generate speech online.

Suppose a user asks, “What did the person say when they opened the door?” The system must locate a visual event, select the corresponding audio interval, understand its speech, and connect that evidence to its answer. An accurate transcript of the entire clip does not establish that the event and utterance were aligned.

<span id="1-от-видеофайла-до-visual-tokens"></span>

A minute sampled at two frames per second contains 120 images. At 256 tokens per frame, that is 30,720 visual tokens before text. Systems use uniform or scene-aware sampling, tubelets, temporal pooling, clip hierarchies, and resamplers. Uniform sampling is reproducible but may miss a brief event; learned selection adds its own computation and bias even when it reduces the answering model's input.

At 8 fps the same minute would occupy $60\cdot8\cdot256=122\,880$ visual positions. Pooling groups of four frames in the 2-fps example reduces the sequence to 7,680 positions, but may erase a brief motion. A projector changes feature width; a position-wise linear map or MLP does **not** reduce frame or patch count. Temporal pooling or a resampler is a separate operation.

Let the door open during $[18.3,18.5)$ seconds. At 2 fps, samples at 18.0 and 18.5 miss the movement itself; changed states may suggest that a door opened, but its exact moment or a brief accompanying action is not observed. At 8 fps, a sample falls at 18.375:

```python
from fractions import Fraction

def sample_times(duration, fps):
    return [Fraction(i, fps) for i in range(duration * fps)]

low, high = sample_times(60, 2), sample_times(60, 8)
event = (Fraction("18.3"), Fraction("18.5"))
inside = lambda times: [t for t in times if event[0] <= t < event[1]]
assert len(low) == 120 and len(high) == 480
assert len(low) * 256 == 30_720
assert len(high) * 256 == 122_880
assert inside(low) == []
assert inside(high) == [Fraction("18.375")]
```

Scene cuts preserve editing boundaries, not necessarily the action needed for a question. Question-aware sampling can prioritize an episode only if it is among the candidates. A downstream resampler cannot reconstruct a missing observation. A coarser frame may preserve a person's position while losing a license plate; rereading a selected frame at higher resolution is a separate operation and cost.

Frame index is not physical time. Qwen2.5-VL scales temporal position IDs with seconds; [Qwen3-VL replaces that physical-time alignment with textual timestamps](https://arxiv.org/html/2511.21631v1#S2.SS3), while retaining interleaved M-RoPE. Reordering frames, deleting the event, and changing sampling rate are necessary causal tests.

<span id="один-интерфейс-для-изображения-набора-изображений-и-видео"></span>

### Shared architecture, different media budgets

OneVision uses the same vision encoder and projector for single images, multiple images, and video, but not the same token allocation per frame. A detailed image can use many crops; dozens of video frames must share the context budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-modalities.png]]

*Stanford CS336, Lecture 17, modality comparison based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Compare per-frame detail with total sequence length rather than treating “video support” as a binary capability.*

<span id="timestamp-token-и-temporal-position--разные-сигналы"></span>

A timestamp string and a temporal rotary coordinate have different roles: physical-time labels are not rotary position IDs. Qwen3-VL uses text to mark time rather than Qwen2.5-VL's time-scaled IDs. Swapping frames while leaving old timestamps creates contradictory evidence; changing sampling requires regenerating timestamps and positions according to the model's processor. Neither representation fixes a missed event between selected frames.

<span id="xgen-video-temporal-compression"></span>

### xGen-MM-Vid: compress representations after sampling

<!-- source_unit_id: meeting-07-slides-xgen-video -->

Even eight sampled frames can occupy too many LM positions.
[xGen-MM-Vid (BLIP-3-Video)](https://arxiv.org/html/2410.16267v2#S2) adds a
temporal encoder after frame-wise SigLIP and a Perceiver Resampler. Its
described configuration maps $8\cdot729=5\,832$ ViT features to
$8\cdot128=1\,024$ frame tokens, then to $M$ video tokens. Phi-3 receives the
last representation with the question. This limits the LM budget differently
from reducing per-frame detail in the OneVision figure above.

[Original Figure 3](https://arxiv.org/html/2410.16267v2#S2.F3) compares temporal
encoders. Learned spatiotemporal pooling forms $M$ weighted representations
over the input positions; a sequential encoder instead updates memory as
frames arrive. The grouped variant keeps four memory slots for each of the
128 frame-token groups, then pools the resulting $128\cdot4=512$ slots into
$M$ outputs. Thus **32 LM input tokens do not mean 32 memory slots in the
whole system**. Grouping and temporal embeddings help retain distinctions
between observations.

[The released 32-token checkpoint was trained with eight frames](https://huggingface.co/Salesforce/xgen-mm-vid-phi3-mini-r-v1.5-32tokens-8frames).
The paper title is not a guarantee that 32 tokens preserve every detail of any
video. Detecting a door opening and reading a tiny label demand different
evidence. Compression cannot recover an event missed by frame sampling, and
a smaller LM input does not eliminate frame encoding or memory-update costs.

Averaging a “door open” feature over $[0,0,1,1]$ and $[1,1,0,0]$ gives 0.5 for both opening and closing. Learned pooling can use unequal weights, but input features also need temporal information to distinguish order:

```python
import numpy as np

opening = np.array([0., 0., 1., 1.])
closing = opening[::-1]
assert opening.mean() == closing.mean() == 0.5
weights = np.array([0.05, 0.05, 0.45, 0.45])
assert np.isclose(weights @ opening, 0.9)
assert np.isclose(weights @ closing, 0.1)
assert 8 * 729 == 5832 and 8 * 128 == 1024
assert 1024 // 32 == 32
assert 128 * 4 == 512
```

These weights are illustrative, not learned xGen weights. Feature-count reduction is not a speed measurement. Temporal information helps distinguish early and late observations, but a fixed output budget does not guarantee retention of every small event in arbitrarily long recordings.

<span id="gens-question-aware-frame-selection"></span>

### GenS: retrieve frames for the question

<!-- source_unit_id: meeting-07-slides-gens -->

Instead of compressing all features into memory, a system can select
observations conditioned on the question. **GenS stands for Generative Frame
Sampler**, not a generalist action agent. In the
[original Aria-based setup](https://arxiv.org/html/2503.09146v1#S2.SS2), the
model reads the question and textually indexed frames, then generates frame
indices or spans with relevance scores. Joint observation can represent
relations such as “immediately after”; independent image–text similarities
do not directly model that relation.

The paper's inference protocol starts at 1 fps and processes windows of up to
256 frames. It selects $K=\min(N_{\mathrm{ret}},N_{\mathrm{budget}})$ frames,
where these terms count retrieved candidates and the answering model's frame
budget. Relevance scores rank evidence; they are not calibrated probabilities
that the final answer is correct.
[GenS-Video-150K](https://generative-sampler.github.io/) supplies synthetic
relevance supervision for this retrieval interface, not verification of
downstream answers.

As a teaching example, 600 seconds at 1 fps yields 600 candidates. Non-overlapping
windows contain 256, 256, and 88 frames; overlap would add work. Retrieving
45 relevant frames under a 32-frame answering budget still leaves only
32 for the answer. Window-local indices must map back to original timestamps:
“frame 1” in the second window is not the first second of the whole video.
Count both the sampler and the answering model when comparing computational
cost, rather than only the latter's shortened input.

For the door question, selected visual evidence must still be aligned with
audio. GenS does not replace the audio pipeline. Removing the selected interval
tests reliance on retrieved frames; shifting the soundtrack separately tests
synchronization. [Original Figure 1](https://arxiv.org/html/2503.09146v1#S0.F1)
illustrates why a temporal relation requires a sequence rather than a similar
single frame. Neither frame retrieval nor xGen memory constitutes action
execution; that requires its own observation–action–state verification loop.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/gens-frame-selection.png]]

*Yao et al., [GenS, Figure 1](https://aclanthology.org/2025.findings-acl.921.pdf#page=2), CC BY 4.0. “Immediately after” requires a sequence, not one frame similar to the question's nouns. Plotted results concern the paper's specified LongVideoBench subset.*

A meeting image alone does not answer a question about subsequent cooking. Joint processing of observations and question supports that relation. Window bookkeeping is also part of retrieval correctness:

```python
windows = [(start, min(start + 256, 600)) for start in range(0, 600, 256)]
assert [end - start for start, end in windows] == [256, 256, 88]
global_index = windows[1][0] + (1 - 1)
assert global_index == 256
timestamp_at_1fps = global_index
assert timestamp_at_1fps == 256

candidates = [(i, 45 - i) for i in range(45)]
budget = 32
selected = sorted(candidates, key=lambda item: (-item[1], item[0]))[:budget]
assert len(selected) == min(len(candidates), budget) == 32
```

Overlapping windows require deduplication by original global frame identity. A local index is not physical time without the source mapping. GenS-Video-150K's synthetic pipeline builds detailed frame captions, generates frame-linked questions, expands candidate sets, and assigns relevance. Teacher mistakes can train a sampler to omit events that the answering VLM could otherwise understand. Count candidate processing as well as the answering model's input.

### Preserve presentation timestamps

Container presentation timestamps multiplied by their time base specify physical time. Selected-frame index divided by assumed FPS fails for irregular timestamps:

```python
from fractions import Fraction

pts = [0, 15_000, 66_000]
time_base = Fraction(1, 30_000)
times = [p * time_base for p in pts]
assert times == [Fraction(0), Fraction("0.5"), Fraction("2.2")]
assumed_constant_rate = Fraction(2, 2)
assert times[2] != assumed_constant_rate
```

The third observation is at 2.2 seconds, not one second. A surviving frame at 18.4 seconds retains that timestamp after thinning. Sampling changes observations and model position sequences, not when the frame occurred. Qwen3-VL's textual timestamps and Qwen3-Omni's absolute 80-ms TM-RoPE grid are different interfaces despite the family name.

<span id="3-от-waveform-до-признаков-и-дискретных-аудиотокенов"></span>

## Audio and audiovisual alignment

Waveforms are commonly resampled, converted to mel spectrograms, and encoded into short-window features. An ASR → LLM cascade makes transcription inspectable, but plain text omits intonation, timing, speaker identity, music, and non-speech events unless these are supplied separately. Direct audio representations can preserve such evidence; what survives and its sequence cost depend on the encoder.

Sample rate describes the waveform grid, not speech speed or LLM token rate: one second of mono audio at 16 kHz contains 16,000 amplitudes. A spectrum of an entire minute would mix phonemes, pauses, and noise. STFT computes spectra over overlapping windows; mel filters combine frequency bins and a logarithm compresses energy range. An encoder contextualizes these features and can reduce temporal frequency through convolutions.

Original [Whisper §2.2](https://arxiv.org/abs/2212.04356v1) uses 16 kHz audio, 25-ms windows, 10-ms hops, 80 mel channels, and a second initial convolution that halves temporal frequency. Qwen2.5-Omni uses 128 mel channels and about 40 ms per output encoder position. These are version-specific. At 16 kHz the window and hop contain 400 and 160 samples. This FFT example precedes mel filtering, not a complete Whisper processor:

```python
import numpy as np

sample_rate = 16_000
window, hop = 400, 160
t = np.arange(window) / sample_rate
signal = np.sin(2 * np.pi * 440 * t)
spectrum = np.abs(np.fft.rfft(signal * np.hanning(window)))
frequencies = np.fft.rfftfreq(window, 1 / sample_rate)
assert np.isclose(frequencies[spectrum.argmax()], 440)
assert window / sample_rate == 0.025
assert hop / sample_rate == 0.01
assert 1 + (16_000 - window) // hop == 98
```

One unpadded second contains 98 complete windows, although a 10-ms hop corresponds to 100 windows/s in a long recording. Padding, truncation, and fixed-size inputs affect exact counts. Preserve feature-to-time mappings including edge rules. Continuous perception features are not automatically codes from which the waveform can be reconstructed.

Audio and frames share a timeline but have different rates. Qwen2.5-Omni interleaves them by time and uses Time-aligned Multimodal RoPE.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure3-tmrope.png]]

*Figure 3 from Xu et al., [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), PDF p. 4. TMRoPE gives audio and video a shared temporal axis while vision retains spatial coordinates.*

Before a shared sequence offset, Qwen2.5-Omni's 40-ms grid gives time IDs 25 at one second, 50 at two seconds, and 460 at 18.4 seconds. Audio uses matching time/height/width IDs; video holds time fixed within a frame while height and width vary. Qwen3-Omni's 80-ms grid assigns 230 to 18.4 seconds:

```python
from fractions import Fraction

time = Fraction("18.4")
assert time / Fraction("0.04") == 460
assert time / Fraction("0.08") == 230
assert Fraction(2) / Fraction("0.04") == 50

sample_rate = 16_000
start, end = Fraction("18.0"), Fraction("18.8")
audio_slice = (int(start * sample_rate), int(end * sample_rate))
assert audio_slice == (288_000, 300_800)
```

Qwen2.5-Omni groups visual features and then audio within two-second intervals. IDs still determine which observations coincide. Qwen3-Omni replaces rigid grouping with alignment through its 80-ms IDs. Ordering, encoding, and time grid form one contract; changing one while retaining old metadata creates misalignment.

### Continuous features and discrete speech codes

At a 20-ms feature stride, one minute of audio yields $60/0.02=3\,000$ positions before further downsampling. These continuous vectors can condition an LLM for perception. To generate sound, the system needs an output representation with a waveform decoder.

A neural codec encodes a waveform into compressed latents, quantizes them into codebook indices, and reconstructs audio from those indices. A speech model predicts the indices, while the codec decoder produces the waveform. The codec's reconstruction and adversarial objectives are not the same loss as the speech model's categorical prediction loss.

For an illustrative codec with 50 frames/s and eight codebooks, ten seconds requires

$$
10\cdot50\cdot8=4\,000
$$

indices. That does not necessarily mean 4,000 steps through the largest Transformer: time and codebook depth can be modeled separately. Always state frame rate, codebook count, and prediction schedule when reporting “audio tokens.”

### Residual vector quantization

A vector quantizer selects the nearest vector in a learned codebook and transmits its index. RVQ's next codebook approximates the residual. Replacing $(0.9,0.2)$ with $(1,0)$ leaves $(-0.1,0.2)$; a second entry $(-0.1,0.25)$ gives combined reconstruction $(0.9,0.25)$ with error norm 0.05:

```python
import numpy as np

z = np.array([0.9, 0.2])
codebooks = [
    np.array([[0., 0.], [1., 0.]]),
    np.array([[0., 0.], [-0.1, 0.25]]),
]
residual = z.copy()
reconstructed = np.zeros_like(z)
indices, errors = [], []
for book in codebooks:
    index = int(np.argmin(((book - residual) ** 2).sum(axis=1)))
    indices.append(index)
    reconstructed += book[index]
    residual = z - reconstructed
    errors.append(float(np.linalg.norm(residual)))
assert indices == [1, 1]
assert np.allclose(reconstructed, [0.9, 0.25])
assert errors[1] < errors[0] and np.isclose(errors[1], 0.05)
```

Real vectors are learned with the codec. [EnCodec §3.1–3.2](https://arxiv.org/abs/2210.13438v1) varies active RVQ levels to trade code volume against quality. Reconstruction alone does not guarantee that the first codebook represents words or phonemes.

### Mimi separates semantic and acoustic quantization

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/mimi-split-rvq.png]]

*Défossez et al., [Moshi, Figure 2](https://arxiv.org/pdf/2410.00037v2#page=10), CC BY-NC-SA 4.0. Semantic VQ and acoustic RVQ are parallel branches; their outputs combine before decoding. Blue paths belong to training.*

Mimi is a causal codec at 12.5 steps/s, or 80 ms per step. One VQ branch is distilled toward WavLM features; a separate seven-level RVQ branch encodes acoustics. Each receives an independently projected encoder representation. Acoustic RVQ does not continue the semantic branch's residual. This split addresses conflicting semantic and acoustic objectives.

WavLM can see future context when producing training targets, but is absent from the deployed codec. Its use as teacher does not grant future-waveform access at inference. Generator and codec objectives also differ: the generator predicts indices with cross-entropy; EnCodec uses reconstruction and adversarial losses, while the described Mimi uses adversarial and feature losses without explicit reconstruction loss, retaining semantic distillation.

Eight 2048-entry codebooks require 11 bits per index: $12.5\cdot8=100$ indices/s, or 1100 bits/s before overhead. Ten seconds gives 125 time steps and 1000 indices. EnCodec's 24-kHz variant instead operates at 75 steps/s:

```python
from math import prod

sample_rate, frame_rate = 24_000, 12.5
assert prod([4, 5, 6, 8, 2]) == 1920
assert sample_rate / 1920 == frame_rate
assert sample_rate / frame_rate == 1920
codebooks, bits_per_index = 8, 11
assert 2 ** bits_per_index == 2048
assert frame_rate * codebooks * bits_per_index == 1100
assert 10 * frame_rate * codebooks == 1000
assert 24_000 / prod([2, 4, 5, 8]) == 75
```

“Eight audio tokens” or “one minute of sound” therefore does not specify sequence size without codec rate and codebook count.

<span id="4-cascade-и-end-to-end-speech-model"></span>

| Pipeline | Inspectable intermediate | What can be lost | Deployment trade-off |
|---|---|---|---|
| ASR → LLM → TTS | transcript and answer text | prosody and non-speech cues | components can be replaced independently |
| Direct audio/omni | continuous states or codec tokens | depends on learned representation | richer signal but coupled models and scheduling |

A cascade remains a strong baseline for summarizing a lecture. For sarcasm, a sound event, or who is speaking, the transcript may have already discarded the relevant evidence.

A streaming cascade can use partial ASR hypotheses, begin generating text, and synthesize completed fragments before the whole utterance is processed. Latency need not equal the sum of three complete stages. But partial ASR can revise a word, while spoken output cannot be retracted as easily as on-screen text. Losing prosody is a limitation of a text-only intermediate, not proof that every cascade is worse.

Suppose “I'm back” spans 18.24–18.72 seconds and “Wait” spans 20.24–20.72. The door at 18.4 overlaps the first. Shifting audio forward by two seconds removes that overlap; shifting backward by two places the second phrase at the event. This fixture retains the original Russian strings:

```python
from fractions import Fraction as F

segments = [
    (F("18.24"), F("18.72"), "Я вернулся"),
    (F("20.24"), F("20.72"), "Подожди"),
]
def words_at(segments, time, shift=F(0)):
    return [text for start, end, text in segments if start + shift <= time < end + shift]

assert words_at(segments, F("18.4")) == ["Я вернулся"]
assert words_at(segments, F("18.4"), F(2)) == []
assert words_at(segments, F("18.4"), F(-2)) == ["Подожди"]
```

This verifies interval arithmetic, not a VLM. A real experiment must modify the soundtrack and invalidate old audio features and transcripts. A small shift leaving the same phrase overlapping the event need not change the answer; demanding a change after every shift is an invalid test.

## Thinker–Talker and streaming speech

Qwen2.5-Omni separates high-level understanding from speech generation. Thinker processes text, image, video, and audio and generates text. Talker consumes Thinker states and predicts speech-codec tokens as a stream.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure2-thinker-talker.png]]

*Figure 2 from [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), PDF p. 3. Thinker produces semantic states and text; Talker turns them into streaming speech tokens.*

Talker receives both Thinker representations and embeddings of the selected text tokens. Semantically similar words can sound different; hidden semantic similarity does not identify which word to pronounce. Talker can begin from available text and state before the entire answer is ready, without one audio token per BPE token. Codes pass through a Flow-Matching DiT to mel features and a modified BigVGAN to waveform.

### What Qwen3-Omni changes

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/qwen3omni-architecture.png]]

*Qwen Team, [Qwen3-Omni, Figure 2](https://arxiv.org/pdf/2509.17765v1#page=3), CC BY 4.0. Talker predicts the first code, a small MTP predicts remaining codes in that frame, and causal Code2Wav reconstructs audio.*

Qwen3-Omni conditions Talker on discrete text and multimodal features, no longer on the hidden states of Thinker's text positions. This allows text intervention before speech, such as external filtering or prepared text. Its AuT input encoder downsamples mel features eightfold to 12.5 positions/s and learns recognition and other audio tasks. The output multi-codebook codec serves synthesis. Equal input/output rates do not make their vectors or indices interchangeable.

Talker chooses each frame's first code; a small MTP Transformer autoregressively predicts the remaining codebooks of that frame. They are not independent simultaneous predictions. A causal ConvNet Code2Wav uses available codes and left context instead of waiting for a future DiT block.

| Component | Qwen2.5-Omni | Qwen3-Omni |
|---|---|---|
| Input audio encoder | Qwen2-Audio, about 25 positions/s | AuT, about 12.5 positions/s |
| Time representation | 40-ms IDs and two-second grouping | 80-ms IDs without rigid grouping |
| Talker text condition | hidden states and selected text | discrete text without former text-position hidden states |
| Code prediction | dual-track autoregression | first code by Talker, remaining codes by MTP |
| Waveform synthesis | Flow-Matching DiT and BigVGAN | causal ConvNet |

Thinker and Talker use MoE, but attention configuration and sequence length determine KV-cache size, not expert count. Lower temporal frequency reduces positions without eliminating AuT, MTP, or Code2Wav work.

Streaming changes the causal contract: input may continue while the model plans and speaks. The system must decide when to start, how to stop on interruption, and how much audio history to retain. Metrics include time to first audio, real-time factor, interruption latency, semantic accuracy, naturalness, and voice stability.

### Moshi: one temporal step, several codebook decisions

[Moshi](https://arxiv.org/html/2410.00037v2#S3.SS4.SSS4) separates time from within-frame code prediction. Mimi's first codebook carries semantic information; the remaining codebooks refine acoustic detail. In the text-conditioned Inner Monologue arrangement, a linear head over the large Temporal Transformer's state predicts the text token first. A smaller Depth Transformer then predicts semantic and acoustic audio codes, conditioned on the temporal state and text. The audio-only introductory RQ formulation must not be confused with this text-first ordering.

Thus the large network advances once per codec time step rather than once per individual code. User audio and model speech occupy parallel streams on a shared timeline. Streaming means progressively emitting output; **full duplex** additionally means continuing to listen while speaking, including overlap and interruption.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/moshi-rq-transformer.png]]

*Défossez et al., [Moshi, Figure 3](https://arxiv.org/pdf/2410.00037v2#page=13), CC BY-NC-SA 4.0. Embeddings from the previous step enter the Temporal Transformer; the Depth Transformer predicts components of the new step. Four illustrated components are schematic, not Moshi's full stream count.*

One Mimi stream requires $12.5\cdot8=100$ code predictions per second. Flattening codes requires 100 large-model steps/s; temporal/depth factorization requires 12.5 large-model steps plus small-model decisions. Codes are not removed. Two audio streams plus one text stream give $K=2\cdot8+1=17$ components per step. In application, the user's stream comes from actual user audio, not substituted model predictions:

```python
from math import log, isclose

seconds, rate, codebooks = 300, 12.5, 8
time_steps = int(seconds * rate)
assert time_steps == 3750
assert time_steps * codebooks == 30_000
streams = 2 * codebooks + 1
assert streams == 17 and time_steps * streams == 63_750

joint = 0.6 * 0.5
assert isclose(joint, 0.3)
assert isclose(-log(joint), -log(0.6) - log(0.5))
```

The conditional product $p(a)p(b\mid a)$ retains within-step dependence; replacing it with $p(a)p(b)$ assumes independence. The illustrative probabilities and step counts are not a Moshi speed benchmark.

### Inner Monologue, delayed acoustics, and duplex streams

Moshi's Inner Monologue is its own speech-aligned text stream, not a requirement to generate hidden reasoning. PAD fills positions between words and EPAD marks the end of padding before the next word.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/moshi-duplex-streams.png]]

*Défossez et al., [Moshi, Figure 4](https://arxiv.org/pdf/2410.00037v2#page=15), CC BY-NC-SA 4.0. Model text/audio appear below the dashed line and user audio above it. Acoustic rows lag semantic rows by one step.*

With acoustic delay $\tau=1$, step $s$ carries semantic code for frame $s$ and acoustic codes for frame $s-1$. Three frames need this schedule:

| Step | Semantic frame | Acoustic frame |
|---|---|---|
| 0 | 0 | not yet available |
| 1 | 1 | 0 |
| 2 | 2 | 1 |
| 3 | completion | 2 |

```python
semantic = ["s0", "s1", "s2"]
acoustic = ["a0", "a1", "a2"]
delayed = [None] + acoustic
semantic_with_flush = semantic + [None]
schedule = list(zip(semantic_with_flush, delayed))
assert schedule == [("s0", None), ("s1", "a0"), ("s2", "a1"), (None, "a2")]
assert delayed[1:] == acoustic
assert 1 / 12.5 == 0.08
```

The final acoustic frame needs a flush step. The simplified zero-based schedule omits actual special symbols and state handling. At 80 ms per step, the delay adds one interval before a complete frame's codes become available. It provides later semantic context for acoustics, not access to user audio that has not arrived.

Moshi §3.4.4 uses relative text/audio delays for ASR and TTS modes. Supplying actual audio codes and selecting text produces streaming recognition; supplying a leading text stream conditions synthesis. Word alignment, padding, and delay determine that condition. Two audio streams represent simultaneous speech and silence without compulsory non-overlapping turns. Chunked TTS with the microphone disabled is not full duplex.

The label “audio head” is consequently ambiguous. An input projector aligns audio feature width. A classification or ASR head predicts labels. A speech generator predicts a long structured codec sequence and may itself be a Transformer.

### Four live-dialogue decisions

An offline model knows when the recording ends. A live system must decide:

1. **Endpointing:** has the user finished or merely paused?
2. **Turn taking:** is it appropriate to begin answering?
3. **Incremental generation:** which audio chunk can be emitted without future context?
4. **Interruption:** how quickly can playback stop and the new input enter state?

Starting early can cut off the user; starting late creates an awkward pause. Semantic accuracy alone cannot measure these failures, and a real-time factor below one does not establish low interruption latency.

<span id="7-streaming-меняет-причинный-контракт"></span>

### A streaming interface can still have lookahead

A bidirectional two-second input block can make its earliest representation depend on the block's end. It can be processed incrementally by blocks but must wait for that context. Qwen2.5-Omni's input audio encoder uses two-second blocks. Its output DiT has a separate rule: current block, two previous blocks, and one future block. Input-block duration is not the output codec-block duration.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/qwen25omni-block-attention.png]]

*Xu et al., [Qwen2.5-Omni, Figure 4](https://arxiv.org/pdf/2503.20215v1#page=5), CC BY 4.0. The sliding-window DiT mask belongs to codec-to-waveform decoding and includes one future block, not the input audio encoder's mask.*

Output block 3 needs blocks 1, 2, 3, and 4. A hypothetical 160-ms output block with one-block lookahead adds 160 ms of waiting. This duration is a teaching assumption, not a recovered Qwen parameter; removing lookahead changes the model's conditioning mask.

```python
import numpy as np

query = np.arange(6)[:, None]
key = np.arange(6)[None, :]
allowed = (key >= query - 2) & (key <= query + 1)
assert np.flatnonzero(allowed[3]).tolist() == [1, 2, 3, 4]
assert allowed[3, 4] and not allowed[3, 5]
assert 1 * 0.160 == 0.160
```

A causal operator's available prefix must not change when future inputs change. A streaming convolution reproduces full causal convolution if it preserves boundary history:

```python
import numpy as np

x = np.array([1., 2., 3., 4., 5., 6.])
kernel = np.array([0.5, 0.3, 0.2])
full = np.convolve(x, kernel)[:len(x)]
history = np.zeros(len(kernel) - 1)
pieces = []
for chunk in [x[:2], x[2:5], x[5:]]:
    joined = np.r_[history, chunk]
    output = np.convolve(joined, kernel)
    start = len(history)
    pieces.append(output[start:start + len(chunk)])
    history = joined[-(len(kernel) - 1):]
assert np.allclose(np.concatenate(pieces), full)

changed = x.copy()
changed[3:] += 100
assert np.allclose(np.convolve(changed, kernel)[:3], full[:3])
centered = np.convolve(x, np.ones(3) / 3, mode="same")
centered_changed = np.convolve(changed, np.ones(3) / 3, mode="same")
assert centered_changed[2] != centered[2]
```

The centered convolution at the end violates that property by using a future neighbor. For a real codec, inspect convolution padding, temporal normalization, attention, cache, and waveform stitching; passing one chunk-size test does not establish correctness at all boundaries.

### First audible sound and uninterrupted playback

First-audio latency includes required input context, feature preparation, Thinker/Talker computation, codec decoding, transport, and playback. A “first token generated” log covers only one stage. Real-time factor divides compute duration by generated audio duration. A mean below one does not prevent an individual late chunk from exhausting the buffer:

```python
import numpy as np

audio_durations = np.array([0.08, 0.08, 0.08, 0.08])
compute_times = np.array([0.02, 0.02, 0.25, 0.02])
arrivals = compute_times.cumsum()
rtf = compute_times.sum() / audio_durations.sum()
playback_start = arrivals[0]
deadlines = playback_start + np.r_[0., audio_durations.cumsum()[:-1]]
late = arrivals > deadlines + 1e-12
assert np.isclose(rtf, 0.96875) and rtf < 1
assert np.flatnonzero(late).tolist() == [2, 3]
```

The third and fourth chunks miss their original deadlines despite RTF 0.96875. This does not count actual audible stalls: after a stall, playback scheduling shifts. Prebuffering can avoid a gap by delaying response onset.

Interruption requires continued microphone input, speech detection, generation cancellation, and removal of unplayed audio. Samples already sent to a device may remain audible after server cancellation. For user speech at 1000 ms, detection at 1040, cancellation at 1055, and final audible model sound at 1180, interruption latency is 180 ms, not 55. The other 125 ms belongs to buffered output and playback. These are illustrative event timestamps, not product measurements.

Conversation state must record what was actually played. If a full sentence was generated but only its first half heard, the next turn should not automatically treat the entire sentence as spoken.

## Shared representations are not any-to-any generation

ImageBind creates one embedding space for text, image, audio, depth, thermal, and IMU data. That supports cross-modal retrieval but does not itself generate each modality. Image or speech output requires a dedicated decoder such as diffusion/flow, discrete image tokens, or an audio codec.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/imagebind-figure1-capabilities.png]]

*Girdhar et al., [ImageBind, Figure 1](https://arxiv.org/abs/2305.05665). Shared representations support retrieval and conditioning of a separate image generator; the diagram is not a streaming speech architecture.*

ImageBind trains modality encoders using pairs linked through images, with a contrastive objective over normalized vectors. A crackling-fire audio vector can retrieve an image by cosine similarity without ASR or answer generation. This numerical example illustrates retrieval, not actual ImageBind embeddings:

```python
import numpy as np

audio = np.array([0.8, 0.6])
images = np.array([[1., 0.], [0., 1.], [-1., 0.]])
scores = images @ audio
assert np.allclose(scores, [0.8, 0.6, -0.8])
assert scores.argmax() == 0
```

The shared space supplies neither phoneme sequences nor waveform codes. Likewise, text-answering multimodal understanding need not support image/audio generation, and adding Talker does not provide video output. Verify the deployed API separately: supported inputs, simultaneous streams, continued listening during speech, and enabled output modes may differ from architectural capabilities. A causal codec does not stop an application from buffering an entire utterance.

<span id="9-матрица-оценивания"></span>

Evaluation must cover each modality and their interaction. Strong image scores cannot compensate for ignored audio; ASR accuracy does not prove audiovisual synchronization. Cost curves should be reported over frames, seconds, and multimodal tokens.

| Capability | Measurement | Intervention |
|---|---|---|
| Temporal order | event-order accuracy | swap event order |
| Moment localization | temporal IoU | remove the target interval |
| Speech recognition | WER/CER | vary noise while retaining words |
| Non-speech audio | event F1 or mAP | replace background sound |
| Audio–video alignment | paired-event accuracy | shift the soundtrack |
| Spoken answer | semantic accuracy and listening tests | compare text and speech outputs |
| Streaming | first-audio time, real-time factor, interruption latency | simulate user barge-in |

For the door question, the answer must identify the correct speech at the correct event, not merely mention something said elsewhere. Report quality as a function of frame rate, visual budget, audio duration, and end-to-end latency.

Temporal IoU divides interval intersection duration by union duration. Ground truth $[18.3,18.7]$ and prediction $[18,19]$ both contain 18.4, but have IoU only 0.4:

```python
from math import isclose

gold, predicted = (18.3, 18.7), (18.0, 19.0)
intersection = max(0, min(gold[1], predicted[1]) - max(gold[0], predicted[0]))
union = (gold[1] - gold[0]) + (predicted[1] - predicted[0]) - intersection
assert isclose(intersection / union, 0.4)
assert gold[0] <= 18.4 < gold[1] and predicted[0] <= 18.4 < predicted[1]
```

Measure audio under specified noise conditions. $\mathrm{SNR}_{dB}=10\log_{10}(P_{signal}/P_{noise})$; 10 dB means signal power is ten times noise power, not that every word remains safe from corruption. Noise spectrum, timing, and event type matter:

```python
import numpy as np

rng = np.random.default_rng(20261020)
signal = np.sin(2 * np.pi * 440 * np.arange(1600) / 16_000)
noise = rng.normal(size=len(signal))
signal_power = np.mean(signal ** 2)
noise *= np.sqrt(signal_power / (10 * np.mean(noise ** 2)))
snr_db = 10 * np.log10(signal_power / np.mean(noise ** 2))
assert np.isclose(snr_db, 10)
```

These checks verify formulas, not model quality. Compare systems on the same clips and questions with equal audio access and comparable frame budgets. Service comparisons additionally include sampler, encoders, speech decoder, resource use, and client playback. Recognition, audiovisual alignment, response semantics, and blind listening quality need separate evidence.

## Sources and next chapter

- [Flamingo](https://arxiv.org/abs/2204.14198), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), and [ImageBind](https://arxiv.org/abs/2305.05665).
- [xGen-MM-Vid §2.1–2.3](https://arxiv.org/abs/2410.16267v2); [GenS §2](https://aclanthology.org/2025.findings-acl.921/); [EnCodec §3.1–3.2](https://arxiv.org/abs/2210.13438v1); [Moshi §3.3–3.4](https://arxiv.org/abs/2410.00037v2); [Whisper §2.2–2.3](https://arxiv.org/abs/2212.04356v1).
- [Qwen2.5-Omni §2.1–2.4](https://arxiv.org/abs/2503.20215v1); [Qwen3-Omni §2.1–2.5](https://arxiv.org/abs/2509.17765v1); [Qwen3-VL §2.3](https://arxiv.org/abs/2511.21631v1); [ImageBind §3](https://arxiv.org/abs/2305.05665v1).
- [Stanford CS336 Lecture 17](https://cs336.stanford.edu/) and [Berkeley Advanced LLM Agents Meeting 7](https://rdi.berkeley.edu/adv-llm-agents/sp25).
- Previous: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|Documents, OCR, and grounding]]. Next: [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|Evaluation and VLM serving]].
