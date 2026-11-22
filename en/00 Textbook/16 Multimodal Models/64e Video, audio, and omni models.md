---
title: "64.5. How models understand video and answer in speech"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "f95b360a17fb01791e6d1076ae0fd32af888a9cf6a2cbf6d3f670a6ff400df8d"
translation_reviewed: "2026-11-22"
translation_of: "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding]]"
next:
  - "[[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving]]"
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
  - https://arxiv.org/abs/2509.17765v1
  - https://arxiv.org/abs/2305.05665
  - https://doi.org/10.1038/s41598-025-17015-z
  - https://arxiv.org/abs/2410.00037
  - https://ffmpeg.org/doxygen/7.1/structAVStream.html
  - https://cs336.stanford.edu/
---

# How models understand video and answer in speech

In the [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|document chapter]],
“14 days” came from a particular cell on a page. Now the same
words appear on a sign in a one-minute video of a returns counter.
Someone opens a door at about 18.4 seconds and says “I'm back.”
Two seconds later we hear “Wait.” To answer what they said as
the door opened, find the movement in the frames and read the
sound from the same time.

A saved clip lets us go back and sample around the door more
closely. In a live call, later frames have not arrived. If the
system must speak, it also needs to turn the chosen words into
audio and decide when playback can start.

<span id="1-от-видеофайла-до-visual-tokens"></span>

## Which frames the model sees

A minute sampled at two frames per second contains 120 images. At 256 tokens per frame, that is 30,720 visual tokens before text. Systems use uniform or scene-aware sampling, tubelets, temporal pooling, clip hierarchies, and resamplers. Uniform sampling is reproducible but may miss a brief event; learned selection adds its own computation and bias even when it reduces the answering model's input.

At 8 fps the same minute would occupy $60\cdot8\cdot256=122\,880$ visual positions. Pooling groups of four frames in the 2-fps example reduces the sequence to 7,680 positions, but may erase a brief motion. A projector changes feature width; a position-wise linear map or MLP does **not** reduce frame or patch count. Temporal pooling or a resampler is a separate operation.

Let the door open during $[18.3,18.5)$ seconds. At 2 fps, samples at 18.0 and 18.5 miss the movement itself; changed states may suggest that a door opened, but its exact moment or a brief accompanying action is not observed. At 8 fps, a sample falls at 18.375:

<details id="omni-code-frame-sampling"><summary>Sampling at 2 and 8 fps</summary>

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

</details>

Scene cuts preserve editing boundaries, which may not match the
action asked about. A question-aware sampler can only select among
its candidates. Reducing a frame may leave the door visible but
erase strokes in “14 days.” Rereading the sign then requires a
higher-resolution frame, not just more frames.

Frame index is not physical time. Qwen2.5-VL scales temporal position IDs with seconds; [Qwen3-VL replaces that physical-time alignment with textual timestamps](https://arxiv.org/html/2511.21631v1#S2.SS3), while retaining interleaved M-RoPE. Reordering frames, deleting the event, and changing sampling rate are necessary causal tests.

<span id="один-интерфейс-для-изображения-набора-изображений-и-видео"></span>

### Shared architecture, different media budgets

OneVision uses the same vision encoder and projector for single images, multiple images, and video, but not the same token allocation per frame. A detailed image can use many crops; dozens of video frames must share the context budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-modalities.png|One encoder and projector across images, image sets, and video, with stronger spatial pooling when more frames share the context.]]

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

<details id="omni-code-temporal-pooling"><summary>Pooling opening and closing</summary>

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

</details>

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

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/gens-frame-selection.png|For a question about what follows a meeting, GenS retrieves meeting and cooking frames. The right panel compares scores on a LongVideoBench subset.]]

*Yao et al., [GenS, Figure 1](https://aclanthology.org/2025.findings-acl.921.pdf#page=2), CC BY 4.0. “Immediately after” requires a sequence, not one frame similar to the question's nouns. Plotted results concern the paper's specified LongVideoBench subset.*

A meeting image alone does not answer a question about subsequent cooking. Joint processing of observations and question supports that relation. Window bookkeeping is also part of retrieval correctness:

<details id="omni-code-frame-windows"><summary>Local and global frame indices</summary>

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

</details>

Overlapping windows require deduplication by original global frame identity. A local index is not physical time without the source mapping. GenS-Video-150K's synthetic pipeline builds detailed frame captions, generates frame-linked questions, expands candidate sets, and assigns relevance. Teacher mistakes can train a sampler to omit events that the answering VLM could otherwise understand. Count candidate processing as well as the answering model's input.

<span id="preserve-presentation-timestamps"></span>

## Time on the original recording

Multiply a frame's presentation timestamp by the duration of
one tick to get its time. [FFmpeg's AVStream](https://ffmpeg.org/doxygen/7.1/structAVStream.html)
specifies the time base per stream. Audio and video can use
different units, so compare seconds on the same timeline, not
raw timestamp integers. Dividing the new frame index by an
assumed frame rate can pick the wrong audio interval:

<details id="omni-code-pts"><summary>From PTS to seconds</summary>

```python
from fractions import Fraction

pts = [0, 15_000, 66_000]
time_base = Fraction(1, 30_000)
times = [p * time_base for p in pts]
assert times == [Fraction(0), Fraction("0.5"), Fraction("2.2")]
assumed_constant_rate = Fraction(2, 2)
assert times[2] != assumed_constant_rate
```

</details>

The third observation is at 2.2 seconds, not one second. A surviving frame at 18.4 seconds retains that timestamp after thinning. Sampling changes observations and model position sequences, not when the frame occurred. Qwen3-VL's textual timestamps and Qwen3-Omni's absolute 80-ms TM-RoPE grid are different interfaces despite the family name.

<span id="3-от-waveform-до-признаков-и-дискретных-аудиотокенов"></span>

<span id="audio-and-audiovisual-alignment"></span>

## Audio features and speech codes

Waveforms are commonly resampled, converted to mel spectrograms, and encoded into short-window features. An ASR → LLM cascade makes transcription inspectable, but plain text omits intonation, timing, speaker identity, music, and non-speech events unless these are supplied separately. Direct audio representations can preserve such evidence; what survives and its sequence cost depend on the encoder.

Sample rate describes the waveform grid, not speech speed or LLM token rate: one second of mono audio at 16 kHz contains 16,000 amplitudes. A spectrum of an entire minute would mix phonemes, pauses, and noise. STFT computes spectra over overlapping windows; mel filters combine frequency bins and a logarithm compresses energy range. An encoder contextualizes these features and can reduce temporal frequency through convolutions.

Original [Whisper §2.2](https://arxiv.org/abs/2212.04356v1) uses 16 kHz audio, 25-ms windows, 10-ms hops, 80 mel channels, and a second initial convolution that halves temporal frequency. Qwen2.5-Omni uses 128 mel channels and about 40 ms per output encoder position. These are version-specific. At 16 kHz the window and hop contain 400 and 160 samples. This FFT example precedes mel filtering, not a complete Whisper processor:

<details id="omni-code-fft"><summary>Frequency of a tone within a window</summary>

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

</details>

One unpadded second contains 98 complete windows, although a 10-ms hop corresponds to 100 windows/s in a long recording. Padding, truncation, and fixed-size inputs affect exact counts. Preserve feature-to-time mappings including edge rules. Continuous perception features are not automatically codes from which the waveform can be reconstructed.

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

<details id="omni-code-rvq"><summary>Two residual quantization steps</summary>

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

</details>

Real vectors are learned with the codec. [EnCodec §3.1–3.2](https://arxiv.org/abs/2210.13438v1) varies active RVQ levels to trade code volume against quality. Reconstruction alone does not guarantee that the first codebook represents words or phonemes.

### Mimi separates semantic and acoustic quantization

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/mimi-split-rvq.png|Mimi's semantic VQ and acoustic RVQ run in parallel. Their vectors are added before decoding.]]

*Défossez et al., [Moshi, Figure 2](https://arxiv.org/pdf/2410.00037v2#page=10), CC BY-NC-SA 4.0. Semantic VQ and acoustic RVQ are parallel branches; their outputs combine before decoding. Blue paths belong to training.*

Mimi is a causal codec at 12.5 steps/s, or 80 ms per step. One VQ branch is distilled toward WavLM features; a separate seven-level RVQ branch encodes acoustics. Each receives an independently projected encoder representation. Acoustic RVQ does not continue the semantic branch's residual. This split addresses conflicting semantic and acoustic objectives.

WavLM can see future context when producing training targets, but is absent from the deployed codec. Its use as teacher does not grant future-waveform access at inference. Generator and codec objectives also differ: the generator predicts indices with cross-entropy; EnCodec uses reconstruction and adversarial losses, while the described Mimi uses adversarial and feature losses without explicit reconstruction loss, retaining semantic distillation.

Eight 2048-entry codebooks require 11 bits per index: $12.5\cdot8=100$ indices/s, or 1100 bits/s before overhead. Ten seconds gives 125 time steps and 1000 indices. EnCodec's 24-kHz variant instead operates at 75 steps/s:

<details id="omni-code-mimi-rate"><summary>Mimi codes per second</summary>

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

</details>

“Eight audio tokens” or “one minute of sound” therefore does not specify sequence size without codec rate and codebook count.

<span id="4-cascade-и-end-to-end-speech-model"></span>

## A transcript or direct audio features

| Pipeline | What reaches the language model |
|---|---|
| ASR → LLM → TTS | A transcript and any extra metadata. You can inspect the ASR output and replace components independently. Prosody and ambient sound need a separate channel. |
| Direct audio | Audio features or codes. These can retain cues absent from text, but you need tests for what the learned representation preserves. The components share a trained interface. |

A cascade remains a strong baseline for summarizing a lecture. For sarcasm, a sound event, or who is speaking, the transcript may have already discarded the relevant evidence.

A streaming cascade can use partial ASR hypotheses, begin generating text, and synthesize completed fragments before the whole utterance is processed. Latency need not equal the sum of three complete stages. But partial ASR can revise a word, while spoken output cannot be retracted as easily as on-screen text. Losing prosody is a limitation of a text-only intermediate, not proof that every cascade is worse.

<span id="shared-representations-are-not-any-to-any-generation"></span>

## Matching sound and images with ImageBind

ImageBind creates one embedding space for text, image, audio, depth, thermal, and IMU data. That supports cross-modal retrieval but does not itself generate each modality. Image or speech output requires a dedicated decoder such as diffusion/flow, discrete image tokens, or an audio codec.

ImageBind trains modality encoders using pairs linked through images,
with a contrastive objective over normalized vectors. In the first
panel of [Girdhar et al.'s Figure 1](https://arxiv.org/pdf/2305.05665v2#page=1), crackling fire is
the audio query for image retrieval. The system ranks image vectors
by cosine similarity without transcribing the audio.

The second panel adds a bird image vector to a wave sound vector
and retrieves images that match the combination. The third produces
a new image with a separate pretrained generator conditioned on
ImageBind's audio embedding.

<a id="imagebind-audio-retrieval"></a>

### Using sound to pick an object in a photo

In the [Panda Act experiment](https://www.nature.com/articles/s41598-025-17015-z#Fig2),
the alarm is recorded separately from the photo of the table. A segmentation
model first identifies candidate objects in the photo and crops them out.
In the lower part of the diagram, the audio encoder takes the recording and
the image encoder takes those crops. For one recording $A_1$, read one row
of the matrix: $A_1\cdot I_1$, $A_1\cdot I_2$, and so on. The largest
similarity selects a crop. The system gets coordinates from the selected
object's mask in the photo, not from the direction of the sound. If the
segmentation model misses the clock, the clock is absent from the candidates.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-connectors-2026-11-13/panda-act-figure2.jpg|Panda Act diagram. At lower right, ImageBind compares audio vectors against vectors for candidate object crops. The segmentation model on the left supplies the crops.]]

*Yifan Li et al., [A framework for robotic manipulation tasks based on multiple zero shot models, Figure 2](https://www.nature.com/articles/s41598-025-17015-z#Fig2),
[original JPEG](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-17015-z/MediaObjects/41598_2025_17015_Fig2_HTML.jpg),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Complete original, unmodified. The lower matching matrix uses ImageBind;
other components handle the rest of the system.*

The row in the example below has three scores: $0.8$, $0.6$, and $-0.8$.
If the alarm clock is the first candidate, it wins. Reordering the crops
would change the winning column number, not the object with the highest score.

The values are illustrative, not actual ImageBind embeddings.

<details id="omni-code-imagebind"><summary>Compare the audio with three objects</summary>

```python
import numpy as np

audio = np.array([0.8, 0.6])
images = np.array([[1., 0.], [0., 1.], [-1., 0.]])
scores = images @ audio
assert np.allclose(scores, [0.8, 0.6, -0.8])
assert scores.argmax() == 0
```

</details>

The shared space supplies neither phoneme sequences nor waveform codes. Likewise, text-answering multimodal understanding need not support image/audio generation, and adding Talker does not provide video output. Verify the deployed API separately: supported inputs, simultaneous streams, continued listening during speech, and enabled output modes may differ from architectural capabilities. A causal codec does not stop an application from buffering an entire utterance.

## Matching audio to video

Audio and frames share a timeline but have different rates. Qwen2.5-Omni interleaves them by time and uses Time-aligned Multimodal RoPE.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure3-tmrope.png|Frames and audio share a time axis. Patches from one frame share their time index; brackets show two-second intervals.]]

*Figure 3 from Xu et al., [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), PDF p. 4. TMRoPE gives audio and video a shared temporal axis while vision retains spatial coordinates.*

Before a shared sequence offset, Qwen2.5-Omni's 40-ms grid gives time IDs 25 at one second, 50 at two seconds, and 460 at 18.4 seconds. Audio uses matching time/height/width IDs; video holds time fixed within a frame while height and width vary. Qwen3-Omni's 80-ms grid assigns 230 to 18.4 seconds:

<details id="omni-code-av-time"><summary>Time IDs and the audio slice</summary>

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

</details>

Qwen2.5-Omni groups visual features and then audio within two-second intervals. IDs still determine which observations coincide. Qwen3-Omni replaces rigid grouping with alignment through its 80-ms IDs. Ordering, encoding, and time grid form one contract; changing one while retaining old metadata creates misalignment.

This calculation put the first audio sample at time zero. If the
decoded PCM starts at 0.12 s on the shared timeline, the interval
$[18.0,18.8)$ becomes $[17.88,18.68)$ relative to the array.
At 16 kHz the slice runs from 286,080 to 298,880 instead of from
288,000 to 300,800. Both bounds move by 1,920 samples without
changing the duration. This assumes continuous PCM; if the recording
has gaps, keep timestamps for the individual segments. Resetting
both stream origins independently would erase their relative offset.

Suppose “I'm back” spans 18.24–18.72 seconds and “Wait” spans 20.24–20.72. The door at 18.4 overlaps the first. Shifting audio forward by two seconds removes that overlap; shifting backward by two places the second phrase at the event. This fixture retains the original Russian strings:

<details id="omni-code-audio-shift"><summary>Check a shifted soundtrack</summary>

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

</details>

This verifies interval arithmetic, not a VLM. A real experiment must modify the soundtrack and invalidate old audio features and transcripts. A small shift leaving the same phrase overlapping the event need not change the answer; demanding a change after every shift is an invalid test.

<span id="thinkertalker-and-streaming-speech"></span>

## How Qwen-Omni speaks its answer

Qwen2.5-Omni separates high-level understanding from speech generation. Thinker processes text, image, video, and audio and generates text. Talker consumes Thinker states and predicts speech-codec tokens as a stream.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure2-thinker-talker.png|Thinker receives text, sound, and vision. Talker uses its states and chosen text tokens to generate speech codes for the decoder.]]

*Figure 2 from [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), PDF p. 3. Thinker produces semantic states and text; Talker turns them into streaming speech tokens.*

Talker receives both Thinker representations and embeddings of the selected text tokens. Semantically similar words can sound different; hidden semantic similarity does not identify which word to pronounce. Talker can begin from available text and state before the entire answer is ready, without one audio token per BPE token. Codes pass through a Flow-Matching DiT to mel features and a modified BigVGAN to waveform.

### What Qwen3-Omni changes

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/qwen3omni-architecture.png|Qwen3-Omni's Talker produces the first code, MTP completes the frame, and causal Code2Wav reconstructs the waveform.]]

*Qwen Team, [Qwen3-Omni, Figure 2](https://arxiv.org/pdf/2509.17765v1#page=3), CC BY 4.0. Talker predicts the first code, a small MTP predicts remaining codes in that frame, and causal Code2Wav reconstructs audio.*

Qwen3-Omni conditions Talker on discrete text and multimodal features, no longer on the hidden states of Thinker's text positions. This allows text intervention before speech, such as external filtering or prepared text.

<a id="omni-aut"></a>

#### How AuT learns to encode audio

The [Qwen3-Omni authors, §2.2](https://arxiv.org/html/2509.17765v1#S2.SS2)
train AuT as an audio encoder and text decoder. At the bottom of the
figure, filterbank features enter three Conv2d layers, pass through
32 encoder attention layers and become `AuT Hidden`. During training,
the eight-layer decoder sees the audio representation and the previous
reference tokens. It predicts the next text token, including the end
marker. The training tasks include recognition and audio understanding;
this decoder does not reconstruct a waveform.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/omni-2026-11-09/qwen3omni-figure3-aut.png|AuT training path from audio features through Conv2d and the audio encoder to the text decoder]]

*Qwen Team, [Qwen3-Omni, Figure 3, p. 4](https://arxiv.org/pdf/2509.17765v1#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Qwen3-Omni uses the path up to `AuT Hidden`; the decoder above it
is used to train AuT.*

A 10-ms input hop means 100 feature positions per second. Eightfold
downsampling gives 12.5 positions per second: about 400 input and 50
output positions for four seconds, ignoring boundary effects. The
resulting 80-ms stride does not limit each feature's context to 80 ms.
The report describes attention windows ranging from one to eight seconds.

In Qwen3-Omni, AuT supplies continuous features to Thinker rather than
a transcript. The output codec instead accepts discrete indices from
Talker and MTP. The two streams can have the same 80-ms stride without
representing the same information.

#### How the first speech frame is generated

Talker chooses each frame's first code. A small MTP Transformer autoregressively predicts the indices for the remaining codebooks of that frame. Code2Wav, a causal ConvNet, needs all codes for the current frame and left context, but does not wait for a future block as the DiT does.

| Component | From Qwen2.5-Omni to Qwen3-Omni |
|---|---|
| Audio input | The Qwen2-Audio design (25 positions/s) gives way to AuT (12.5 positions/s). |
| Time grid | 40-ms IDs with two-second grouping become 80-ms IDs without that grouping. |
| Talker text input | The selected tokens remain; the hidden states for text positions are removed. |
| Code prediction | Instead of dual-track autoregression, Talker picks the first code and MTP completes the frame. |
| Waveform | A causal ConvNet replaces Flow-Matching DiT and BigVGAN. |

The Qwen2.5-Omni authors specify Whisper-large-v3 for encoder initialization
([report, §3.1](https://arxiv.org/html/2503.20215v1#S3.SS1)); the description
of the audio feature interface cites Qwen2-Audio (§2.2).

Thinker and Talker use MoE, but attention configuration and sequence length determine KV-cache size, not expert count. Lower temporal frequency reduces positions without eliminating AuT, MTP, or Code2Wav work.

Streaming changes the causal contract: input may continue while the model plans and speaks. The system must decide when to start, how to stop on interruption, and how much audio history to retain. Metrics include time to first audio, real-time factor, interruption latency, semantic accuracy, naturalness, and voice stability.

## How Moshi listens while speaking

### Moshi: one temporal step, several codebook decisions

[Moshi](https://arxiv.org/html/2410.00037v2#S3.SS4.SSS4) separates time from within-frame code prediction. Mimi's first codebook carries semantic information; the remaining codebooks refine acoustic detail. In the text-conditioned Inner Monologue arrangement, a linear head over the large Temporal Transformer's state predicts the text token first. A smaller Depth Transformer then predicts semantic and acoustic audio codes, conditioned on the temporal state and text. The audio-only introductory RQ formulation must not be confused with this text-first ordering.

Thus the large network advances once per codec time step rather than once per individual code. User audio and model speech occupy parallel streams on a shared timeline. Streaming means progressively emitting output; **full duplex** additionally means continuing to listen while speaking, including overlap and interruption.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/moshi-rq-transformer.png|The large Transformer advances in time; the smaller one predicts components within the current step.]]

*Défossez et al., [Moshi, Figure 3](https://arxiv.org/pdf/2410.00037v2#page=13), CC BY-NC-SA 4.0. Embeddings from the previous step enter the Temporal Transformer; the Depth Transformer predicts components of the new step. Four illustrated components are schematic, not Moshi's full stream count.*

One Mimi stream requires $12.5\cdot8=100$ code predictions per second. Flattening codes requires 100 large-model steps/s; temporal/depth factorization requires 12.5 large-model steps plus small-model decisions. Codes are not removed. Two audio streams plus one text stream give $K=2\cdot8+1=17$ components per step. In application, the user's stream comes from actual user audio, not substituted model predictions:

<details id="omni-code-moshi-steps"><summary>Moshi time steps and codes</summary>

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

</details>

The conditional product $p(a)p(b\mid a)$ retains within-step dependence; replacing it with $p(a)p(b)$ assumes independence. The illustrative probabilities and step counts are not a Moshi speed benchmark.

<a id="omni-duplex"></a>

### Inner Monologue, delayed acoustics, and duplex streams

Moshi's Inner Monologue is its own speech-aligned text stream, not a requirement to generate hidden reasoning. PAD fills positions between words and EPAD marks the end of padding before the next word.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/moshi-duplex-streams.png|User audio runs above the dashed line, Moshi's text and audio below it. Acoustic codes lag by one step.]]

*Défossez et al., [Moshi, Figure 4](https://arxiv.org/pdf/2410.00037v2#page=15), CC BY-NC-SA 4.0. Model text/audio appear below the dashed line and user audio above it. Acoustic rows lag semantic rows by one step.*

With acoustic delay $\tau=1$, step $s$ carries semantic code for frame $s$ and acoustic codes for frame $s-1$. Three frames need this schedule:

| Step | Semantic frame | Acoustic frame |
|---|---|---|
| 0 | 0 | not yet available |
| 1 | 1 | 0 |
| 2 | 2 | 1 |
| 3 | completion | 2 |

<details id="omni-code-acoustic-delay"><summary>Schedule delayed acoustic codes</summary>

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

</details>

The final acoustic frame needs a flush step. The simplified zero-based schedule omits actual special symbols and state handling. At 80 ms per step, the delay adds one interval before a complete frame's codes become available. It provides later semantic context for acoustics, not access to user audio that has not arrived.

Moshi §3.4.4 uses relative text/audio delays for ASR and TTS modes. Supplying actual audio codes and selecting text produces streaming recognition; supplying a leading text stream conditions synthesis. Word alignment, padding, and delay determine that condition. Two audio streams represent simultaneous speech and silence without compulsory non-overlapping turns. Chunked TTS with the microphone disabled is not full duplex.

The label “audio head” is consequently ambiguous. An input projector aligns audio feature width. A classification or ASR head predicts labels. A speech generator predicts a long structured codec sequence and may itself be a Transformer.

<span id="four-live-dialogue-decisions"></span>

## When playback can start and stop

An offline model knows when the recording ends. A live system must decide:

1. **Endpointing:** has the user finished or merely paused?
2. **Turn taking:** is it appropriate to begin answering?
3. **Incremental generation:** which audio chunk can be emitted without future context?
4. **Interruption:** how quickly can playback stop and the new input enter state?

Starting early can cut off the user; starting late creates an awkward pause. Semantic accuracy alone cannot measure these failures, and a real-time factor below one does not establish low interruption latency.

<span id="7-streaming-меняет-причинный-контракт"></span>

### A streaming interface can still have lookahead

A bidirectional two-second input block can make its earliest representation depend on the block's end. It can be processed incrementally by blocks but must wait for that context. Qwen2.5-Omni's input audio encoder uses two-second blocks. Its output DiT has a separate rule: current block, two previous blocks, and one future block. Input-block duration is not the output codec-block duration.

![[00 Учебник/Assets/Figures/curated/omni-review-2026-10/qwen25omni-block-attention.png|The output DiT reads the current speech-code block, two previous blocks, and one future block before reconstructing mel features.]]

*Xu et al., [Qwen2.5-Omni, Figure 4](https://arxiv.org/pdf/2503.20215v1#page=5), CC BY 4.0. The sliding-window DiT mask belongs to codec-to-waveform decoding and includes one future block, not the input audio encoder's mask.*

Output block 3 needs blocks 1, 2, 3, and 4. A hypothetical 160-ms output block with one-block lookahead adds 160 ms of waiting. This duration is a teaching assumption, not a recovered Qwen parameter; removing lookahead changes the model's conditioning mask.

<details id="omni-code-lookahead"><summary>DiT attention window</summary>

```python
import numpy as np

query = np.arange(6)[:, None]
key = np.arange(6)[None, :]
allowed = (key >= query - 2) & (key <= query + 1)
assert np.flatnonzero(allowed[3]).tolist() == [1, 2, 3, 4]
assert allowed[3, 4] and not allowed[3, 5]
assert 1 * 0.160 == 0.160
```

</details>

A causal operator's available prefix must not change when future inputs change. A streaming convolution reproduces full causal convolution if it preserves boundary history:

<details id="omni-code-causal-conv"><summary>Causal convolution by chunks</summary>

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

</details>

The centered convolution at the end violates that property by using a future neighbor. For a real codec, inspect convolution padding, temporal normalization, attention, cache, and waveform stitching; passing one chunk-size test does not establish correctness at all boundaries.

### First audible sound and uninterrupted playback

First-audio latency includes required input context, feature preparation, Thinker/Talker computation, codec decoding, transport, and playback. A “first token generated” log covers only one stage.

<a id="omni-first-packet"></a>

#### What the 234 ms in the Qwen3-Omni report measures

[Table 2](https://arxiv.org/html/2509.17765v1#S2.T2) adds five stages:
preprocessing the final input chunk, Thinker's first token, Talker's
first code, MTP and waveform decoding. For one audio stream, the sum is
$72+88+57+14+3=234$ ms. For one video stream with audio it is
$160+160+210+14+3=547$ ms. With four concurrent audio streams the
first-packet estimate rises to 728 ms, of which 468 ms is Thinker's
first-token time. Shaving a few milliseconds off the final decoder
would barely move that result.

The authors derive the first-packet estimate from stage timings on
vLLM, with `torch.compile` and CUDA Graph for MTP and the decoder.
It does not cover the full path from the start of speech to sound from
the speaker. For a client measurement, state the start event, such as
the end of the user's utterance, and record when playback actually starts.
Track endpointing, transport and playback buffering separately from
packet generation.

The reported generation RTF of 0.47 is a separate calculation.
At one concurrent stream, a Thinker token takes about 13.3 ms
and a Talker token about 7.1 ms at the reported rates. Add 14 ms
for MTP and 3 ms for decoding, then divide by 80 ms of audio.
This estimate does not check that each chunk arrives by its
playback deadline.

<a id="omni-playback-gaps"></a>

#### A playback stall even when RTF is below one

Real-time factor divides compute duration by generated audio duration. A mean below one does not prevent an individual late chunk from exhausting the buffer:

<details id="omni-code-rtf"><summary>RTF and chunk deadlines</summary>

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

</details>

The third and fourth chunks miss their original deadlines, but there is
only one stall. Playback runs out after the second chunk at 180 ms and
resumes at 290 ms. The third chunk ends at 370 ms, by which time the
fourth has already arrived at 310 ms. A player that does not drop chunks
starts each chunk no earlier than its arrival and the previous chunk's end.

<details id="omni-code-playback"><summary>Playback buffering and stalls</summary>

```python
def playback(arrivals_ms, durations_ms, first_start_ms):
    end = first_start_ms
    starts, gaps = [], []
    for arrival, duration in zip(arrivals_ms, durations_ms, strict=True):
        start = max(arrival, end)
        starts.append(start)
        gaps.append(start - end)
        end = start + duration
    return starts, gaps, end

arrivals_ms = [20, 40, 290, 310]
durations_ms = [80] * 4
assert playback(arrivals_ms, durations_ms, 20) == (
    [20, 100, 290, 370], [0, 0, 110, 0], 450
)
uninterrupted_start = max(a - 80 * i for i, a in enumerate(arrivals_ms))
assert uninterrupted_start == 130
assert playback(arrivals_ms, durations_ms, uninterrupted_start) == (
    [130, 210, 290, 370], [0, 0, 0, 0], 450
)
```

</details>

Starting at 130 ms makes this particular stream play continuously. Both
strategies finish at 450 ms: starting earlier trades 110 ms of waiting
before playback for a 110-ms stall during it. The buffer calculation uses
known arrivals and does not guarantee uninterrupted playback on new requests.

Interruption requires continued microphone input, speech detection, generation cancellation, and removal of unplayed audio. Samples already sent to a device may remain audible after server cancellation. For user speech at 1000 ms, detection at 1040, cancellation at 1055, and final audible model sound at 1180, interruption latency is 180 ms, not 55. The other 125 ms belongs to buffered output and playback. These are illustrative event timestamps, not product measurements.

Conversation state must record what was actually played. If a full sentence was generated but only its first half heard, the next turn should not automatically treat the entire sentence as spoken.

<span id="9-матрица-оценивания"></span>

## How to score the answer

Evaluation must cover each modality and their interaction. Strong image scores cannot compensate for ignored audio; ASR accuracy does not prove audiovisual synchronization. Cost curves should be reported over frames, seconds, and multimodal tokens.

| What to test | What to measure or change |
|---|---|
| Event order | Swap the events and check their predicted order. |
| Moment localization | Compare intervals with temporal IoU. Then remove the event and check for a false report. |
| Speech recognition | Measure WER or CER as noise changes but the spoken words stay fixed. |
| Non-speech audio | Measure event F1 or mAP when the background sound changes. |
| Audio–video alignment | Shift the soundtrack and score whether the answer follows the new pairing. |
| Spoken answer | Compare the text and spoken meanings; evaluate sound in a listening test. |
| Streaming | Record first sound, stalls, RTF, and the time to stop after an interruption. |

For the door question, the answer must identify the correct speech at the correct event, not merely mention something said elsewhere. Report quality as a function of frame rate, visual budget, audio duration, and end-to-end latency.

Temporal IoU divides interval intersection duration by union duration. Ground truth $[18.3,18.7]$ and prediction $[18,19]$ both contain 18.4, but have IoU only 0.4:

<details id="omni-code-interval-iou"><summary>Temporal overlap of two intervals</summary>

```python
from math import isclose

gold, predicted = (18.3, 18.7), (18.0, 19.0)
intersection = max(0, min(gold[1], predicted[1]) - max(gold[0], predicted[0]))
union = (gold[1] - gold[0]) + (predicted[1] - predicted[0]) - intersection
assert isclose(intersection / union, 0.4)
assert gold[0] <= 18.4 < gold[1] and predicted[0] <= 18.4 < predicted[1]
```

</details>

Measure audio under specified noise conditions. $\mathrm{SNR}_{dB}=10\log_{10}(P_{signal}/P_{noise})$; 10 dB means signal power is ten times noise power, not that every word remains safe from corruption. Noise spectrum, timing, and event type matter:

<details id="omni-code-snr"><summary>Set the signal-to-noise ratio</summary>

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

</details>

These checks verify formulas, not model quality. Compare systems on the same clips and questions with equal audio access and comparable frame budgets. Service comparisons additionally include sampler, encoders, speech decoder, resource use, and client playback. Recognition, audiovisual alignment, response semantics, and blind listening quality need separate evidence.

The [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|next chapter]]
separates errors in the answer from cases where a correct answer
reaches the client too late.

## Sources and next chapter

- [Flamingo](https://arxiv.org/abs/2204.14198), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), and [ImageBind](https://arxiv.org/abs/2305.05665).
- [xGen-MM-Vid §2.1–2.3](https://arxiv.org/abs/2410.16267v2); [GenS §2](https://aclanthology.org/2025.findings-acl.921/); [EnCodec §3.1–3.2](https://arxiv.org/abs/2210.13438v1); [Moshi §3.3–3.4](https://arxiv.org/abs/2410.00037v2); [Whisper §2.2–2.3](https://arxiv.org/abs/2212.04356v1).
- [Qwen2.5-Omni §2.1–2.4](https://arxiv.org/abs/2503.20215v1); [Qwen3-Omni §2.1–2.5](https://arxiv.org/abs/2509.17765v1); [Qwen3-VL §2.3](https://arxiv.org/abs/2511.21631v1); [ImageBind §3](https://arxiv.org/abs/2305.05665v1).
- [Stanford CS336 Lecture 17](https://cs336.stanford.edu/) and [Berkeley Advanced LLM Agents Meeting 7](https://rdi.berkeley.edu/adv-llm-agents/sp25).
- Previous: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|Documents, OCR, and grounding]]. Next: [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|Evaluation and VLM serving]].
