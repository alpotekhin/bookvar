---
title: "64.5. Video, audio, and omni models"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
last_updated: 2026-09-15
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

A minute sampled at two frames per second contains 120 images. At 256 tokens per frame, that is 30,720 visual tokens before text. Systems use uniform or scene-aware sampling, tubelets, temporal pooling, clip hierarchies, and resamplers. Uniform sampling is reproducible but may miss a brief event; learned keyframes are cheaper but add another model and bias.

At 8 fps the same minute would occupy $60\cdot8\cdot256=122\,880$ visual positions. Pooling groups of four frames in the 2-fps example reduces the sequence to 7,680 positions, but may erase a brief motion. A projector changes feature width; a position-wise linear map or MLP does **not** reduce frame or patch count. Temporal pooling or a resampler is a separate operation.

Frame index is not physical time. Qwen2.5-VL scales temporal position IDs with seconds; [Qwen3-VL replaces that physical-time alignment with textual timestamps](https://arxiv.org/html/2511.21631v1#S2.SS3), while retaining interleaved M-RoPE. Reordering frames, deleting the event, and changing sampling rate are necessary causal tests.

### Shared architecture, different media budgets

OneVision uses the same vision encoder and projector for single images, multiple images, and video, but not the same token allocation per frame. A detailed image can use many crops; dozens of video frames must share the context budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-modalities.png]]

*Stanford CS336, Lecture 17, modality comparison based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Compare per-frame detail with total sequence length rather than treating “video support” as a binary capability.*

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

## Audio and audiovisual alignment

Waveforms are commonly resampled, converted to mel spectrograms, and encoded into short-window features. `ASR → LLM` is a strong, inspectable baseline, but transcription loses intonation, pauses, speaker identity, music, and non-speech events. Direct audio models retain them at higher sequence cost.

Audio and frames share a timeline but have different rates. Qwen2.5-Omni interleaves them by time and uses Time-aligned Multimodal RoPE.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure3-tmrope.png]]

*Figure 3 from Xu et al., [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), PDF p. 4. TMRoPE gives audio and video a shared temporal axis while vision retains spatial coordinates.*

### Continuous features and discrete speech codes

At a 20-ms feature stride, one minute of audio yields $60/0.02=3\,000$ positions before further downsampling. These continuous vectors can condition an LLM for perception. To generate sound, the system needs an output representation with a waveform decoder.

A neural codec encodes a waveform into compressed latents, quantizes them into codebook indices, and reconstructs audio from those indices. A speech model predicts the indices, while the codec decoder produces the waveform. The codec's reconstruction and adversarial objectives are not the same loss as the speech model's categorical prediction loss.

For an illustrative codec with 50 frames/s and eight codebooks, ten seconds requires

$$
10\cdot50\cdot8=4\,000
$$

indices. That does not necessarily mean 4,000 steps through the largest Transformer: time and codebook depth can be modeled separately. Always state frame rate, codebook count, and prediction schedule when reporting “audio tokens.”

| Pipeline | Inspectable intermediate | What can be lost | Deployment trade-off |
|---|---|---|---|
| ASR → LLM → TTS | transcript and answer text | prosody and non-speech cues | components can be replaced independently |
| Direct audio/omni | continuous states or codec tokens | depends on learned representation | richer signal but coupled models and scheduling |

A cascade remains a strong baseline for summarizing a lecture. For sarcasm, a sound event, or who is speaking, the transcript may have already discarded the relevant evidence.

For the door example, shift the soundtrack relative to the video while retaining the same frames and words. A question about simultaneous events should change its answer or become uncertain. This tests synchronization rather than separate ASR and image quality.

## Thinker–Talker and streaming speech

Qwen2.5-Omni separates high-level understanding from speech generation. Thinker processes text, image, video, and audio and generates text. Talker consumes Thinker states and predicts speech-codec tokens as a stream.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure2-thinker-talker.png]]

*Figure 2 from [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), PDF p. 3. Thinker produces semantic states and text; Talker turns them into streaming speech tokens.*

Streaming changes the causal contract: input may continue while the model plans and speaks. The system must decide when to start, how to stop on interruption, and how much audio history to retain. Metrics include time to first audio, real-time factor, interruption latency, semantic accuracy, naturalness, and voice stability.

### Moshi: one temporal step, several codebook decisions

[Moshi](https://arxiv.org/html/2410.00037v2#S3.SS4.SSS4) separates time from within-frame code prediction. Mimi's first codebook carries semantic information; the remaining codebooks refine acoustic detail. In the text-conditioned Inner Monologue arrangement, a linear head over the large Temporal Transformer's state predicts the text token first. A smaller Depth Transformer then predicts semantic and acoustic audio codes, conditioned on the temporal state and text. The audio-only introductory RQ formulation must not be confused with this text-first ordering.

Thus the large network advances once per codec time step rather than once per individual code. User audio and model speech occupy parallel streams on a shared timeline. Streaming means progressively emitting output; **full duplex** additionally means continuing to listen while speaking, including overlap and interruption.

The label “audio head” is consequently ambiguous. An input projector aligns audio feature width. A classification or ASR head predicts labels. A speech generator predicts a long structured codec sequence and may itself be a Transformer.

### Four live-dialogue decisions

An offline model knows when the recording ends. A live system must decide:

1. **Endpointing:** has the user finished or merely paused?
2. **Turn taking:** is it appropriate to begin answering?
3. **Incremental generation:** which audio chunk can be emitted without future context?
4. **Interruption:** how quickly can playback stop and the new input enter state?

Starting early can cut off the user; starting late creates an awkward pause. Semantic accuracy alone cannot measure these failures, and a real-time factor below one does not establish low interruption latency.

## Shared representations are not any-to-any generation

ImageBind creates one embedding space for text, image, audio, depth, thermal, and IMU data. That supports cross-modal retrieval but does not itself generate each modality. Image or speech output requires a dedicated decoder such as diffusion/flow, discrete image tokens, or an audio codec.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/imagebind-figure1-capabilities.png]]

*Girdhar et al., [ImageBind, Figure 1](https://arxiv.org/abs/2305.05665). Shared representations support retrieval and conditioning of a separate image generator; the diagram is not a streaming speech architecture.*

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

## Sources and next chapter

- [Flamingo](https://arxiv.org/abs/2204.14198), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), and [ImageBind](https://arxiv.org/abs/2305.05665).
- Previous: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|64.4]]. Next: [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|64.6]].
