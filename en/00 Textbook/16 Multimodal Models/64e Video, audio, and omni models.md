---
title: "64.5. Video, audio, and omni models"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
last_updated: 2026-07-23
---

# Video, audio, and omni models

Video adds time to images; audio makes time continuous and sampling-rate dependent. Encoding every frame and short audio window quickly exhausts context. An omni model must select evidence, synchronize modalities, and often generate speech online.

A minute sampled at two frames per second contains 120 images. At 256 tokens per frame, that is 30,720 visual tokens before text. Systems use uniform or scene-aware sampling, tubelets, temporal pooling, clip hierarchies, and resamplers. Uniform sampling is reproducible but may miss a brief event; learned keyframes are cheaper but add another model and bias.

Frame index is not physical time. Qwen2.5-VL aligns temporal positions with seconds; Qwen3-VL adds timestamp tokens. Reordering frames, deleting the event, and changing sampling rate are necessary causal tests.

## Audio and audiovisual alignment

Waveforms are commonly resampled, converted to mel spectrograms, and encoded into short-window features. `ASR → LLM` is a strong, inspectable baseline, but transcription loses intonation, pauses, speaker identity, music, and non-speech events. Direct audio models retain them at higher sequence cost.

Audio and frames share a timeline but have different rates. Qwen2.5-Omni interleaves them by time and uses Time-aligned Multimodal RoPE.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure3-tmrope.png]]

*Figure 3 from Xu et al., [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), PDF p. 4. TMRoPE gives audio and video a shared temporal axis while vision retains spatial coordinates.*

## Thinker–Talker and streaming speech

Qwen2.5-Omni separates high-level understanding from speech generation. Thinker processes text, image, video, and audio and generates text. Talker consumes Thinker states and predicts speech-codec tokens as a stream.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure2-thinker-talker.png]]

*Figure 2 from [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), PDF p. 3. Thinker produces semantic states and text; Talker turns them into streaming speech tokens.*

Streaming changes the causal contract: input may continue while the model plans and speaks. The system must decide when to start, how to stop on interruption, and how much audio history to retain. Metrics include time to first audio, real-time factor, interruption latency, semantic accuracy, naturalness, and voice stability.

## Shared representations are not any-to-any generation

ImageBind creates one embedding space for text, image, audio, depth, thermal, and IMU data. That supports cross-modal retrieval but does not itself generate each modality. Image or speech output requires a dedicated decoder such as diffusion/flow, discrete image tokens, or an audio codec.

Evaluation must cover each modality and their interaction. Strong image scores cannot compensate for ignored audio; ASR accuracy does not prove audiovisual synchronization. Cost curves should be reported over frames, seconds, and multimodal tokens.

## Sources and next chapter

- [Flamingo](https://arxiv.org/abs/2204.14198), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), and [ImageBind](https://arxiv.org/abs/2305.05665).
- Previous: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|64.4]]. Next: [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|64.6]].
