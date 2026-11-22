---
title: "64.3. Training vision-language models"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "66f36ca58a55b60c092ae7c6f2d626264e4f89890e7c549f167c083fd617818d"
translation_reviewed: "2026-11-22"
translation_of: "00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions]]"
next:
  - "[[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding]]"
primary_sources:
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2304.08485
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2301.12597
  - https://arxiv.org/abs/2308.12966
  - https://arxiv.org/abs/2511.21631
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2504.10479v1
  - https://arxiv.org/abs/2508.18265v1
  - https://arxiv.org/abs/2409.17146
  - https://proceedings.mlr.press/v235/karamcheti24a.html
  - https://cs336.stanford.edu/
---

# Training vision-language models

On the [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions#vlm-pixel-budget|scanned form from the previous chapter]],
the cell at the right gives the return period: “14 days.” A question
about the return period calls for “14 days.” A transcription task
can ask for both the field's name and its value, exactly as printed.
A request to point out where the period is printed calls for
the cell's coordinates. The same image supports all three tasks,
but each has its own target and way to check the answer.

A [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|projector and a language decoder]]
let the LLM receive the image features. Training shows it an image
and a prompt, then computes the loss against the annotated answer.
The projector can be the only component trained at one stage; the
vision encoder or the LLM may also be updated at another.

<span id="what-changes-at-each-stage"></span>

## LLaVA's two training stages

LLaVA starts with an already trained vision encoder. Models such
as CLIP train that encoder by matching images to text. BLIP-2 then
trains its Q-Former with contrastive, matching, and generation
objectives. LLaVA's projector learns from prediction errors on
caption tokens passed through the language model.

[Original LLaVA, §4.2](https://arxiv.org/html/2304.08485v2#S4.SS2),
uses next-token prediction at both stages. The first trains only
the projector on 595,000 filtered CC3M image–caption pairs. The
second trains the projector and LLM on visual instructions,
including LLaVA-Instruct-158K for the conversational variant.
The vision encoder stays frozen. A caption might describe the
scan as a return form without mentioning the deadline. A question
about the deadline makes the value of that field the target.
LLaVA-1.5 and OneVision use different training mixtures.

## Backpropagation through the frozen LLM

Let $Z_v$ be frozen vision features and $H_v=g_\phi(Z_v)$ the connector output. Caption alignment can use

$$
\mathcal L_{\mathrm{caption}}
=-\sum_{t=1}^{T}\log p_\theta(y_t\mid H_v,q,y_{<t}).
$$

The LLM parameters $\theta$ stay fixed while $\phi$ changes. Backpropagation still passes through the LLM to $H_v$: otherwise the projector receives no learning signal. Dimension matching is necessary, but the real task is finding continuous inputs that the pretrained language model can use.

Here $q$ is the prompt. The conditional probability factors into
next-token probabilities, including the answer terminator if used
in the target. With teacher forcing, “days” is predicted from the
image, the question, and the annotated token “14,” even if the
model assigned another number a higher probability. The connector
gradient is

$$
\nabla_\phi\mathcal L
=\left(\frac{\partial H_v}{\partial\phi}\right)^\top\nabla_{H_v}\mathcal L.
$$

This does not require each visual vector to coincide with a word embedding. It requires useful conditioning of the fixed decoder.

### One projector update

Use two possible answers, red and blue, a visual feature $(1,0)$, an identity frozen output head, and an initially zero projector. Both logits are zero, so the correct answer has probability 0.5 and loss $-\log0.5\approx0.6931$. The logit gradient is $(-0.5,0.5)$. An SGD step of 0.2 changes the projector's first column to $(0.1,-0.1)$; the new correct probability is about 0.5498 and loss 0.5981:

<details id="training-code-projector"><summary>One projector update</summary>

```python
import torch
import torch.nn.functional as F

projector = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
frozen_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
with torch.no_grad():
    projector.weight.zero_()
    frozen_head.weight.copy_(torch.eye(2))
frozen_head.requires_grad_(False)

visual = torch.tensor([[1., 0.]], dtype=torch.float64)
target = torch.tensor([0])
optimizer = torch.optim.SGD(projector.parameters(), lr=.2)
before = F.cross_entropy(frozen_head(projector(visual)), target)
before.backward()
assert frozen_head.weight.grad is None
assert torch.allclose(
    projector.weight.grad,
    torch.tensor([[-.5, 0.], [.5, 0.]], dtype=torch.float64),
)
optimizer.step()
after = F.cross_entropy(frozen_head(projector(visual)), target)
assert abs(before.item() - .69314718) < 1e-7
assert abs(after.item() - .59813887) < 1e-7
```

</details>

This tests one conditional token and a linear output head, not visual color recognition. A real LLM inserts decoder blocks into the same chain rule. Setting its weights' `requires_grad=False` preserves input differentiation; placing the entire LLM forward in `no_grad()` would break it. A frozen vision encoder can run without a graph if there are no trainable modules before it.

<a id="vlm-training-targets"></a>

<span id="loss-masks-are-not-attention-masks"></span>

## Which tokens contribute to the loss

Take the question “What is the return period?” with the answer
“14 days.” and an end marker. The example assigns one position to
each number, word, and punctuation mark for illustration.

| Position | Input | Label |
| ---: | --- | --- |
| 0 | `<user>` | ignore |
| 1 | visual A | ignore |
| 2 | visual B | ignore |
| 3 | “What is the return period?” | ignore |
| 4 | `<assistant>` | ignore |
| 5 | `14` | `14` |
| 6 | `days` | `days` |
| 7 | `.` | `.` |
| 8 | `<eos>` | `<eos>` |

The response loss is

$$
\begin{gathered}
\mathcal L=-\sum_t m_t\log p(y_t\mid y_{<t},I),
\\
m_t=\mathbf1[\text{assistant target at }t].
\end{gathered}
$$

A zero loss mask does not hide an input from attention. Image and question positions must remain usable as conditions. In a multi-turn conversation, several assistant responses can contribute loss, but no response may see its future tokens. When packing independent dialogues, separate their attention regions so an earlier example does not silently become another example's context.

### Shift targets before computing cross-entropy

The first five labels are ignored. Logits at position 4 predict `14` at position
5. The next three positions predict `days`, the full stop, and EOS.
Shifting the labels keeps the answer token out of the context used
to predict it. If the model assigns the four correct tokens
probabilities 0.5, 0.25, 0.125, and 0.5, the negative
log-probabilities sum to about 4.8520 and average to 1.2130:

<details id="training-code-targets"><summary>Loss after shifting the labels</summary>

```python
import torch
import torch.nn.functional as F

labels = torch.tensor([-100] * 5 + [0, 1, 2, 3])
prob = torch.full((9, 4), .25, dtype=torch.float64)
for row, target, p in [(4, 0, .5), (5, 1, .25),
                       (6, 2, .125), (7, 3, .5)]:
    prob[row].fill_((1 - p) / 3)
    prob[row, target] = p
logits = prob.log().requires_grad_()
loss = F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)
loss.backward()
active_rows = logits.grad.abs().sum(dim=1).nonzero().flatten().tolist()
assert active_rows == [4, 5, 6, 7]
assert abs(loss.item() - 1.21300757) < 1e-7
```

</details>

Zero gradient on an ignored row's logits does not imply zero gradient on its input embedding in a Transformer: later answers can use that input through attention. Original LLaVA supervises answer terminators. Qwen-VL §3.3 supervises answers and designated special tokens, but not role names or user questions. The chat template determines exact boundaries; “assistant-only loss” does not specify them completely.

<a id="vlm-training-packing"></a>

### Independent packed samples and shared-image branches

For independent sequences of lengths five and four, packed boundaries are `[0,5,9]` and positions restart at each sequence. A plain causal mask would let the second dialogue read the first one's answer, so attention must also respect sample identity. Image indexing, target boundaries, and loss denominators must follow the same separation.

Multimodal packing has a second budget: under limits of 2048 positions and eight tiles, requests of 700 positions/three tiles and 1000 positions/six tiles fit by length but exceed the visual budget. Pairing the first with 600 positions/four tiles fits both, at 1300 positions and seven tiles. [InternVL 2.5 §4.2](https://arxiv.org/abs/2412.05271v1) describes this dual constraint.

<details id="training-code-packing"><summary>Two masks for packed sequences</summary>

```python
import numpy as np

sample = np.repeat([0, 1], [5, 4])
index = np.arange(9)
allowed = ((sample[:, None] == sample[None, :]) &
           (index[:, None] >= index[None, :]))
assert not allowed[5:, :5].any()
assert allowed.sum() == 5 * 6 // 2 + 4 * 5 // 2
assert 700 + 1000 <= 2048 and 3 + 6 > 8
assert 700 + 600 <= 2048 and 3 + 4 <= 8

branch = np.array([-1, -1, -1, 0, 0, 1, 1])
causal = np.arange(7)[:, None] >= np.arange(7)[None, :]
shared = causal & ((branch[:, None] == branch[None, :]) |
                   (branch[None, :] == -1))
assert shared[5:, :3].all()
assert not shared[5:, 3:5].any()
assert shared.sum() == 24
```

</details>

The second mask models Molmo's shared-image prefix. Positions
5–6 see image positions 0–2 but not the other question and answer
at 3–4. In one multi-turn conversation, the second answer should
see the first.

For two independent dialogues, shifting labels would make position
4 predict position 5, the start of the next dialogue. Ignore the
label at 5 if it is not already masked. The attention mask does not
remove that target. Assign the image vectors to the correct
positions and count the remaining supervised tokens separately.

## Frozen and trainable components

Connector-only training needs no weight gradients or optimizer
state for the frozen encoder and LLM. Backpropagating to the
projector can still require the LLM's activations. If preprocessing
crops out the return cell, no weight update puts those pixels
back. If the cell is visible but the encoder cannot distinguish
“7” from “14,” training the encoder can change its visual features.
Training the LLM can change how it reads those features and answers.
Check for regressions on text-only and earlier visual tasks.

<a id="vlm-internvl25-stages"></a>

### InternVL 2.5: transferring the vision encoder

[InternVL 2.5, §3.2 and Table 3](https://arxiv.org/html/2412.05271v1#S3.SS2)
uses the same pre-training mixture in stages 1 and 1.5. What changes
is which weights the next-token prediction loss can update. Stage 2
switches to curated multimodal instructions.

| Stage | Updated | Frozen |
| --- | --- | --- |
| 1. Warmup | MLP | ViT, LLM |
| 1.5. ViT | ViT, MLP | LLM |
| 2. SFT | ViT, MLP, LLM | — |

Consider the return-period answer. In stage 1 its error
backpropagates through the LLM to the projector: it can change the
vectors the LLM receives, but not the features computed by the ViT.
In stage 1.5, the ViT features can change too. The authors used that
stage to adapt vision to domains such as multilingual OCR and charts
without updating the language model.

InternVL2.5-26B includes stage 1.5; 78B skips it. In
[§3.3](https://arxiv.org/html/2412.05271v1#S3.SS3), the authors describe
the transfer with 20B and 72B language models: adapt InternViT with the
smaller LLM, then reuse those vision weights with the larger one. The
new pair still needs stage 1 to fit the projector to the new LLM's
input space. There is no separate encoder adaptation stage, but its
weights are trainable again in stage 2. In
[Zhe Chen et al., original Figure 4(b)](https://arxiv.org/pdf/2412.05271v1#page=6),
the `shared weights` arrow starts at the ViT, not the projector.

Molmo [§4](https://arxiv.org/abs/2409.17146v2) skips connector-only
warmup and trains all components directly on PixMo-Cap. The connector
uses learning rate $2\cdot10^{-4}$ with 200 warmup steps, the ViT
$6\cdot10^{-6}$ and the LLM $2\cdot10^{-5}$ with 2000 warmup steps
each. InternVL 2.5 instead uses one learning rate across the model in
stage 2. The freeze schedule and the learning rates are separate choices.

<a id="vlm-internvl-native-training"></a>

### InternVL 3 and 3.5: joint updates from the first stage

[InternVL 3, §2.2](https://arxiv.org/html/2504.10479v1#S2.SS2)
updates the ViT, projector and LLM together from the start of
multimodal pre-training. Its mixture includes both text-only and
multimodal examples. [InternVL 3.5, §§2.1–2.2](https://arxiv.org/html/2508.18265v1#S2.SS2)
keeps this schedule. Both the vision encoder and language model
start from pretrained weights. Here `native` describes when
joint training starts, not random initialization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-training-2026-11-13/internvl35-training-process.png|Left to right: joint pre-training, SFT, and Cascade RL. The Flash variant adds ViCO to support different levels of visual compression.]]

*Weiyun Wang et al., [InternVL3.5, Figure 3](https://arxiv.org/html/2508.18265v1#S2.F3),
[original PNG](https://arxiv.org/html/2508.18265v1/training_process.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified.*

In the first block, the error on the return-period answer can already
update all three components. Visual vectors still act as inputs.
The model predicts text and, in dialogue examples, loss is applied
to response tokens. In the second block, SFT keeps that objective
and switches to curated conversations. The labels `250B tokens`
and `130B tokens` count tokens at those two stages; Cascade RL's
`270K samples` counts examples. The final ViCO block is for
[[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions#vlm-internvl-resolution-router|the Flash variant with a resolution router]].

In the [InternVL 3 ablation, §3.14 and Figure 3](https://arxiv.org/html/2504.10479v1#S3.SS14),
the authors kept the architecture, initialization, and training data
of InternVL2-8B fixed and replaced MLP-only warmup with joint
pre-training. On most of the benchmarks shown, pre-training alone
gave results comparable to the full two-stage baseline. Instruction
tuning improved them further. The ablation compares schedules
for InternVL2-8B, not the performance of releases 2.5 and 3.5.

<a id="vlm-training-freeze"></a>

### What changed with the same starting checkpoints

In [Prismatic VLMs, §4.1](https://proceedings.mlr.press/v235/karamcheti24a.html), the authors reproduce
the two-stage LLaVA-1.5 recipe and then remove the projector-only stage.
The single-stage model still trains the projector, now together with
the LLM on instruction data. The ViT stays frozen.

![[00 Учебник/Assets/Figures/curated/vlm-training-2026-11-09/prismatic-figure4.png|Two charts compare the published, reproduced, and single-stage LLaVA results for 7B and 13B. The panel on the right gives training times on eight A100s.]]

*Siddharth Karamcheti et al., [Prismatic VLMs, Figure 4, p. 5](https://raw.githubusercontent.com/mlresearch/v235/main/assets/karamcheti24a/karamcheti24a.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Compare orange with green on a given axis: that matches the single-stage
run against the reproduced two-stage run. The gray part of each bar
on the right is the time spent on the removed stage.*

For 7B, skipping the first stage reduces time on eight A100s from
$2.72+8.56=11.28$ to 8.56 hours, about 24%. In Table 2,
VizWiz rises from 51.26 to 54.33; TextVQA without OCR prompts falls
from 46.44 to 44.45. The authors aggregate normalized scores over 12
benchmarks, not the area of the radar plot.

In a separate ablation they freeze the ViT or train it alongside
the projector and LLM. For single-stage training, unfreezing takes
RefCOCO accuracy from 64.08 to 42.56 under the protocol's 0.5 IoU
threshold. VizWiz changes from 54.33 to 55.26 (Tables 2–3).
The two tasks move in opposite directions.

![[00 Учебник/Assets/Figures/curated/vlm-training-2026-11-09/prismatic-figure5.png|The orange line shows a frozen ViT. The dashed gray line shows a fine-tuned ViT under the same single-stage schedule. The largest differences are on localization tasks.]]

*Karamcheti et al., [Figure 5, p. 5](https://raw.githubusercontent.com/mlresearch/v235/main/assets/karamcheti24a/karamcheti24a.pdf#page=5),
CC BY 4.0. To compare freezing and unfreezing within one schedule,
use the orange and dashed curves.*

These results hold for the data, checkpoints and settings in Prismatic.
The InternVL and Molmo recipes change both data and schedules, so
they are not counterexamples to this comparison.

<span id="5-какие-данные-создают-какие-способности"></span>

<span id="data-creates-different-visual-behaviors"></span>

## How visual annotations are collected

A short caption can describe a scan without naming any of its
fields. A document question can ask for the value of one field.
Transcription requires copying characters as printed, and grounding
pairs a reference to a field with a point or a box. These targets
can all be prepared for the same image.

[Figure 1 of Molmo and PixMo](https://arxiv.org/pdf/2409.17146v2#page=2)
shows how annotations are obtained on the left and the resulting tasks on
the right. A person supplies the coordinates in PixMo-Points; in
PixMo-Clocks, the time is set before the clock face is drawn.

PixMo's dense descriptions were collected by asking people to speak about an image and transcribing that speech. Point annotations provide a different supervision channel: the output location can be checked, and grounded counting can be audited by inspecting the individual points. The annotation interface therefore shapes what the model learns, not only how quickly examples are collected.

Initially PixMo-Cap requested at least a minute of speech; later collection used one annotator with a 90-second minimum. Transcription and language-model processing converted speech into captions. The paper reports 712,000 images, 1.3 million transcripts/captions, and a mean final caption length of about 196 words. Recording duration describes collection, not a guarantee that every fact is correct.

In PixMo-AskModelAnything, a person selected an image and asked a question. A language model received OCR and a detailed caption, proposed an answer, and the person accepted or corrected it against the image. The human saw pixels; the text assistant depended on the supplied evidence. PixMo-Points uses human point annotations. Counting examples list points before the count, allowing inspection of the counted objects, but valid coordinates alone do not prove that every object was found or that none was counted twice.

PixMo-Docs renders generated code into tables and diagrams; that code
supplies values and structure for questions. PixMo-Clocks fixes the time
when constructing a clock face. PixMo-Count relies on detections, and
PixMo-CapQA on captions, so their annotations can inherit errors from
the detector or the caption.

<a id="vlm-pixmo-code-guided"></a>

Code can also supply the geometry of an interface. In the later
**MolmoPoint-GUISyn** dataset, an LLM writes HTML to imitate an AutoCAD
window. A browser renders it, and a Playwright script extracts bounding
boxes for the visible elements. Follow the `Measure` button in the figure:
its location is already known when the text model writes an instruction
about what to do with it, such as measuring a length. In the training
example the VLM receives the image and the instruction, and the center
of that bounding box is the target point.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-training-2026-11-13/molmopoint-guisyn-figure3.png|An LLM writes HTML for a simulated AutoCAD interface. The browser renders the interface and a script extracts bounding boxes. The Measure, Add Pick Points, and Cancel boxes are paired with natural-language intents.]]

*Christopher Clark et al., [MolmoPoint, Figure 3](https://arxiv.org/html/2603.28069v1#S4.F3),
[§4.4 and Appendix 10](https://arxiv.org/html/2603.28069v1#S10),
[original PNG](https://arxiv.org/html/2603.28069v1/MolmoPSyn_overview.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified figure.*

The authors report 36,000 screenshots, 2 million annotated points,
and over 10 million instructions. The script obtains locations
without inferring them from pixels. It does not check what an instruction means: if two
buttons can measure a length, a correct bounding-box center still leaves
the task ambiguous.

Task or style tags distinguish short benchmark answers from open
conversation; without that distinction, a strict extraction format can
become the default for all requests.

<span id="instruction-and-preference-tuning"></span>

### What LLaVA's text-only teacher received

LLaVA used GPT-4 over captions and symbolic descriptions to produce
conversations, detailed descriptions, and reasoning prompts. Those inputs
were its only evidence about the image. The teacher could still invent
details that the captions and boxes did not support.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-gen.png|A photo shows people loading luggage into an SUV. GPT-4 receives only the captions and object boxes; three types of generated tasks appear beside them.]]

*[Stanford CS336, Lecture 17](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-gen.png), based on Liu et al., [Visual Instruction Tuning, Table 1](https://arxiv.org/abs/2304.08485v2). The teacher received the captions and boxes, not the photograph.*

A description such as “people loading luggage into a black SUV” supports a question about vehicle type, but not its license plate. A plausible plate generated from that description is not verified OCR data. OneVision instead used LLaVA-NeXT-34B for recaptioning and GPT-4V for some data. Pixel access distinguishes those teachers from original LLaVA's text-only teacher, but does not guarantee correctness. Record teacher version, input evidence, and answer-validation procedure.

### Filtering and leakage

InternVL 2.5 §4.3 combines text-quality scoring, repetition detection, and heuristics for abnormal responses, with manual review of flagged examples. Its multimodal filtering emphasizes repetition and heuristics rather than applying every text-only quality rule unchanged. Removing every repeated line would also remove valid OCR from repeated form fields. Review candidate removals against the task.

Check image–question correspondence, source page, object identity, and whether the answer leaked into prompt metadata. Split by original image or document before generating crops, recaptions, and questions. Otherwise derivatives of the same document can appear under different filenames in both train and test. Image/document similarity and duplicate-answer checks detect different leakage paths.

For the scanned form, keep all derived crops and questions on the
same side of the split. Keep extracted values out of the prompt:
if “14 days” appears in metadata passed to the model, the example
can be solved without reading the scan.

<a id="vlm-training-mix"></a>

<span id="sampling-mixtures-changes-both-learning-and-cost"></span>

## Examples and tokens in a training mixture

For dataset sizes $n_1,\ldots,n_K$, choose the dataset $k$ with
probability $p_k$ before drawing an example from it:

$$
p_k=\frac{n_k^\alpha}{\sum_j n_j^\alpha}.
$$

At $\alpha=1$, examples are sampled proportionally to dataset size; at $\alpha=0$, datasets receive equal probability. With 10 million captions, 1 million document questions, and 100,000 grounding examples, proportional sampling gives about 90.1%, 9.0%, and 0.9%. At $\alpha=0.5$, the shares become 70.6%, 22.3%, and 7.1%. Grounding is now encountered almost eight times as frequently, without making the three datasets equally weighted.

Equal example shares still do not imply equal compute. A short caption may use a hundred visual vectors; a video example may require tens of thousands. Report mixture fractions by examples, supervised text tokens, visual tokens, pixels, and frames as appropriate.

For mean full lengths 200, 2000, and 1000 positions, the square-root mixture gives document QA 22.3% of examples but about 67.8% of input tokens. Grounding receives approximately 706 presentations per 10,000 selections, not necessarily 706 unique images:

<details id="training-code-mixture"><summary>Shares of examples and tokens</summary>

```python
import numpy as np

sizes = np.array([10_000_000, 1_000_000, 100_000], dtype=float)
weights = np.sqrt(sizes)
example_share = weights / weights.sum()
mean_length = np.array([200, 2000, 1000])
token_mass = example_share * mean_length
token_share = token_mass / token_mass.sum()
assert np.allclose(example_share, [.706101, .223289, .070610], atol=1e-6)
assert .677 < token_share[1] < .679
assert 705 < 10_000 * example_share[2] < 707
```

</details>

Molmo §4 combines square-root sampling with manual corrections, reducing some large synthetic sources and increasing pointing. InternVL's [pinned training script](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/internvl_chat_finetune.py#L749-L784) also uses square roots when `use_data_resampling` is enabled. Preserve the flag and coefficients; example shares do not determine FLOPs because visual prefixes, tiles, and attention costs differ.

### Sampling a dataset and weighting its loss are separate choices

Suppose one answer has four supervised tokens with mean loss 2, and another has 16 with mean loss 1. Token averaging gives $(4\cdot2+16\cdot1)/20=1.2$; sample averaging gives $(2+1)/2=1.5$. InternVL 2.5 §3.4 uses per-target weight $1/\sqrt{n_i}$, so each example's total weight is $\sqrt{n_i}$ and the result is $(2\cdot2+4\cdot1)/6=4/3$.

$$
\mathcal L_\gamma=
\frac{\sum_i n_i^{-\gamma}\sum_{t=1}^{n_i}\ell_{it}}
{\sum_i n_i^{1-\gamma}}.
$$

Here $\gamma=0$ is token averaging, $\gamma=1$ sample averaging, and $\gamma=1/2$ square averaging. The length $n_i$ counts supervised targets, excluding visual positions, user questions, and padding. InternVL counts the unmasked labels across a multi-turn example. A batch with no targets needs separate handling rather than division by zero.

<details id="training-code-length-weights"><summary>Three ways to average the loss</summary>

```python
import numpy as np

length = np.array([4., 16.])
mean_loss = np.array([2., 1.])

def length_weighted_loss(gamma):
    mass = length ** (1 - gamma)
    return mean_loss @ mass / mass.sum()

assert np.isclose(length_weighted_loss(0), 1.2)
assert np.isclose(length_weighted_loss(1), 1.5)
assert np.isclose(length_weighted_loss(.5), 4 / 3)
```

</details>

[`len2weight`](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/internvl_chat_finetune.py#L786-L795) constructs weights, and the [packed collator](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/dataset_packed.py#L517-L545) counts targets within original examples before zeroing ignored weights. Gradient accumulation must aggregate compatible denominators: averaging two already normalized microbatch losses need not equal normalization over all targets in the step. The precise formula here is verified for InternVL; the phrase “square-root reweighting” alone does not establish identical implementation details in Qwen3-VL.

<a id="vlm-training-reduction"></a>

### One optimizer step across several microbatches

Put the four-token and sixteen-token answers in separate microbatches.
The mean of their local losses is $(2+1)/2=1.5$. To weight all
supervised tokens equally, add $8+16=24$ first, then divide
by $4+16=20$, giving 1.2. Splitting the same batch should
not change the objective.

The mistake can reverse the update. Suppose the average derivatives
on one parameter are $+1$ and $-0.75$ in the two microbatches.
Their unweighted mean is $+0.125$. Giving each token the same weight
yields $(4-12)/20=-0.4$. Under square averaging, the example
weights are 2 and 4, and the derivative is $(2-3)/6=-1/6$.

Keep the sum of weighted errors $S$ and the sum of target weights $W$.
Add the numerators across the step and divide by the global $W$.
If standard DDP averages gradients from $R$ replicas and the trainer
does not apply additional divisors, scale each replica's numerator
by $R/W$. This cancels the factor $1/R$ in gradient synchronization.
Count all microbatches and replicas in $W$ and exclude positions
without a target. If $W=0$ for the entire step, all replicas skip it.

Put the short answer on one replica and the long answer on another.
Their numerators are 8 and 16. Both divide by $W/R=20/2=10$
and differentiate 0.8 and 1.6 respectively. Averaging those gradients
gives the gradient of the global mean loss, whose value here is 1.2.
[Molmo, Appendix A.3](https://arxiv.org/html/2409.17146v2#A1.SS3),
describes the same division for FSDP: each device divides its loss
sum by the mean target count per device.

[The Transformers team's write-up](https://huggingface.co/blog/gradient_accumulation)
shows the same bug for unweighted causal LM loss: sum losses with
`ignore_index` and divide by the global target count, rather
than averaging the local means.

<span id="llava-и-onevision-как-два-конкретных-curriculum"></span>

<span id="concrete-curricula-llava-onevision-and-qwen"></span>

## Longer inputs and new task mixtures

OneVision extends training from an initial connector stage through high-quality single-image learning to mixtures including multiple images and video. The visual budget and task distribution change along with the trainable components.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-training.png|LLaVA-OneVision training table: projector training, richer annotations, single-image instruction tuning, and a mixture of images and video.]]

*[Stanford CS336, Lecture 17](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-training.png), based on [LLaVA-OneVision, Table 1](https://arxiv.org/html/2408.03326v1#S5.T1). The columns give image sizes, mixture sizes, learning rates, and updated components.*

In OneVision §5 and Table 1, stage one trains the projector on 558,000 pairs at $384\times384$. Stage 1.5 trains all components on roughly four million examples of detailed descriptions, OCR, and language tasks. Next come 3.2 million single-image instructions and the final mixture of single images, multiple images, and video. The listed final counts, 560,000 multi-image, 350,000 video, and 800,000 single-image examples, sum to about 1.71 million rather than the report's “1.6M” label. A reproduction needs the actual manifest, not a guessed reconciliation of summary counts.

After the first stage, the table gives vision learning rate $2\cdot10^{-6}$ versus $10^{-5}$ for the LLM and projector. Visual budgets grow by stage. Packing unrelated images into long sequences with isolated attention improves utilization but does not itself teach reasoning across a long document; training examples must require those cross-page comparisons.

<span id="какие-способности-переносятся-между-сценариями"></span>

The intended transfer is compositional: reading one chart can help compare two charts; OCR can help interpret several GUI states. This is an explanation to test, not proof that transfer always occurs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s1.png|Two inputs: a sector of radius 11 and angle 40 degrees, and a table giving Allstate price 63. The model's displayed area is incorrect.]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 1](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s1.png), revision `8b59b507`.*

*The original model answer is retained. It contains arithmetic errors; it is not a verified worked solution.*

The diagram gives radius 11 and angle $40^\circ$, while the table gives Allstate price 63. The sector area is

$$
A=\frac{40}{360}\pi\,11^2\approx42.24.
$$

Multiplying before rounding gives 2660.93; rounding the area to 42.24 first gives 2661.12. The model's displayed 38.01 and 2386.03 are inconsistent with the inputs and with each other:

<details id="training-code-sector"><summary>Sector area and cost</summary>

```python
import math

area = 40 / 360 * math.pi * 11**2
assert round(area, 2) == 42.24
assert round(area * 63, 2) == 2660.93
assert round(round(area, 2) * 63, 2) == 2661.12
assert round(38.01 * 63, 2) == 2394.63
```

</details>

The example illustrates a two-image task format, not a wholly correct answer or causal evidence that one training stage produced transfer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s2.png|Four screenshots show a TikTok search, the app listing, and launch. The text on the right infers actions between screenshots.]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 2](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s2.png), revision `8b59b507`.*

*Four interface states show a TikTok query, an app listing, and launch. The generated description infers transitions from screenshots; the images are not a complete event log and do not prove that an agent executed every stated click.*

Test the explanation by removing the relevant training category while controlling total updates, visual budget, and backbone. Otherwise direct exposure to similar examples or a higher resolution can masquerade as transfer.

<span id="qwen-vl-почему-меняются-и-данные-и-замороженные-модули"></span>

Original [Qwen-VL](https://arxiv.org/abs/2308.12966) uses another schedule. Its initial image–text stage updates vision and adapter while freezing the LLM. A subsequent multitask stage trains all components at higher resolution. During instruction tuning the vision encoder is frozen again, while the adapter and language model learn the dialogue format.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen-vl-stages.png|Qwen-VL's training stages change both data and trainable components: freeze the LLM, then train all components, then freeze the ViT.]]

*Stanford CS336, Lecture 17, based on Bai et al., [Qwen-VL, Figure 2](https://arxiv.org/abs/2308.12966). Follow resolution, dataset, and freeze-state changes together.*

Qwen-VL's first stage uses 1.4 billion cleaned image–text pairs at $224\times224$. The second raises resolution to $448\times448$ and includes seven task types spanning captions, VQA, grounding variants, OCR, and text generation. The third uses 350,000 dialogue examples, including text-only data, with the vision encoder frozen again. Corpus size is not exposure count: batch size 30,720 over 50,000 steps gives 1.536 billion presentations, not 1.4 billion distinct images per step or epoch.

<span id="qwen3-vl-четыре-длины-pre-training-и-post-training"></span>

[Qwen3-VL](https://arxiv.org/abs/2511.21631) extends its pre-training curriculum through connector warm-up and joint stages at 8K, 32K, and 256K context. Its report also describes square-root-normalized per-token loss and separate thinking/non-thinking post-training. The length extension requires a corresponding data mixture; changing a context-limit flag is not equivalent training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen3-vl-pretraining.png|Four Qwen3-VL stages: train the merger at S0, then all components from S1. The maximum sequence length rises from 8K to 32K and 256K.]]

*Stanford CS336, Lecture 17, pre-training schedule based on the [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). The context stages expand the training distribution as well as tensor size.*

The [Qwen3-VL introduction](https://arxiv.org/html/2511.21631v1#S1) names square-root-normalized per-token loss without specifying the exact denominator or masking rule. The InternVL formula above should not be attributed to Qwen3-VL on that basis; reproduction needs its handling of padding, visual positions, and example boundaries.

The report does specify four pretraining stages. S0 trains the merger on 67 billion tokens of captions, visual knowledge, and OCR. S1 trains all components on roughly one trillion tokens, including text-only material, at the same 8192-position limit. S2 raises the limit to 32,768, adds more video and agent-oriented data, and increases text-only sampling. S3 uses 100 billion tokens at up to 262,144 positions, emphasizing long documents and video. A maximum of 256K does not mean every example is that long.

Post-training then includes SFT, strong-to-weak distillation, and RL.
SFT progresses from up to 32K to up to 256K; the thinking and
non-thinking variants train on different target formats. The report's
§3.2.3 describes questions spanning multiple pages. The model has to
combine facts that no single page contains.

<span id="does-post-training-improve-use-of-the-image"></span>

## Testing whether the model read the image

“14 days” is a plausible guess at a return period. Make two copies of
the form and change only the cell, from “14 days” to “7 days.” Keep
the rest of the scan and the question fixed. The two inputs now
require different answers.

<a id="vlm-training-image-control"></a>

### Scoring paired image tests

Make four pairs from different forms, each with a 14-day and a 7-day
version. Suppose the model reads “14” correctly in pairs 1, 2, and 3,
and “7” correctly in pairs 2, 3, and 4. It gets six of eight scans
right (75%), but it gets both answers right only in pairs 2 and 3
(50%). Always answering “14 days” scores 50% per scan and 0% per pair.

[MMVP](https://arxiv.org/abs/2401.06209) also credits a pair only
when both answers are correct. Its images and questions cover
more differences than this scan test. Shuffle
the choices and check that file names, backgrounds and prompt templates
do not give away answers. The paired score describes behavior
on these visual contrasts, not whether the model can read an
unfamiliar diagram.

On a third copy of each form, cover the cell and ask the same
question. “The return period is not visible” is now the correct
answer. A number has no support in this image even if it happens
to match the original. Record this guess rate separately: a model
that always declines passes the covered-cell test but scores
zero on pairs with two visible values.

In a preference pair with the visible 7-day scan, “7 days” should
beat the equally fluent “14 days.” The reward comparison then
concerns what the model read rather than how it worded the answer.

### A reward for object localization

For a box target $[0,0,2,2]$, the exact box has IoU 1; $[1,0,3,2]$ has IoU $1/3$; a non-overlapping box has IoU 0. An illustrative reward threshold of 0.5 accepts only the first:

<details id="training-code-iou"><summary>Box overlap</summary>

```python
import numpy as np

def iou(box, target):
    box, target = np.array(box), np.array(target)
    lo = np.maximum(box[:2], target[:2])
    hi = np.minimum(box[2:], target[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    a = np.maximum(box[2:] - box[:2], 0).prod()
    b = np.maximum(target[2:] - target[:2], 0).prod()
    return float(intersection / (a + b - intersection))

target = [0, 0, 2, 2]
scores = [iou(box, target) for box in
          [[0, 0, 2, 2], [1, 0, 3, 2], [3, 0, 5, 2]]]
assert np.allclose(scores, [1, 1 / 3, 0])
assert [s >= .5 for s in scores] == [True, False, False]
```

</details>

This reward evaluates geometry for an already selected object. It does not check object class, text reading, or JSON validity. Those conditions need their own checks. OCR exact match depends on case, whitespace, and number normalization; an extra explanation can violate an extraction format even when the characters are correct.

[Qwen3-VL §4.4](https://arxiv.org/abs/2511.21631v1) uses rule- or code-verifiable tasks for reasoning RL and names SAPO in this report version, not GRPO. General RL combines rule-based and model-based rewards. Judges can help score open responses, but must themselves be checked for missed visual errors, verbosity bias, and format exploits.

### Distilling responses versus distributions

Off-policy response distillation trains on answers already generated by a teacher. In an on-policy procedure, the student generates continuations and the teacher evaluates the resulting prefixes, including imperfect states the student actually visits. Qwen3-VL §4.3 describes response distillation followed by on-policy logit alignment with KL divergence and explicitly identifies text-only data for the strong-to-weak stage. That does not establish that a teacher checked each example's image pixels.

For next-token distributions $p=(0.8,0.2)$ from a teacher and $q=(0.6,0.4)$ from a student,

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_jp_j\log\frac{p_j}{q_j}\approx0.0915.
$$

Both probabilities contribute. Keeping only the teacher's most likely token discards its alternative distribution and gives a different target.

<details id="training-code-kl"><summary>KL divergence calculation</summary>

```python
import numpy as np

teacher = np.array([.8, .2])
student = np.array([.6, .4])
kl = np.sum(teacher * np.log(teacher / student))
assert abs(kl - .09151622) < 1e-7
assert np.isclose(np.sum(teacher * np.log(teacher / teacher)), 0)
```

</details>

This numerical example illustrates teacher-to-student KL, not undisclosed temperatures or reduction rules of a closed training pipeline. Better text-only reasoning after distillation is not evidence of better OCR or localization; each requires its own before/after visual evaluation.

<span id="mixtures-and-reproducibility"></span>

## What to record to repeat the run

| Record | Required detail |
|---|---|
| Components | encoder, connector, LLM revisions and initialization |
| Gradient path | freeze state at every stage, checkpointing and precision |
| Input | resizing, tiling, frame rate, token budgets |
| Data | origin, licensing, deduplication, filters, synthetic teacher |
| Mixture | formula, caps, example and token shares |
| Objective | target masks, packing boundaries, loss weights |
| Optimization | learning rates by component, batch units, steps |
| Validation | text-only retention, modality-specific tests, checkpoints |

Save example exposures and consumed tokens, not just downloaded corpus size. Retain target shifts, ignored positions, target weights, denominator, original sample boundaries, microbatch size, gradient accumulation, and distributed reduction. A few anonymized fully prepared sequences with masks and image-to-position mappings make these contracts inspectable. Check text-only retention, OCR, grounding, multi-image tasks, and long documents on source-disjoint held-out data.

The [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|next chapter]]
returns to the scanned form and follows the extracted field back
to its location on the page. The right number is not enough if
the model read it from the wrong row.

## Sources and next chapter

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), [InternVL 2.5](https://arxiv.org/abs/2412.05271), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Jinguo Zhu et al., [InternVL 3](https://arxiv.org/abs/2504.10479v1), §§2.2 and 3.14;
  Weiyun Wang et al., [InternVL 3.5](https://arxiv.org/abs/2508.18265v1), §§2.1–2.3.
- Detailed recipes: [LLaVA §3–4.2](https://arxiv.org/abs/2304.08485v2), [InternVL §3.2–3.4 and §4.2–4.5](https://arxiv.org/abs/2412.05271v1), [Molmo §2–4](https://arxiv.org/abs/2409.17146v2), [OneVision §4–5 and Table 1](https://arxiv.org/abs/2408.03326v1), [Qwen-VL §3.1–3.3](https://arxiv.org/abs/2308.12966v1), [Qwen3-VL §3.1 and §4](https://arxiv.org/abs/2511.21631v1).
- Previous: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|Resolution and spatial positions]]. Next: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|Documents, OCR, and grounding]].
