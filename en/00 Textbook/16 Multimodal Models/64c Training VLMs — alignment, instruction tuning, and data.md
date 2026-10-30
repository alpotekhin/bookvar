---
title: "64.3. Training VLMs: alignment, instruction tuning, and data"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "12e614a8be23e31ed608433c64be1f4b049c39af27cb9d4092080576a902921d"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2304.08485
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2301.12597
  - https://arxiv.org/abs/2308.12966
  - https://arxiv.org/abs/2511.21631
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2409.17146
  - https://cs336.stanford.edu/
---

# Training VLMs: alignment, instruction tuning, and data

A training example pairs a car photograph with “What color is the car?” and the answer “Red.” The model receives the image and question, then predicts answer tokens using the annotated previous tokens during training. Prediction error updates whichever components are trainable at that stage. The same photograph can support a caption, a question about its license plate, or a request to locate the car. Those annotations demand different evidence: a color, literal characters, or coordinates. The loss, masks, and trainable parameters determine how each demand affects learning.

Alignment does not always mean equal embeddings. LLaVA trains a projector with next-token prediction on image-caption pairs. BLIP-2 combines contrastive, matching, and grounded-generation objectives for Q-Former. The relevant questions are the objective, data, and trainable parameters—not the stage name.

## What changes at each stage?

| Stage | Prediction | Typical data | Parameters that move |
|---|---|---|---|
| Vision pre-training | class, masked content, or paired text | images and image–text pairs | vision encoder |
| Connector alignment | caption or pair relationship | image–caption pairs | connector/Q-Former, according to recipe |
| Visual instruction tuning | answer conditioned on media and a question | VQA, OCR, documents, grounding, dialogue | connector and selected backbones |
| Preference or reward training | preferred response or verifiable result | response pairs, points, boxes, exact answers | policy; reward model only where the method requires one |

This table describes roles, not four mandatory stages. A BLIP-2 model and a LLaVA model can both say “alignment” while using different objectives and gradient paths.

Original LLaVA's two stages use the same next-token objective. Stage one trains only the projector on 595,000 filtered CC3M image–caption pairs; stage two trains the projector and LLM on visual instructions, including LLaVA-Instruct-158K for the conversational variant, while vision remains frozen. Data, prompts, answer lengths, and trainable parameters change, not the basic factorization of the caption probability. These counts belong to the original model, not LLaVA-1.5 or OneVision.

Let $Z_v$ be frozen vision features and $H_v=g_\phi(Z_v)$ the connector output. Caption alignment can use

$$
\mathcal L_{\mathrm{caption}}
=-\sum_{t=1}^{T}\log p_\theta(y_t\mid H_v,q,y_{<t}).
$$

The LLM parameters $\theta$ stay fixed while $\phi$ changes. Backpropagation still passes through the LLM to $H_v$: otherwise the projector receives no learning signal. Dimension matching is necessary, but the real task is finding continuous inputs that the pretrained language model can use.

Here $q$ is the request to describe the image. The conditional probability is the product of the individual next-token probabilities, including the answer terminator. With teacher forcing, “Red car” predicts its first token from image and prompt, then its second from those inputs plus the annotated “Red.” The connector gradient is

$$
\nabla_\phi\mathcal L
=\left(\frac{\partial H_v}{\partial\phi}\right)^\top\nabla_{H_v}\mathcal L.
$$

This does not require each visual vector to coincide with a word embedding. It requires useful conditioning of the fixed decoder.

### One projector update

Use two possible answers, red and blue, a visual feature $(1,0)$, an identity frozen output head, and an initially zero projector. Both logits are zero, so the correct answer has probability 0.5 and loss $-\log0.5\approx0.6931$. The logit gradient is $(-0.5,0.5)$. An SGD step of 0.2 changes the projector's first column to $(0.1,-0.1)$; the new correct probability is about 0.5498 and loss 0.5981:

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

This tests one conditional token and a linear output head, not visual color recognition. A real LLM inserts decoder blocks into the same chain rule. Setting its weights' `requires_grad=False` preserves input differentiation; placing the entire LLM forward in `no_grad()` would break it. A frozen vision encoder can run without a graph if there are no trainable modules before it.

## Frozen and trainable components

Connector-only training leaves the original vision and language weights unchanged and avoids their weight gradients and optimizer state. It can still require LLM activations for input gradients. If the encoder output lacks the information needed for OCR, the projector cannot invent it; unfreezing vision allows those features to change. Changing the LLM can adapt how it uses visual inputs, but may degrade previous abilities. Retention must be measured on text-only and other preserved tasks, not inferred from the freeze schedule.

InternVL makes the schedule explicit: warm up the MLP with frozen backbones, optionally adapt the ViT, then perform full-model instruction tuning.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure4-training-pipeline.png]]

*Figure 4 from Zhe Chen et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 6. Panel (a) shows the trainable components; panel (b) transfers an adapted InternViT to a larger LLM and trains a new projector.*

Stage 1.5 trains vision and projector while freezing the LLM; its optional use concerns reuse of an already adapted vision encoder, not arbitrary removal of perception training. Moving that encoder from a smaller to a larger language model requires realigning the connector to the new width and input space.

Molmo makes a different choice: [§4](https://arxiv.org/abs/2409.17146v2) omits a connector-only warmup and trains all components directly on PixMo-Cap. The connector uses learning rate $2\cdot10^{-4}$ with 200 warmup steps, the ViT $6\cdot10^{-6}$ and LLM $2\cdot10^{-5}$ with 2000 warmup steps each. InternVL 2.5 instead uses one learning rate across the model during full instruction tuning. Neither recipe can be reconstructed from the general advice to update pretrained components more slowly.

## Loss masks are not attention masks

Suppose the user asks an invoice question and the assistant answers `12 480 RUB`. Token boundaries below are illustrative, not a tokenizer claim.

| Segment | system instruction | image vectors | user question | assistant answer | EOS |
|---|---|---|---|---|---|
| Available as causal context | yes | yes | yes | previous answer tokens | as configured |
| Target loss | no | no | no | yes | yes |

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

Take nine positions: user marker, visual A, visual B, question, assistant marker, `12`, `480`, currency, EOS. The first five labels are ignored; the four answer labels are active. Logits at position 4 predict the first answer token at position 5. Position 5 predicts the next token, not the `12` already present in its input. For correct-target probabilities 0.5, 0.25, 0.125, and 0.5, the negative log-probabilities sum to about 4.8520 and average to 1.2130:

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

Zero gradient on an ignored row's logits does not imply zero gradient on its input embedding in a Transformer: later answers can use that input through attention. Original LLaVA supervises answer terminators. Qwen-VL §3.3 supervises answers and designated special tokens, but not role names or user questions. The chat template determines exact boundaries; “assistant-only loss” does not specify them completely.

### Independent packed samples and shared-image branches

For independent sequences of lengths five and four, packed boundaries are `[0,5,9]` and positions restart at each sequence. A plain causal mask would let the second dialogue read the first one's answer, so attention must also respect sample identity. Image indexing, target boundaries, and loss denominators must follow the same separation.

Multimodal packing has a second budget: under limits of 2048 positions and eight tiles, requests of 700 positions/three tiles and 1000 positions/six tiles fit by length but exceed the visual budget. Pairing the first with 600 positions/four tiles fits both, at 1300 positions and seven tiles. [InternVL 2.5 §4.2](https://arxiv.org/abs/2412.05271v1) describes this dual constraint.

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

The second mask models Molmo's shared-image prefix with independent annotations. Each question sees the image and its own preceding tokens but not neighboring annotations. Positions 5–6 see image positions 0–2, not annotation positions 3–4. This is different from a real multi-turn dialogue, where the second answer should see the first. Packing must also avoid training the first token of a new sample from the final token of the preceding sample.

<span id="5-какие-данные-создают-какие-способности"></span>

## Data creates different visual behaviors

Web captions name salient objects but omit detail. Dense captions cover background, attributes, and relations. OCR and document data require literal accuracy and layout. Grounding pairs language with points or boxes. Interleaved documents teach correspondence across several images and text spans.

Molmo’s PixMo collection connects separate human-collected datasets to dense captioning, question answering, pointing, and grounded counting.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure1-data-and-capabilities.png]]

*Figure 1 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 2. Different data collections support different model capabilities.*

PixMo's dense descriptions were collected by asking people to speak about an image and transcribing that speech. Point annotations provide a different supervision channel: the output location can be checked, and grounded counting can be audited by inspecting the individual points. The annotation interface therefore shapes what the model learns, not only how quickly examples are collected.

Initially PixMo-Cap requested at least a minute of speech; later collection used one annotator with a 90-second minimum. Transcription and language-model processing converted speech into captions. The paper reports 712,000 images, 1.3 million transcripts/captions, and a mean final caption length of about 196 words. Recording duration describes collection, not a guarantee that every fact is correct.

In PixMo-AskModelAnything, a person selected an image and asked a question. A language model received OCR and a detailed caption, proposed an answer, and the person accepted or corrected it against the image. The human saw pixels; the text assistant depended on the supplied evidence. PixMo-Points uses human point annotations. Counting examples list points before the count, allowing inspection of the counted objects, but valid coordinates alone do not prove that every object was found or that none was counted twice.

PixMo-Docs renders generated code into tables and diagrams; that code supplies values and structure for questions. PixMo-Clocks fixes the time when constructing a clock face. PixMo-Count relies on detections, and PixMo-CapQA on captions. The latter annotations inherit the detector's or caption's errors, unlike a generator-defined value. Task/style tags distinguish benchmark-specific short answers from open conversation; otherwise strict extraction format can become an unwanted default for all requests.

Synthetic data scales strict formats but inherits teacher errors. Provenance matters: a confident wrong OCR label from a proprietary teacher can be distilled into an open-weight student.

## Instruction and preference tuning

LLaVA used GPT-4 over captions and symbolic descriptions to produce conversations, detailed descriptions, and reasoning prompts. GPT-4 did not inspect the original image, so generated instructions could contain only information already present in those inputs.

A description such as “people loading luggage into a black SUV” supports a question about vehicle type, but not its license plate. A plausible plate generated from that description is not verified OCR data. OneVision instead used LLaVA-NeXT-34B for recaptioning and GPT-4V for some data. Pixel access distinguishes those teachers from original LLaVA's text-only teacher, but does not guarantee correctness. Record teacher version, input evidence, and answer-validation procedure.

### Filtering and leakage

InternVL 2.5 §4.3 combines text-quality scoring, repetition detection, and heuristics for abnormal responses, with manual review of flagged examples. Its multimodal filtering emphasizes repetition and heuristics rather than applying every text-only quality rule unchanged. Removing every repeated line would also remove valid OCR from repeated form fields. Review candidate removals against the task.

Check image–question correspondence, source page, object identity, and whether the answer leaked into prompt metadata. Split by original image or document before generating crops, recaptions, and questions. Otherwise derivatives of the same document can appear under different filenames in both train and test. Image/document similarity and duplicate-answer checks detect different leakage paths.

Instruction loss is normally applied to assistant tokens, with image and user tokens acting as conditions. Packing and masks must prevent future answers from leaking across examples. Multimodal preference pairs should differ in visual correctness, not merely style; otherwise preference optimization rewards fluent language without better grounding.

Verifiable rewards are useful for points, boxes, OCR, and GUI actions. Open descriptions still require human audits because an automatic judge may reward verbosity while missing a visual error.

## Sampling mixtures changes both learning and cost

For dataset sizes $n_1,\ldots,n_K$, a common mixture is

$$
p_k=\frac{n_k^\alpha}{\sum_j n_j^\alpha}.
$$

At $\alpha=1$, examples are sampled proportionally to dataset size; at $\alpha=0$, datasets receive equal probability. With 10 million captions, 1 million document questions, and 100,000 grounding examples, proportional sampling gives about 90.1%, 9.0%, and 0.9%. At $\alpha=0.5$, the shares become 70.6%, 22.3%, and 7.1%. Grounding is now encountered almost eight times as frequently, without making the three datasets equally weighted.

Equal example shares still do not imply equal compute. A short caption may use a hundred visual vectors; a video example may require tens of thousands. Report mixture fractions by examples, supervised text tokens, visual tokens, pixels, and frames as appropriate.

For mean full lengths 200, 2000, and 1000 positions, the square-root mixture gives document QA 22.3% of examples but about 67.8% of input tokens. Grounding receives approximately 706 presentations per 10,000 selections, not necessarily 706 unique images:

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

Molmo §4 combines square-root sampling with manual corrections, reducing some large synthetic sources and increasing pointing. InternVL's [pinned training script](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/internvl_chat_finetune.py#L749-L784) also uses square roots when `use_data_resampling` is enabled. Preserve the flag and coefficients; example shares do not determine FLOPs because visual prefixes, tiles, and attention costs differ.

### Sampling a dataset and weighting its loss are separate choices

Suppose one answer has four supervised tokens with mean loss 2, and another has 16 with mean loss 1. Token averaging gives $(4\cdot2+16\cdot1)/20=1.2$; sample averaging gives $(2+1)/2=1.5$. InternVL 2.5 §3.4 uses per-target weight $1/\sqrt{n_i}$, so each example's total weight is $\sqrt{n_i}$ and the result is $(2\cdot2+4\cdot1)/6=4/3$.

$$
\mathcal L_\gamma=
\frac{\sum_i n_i^{-\gamma}\sum_{t=1}^{n_i}\ell_{it}}
{\sum_i n_i^{1-\gamma}}.
$$

Here $\gamma=0$ is token averaging, $\gamma=1$ sample averaging, and $\gamma=1/2$ square averaging. The length $n_i$ counts supervised targets, excluding visual positions, user questions, and padding. InternVL counts the unmasked labels across a multi-turn example. A batch with no targets needs separate handling rather than division by zero.

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

[`len2weight`](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/internvl_chat_finetune.py#L786-L795) constructs weights, and the [packed collator](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/dataset_packed.py#L517-L545) counts targets within original examples before zeroing ignored weights. Gradient accumulation must aggregate compatible denominators: averaging two already normalized microbatch losses need not equal normalization over all targets in the step. The precise formula here is verified for InternVL; the phrase “square-root reweighting” alone does not establish identical implementation details in Qwen3-VL.

<span id="llava-и-onevision-как-два-конкретных-curriculum"></span>

## Concrete curricula: LLaVA, OneVision, and Qwen

Original LLaVA first trains the projector with both backbones frozen. It then trains the projector and LLM on visual instructions while retaining the frozen vision encoder. Its synthetic instruction pipeline supplies captions and box descriptions to GPT-4, not image pixels:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-gen.png]]

*Stanford CS336, Lecture 17, based on Liu et al., [Visual Instruction Tuning, Table 1](https://arxiv.org/abs/2304.08485v2). The image is shown to the reader, not supplied to the text-only teacher; captions and boxes are the teacher's evidence.*

OneVision extends training from an initial connector stage through high-quality single-image learning to mixtures including multiple images and video. The visual budget and task distribution change along with the trainable components.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-training.png]]

*Stanford CS336, Lecture 17, training illustration based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Read each stage as a combination of data, trainable modules, and supported input structure.*

In OneVision §5 and Table 1, stage one trains the projector on 558,000 pairs at $384\times384$. Stage 1.5 trains all components on roughly four million examples of detailed descriptions, OCR, and language tasks. Next come 3.2 million single-image instructions and the final mixture of single images, multiple images, and video. The listed final counts, 560,000 multi-image, 350,000 video, and 800,000 single-image examples, sum to about 1.71 million rather than the report's “1.6M” label. A reproduction needs the actual manifest, not a guessed reconciliation of summary counts.

After the first stage, the table gives vision learning rate $2\cdot10^{-6}$ versus $10^{-5}$ for the LLM and projector. Visual budgets grow by stage. Packing unrelated images into long sequences with isolated attention improves utilization but does not itself teach reasoning across a long document; training examples must require those cross-page comparisons.

<span id="какие-способности-переносятся-между-сценариями"></span>

The intended transfer is compositional: reading one chart can help compare two charts; OCR can help interpret several GUI states. This is an explanation to test, not proof that transfer always occurs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s1.png]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 1](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s1.png), revision `8b59b507`.*

*The original model answer is retained. It contains arithmetic errors; it is not a verified worked solution.*

The diagram gives radius 11 and angle $40^\circ$, while the table gives Allstate price 63. The sector area is

$$
A=\frac{40}{360}\pi\,11^2\approx42.24.
$$

Multiplying before rounding gives 2660.93; rounding the area to 42.24 first gives 2661.12. The model's displayed 38.01 and 2386.03 are inconsistent with the inputs and with each other:

```python
import math

area = 40 / 360 * math.pi * 11**2
assert round(area, 2) == 42.24
assert round(area * 63, 2) == 2660.93
assert round(round(area, 2) * 63, 2) == 2661.12
assert round(38.01 * 63, 2) == 2394.63
```

The example illustrates a two-image task format, not a wholly correct answer or causal evidence that one training stage produced transfer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s2.png]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 2](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s2.png), revision `8b59b507`.*

*Four interface states show a TikTok query, an app listing, and launch. The generated description infers transitions from screenshots; the images are not a complete event log and do not prove that an agent executed every stated click.*

Test the explanation by removing the relevant training category while controlling total updates, visual budget, and backbone. Otherwise direct exposure to similar examples or a higher resolution can masquerade as transfer.

<span id="qwen-vl-почему-меняются-и-данные-и-замороженные-модули"></span>

Original [Qwen-VL](https://arxiv.org/abs/2308.12966) uses another schedule. Its initial image–text stage updates vision and adapter while freezing the LLM. A subsequent multitask stage trains all components at higher resolution. During instruction tuning the vision encoder is frozen again, while the adapter and language model learn the dialogue format.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen-vl-stages.png]]

*Stanford CS336, Lecture 17, based on Bai et al., [Qwen-VL, Figure 2](https://arxiv.org/abs/2308.12966). Follow resolution, dataset, and freeze-state changes together.*

Qwen-VL's first stage uses 1.4 billion cleaned image–text pairs at $224\times224$. The second raises resolution to $448\times448$ and includes seven task types spanning captions, VQA, grounding variants, OCR, and text generation. The third uses 350,000 dialogue examples, including text-only data, with the vision encoder frozen again. Corpus size is not exposure count: batch size 30,720 over 50,000 steps gives 1.536 billion presentations, not 1.4 billion distinct images per step or epoch.

<span id="qwen3-vl-четыре-длины-pre-training-и-post-training"></span>

[Qwen3-VL](https://arxiv.org/abs/2511.21631) extends its pre-training curriculum through connector warm-up and joint stages at 8K, 32K, and 256K context. Its report also describes square-root-normalized per-token loss and separate thinking/non-thinking post-training. The length extension requires a corresponding data mixture; changing a context-limit flag is not equivalent training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen3-vl-pretraining.png]]

*Stanford CS336, Lecture 17, pre-training schedule based on the [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). The context stages expand the training distribution as well as tensor size.*

The [Qwen3-VL introduction](https://arxiv.org/html/2511.21631v1#S1) names square-root-normalized per-token loss without specifying the exact denominator or masking rule. The InternVL formula above should not be attributed to Qwen3-VL on that basis; reproduction needs its handling of padding, visual positions, and example boundaries.

The report does specify four pretraining stages. S0 trains the merger on 67 billion tokens of captions, visual knowledge, and OCR. S1 trains all components on roughly one trillion tokens, including text-only material, at the same 8192-position limit. S2 raises the limit to 32,768, adds more video and agent-oriented data, and increases text-only sampling. S3 uses 100 billion tokens at up to 262,144 positions, emphasizing long documents and video. A maximum of 256K does not mean every example is that long.

Post-training then includes SFT, strong-to-weak distillation, and RL. SFT itself progresses from 32K to 256K, with different target formats for thinking and non-thinking variants. These choices cannot be represented reproducibly as a single instruction to add longer reasoning traces.

## Does post-training improve use of the image?

If every car-color question in training has answer “red,” low loss can come from answer frequency without reading the car. Change the image while keeping the question fixed and check whether the answer changes appropriately. A correct answer after replacing the image with an irrelevant one can indicate a text shortcut or leakage.

For a box target $[0,0,2,2]$, the exact box has IoU 1; $[1,0,3,2]$ has IoU $1/3$; a non-overlapping box has IoU 0. An illustrative reward threshold of 0.5 accepts only the first:

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

This reward evaluates geometry for an already selected object. It does not check object class, text reading, or JSON validity. Those conditions need their own checks. OCR exact match depends on case, whitespace, and number normalization; an extra explanation can violate an extraction format even when the characters are correct.

[Qwen3-VL §4.4](https://arxiv.org/abs/2511.21631v1) uses rule- or code-verifiable tasks for reasoning RL and names SAPO in this report version, not GRPO. General RL combines rule-based and model-based rewards. Judges can help score open responses, but must themselves be checked for missed visual errors, verbosity bias, and format exploits.

### Distilling responses versus distributions

Off-policy response distillation trains on answers already generated by a teacher. In an on-policy procedure, the student generates continuations and the teacher evaluates the resulting prefixes, including imperfect states the student actually visits. Qwen3-VL §4.3 describes response distillation followed by on-policy logit alignment with KL divergence and explicitly identifies text-only data for the strong-to-weak stage. That does not establish that a teacher checked each example's image pixels.

For next-token distributions $p=(0.8,0.2)$ from a teacher and $q=(0.6,0.4)$ from a student,

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_jp_j\log\frac{p_j}{q_j}\approx0.0915.
$$

Both probabilities contribute. Keeping only the teacher's most likely token discards its alternative distribution and gives a different target.

```python
import numpy as np

teacher = np.array([.8, .2])
student = np.array([.6, .4])
kl = np.sum(teacher * np.log(teacher / student))
assert abs(kl - .09151622) < 1e-7
assert np.isclose(np.sum(teacher * np.log(teacher / teacher)), 0)
```

This numerical example illustrates teacher-to-student KL, not undisclosed temperatures or reduction rules of a closed training pipeline. Better text-only reasoning after distillation is not evidence of better OCR or localization; each requires its own before/after visual evaluation.

## Mixtures and reproducibility

Simple concatenation lets the largest web corpus overwhelm rare grounding or document examples. Sampling weights should consider examples, tokens, pixels, and frames. A complete recipe reports data provenance and filtering; component freeze states; resolution and token budget; objectives and masks; mixture weights; optimizer, per-component learning rates, curriculum, and evaluation checkpoints.

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

Preference data must also distinguish visual correctness from prose quality. A fluent negative answer naming an absent object is useful for a different reason than a poorly written answer: only the former directly tests whether the policy follows the image.

Save example exposures and consumed tokens, not just downloaded corpus size. Retain target shifts, ignored positions, target weights, denominator, original sample boundaries, microbatch size, gradient accumulation, and distributed reduction. A few anonymized fully prepared sequences with masks and image-to-position mappings make these contracts inspectable. Check text-only retention, OCR, grounding, multi-image tasks, and long documents on source-disjoint held-out data.

## Sources and next chapter

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), [InternVL 2.5](https://arxiv.org/abs/2412.05271), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Detailed recipes: [LLaVA §3–4.2](https://arxiv.org/abs/2304.08485v2), [InternVL §3.2–3.4 and §4.2–4.5](https://arxiv.org/abs/2412.05271v1), [Molmo §2–4](https://arxiv.org/abs/2409.17146v2), [OneVision §4–5 and Table 1](https://arxiv.org/abs/2408.03326v1), [Qwen-VL §3.1–3.3](https://arxiv.org/abs/2308.12966v1), [Qwen3-VL §3.1 and §4](https://arxiv.org/abs/2511.21631v1).
- Previous: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|Resolution and spatial positions]]. Next: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|Documents, OCR, and grounding]].
