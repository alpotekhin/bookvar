---
title: "64.6. Evaluation, failure modes, and VLM serving"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "7158736c550f13fbe8deca537fc369ef87c94a38586dae167fbdc78d215e625a"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
primary_sources:
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
  - https://arxiv.org/abs/2204.03162
  - https://arxiv.org/abs/2305.10355
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2502.13923
---

# Evaluation, failure modes, and VLM serving

One aggregate multimodal score mixes recognition, OCR, knowledge, reasoning, grounding, and response format. A useful evaluation separates those abilities and then tests whether the answer causally depends on visual evidence.

MMMU covers expert disciplines; MathVista visual mathematics; OCRBench, DocVQA, and ChartQA text and layout; Video-MME and LongVideoBench temporal understanding. None is a complete measure of “multimodal intelligence.”

| Ability | Example test | Useful measurement | Not established by success |
|---|---|---|---|
| Objects and attributes | VQAv2, perception tasks | accuracy, F1 | relation understanding or OCR |
| Composition | Winoground | text/image/group scores | unrestricted scene descriptions |
| Documents | OCRBench, DocVQA, ChartQA | exact match, ANLS, CER/WER | resistance to template shortcuts |
| Grounding | referring expressions, points | IoU, point accuracy | completeness of an answer |
| Visual mathematics | MathVista | answer accuracy | reliable reading of any document |
| Expert reasoning | MMMU | accuracy by discipline | new-domain reliability |
| Video | Video-MME, LongVideoBench | accuracy, localization metrics where annotated | audio–video synchronization |
| Object hallucination | POPE | precision, recall, F1, yes ratio | attribute, count, or OCR correctness |

Metrics apply to particular tasks and annotations; not every listed video benchmark uses temporal IoU. The table is an evaluation map, not a ranking.

### Distinguish the failed step

![[00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/mmvet-integrated-capabilities.png]]

*Yu et al., [MM-Vet, Figure 1](https://raw.githubusercontent.com/mlresearch/v235/main/assets/yu24o/yu24o.pdf#page=1), ICML 2024, CC BY 4.0. In the lower-left example, the indicated person's expression is $7\times2$, giving the reference 14; the neighboring $11-2$ does not answer that question.*

An answer of 9 can arise from correctly calculating the wrong expression $11-2$, misreading the correct region as $7+2$, or computing $7\times2$ incorrectly. Start by checking the selected region, then transcription, then calculation. MM-Vet labels recognition, knowledge, OCR, spatial relations, language generation, and mathematics as overlapping capabilities; a single question can require several.

Dataset composition can reverse a ranking. With 900 photos and 100 documents, model A's accuracies 95% and 50% give 90.5% overall; model B's 90% and 80% give 89%. Equal category weighting instead gives 72.5% versus 85%:

```python
import numpy as np

counts = np.array([900, 100])
a = np.array([0.95, 0.50])
b = np.array([0.90, 0.80])
micro_a, micro_b = (counts @ a) / counts.sum(), (counts @ b) / counts.sum()
macro_a, macro_b = a.mean(), b.mean()
assert np.isclose(micro_a, 0.905) and np.isclose(micro_b, 0.89)
assert micro_a > micro_b and macro_a < macro_b
```

Micro averaging describes the chosen request mix; macro averaging gives each category equal influence. Both are legitimate questions, but a document service should not treat many easy photos as compensation for wrong invoice amounts. Publish category counts and results alongside the aggregate.

<span id="2-как-winoground-изолирует-композиционность"></span>

## Composition and hallucination

Winoground uses image-caption pairs with the same words and entities but different relations. It tests whether the representation preserves composition rather than a bag of concepts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/cs231n-lecture16-winoground.png]]

*Stanford CS231n 2025, [Lecture 16, slide 50](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf#page=50). Winoground, CREPE, and ARO examples are shown. The same-word constraint belongs to Winoground here; CREPE also substitutes words.*

For a Winoground item, let $I_0,I_1$ be images and $T_0,T_1$ their matching captions. Four similarities expose two different directions:

$$
\begin{gathered}
\text{text:}\\
s(I_0,T_0)>s(I_0,T_1),\\
s(I_1,T_1)>s(I_1,T_0),
\end{gathered}
$$

$$
\begin{gathered}
\text{image:}\\
s(I_0,T_0)>s(I_1,T_0),\\
s(I_1,T_1)>s(I_0,T_1).
\end{gathered}
$$

Group success requires all four inequalities. For the illustrative matrix
$S=\begin{pmatrix}0.9&0.8\\0.7&0.6\end{pmatrix}$, the first image chooses its caption correctly but the second does not. Two relatively high diagonal scores do not imply group success. See [Thrush et al., Winoground](https://arxiv.org/abs/2204.03162).

Winoground contains 400 two-image/two-caption groups. In $S=\bigl(\begin{smallmatrix}.9&.8\\.1&.2\end{smallmatrix}\bigr)$, each image prefers its correct caption, so text score is one. But caption 1 prefers image 0 because $.8>.2$, making image and group scores zero. Ties fail the strict comparisons:

```python
import numpy as np

def winoground_scores(s):
    text = bool(s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0])
    image = bool(s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1])
    return text, image, text and image

assert winoground_scores(np.array([[.9, .8], [.1, .2]])) == (True, False, False)
assert winoground_scores(np.array([[.9, .1], [.2, .8]])) == (True, True, True)
assert winoground_scores(np.ones((2, 2))) == (False, False, False)
```

For independent identically distributed continuous random scores, expected text/image scores are each $1/4$, and group score is $1/6$. The four group comparisons are dependent: both diagonal entries must outrank both off-diagonal entries, true in four of the 24 orderings. This baseline does not automatically apply to a generative model choosing one of two letters. Fix how its outputs become four comparable scores. The dataset also includes pragmatics, symbolic depictions, and same-series photographs; failure is not a complete diagnosis of every kind of compositional understanding.

POPE probes object hallucination with absent categories sampled randomly, by dataset frequency, or by co-occurrence with present objects. It measures one failure mode, not all hallucination: attributes, relations, counts, and OCR can also be false. “Adversarial” names the co-occurrence sampling procedure, not proof that every such question is harder for every model.

![[00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/pope-question-construction.png]]

*Li et al., [POPE, Figure 3](https://aclanthology.org/2023.emnlp-main.20.pdf#page=6), CC BY 4.0. Object annotation, absent-category selection, and binary question construction are separate stages.*

The original MSCOCO evaluation uses 500 images with six questions each, balancing positive and negative labels. Questions sharing an image are correlated; an image-level resampling unit can be appropriate for uncertainty estimates rather than assuming every question independent.

For 100 binary questions, suppose $TP=40$, $FP=20$, $FN=10$, and $TN=30$. Then precision is $40/60\approx0.667$, recall $40/50=0.8$, F1 $80/110\approx0.727$, accuracy 70%, and the yes ratio 60%. Reporting recall alone hides the false-positive problem. Always include the negative sampling strategy; frequent and co-occurring absent categories test different biases.

Another 100-question fixture uses $TP=45,FP=15,FN=5,TN=35$, giving precision 0.75, positive-class recall 0.90, F1 $9/11$, accuracy 0.80, and yes ratio 0.60. Recall is about present objects, not correct “no” answers. An always-yes model reaches recall one but accuracy only 0.5; an always-no model has undefined precision because there are no positive predictions:

```python
from math import isclose

def binary_metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return precision, recall, f1, (tp + tn) / n, (tp + fp) / n

p, r, f1, accuracy, yes = binary_metrics(45, 15, 5, 35)
assert isclose(p, .75) and isclose(r, .9)
assert isclose(f1, 9 / 11) and isclose(accuracy, .8) and isclose(yes, .6)
assert binary_metrics(50, 50, 0, 0) == (.5, 1., 2/3, .5, 1.)
assert binary_metrics(0, 0, 50, 50) == (None, 0., 0., .5, 0.)
```

The example uses `None` for undefined precision. If an evaluator substitutes zero, document that convention. The [pinned POPE evaluator](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/evaluate.py) keeps the first sentence and labels it negative when separate words `No`, `no`, or `not` occur, positive otherwise. “I am not sure” becomes negative while “I am unsure” becomes positive. That is parser behavior, not perception. Reproduce the original parser for reported results; a new evaluation should separately represent abstentions and malformed answers.

### CHAIR checks object mentions in free captions

![[00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/chair-caption-errors.png]]

*Rohrbach et al., [Object Hallucination in Image Captioning, Figure 1](https://aclanthology.org/D18-1437.pdf#page=1), CC BY 4.0. One original caption adds an unsupported bench despite a favorable text-similarity score.*

CHAIRi divides unsupported object mentions by all object mentions; CHAIRs divides captions containing at least one unsupported mention by all captions. With three captions of three mentions each and one unsupported mention per caption, CHAIRi is $1/3$ and CHAIRs is one:

```python
from math import isclose

mentions = [
    ["person", "bicycle", "dog"],
    ["person", "dog", "chair"],
    ["person", "bicycle", "car"],
]
ground_truth = [
    {"person", "bicycle"},
    {"person", "dog"},
    {"person", "bicycle"},
]
bad = [sum(obj not in truth for obj in caption)
       for caption, truth in zip(mentions, ground_truth)]
chair_i = sum(bad) / sum(map(len, mentions))
chair_s = sum(count > 0 for count in bad) / len(mentions)
assert bad == [1, 1, 1]
assert isclose(chair_i, 1/3) and chair_s == 1
```

Concentrating the same three unsupported mentions in one caption leaves CHAIRi unchanged at fixed total mention count but lowers CHAIRs to $1/3$. This code starts from already extracted categories, not the real NLP parser. Original CHAIR uses 80 MSCOCO categories, synonym normalization, and compound handling so “hot dog” does not add a dog. Ground truth combines object annotations and human-caption categories, each of which can omit evidence. CHAIR and POPE do not cover every color, relation, OCR claim, or annotation error.

<span id="4-проверка-причинной-опоры-на-изображение"></span>

## Counterfactuals need paired expectations

Moving invoice rows together with their labels should preserve a sum. Changing tax from 1480 to 2480 while retaining goods at 11,000 should change the total from 12,480 to 13,480:

```python
from decimal import Decimal

def total(rows):
    return sum((Decimal(value) for _, value in rows), start=Decimal(0))

original = [("Товары", "11000"), ("Налог", "1480")]
reordered = list(reversed(original))
changed = [("Товары", "11000"), ("Налог", "2480")]
assert total(original) == total(reordered) == Decimal("12480")
assert total(changed) == Decimal("13480")
```

This checks expected answers, not a VLM. A visual test must actually edit pixels and retain the new file, not merely change the answer key. Hiding a printed total need not force refusal if visible summands still permit calculation. To test reading that particular field, constrain the question or remove alternative recovery routes. Masks must not accidentally introduce unreadability or answer-class cues.

Three correct originals out of four and three correct edits out of four do not imply three correct pairs:

```python
import numpy as np

original_ok = np.array([True, True, True, False])
changed_ok = np.array([True, False, True, True])
assert original_ok.mean() == changed_ok.mean() == .75
pair_accuracy = np.mean(original_ok & changed_ok)
assert pair_accuracy == .5
```

Only two pairs are jointly correct. Keep identical images and edits when comparing model versions. Such interventions establish behavior on those changes, not a unique internal cause of failure.

## Judges, contamination, and openness

An LLM judge can prefer detail and style while missing a small visual error. Its prompt, version, and access to the image are part of the protocol. Use deterministic metrics for exact text, boxes, numbers, and actions, and audit open-ended scores with humans.

MM-Vet §3.2 supplies GPT-4 with question, reference, prediction, and few-shot examples, but not the image. These original English examples from its Table 1 illustrate partial credit:

| Question | Ground truth | Prediction | Correctness |
|---|---|---|---:|
| What is x in the equation? | -1 &lt;AND&gt; -5 | x = 3 | 0.0 |
| What is x in the equation? | -1 &lt;AND&gt; -5 | x = -1 | 0.5 |
| What is x in the equation? | -1 &lt;AND&gt; -5 | x = -1 or x = -5 | 1.0 |

*Yu et al., [MM-Vet, Table 1 and full prompt](https://raw.githubusercontent.com/mlresearch/v235/main/assets/yu24o/yu24o.pdf#page=4), CC BY 4.0. Original wording and notation retained.*

`<AND>` requires both reference elements; `<OR>` permits alternatives. Both roots are required here, so naming one earns half credit. It does not mean $x$ simultaneously equals two numbers, and ordinary “or” can correctly enumerate the roots. A text-only judge checks agreement with a reference but cannot independently see a mistaken reference or an unmentioned visual detail. An image-enabled judge adds its own OCR and spatial failure modes.

Record judge version, exact prompt, available inputs, repetitions, and reduction. Test answer-order sensitivity in pairwise comparisons and length, extra claims, and missing facts in reference grading. Human-labeled audits should separate false acceptance and false rejection; judge self-consistency is not a substitute. Normalize numeric answers where the task permits it and specify rounding, units, and completeness rules.

Contamination includes perceptual duplicates, OCR text embedded in images, and published solutions. Post-cutoff or parameterized tests help but trade reproducibility for secrecy.

[MMStar §3](https://arxiv.org/abs/2403.20330v1) distinguishes questions answerable without images from suspected training leakage. “What is Nebraska's capital?” needs no map; “What is the capital of the highlighted state?” needs visual identification unless the name leaks elsewhere. Its §4.3 defines

$$
\begin{gathered}
MG=S_v-S_{wv},\\
ML=\max(0,S_{wv}-S_t),
\end{gathered}
$$

where $S_v$ is VLM performance with the image, $S_{wv}$ the same VLM without it, and $S_t$ the original text backbone. Scores 70%, 45%, and 30% give 25 and 15 percentage points:

```python
from math import isclose

visual, without_visual, text_backbone = .70, .45, .30
mg = visual - without_visual
ml = max(0., without_visual - text_backbone)
assert isclose(mg, .25) and isclose(ml, .15)
```

These differences do not prove that a particular test image appeared in training. Language improvement, instruction tuning, formatting, and behavior on missing images also contribute. Corpus evidence or additional checks are needed for a leakage claim. A text-only condition must remove OCR, captions, answer-bearing filenames, and old image features. A blank required image is a distinct protocol from removing visual tokens.

Check exact hashes, perceptually similar images, OCR text, questions, and solutions; redrawn charts can preserve the same task and answer. A new benchmark date reduces some risks without proving every original or synthetic source clean.

Open weights do not imply an open training pipeline. Molmo separates the final VLM, LLM backbone, vision encoder, and data/code.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png]]

*Figure 11 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 19. The table is a dated comparison of openness across model components and data.*

The figure belongs to the December 5, 2024 report, not a current status table for every listed family. “Distilled” identifies dependence on proprietary-VLM-generated data. Reproducing evaluation requires encoder/processor revisions, templates, media, questions, and evaluators; reproducing training additionally requires provenance, licenses, filtering, teachers, and recipe. Open weights alone provide neither missing corpus details nor all reproduction inputs.

## Serving path and observability

For a two-page comparison request, limiting the answer to 200 tokens does not limit image cost. Before model execution, the service must validate the two media inputs, decode them, estimate their resized pixel and visual-token counts, and apply an admission policy.

A VLM request passes through media loading, validation, resize/cropping, vision encoding, connector/resampler, LLM prefill, and autoregressive decode. Time to first token includes every preceding stage. Logging only the LLM engine hides media and vision latency.

The OpenAI-compatible request below is a structural example with fictitious model and URLs, not an executed API call:

```json
{
  "model": "served-vlm",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.org/page-1.png"}},
    {"type": "image_url", "image_url": {"url": "https://example.org/page-2.png"}},
    {"type": "text", "text": "Compare the totals and identify their pages."}
  ]}],
  "max_tokens": 200
}
```

The [pinned vLLM media-input documentation](https://github.com/vllm-project/vllm/blob/7b569ec8953cf8099638a6d2ee1e915b75867c3a/docs/features/multimodal_inputs.md#image-inputs) separates this client format from the model chat template. Clients do not manually repeat placeholders per patch; processing expands media according to the model. Identical image-marker IDs do not mean identical content. Media order, expanded positions, and separators must match the actual features.

### Count image positions, context reserve, and tensors

For two prepared pages of $1008\times1344$, patch size 14 and merge factor two give a $36\times48$ merged grid, or 1728 positions per page. Add 256 text/chat positions and four image boundaries: input length is 3716; reserving 200 output tokens requires capacity for 3916.

```python
height, width, patch, merge = 1008, 1344, 14, 2
assert height % (patch * merge) == width % (patch * merge) == 0
per_page = (height // (patch * merge)) * (width // (patch * merge))
visual_tokens = 2 * per_page
text_and_chat_tokens, image_boundaries, output_budget = 256, 4, 200
input_length = visual_tokens + text_and_chat_tokens + image_boundaries
assert per_page == 1728 and visual_tokens == 3456
assert input_length == 3716
assert input_length + output_budget == 3916
```

These are explicit teaching parameters based on the Qwen2/2.5-VL scheme, not a universal pixels-to-tokens conversion. Use actual processor output to avoid omitting or double-counting separators. `max_tokens` limits output, not 3456 visual positions or encoder work. Before GPU admission, also check image count, decoded pixels, video frames, full sequence length, and generation reserve; reject excessive requests or apply a stated resizing policy.

For a hypothetical 32-layer decoder with eight KV heads, head width 128, and BF16, each position requires $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ cache bytes. Width-4096 projected features for those pages require 27 MiB; input KV requires 464.5 MiB, or 489.5 MiB with the output reserve:

```python
visual_tokens, input_length = 3456, 3716
embedding_bytes = visual_tokens * 4096 * 2
kv_per_token = 2 * 32 * 8 * 128 * 2
input_kv_bytes = input_length * kv_per_token
reserved_kv_bytes = (input_length + 200) * kv_per_token
assert embedding_bytes / 2**20 == 27
assert kv_per_token == 131_072
assert input_kv_bytes / 2**20 == 464.5
assert reserved_kv_bytes / 2**20 == 489.5
```

These are tensor sizes, not peak memory or a particular Qwen configuration. They exclude weights, temporary vision/prefill activations, block alignment, transfer buffers, and concurrent requests. Total parameter count does not determine KV-head count or projected feature width.

Admission control should cap pixels, images, frames, and total visual tokens. Variable-length media causes batch imbalance and head-of-line blocking. Image-embedding cache keys must include content hash, preprocessing configuration, encoder, and connector version; video keys also include sampling policy.

### Follow the complete latency clock

For a sequential illustrative path,

$$
\begin{aligned}
T_{\mathrm{TTFT}}={}&T_{\mathrm{request}}+T_{\mathrm{queue}}\\
&+T_{\mathrm{fetch/decode}}+T_{\mathrm{preprocess}}\\
&+T_{\mathrm{vision}}+T_{\mathrm{connector}}\\
&+T_{\mathrm{LLM\ prefill}}+T_{\mathrm{sample/deliver},1}.
\end{aligned}
$$

The request term measures client start to server receipt. If the seven subsequent intervals are 20, 80, 35, 120, 10, 180, and 15 ms, server-receipt-to-first-token is 460 ms. Another 40 ms of inbound request time makes client TTFT 500 ms. The last two intervals alone give 195 ms. First-token sampling uses prefill's final logits; it need not add a complete decoder pass.

```python
stages_ms = {
    "queue": 20, "fetch_and_decode": 80, "processor": 35,
    "vision": 120, "connector": 10, "prefill": 180, "sample_and_deliver": 15,
}
assert sum(stages_ms.values()) == 460
request_to_server_ms = 40
client_ttft = request_to_server_ms + sum(stages_ms.values())
assert client_ttft == 500
engine_slice = stages_ms["prefill"] + stages_ms["sample_and_deliver"]
assert engine_slice == 195
assert client_ttft - engine_slice == 305
```

Overlapping stages require a critical path, not the sum of all work. Suppose independent media branch A spends 80/35/120 ms on loading/processing/encoding, and branch B 150/20/80. After a shared 20-ms queue, the connector waits for both. Adding connector/prefill/delivery 10/180/15 gives $20+\max(235,250)+10+180+15=475$ ms from server receipt, versus 710 ms if all operations were incorrectly summed:

```python
queue_ms = 20
branches = {"A": [80, 35, 120], "B": [150, 20, 80]}
after_join = [10, 180, 15]
critical_path = queue_ms + max(map(sum, branches.values())) + sum(after_join)
all_work = queue_ms + sum(map(sum, branches.values())) + sum(after_join)
assert critical_path == 475 and all_work == 710
```

This assumes independent resources. Competition for a GPU or decoder thread can change branch durations. Real traces need start/end times, dependency joins, and client boundaries; a Python function returning after an asynchronous CUDA launch does not establish device completion. Speech output additionally needs first-audible-sample and playback-continuity measurements.

### Avoid head-of-line blocking before the LLM

Ten short photo requests and one 200-page document should not be treated as eleven equal jobs. The document can occupy media decoding, vision execution, and prefill long enough to delay all the photos. Continuous batching inside the LLM cannot fix an upstream media queue on its own.

Estimate cost before admission, cap media counts and pixels, and group compatible workloads by image/document/video or expected cost. Define separate SLOs for these classes. Where supported, packed vision attention prevents images from attending to one another while avoiding padding every image to the largest dimensions.

On one non-preemptive executor, a 200-ms job followed by ten 10-ms jobs makes the short jobs finish at 210 through 300 ms, mean 255. Serving the short jobs first gives completions 10 through 100 ms, mean 55:

```python
import numpy as np

def completion_times(jobs):
    elapsed, result = 0, {}
    for name, duration in jobs:
        elapsed += duration
        result[name] = elapsed
    return result

short = [(f"s{i}", 10) for i in range(10)]
fifo = completion_times([("large", 200)] + short)
short_first = completion_times(short + [("large", 200)])
assert np.mean([fifo[name] for name, _ in short]) == 255
assert np.mean([short_first[name] for name, _ in short]) == 55
assert fifo["s9"] == short_first["large"] == 300
```

This is a single-queue calculation, not a vLLM simulator or load test. Under continued short-job arrivals, always-shortest-first can starve a large document, so classes need resource shares, waiting limits, or aging. A batch that advances only when every member is ready also makes fast examples wait for the slowest.

![[00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/epd-stage-interference.png]]

*Singh et al., [EPD Disaggregation, Figure 1](https://raw.githubusercontent.com/mlresearch/v267/main/assets/singh25d/singh25d.pdf#page=1), ICML 2025, CC BY 4.0. LLM4 delays E5 on a shared executor; a separate encoder can process E5 independently. The horizontal axis is a conceptual schedule, not our millisecond measurements.*

Continuous LLM batching does not fix queues before the LLM. Chunked prefill does not make an arbitrary vision encoder preemptible. Padding can waste computation; packed independent images need isolation masks and compatible kernels.

### Encode–Prefill–Decode disaggregation

EPD §3.1–3.2 assigns different workers to encoding, prefill, and decode. Encode holds multimodal encoder weights and results; prefill holds LLM weights and builds KV from text plus features; decode continues from transferred KV and the chosen token. Splitting only prefill/decode can leave vision on the prefill GPUs, which is a different allocation.

![[00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/epd-system-architecture.png]]

*Singh et al., [EPD Disaggregation, Figure 4](https://raw.githubusercontent.com/mlresearch/v267/main/assets/singh25d/singh25d.pdf#page=4), CC BY 4.0. E→P transfers multimodal features; P→D transfers KV. Weights, MM-cache, KV-cache, and queues are separate objects.*

The asynchronous handoff retains results in encoder MM-cache until prefill acknowledges transfer. Only then can source storage be released. This transfer buffer is not necessarily a persistent cache for repeated user requests; early release or overwrite risks corrupted reads.

At an illustrative useful bandwidth of 25 GiB/s, the earlier 27-MiB feature tensor and 464.5-MiB KV tensor imply different transfer lower bounds:

```python
import numpy as np

bytes_per_second = 25 * 2**30
embedding_ms = (27 * 2**20) / bytes_per_second * 1000
kv_ms = (464.5 * 2**20) / bytes_per_second * 1000
assert np.isclose(embedding_ms, 1.0546875)
assert np.isclose(kv_ms, 18.14453125)
```

The results are approximately 1.05 and 18.14 ms, excluding protocol, queueing, packing, topology, and transfers across TP ranks. Connector placement affects the actual tensor boundary. These are not measured network latencies or proof that disaggregation always helps.

Intra-request parallelism distributes independently encodable media fragments across encoder workers, then restores order and required projection/merging. Arbitrary patches of one globally attentive ViT image are not independent; splitting them without modifying computation does not reproduce the original output.

### Allocation depends on the bottleneck

For four workers with per-replica rates E=3, P=5, D=2 requests/s, allocation 1E/1P/2D has capacity bound $\min(3,5,4)=3$. Moving one worker to E gives 2E/1P/1D with bound $\min(6,5,2)=2$. If image-heavy requests reduce E's rate to 1.5, the better split changes:

```python
from itertools import product

allocations = [a for a in product(range(1, 3), repeat=3) if sum(a) == 4]
def capacity(allocation, rates):
    return min(n * rate for n, rate in zip(allocation, rates))

rates = (3, 5, 2)
assert capacity((1, 1, 2), rates) == 3
assert capacity((2, 1, 1), rates) == 2
assert max(allocations, key=lambda a: capacity(a, rates)) == (1, 1, 2)
image_heavy_rates = (1.5, 5, 2)
assert max(allocations, key=lambda a: capacity(a, image_heavy_rates)) == (2, 1, 1)
```

This model assumes linear replication; real queues, memory, and communication add constraints. Stage capacity does not prove tail-latency SLO compliance. EPD uses a workload-sample simulator and Bayesian optimization to choose resources, batches, and scheduling under cost constraints. Role changes stop new admissions, redistribute queued work, change weights/cache roles, then resume. P↔D can reuse some state; moving from E to the LLM requires different weights.

### Cache identity depends on the cached object

An encoder-feature cache needs the image bytes, preprocessing, and vision-checkpoint identity. A cache of **projected** visual embeddings additionally depends on connector weights. A full LLM KV cache also depends on text prefix, LLM revision, adapters, positions, and multimodal ordering. Identical visible text does not justify reusing KV from another image.

For video, retain selected frame timestamps and sampling configuration in identity. A preprocessing change is a representation change even when the media file is unchanged. Logs should distinguish encoder-cache hits from LLM prefix-cache hits.

| Cache layer | Stored object | Reuse dependencies |
|---|---|---|
| Media fetch | bytes and retrieval metadata | content, authorization, freshness |
| Processor | tensors and preprocessing metadata | bytes, processor revision, resize/tiling/sampling, orientation |
| Encoder/connector | features at a declared boundary | prepared input, weights, dtype, computation configuration |
| LLM prefix | K/V of completed prefix blocks | prior context, model, positions, adapter, media identity |

Changing a connector invalidates post-connector features, but not independent pre-connector encoder outputs. Question-conditioned visual cross-attention can make features depend on the question; an independent image encoder does not have that dependency. vLLM's [processor cache](https://github.com/vllm-project/vllm/blob/7b569ec8953cf8099638a6d2ee1e915b75867c3a/docs/design/mm_processing.md#processor-output-caching) stores HF-processor results, while [EncoderCacheManager](https://github.com/vllm-project/vllm/blob/7b569ec8953cf8099638a6d2ee1e915b75867c3a/vllm/v1/core/encoder_cache_manager.py) tracks embeddings, active references, and eviction. Their hit rates describe different skipped work.

Matching placeholder IDs with different pixels must not reuse the same KV. The [pinned prefix-cache design](https://github.com/vllm-project/vllm/blob/7b569ec8953cf8099638a6d2ee1e915b75867c3a/docs/design/prefix_caching.md) adds multimodal hashes, adapter identifiers, and salt alongside the previous block's hash. The following explicit keys illustrate dependencies, not vLLM's internal serialization:

```python
import hashlib
import json

def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

def processor_key(data, max_pixels):
    return digest({"bytes": hashlib.sha256(data).hexdigest(),
                   "processor": "processor-v1", "max_pixels": max_pixels})

def feature_key(processed, vision="vision-v1", connector="connector-v1"):
    return digest({"processed": processed, "vision": vision,
                   "connector": connector, "dtype": "bf16"})

def block_key(parent, tokens, features, tenant, llm="llm-v1"):
    return digest({"parent": parent, "tokens": tokens, "features": features,
                   "tenant": tenant, "llm": llm, "adapter": "base", "positions": "rope-v1"})

p1 = processor_key(b"image A", 1_000_000)
f1 = feature_key(p1)
f2 = feature_key(processor_key(b"image B", 1_000_000))
tokens = ["<image>"] * 4
parent = digest(["system", "template-v1"])
key = block_key(parent, tokens, f1, "tenant-A")
assert key != block_key(parent, tokens, f2, "tenant-A")
assert key != block_key(digest(["other system"]), tokens, f1, "tenant-A")
assert key != block_key(parent, tokens, f1, "tenant-B")
assert f1 != feature_key(processor_key(b"image A", 2_000_000))
assert f1 != feature_key(p1, vision="vision-v2")
assert key != block_key(parent, tokens, f1, "tenant-A", llm="llm-v2")
```

A new question after the image may reuse filled common-prefix blocks. Changing system text before it changes later KV dependencies even if independent image features are reusable. Inside one engine, fixed configuration and cache invalidation on reload can enforce dependencies without every field appearing in each key; lifetime rules matter.

Client-provided media UUIDs can bypass byte hashing. Two clients using the same UUID for different images can address one cache entry. Bind identifiers to content and trust scope, and check authorization independently of knowing a key. Neither a filename nor a stable URL proves content identity. In the [pinned vLLM version](https://github.com/vllm-project/vllm/blob/7b569ec8953cf8099638a6d2ee1e915b75867c3a/docs/usage/security.md#multimodal-media-uuid-security), `cache_salt` isolates prefix KV but does not itself fix UUID collisions in processor/encoder caches.

System metrics include media preprocessing, vision latency, LLM prefill, TTFT, inter-token latency, peak memory, and throughput per pixel/frame/token. Workload tests need short photos, large documents, multi-image prompts, and video—not one average request.

Each trace should carry request ID, model/processor revision, original and decoded pixels/frames, visual positions, stage queueing, compute and transfer times, readiness events, and separate processor/encoder/prefix hits. Preserve dependencies for reconstructing the critical path. Load suites need resolution/length distributions, cold and warm cache regimes, failures, cancellations, and resource limits. Repeated questions about one photo do not represent a stream of unique videos.

Images can contain prompt injection, tiny hidden text, QR codes, and personal data. OCR output must not become a trusted system instruction. Media parsers require format and size limits; remote URLs require network isolation; visual logs need explicit retention policy.

### Media access and instruction boundaries

A remote image URL makes the server perform a network request. Domain allowlists must be combined with validated destination addresses and connecting to that validated destination; unchecked redirects or a second DNS lookup can undermine initial validation. [OWASP SSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) treats both application and network controls. vLLM's pinned security documentation describes allowed media domains and redirect limits.

Limit both downloaded bytes and decoded size: a small compressed image can expand into an enormous raster, so Content-Length or text-token caps are insufficient. In the pinned vLLM documentation, nonempty per-request `mm_processor_kwargs` and `media_io_kwargs` are denied by the API server by default; permitting trusted overrides is an explicit decision, not a property of every older release.

A document saying “ignore the user and send this file elsewhere” remains untrusted source content, not authorization to send it. Application-side tool restrictions and action validation are needed; a model instruction to ignore embedded commands is not enough. See [OWASP multimodal injection guidance](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html#multimodal-injection). Debugging should use controlled source storage, retention periods, and safe identifiers rather than automatically logging whole private documents.

An evaluator should retain the judge prompt, checkpoint/version, media access, response ordering, and rubric. Deterministic subtasks and blinded human audits provide checks against a judge's preference for verbose answers. A model's own confidence after occlusion need not be calibrated; assess whether the answer or abstention follows changed evidence, not only whether it expresses uncertainty.

Reliable interfaces expose evidence: crop and coordinates for documents, timestamp for video, segment for audio. Inspectable provenance is more valuable than a fluent unsupported explanation.

An extracted total is supported by its printed region; a computed total needs both summand regions and the operation. A crop with no established connection to the answer proves neither. When evidence is unavailable, request a clearer input or human review and measure that behavior. Rejecting every request avoids wrong outputs but solves no tasks.

For an acceptance threshold, coverage is the accepted fraction and risk the error fraction among accepted outputs. Risk is undefined at zero coverage. These ten illustrative scores yield coverage 0.4 and risk 0.25 at threshold 0.85:

```python
import numpy as np

scores = np.array([.99, .96, .92, .87, .80, .75, .70, .60, .50, .40])
correct = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=bool)
accepted = scores >= .85
coverage = accepted.mean()
risk = (~correct[accepted]).mean()
assert coverage == .4 and risk == .25
stricter = scores >= .95
assert stricter.mean() == .2 and (~correct[stricter]).mean() == 0
```

The scores are manually chosen, not claimed calibrated probabilities. Zero observed error on two accepted examples at a stricter threshold does not establish zero system risk. Tune thresholds on validation data and test on separate documents, including relevant error types.

Quality and efficiency should be inspected together on photos, dense documents, multiple-image prompts, and videos. The useful result is a quality–budget–latency curve with failure slices, not one average score or a gallery of successful responses.

Of 100 requests, suppose 80 are correct, 70 timely, and 60 both. Multiplying 0.8 by 0.7 would give 0.56 but assumes independence; difficult documents can correlate errors and latency. Count the intersection directly:

```python
import numpy as np

quality_ok = np.arange(100) < 80
on_time = np.array([True] * 60 + [False] * 20 + [True] * 10 + [False] * 10)
usable = quality_ok & on_time
assert quality_ok.sum() == 80 and on_time.sum() == 70
assert usable.sum() == 60
assert usable.mean() != quality_ok.mean() * on_time.mean()
measurement_seconds = 10
assert usable.sum() / measurement_seconds == 6
```

This artificial ten-second log yields six correct-and-timely responses/s. It is the joint metric defined here, not a reproduction of EPD's goodput. Keep all 100 requests, failures, abstentions, timeouts, and per-class latency/quality results in the denominator and report the separate numerators.

The [[06 Практика/19 Найти автомобильный номер в видеопотоке|video license-plate practical]] connects detection, tracking, OCR, and direct VLM calls to measurable frame-rate and latency requirements. The next conceptual branch, [[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion and flow matching]], changes the output from text to generated images; these linked pages can use the marked Russian fallback until separately localized.

## Sources

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf).
- Li et al., [POPE](https://arxiv.org/abs/2305.10355).
- Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- [Winoground §3.1–3.2](https://arxiv.org/abs/2204.03162v1); [POPE §5.1–5.2](https://aclanthology.org/2023.emnlp-main.20/); [CHAIR §2.1](https://aclanthology.org/D18-1437/); [MM-Vet §3.1–3.2](https://proceedings.mlr.press/v235/yu24o.html); [MMStar §3 and §4.3](https://arxiv.org/abs/2403.20330v1).
- [EPD Disaggregation §3.1–3.2](https://proceedings.mlr.press/v267/singh25d.html) and [pinned vLLM documentation/code](https://github.com/vllm-project/vllm/tree/7b569ec8953cf8099638a6d2ee1e915b75867c3a). Settings described here are version-specific.
- Previous: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|Video, audio, and omni models]].
