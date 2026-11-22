---
title: "64.8. Chameleon: text and images in one sequence"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "bf04d0560f91edfc2a501db1e54119cf6f4ec679764eec524a429ace5521d11e"
translation_reviewed: "2026-11-22"
translation_of: "00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving]]"
next:
  - "[[00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля]]"
primary_sources:
  - https://arxiv.org/abs/2405.09818v2
  - https://arxiv.org/abs/1711.00937v2
  - https://aclanthology.org/2023.findings-acl.890/
  - https://arxiv.org/abs/2203.13131v1
  - https://arxiv.org/abs/2409.18869v1
  - https://github.com/facebookresearch/chameleon/tree/dbbe47fdbc7a3cc4982ae931fea75e227262e70a
  - https://cs336.stanford.edu/
source_unit_id:
  - lecture-17-figure-step-210-rendering-1
  - lecture-17-figure-step-211-rendering-1
---

# Chameleon: text and images in one sequence

A user sends a photo of bananas and asks for a recipe with a picture of the finished loaf. Chameleon writes the recipe. When the Transformer predicts the start-of-image marker, the decoder switches to the image vocabulary and collects 1,024 codes. The image tokenizer's decoder turns those codes into pixels, and the Transformer can continue writing after the image block.

A system with a separate image generator could send it a written description and insert the resulting picture into the recipe. Chameleon generates the entire response with one Transformer. It still needs a separate tokenizer to turn input pixels into codes and output codes back into pixels.

<a id="chameleon-mixed-response"></a>

<span id="где-заканчивается-рецепт-и-начинается-картинка"></span>

## Where the recipe gives way to the picture

In the example below, the paper places the loaf next to the instructions to save space. The <code>&lt;img&gt;</code> marker comes after the last instruction. It stands for an entire image block in the response, not a single token containing the picture.

![[00 Учебник/Assets/Figures/curated/chameleon-decoding-2026-11-09/chameleon-recipe-response.png|Chameleon's recipe ends with an image marker. The generated banana bread is printed beside the text, but follows it in the sequence.]]

*Chameleon Team, [Mixed-Modal Early-Fusion Foundation Models, Figure 4, p. 5](https://arxiv.org/pdf/2405.09818v2#page=5), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Extract from the original response; open the paper for the input photo and the start of the recipe.*

<span id="1-две-независимые-оси-представление-и-место-объединения"></span>

## What enters and leaves the Transformer

In a LLaVA-style model, a vision encoder returns a grid of continuous vectors. A projector puts those vectors in the language model's hidden width. They sit alongside text embeddings and participate in self-attention from the first language layer. The output head still predicts entries in a text vocabulary.

Chameleon turns an image into integer codebook indices, each mapped to a token in the shared vocabulary. The Transformer can predict that token at the output as well as read it at the input. After embedding lookup, the Transformer still works with ordinary continuous tensors. The discrete part is the interface between the image tokenizer and the language model.

Where the streams meet is a separate choice. In early fusion, the modalities enter the same sequence before the main stack. In cross-attention, language states read separate visual memory; other architectures inject more visual states into intermediate layers. See [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|Connectors and fusion]].

| Architecture | Input and output |
|---|---|
| LLaVA-style VLM | The model reads text embeddings and projected visual features. Its text head needs a separate generator to produce a picture. |
| Chameleon | The model reads embeddings of text and image codes, and can predict both. The tokenizer's decoder turns image codes into pixels. |
| LLM with a separate diffusion or flow model | The language model conditions a generator. The generator produces a continuous latent that an autoencoder decodes into an image. |

Both LLaVA and Chameleon bring the modalities together before the main language stack. The phrase *early fusion* doesn't say which values the output head can predict.

<span id="2-почему-непрерывный-зрительный-кодировщик-не-даёт-генерацию-изображения-автоматически"></span>

### Why a text head doesn't produce pixels

If a vision encoder returns 576 features and the projector maps them to width 4,096, the result is a $576\times4096$ input tensor. A text head with 32,000 rows produces 32,000 logits. None of those rows means “emit the fourteenth visual feature” or “draw a pixel.” These dimensions are illustrative.

CLIP features alone don't define an inverse image transform either. A contrastive encoder aligns images with captions, not every letter, texture, and background pixel with a unique state. A generator conditioned on the features can learn a distribution of plausible images; it cannot recover which of several candidates was the original.

A discrete image tokenizer instead learns to reconstruct pixels from a short grid of code indices. The Transformer learns to sample grids of those indices. Independent random indices may combine into grids never seen by the decoder, even if it reconstructs real images well. Unlike [[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion in continuous latents]], here the prior is a product of conditional probabilities over discrete codes.

<span id="3-как-изображение-становится-дискретными-кодами"></span>

## From pixels to codebook indices

<a id="chameleon-vq-codebook"></a>

<span id="поиск-ближайшего-вектора"></span>

### Finding a nearest vector

Chameleon's tokenizer maps a $512\times512$ image to a $32\times32$ grid. There is a vector of width $d_z$ at each location and a codebook of $K=8192$ vectors of that width. Quantization takes the nearest codebook entry at each location:

$$
\begin{aligned}
u_{r,c}&=E(x)_{r,c},\\
k_{r,c}&=\arg\min_{j\in\{0,\ldots,K-1\}}
            \|u_{r,c}-e_j\|_2^2,\\
z^q_{r,c}&=e_{k_{r,c}},\\
\hat x&=D(z^q).
\end{aligned}
$$

$E$ is the image encoder, $D$ the image decoder, and $e_j$ a codebook vector. The decoder receives vectors. The sequence model needs only the indices, which it can map to its own embeddings at a different width.

Take a four-entry book, $(0,0)$, $(1,0)$, $(0,1)$, and $(1,1)$. For $(0.9,0.2)$ the squared distances are $0.85$, $0.05$, $1.45$, and $0.65$, so index 1 wins. An input of $(0.7,0.1)$ also gets index 1. The reconstruction cannot tell those two inputs apart from the code alone.

<details id="chameleon-code-nearest"><summary>Distances to four codebook vectors</summary>

~~~python
import numpy as np

book = np.array([[0., 0.], [1., 0.],
                 [0., 1.], [1., 1.]])
encoded = np.array([[.9, .2], [.1, .8],
                    [.8, .9], [.1, .1]])
distances = ((encoded[:, None, :] - book[None, :, :]) ** 2).sum(-1)
codes = distances.argmin(-1)
quantized = book[codes]

assert np.allclose(distances[0], [.85, .05, 1.45, .65])
assert codes.tolist() == [1, 2, 3, 0]
assert quantized.tolist() == [[1., 0.], [0., 1.],
                             [1., 1.], [0., 0.]]
~~~

</details>

Flattening a $2\times2$ grid in row order gives $[1,2,3,0]$. Chameleon flattens 1,024 positions in the same order, then reverses that reshape before decoding. A position is not a label for an independent patch or object: the encoder's receptive field can extend over neighboring regions.

![[00 Учебник/Assets/Figures/curated/chameleon-tokenizers-2026-11-10/samuel-ovrelid-vq-figure1.png|The encoder produces e(x), the quantizer chooses codebook row z₂, and the decoder receives that vector. The dashed gradient shortcut bypasses the lookup.]]

*David Samuel and Lilja Øvrelid, [Tokenization with Factorized Subword Encoding, Figure 1](https://aclanthology.org/2023.findings-acl.890.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This example encodes subwords; it shows the same lookup and gradient path used for one vector in an image grid.*

Follow the solid arrows from $e(x)$ to codebook row $z_2$, then to the decoder. Keeping the number 2 is enough to recover that row's vector. The [original VQ-VAE figure](https://arxiv.org/html/1711.00937v2/figures/Figure1_9.png) shows the same selection as a nearest-point problem.

<a id="chameleon-vq-gradient"></a>

<span id="обучение-через-недифференцируемый-выбор"></span>

### Training across a discrete lookup

Small moves from $(0.9,0.2)$ keep $(1,0)$ as the nearest vector. The lookup is constant inside a nearest-neighbor region, so its true derivative is zero almost everywhere and undefined at borders. In [VQ-VAE §3.2](https://arxiv.org/html/1711.00937v2#S3.SS2), the dashed path copies the reconstruction gradient from the decoder input to the encoder output. Written with stop-gradient, the straight-through estimate is

$$
z^{\mathrm{ST}}=u+\operatorname{sg}(e_k-u).
$$

The decoder sees $e_k$ on the forward pass. On the backward pass the shortcut treats the derivative with respect to $u$ as the identity and sends no gradient to $e_k$. The lookup hasn't become differentiable; this is a deliberate approximation.

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{rec}}(x,D(z^{\mathrm{ST}}))\\
 &+\|\operatorname{sg}(u)-e_k\|_2^2\\
 &+\beta\|u-\operatorname{sg}(e_k)\|_2^2 .
\end{aligned}
$$

Reconstruction updates the decoder and, through the shortcut, the encoder. The codebook loss moves the chosen vector toward the fixed encoder output. The commitment loss moves the encoder output toward the fixed codebook vector, so the encoder cannot drift away without paying a penalty. The original VQ-VAE used $\beta=0.25$; that is not a specification of the Chameleon tokenizer.

Set the decoder to the identity for a worked case: $u=(0.9,0.2)$, $e_k=(1,0)$, and the reconstruction target is $(0.8,0.4)$. With squared errors summed over the two coordinates, reconstruction contributes $0.20$, the codebook term $0.05$, and commitment $0.0125$. The reconstruction gradient reaches $u$ through the shortcut but not $e_k$. In the codebook term, the stop-gradient on $u$ does the opposite.

<details id="chameleon-code-ste"><summary>Gradients for the encoder and codebook</summary>

~~~python
import numpy as np

u = np.array([.9, .2])
e = np.array([1., 0.])
target = np.array([.8, .4])
beta = .25

reconstruction = ((e - target) ** 2).sum()  # .20
dictionary = ((u - e) ** 2).sum()          # .05
commitment = beta * dictionary            # .0125

encoder_grad = 2 * (e - target) + 2 * beta * (u - e)
book_grad = 2 * (e - u)

assert np.isclose(reconstruction + dictionary + commitment, .2625)
assert np.allclose(encoder_grad, [.35, -.7])
assert np.allclose(book_grad, [.2, -.4])
~~~

</details>

<span id="от-vq-vae-к-токенизатору-chameleon"></span>

### Chameleon's image tokenizer

A reconstruction loss on pixels can average fine textures; perceptual losses compare features, and adversarial losses can favor realistic local detail. Chameleon builds on [Make-A-Scene, §§3.2–3.5](https://arxiv.org/abs/2203.13131v1), whose reconstruction training uses additional face and object features. That citation does not establish that Chameleon copied each loss. Chameleon §2.1 specifies a new tokenizer trained on licensed images, with images of faces upsampled by a factor of two.

The tokenizer learns a code that its decoder can reconstruct. The Transformer then learns to predict code indices alongside text. Loss on these fixed indices cannot backpropagate into the image encoder. Replacing the tokenizer later can also change what an index means: a Transformer trained to emit index 431 would now refer to a different vector.

One way to check codebook use is perplexity, $\exp(-\sum_jp_j\log p_j)$. Four equally used entries give 4, while using one entry gives 1. Uniform random codes can give a large value and still reconstruct poorly.

An index from a book of 8,192 requires 13 bits. Packing 1,024 such indices at fixed width gives 1,664 bytes. Raw 8-bit RGB at $512\times512$ uses 786,432 bytes. This comparison leaves out the learned decoder, metadata, and file formats; it says how much the index stream carries, not whether it beats JPEG.

<details id="chameleon-code-bits"><summary>Size of the packed image codes</summary>

~~~python
import math

codes_per_image = 32 * 32
bits_per_code = math.ceil(math.log2(8192))
payload_bytes = codes_per_image * bits_per_code // 8
raw_rgb_bytes = 512 * 512 * 3

assert codes_per_image == 1024
assert bits_per_code == 13
assert payload_bytes == 1664
assert raw_rgb_bytes / payload_bytes > 472
~~~

</details>

Distinguish the 8,192 categories from the 1,024 positions and from the codebook vector width $d_z$. Only the second number is the number of image-code positions used in the Transformer context.

<span id="4-одна-авторегрессионная-последовательность"></span>

## Predicting text and image codes

<a id="chameleon-token-sequence"></a>

<span id="словарь-embeddings-и-сдвиг-целей"></span>

### Vocabulary and shifted targets

Chameleon has a vocabulary of 65,536 entries, including 8,192 image codes and special tokens. In the pinned code the image boundaries are <code>&lt;racm3:break&gt;</code> and <code>&lt;eoss&gt;</code>. The following sketch uses readable names:

<details id="chameleon-code-sequence"><summary>A response with an image block</summary>

~~~text
<bos> text prompt
<image_start> image-code-0 ... image-code-1023 <image_end>
answer text <image_start> 1024 new codes <image_end>
continued answer <eos>
~~~

</details>

An index in the image tokenizer and its ID in the Transformer vocabulary need not match. VocabTranslation derives the mapping from names that start with IMGIMG; it does not assume a constant offset. A tokenizer codebook vector and a Transformer embedding are also separate parameters.

![[00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-architecture.png|On the left, text and images become tokens for training. On the right, one Transformer emits text and image codes, and an image decoder turns the latter into pixels.]]

*Chameleon Team, [Figure 1, p. 2](https://arxiv.org/pdf/2405.09818v2#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Training uses a mixed sequence. Generation produces text and image codes, which have separate detokenizers.*

The lookup starts from a $B\times L$ tensor of IDs and returns $B\times L\times d_{\mathrm{model}}$ states. The output is $B\times L\times65536$ logits. A single classifier scores both text and image categories; no separately selected output head is needed. The objective remains next-token prediction:

$$
\begin{aligned}
p_\theta(s)&=\prod_{i=1}^{L}p_\theta(s_i\mid s_{<i}),\\
\mathcal L_{\mathrm{NTP}}&=-\sum_{i=1}^{L}\log p_\theta(s_i\mid s_{<i}).
\end{aligned}
$$

In the short sequence below, the state at A predicts image_start and image_start predicts I1. During training the known I1 is already at the next input position. At generation time the model samples it before it can score I2. These are the same shifted targets and teacher forcing used for text.

<details id="chameleon-code-targets"><summary>Shifted targets and index mapping</summary>

~~~python
sequence = ["bos", "A", "image_start", "I1",
            "I2", "image_end", "B", "eos"]
inputs, targets = sequence[:-1], sequence[1:]
assert list(zip(inputs, targets))[2] == ("image_start", "I1")
assert list(zip(inputs, targets))[5] == ("image_end", "B")

# Illustrative mapping, not the IDs in Chameleon's vocabulary.
book_to_lm = {0: 11, 1: 7, 2: 19, 3: 5}
lm_to_book = {value: key for key, value in book_to_lm.items()}
generated_lm_ids = [7, 19, 5, 11]
recovered_codes = [lm_to_book[i] for i in generated_lm_ids]
assert recovered_codes == [1, 2, 3, 0]
~~~

</details>

<span id="что-видит-причинное-внимание"></span>

### What causal attention can read

Text following an input picture can attend to all its codes. An early code in an output picture can attend to the prompt and earlier pictures, but not to its own future codes. When a grid is flattened by rows, the lower-left code in a $2\times2$ grid sees both upper codes. The upper-right code cannot see the lower-left even though their pixels are close. The image decoder, which sees the whole grid, need not reconstruct in this order.

<details id="chameleon-code-causal"><summary>Causal mask for a 2 × 2 grid</summary>

~~~python
import numpy as np

causal_mask = np.tril(np.ones((4, 4), dtype=bool))
# Order: top left, top right, bottom left, bottom right.
assert causal_mask[2].tolist() == [True, True, True, False]
assert not causal_mask[1, 2]
assert causal_mask[3].all()
~~~

</details>

Causal masking controls which positions attention can read. Loss masking controls which shifted targets are scored. Generation masking controls which categories can be sampled. Excluding the prompt from the loss does not stop the answer from attending to it.

<span id="5-что-происходит-при-генерации"></span>

## Switching between text and image generation

<a id="chameleon-output-mask"></a>

<span id="разрешённые-токены-и-границы-изображения"></span>

### The vocabulary mask and image boundaries

While writing text, the decoder allows text tokens, end of response, and the start-of-image marker if images are enabled. It excludes the image codes themselves. Once the image begins, it permits the 8,192 image codes and samples exactly 1,024 of them.

After the last code, the pinned [ImageDecoder](https://github.com/facebookresearch/chameleon/blob/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference/chameleon.py#L356-L371) inserts the end marker without another model call for that choice. The [TextDecoder](https://github.com/facebookresearch/chameleon/blob/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference/chameleon.py#L251-L286) only switches to image generation if the image-start marker is not treated as EOS. That marker is in the default extra_eos_tokens. Allowing the marker and letting it trigger an image are different settings.

The mask sets forbidden logits to $-\infty$ before softmax. For logits $[2,1,3,0]$ and allowed IDs 1 and 2, the probabilities become about $0.1192$ and $0.8808$. The other IDs receive no mass.

<details id="chameleon-code-allowed"><summary>Probabilities of the allowed codes</summary>

~~~python
import numpy as np

logits = np.array([2., 1., 3., 0.])
allowed = np.array([False, True, True, False])
masked = np.where(allowed, logits, -np.inf)
weights = np.exp(masked - masked.max())
probabilities = weights / weights.sum()

assert np.allclose(probabilities, [0., .119202922, .880797078, 0.])
assert np.isclose(probabilities.sum(), 1.)

raw_weights = np.exp(logits - logits.max())
raw = raw_weights / raw_weights.sum()
allowed_mass = raw[allowed].sum()
assert np.allclose(probabilities[allowed], raw[allowed] / allowed_mass)
assert 1.36 < 1 / allowed_mass < 1.37
~~~

</details>

This is the original distribution conditioned on landing in the allowed set. Independent rejection sampling until success would give the same result for a fixed distribution, but would take about 1.37 draws on average for this example. Dropping the first forbidden draw without retry can leave no output. Applying top-p before checking the mask can change which tokens remain. If the mask leaves no tokens, softmax is undefined.

The four-code case below starts in text, collects four integers, and requires an end marker before text resumes. The real decoder makes 1,024 sequential draws, maps them back to codebook indices, and passes the $32\times32$ grid to the image decoder.

<details id="chameleon-code-switch"><summary>Switching between text and image</summary>

~~~python
events = ["text:A", "begin", 1, 2, 3, 0, "end", "text:B"]
state, count = "text", 0
collected, transitions = [], []

for token in events:
    if state == "text":
        if token == "begin":
            state, count = "image", 0
        else:
            assert isinstance(token, str) and token.startswith("text:")
    elif count < 4:
        assert isinstance(token, int) and 0 <= token < 4
        collected.append(token)
        count += 1
    else:
        assert token == "end"
        state = "text"
    transitions.append((state, count))

assert collected == [1, 2, 3, 0]
assert transitions[4] == ("image", 3)
assert transitions[6] == ("text", 4)
~~~

</details>

<a id="chameleon-image-cfg"></a>

<span id="guidance-по-тексту-и-входным-изображениям"></span>

### Guidance from the prompt and earlier images

The pinned implementation prepares three sequences for image generation: the full prompt; previous image codes with their boundaries, but no ordinary text; and a sequence with only beginning-of-sequence and image-start markers. All three receive the same generated prefix of the current image.

Let $z_{\mathrm{full}},z_{\mathrm{img}},z_{\varnothing}$ be the three vectors of logits. InBatchInstructCFGLogitsProcessor combines them as

$$
\begin{aligned}
z_{\mathrm{guided}}={}&z_{\varnothing}\\
 &+s_{\mathrm{img}}(z_{\mathrm{img}}-z_{\varnothing})\\
 &+s_{\mathrm{text}}(z_{\mathrm{full}}-z_{\mathrm{img}}).
\end{aligned}
$$

Both coefficients equal to one recover the full branch; both zero recover the unconditioned branch. Increasing only $s_{\mathrm{text}}$ strengthens the change from adding text to the image condition. [ImageDecoder](https://github.com/facebookresearch/chameleon/blob/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference/chameleon.py#L297-L354) combines logits, masks out non-image codes, then applies temperature and top-p. Masking first could give <code>-inf - -inf</code> when subtracting branches.

<details id="chameleon-code-cfg"><summary>Combining three branches before masking</summary>

~~~python
import numpy as np

unconditional = np.array([0., 0., 0.])
image_only = np.array([1., 0., -.5])
full = np.array([1., 2., -.5])

def guided(s_text, s_image):
    return (unconditional
            + s_image * (image_only - unconditional)
            + s_text * (full - image_only))

assert np.allclose(guided(1., 1.), full)
assert np.allclose(guided(0., 0.), unconditional)
assert np.allclose(guided(2., 1.), [1., 4., -.5])

visual_only = np.array([True, True, False])
masked_after = np.where(visual_only, guided(2., 1.), -np.inf)
with np.errstate(invalid="ignore"):
    masked_before = (np.where(visual_only, full, -np.inf)
                     - np.where(visual_only, image_only, -np.inf))
assert np.isneginf(masked_after[2]) and np.isnan(masked_before[2])
~~~

</details>

The three branch states are batched for each step. [ReplicatedInputTokenSelector](https://github.com/facebookresearch/chameleon/blob/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference/token_selector.py#L34-L47) samples one new code per output image and appends it to all three branches. Otherwise the differences at the next step would compare both different conditioning and different generated images. The fixed image length allows fused control flow in §2.4; it does not make 1,024 dependent steps parallel.

<a id="chameleon-image-budget"></a>

<span id="длина-контекста-и-выдача-пикселей"></span>

### Context space and time to the finished picture

With a 4,096-position context, a full image plus its two markers occupies 1,026 positions. Two input images use 2,052. If text and other markers use another 1,500, the prompt uses 3,552, leaving 544. There is no room for an output image block. Truncating the code grid would not produce the representation the image decoder expects.

<details id="chameleon-code-context"><summary>Context space for a new image</summary>

~~~python
context_limit = 4096
image_block = 1024 + 2
input_positions = 1500 + 2 * image_block
remaining = context_limit - input_positions

assert input_positions == 3552
assert remaining == 544
assert remaining < image_block
assert 3 * image_block == 3078
~~~

</details>

The pinned [length filter](https://github.com/facebookresearch/chameleon/blob/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference/logits_processor.py#L52-L96) checks the current input length. TextDecoder forbids image-start at or above max_seq_len minus 1026. With a single sequence and limit 4096, the filter allows the marker at length 3069, leaving a spare position after the full block; it forbids the marker at 3070. EOS and the new-token limit still apply independently.

<details id="chameleon-code-bound"><summary>Where the start marker becomes unavailable</summary>

~~~python
def admits_begin_marker(input_len, max_seq_len=4096, block_len=1026):
    return input_len < max_seq_len - block_len

assert admits_begin_marker(3069)
assert not admits_begin_marker(3070)
assert 4096 - (3069 + 1026) == 1
assert 4096 - (3070 + 1026) == 0
~~~

</details>

The first visual code cannot be displayed as a completed image. Suppose the client gets the first text token at 500 ms and the first image code is ready at 800 ms. At 5 ms between successive codes, the last code is ready after $800+1023\cdot5=5915$ ms. With another 85 ms to decode and deliver the pixels, the image appears at 6 seconds. This is a timing example, not a measurement. Like [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving#vlm-eval-ttft|time to first token versus time to a complete answer]], the two numbers have different endpoints.

CPU checks of every GPU token can impose synchronization costs. Fixed-length image blocks reduce the need for branching in some paths, but the next code still depends on the last. Counting 1,024 autoregressive steps against, say, 30 diffusion steps does not compare latency: one diffusion step processes a spatial grid, whereas an autoregressive step can reuse KV. Model size, guidance, hardware, and final decoding also matter.

<span id="6-почему-совместное-обучение-становится-нестабильным"></span>

## Stability when training on both modalities

<span id="сдвиг-логитов-и-их-масштаб"></span>

### Logit shifts and changes of scale

Chameleon uses a Llama-like Transformer with RMSNorm, SwiGLU, and RoPE. In [§2.3](https://arxiv.org/html/2405.09818v2#S2.SS3) the authors report slow growth in activation norms and late divergence, particularly when scaling above 8B parameters and a trillion training tokens. They connect the observation to shared weights, different token statistics, and logit drift. The experiments do not establish that all mixed-modal models must diverge.

Softmax assigns the same probabilities to $(1,2)$ and $(101,102)$. Scaling to $(100,200)$ is different: the distribution nearly collapses to the second entry.

<details id="chameleon-code-scale"><summary>Shifting and scaling the logits</summary>

~~~python
import numpy as np

def softmax(values):
    values = np.asarray(values, dtype=float)
    weights = np.exp(values - values.max())
    return weights / weights.sum()

assert np.allclose(softmax([1., 2.]), softmax([101., 102.]))
assert np.allclose(softmax([1., 2.]), [.268941421, .731058579])
assert softmax([100., 200.])[1] > 1 - 1e-12
~~~

</details>

Cross-entropy is insensitive to a common shift of all its logits. Shifting the image categories alone would change their probability relative to text. In the reported experiments, the useful evidence is the activation and loss history, not the shift identity by itself.

![[00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-norm-qk-dropout.png|Three plots compare growth of output norms, training with and without QK-Norm, and training with and without dropout.]]

*Chameleon Team, [Figure 5, p. 6](https://arxiv.org/pdf/2405.09818v2#page=6), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Norm growth in (a) can precede divergence. Panels (b) and (c) compare QK-Norm and dropout in separate runs with different horizontal scales.*

<a id="chameleon-qknorm"></a>

<span id="qk-norm-внутри-внимания"></span>

### QK-Norm in attention

The attention logit is $q^\top k/\sqrt{d_h}$. Multiplying each vector by ten multiplies the logit by a hundred even when the directions are unchanged. Chameleon applies LayerNorm separately to query and key within each head before RoPE. The residual branches still use RMSNorm.

<details id="chameleon-code-qk"><summary>Dot products after LayerNorm</summary>

~~~python
import numpy as np

q = np.array([1., 2., 0., -1.])
k = np.array([2., 0., 1., -1.])

def unit_layer_norm(vector):
    return (vector - vector.mean()) / vector.std()

raw = q @ k / np.sqrt(len(q))
scaled = (10 * q) @ (10 * k) / np.sqrt(len(q))
normalized = unit_layer_norm(q) @ unit_layer_norm(k) / np.sqrt(len(q))

assert np.isclose(scaled, 100 * raw)
assert np.allclose(unit_layer_norm(10 * q), unit_layer_norm(q))
assert np.isclose(
    normalized,
    unit_layer_norm(10 * q) @ unit_layer_norm(10 * k) / np.sqrt(len(q))
)
~~~

</details>

With no learned affine parameters or epsilon, LayerNorm gives identical outputs for $q$ and $10q$. The real layer includes both; it suppresses logit growth along this route without imposing an unconditional bound on every logit. The vocabulary softmax is a different operation.

<a id="chameleon-zloss"></a>

<span id="z-loss-на-выходе"></span>

### Z-loss at the output

For vocabulary logits $z_j$, Chameleon adds

$$
\begin{aligned}
\log Z&=\log\sum_j e^{z_j},\\
\mathcal L_z&=10^{-5}(\log Z)^2.
\end{aligned}
$$

Adding $c$ to every logit adds $c$ to $\log Z$ while leaving the probabilities unchanged. The penalty changes, so the total objective is no longer indifferent to that shift. Its derivative with respect to logit $j$ is $2\lambda\log Z\,p_j$. It does not fix every pairwise difference or guarantee a small spread.

<details id="chameleon-code-zloss"><summary>Log partition and its gradient</summary>

~~~python
import numpy as np

def log_partition(values):
    values = np.asarray(values, dtype=float)
    maximum = values.max()
    return maximum + np.log(np.exp(values - maximum).sum())

z = np.array([1., 2.])
a, b = log_partition(z), log_partition(z + 100)
lam = 1e-5
weights = np.exp(z - z.max())
probabilities = weights / weights.sum()

assert np.isclose(b - a, 100.)
assert np.isclose(a, 2.3132616875)
assert lam * b ** 2 > 1900 * lam * a ** 2
zloss_gradient = 2 * lam * a * probabilities
assert np.isclose(zloss_gradient.sum(), 2 * lam * a)
~~~

</details>

Computing log-sum-exp after subtracting the maximum avoids exponentiating large numbers directly. It cannot recover differences already lost in earlier operations. Near 1,024, adjacent BF16 numbers on the upper side are eight apart, so a difference of one can disappear. That is a precision illustration, not a diagnosis of the first failing operation in Chameleon's training runs.

<span id="положение-нормализации-и-масштаб-модели"></span>

### Normalization before or after each sublayer

In pre-norm Llama, the sublayer reads a normalized state and adds its unnormalized output to the residual. In Chameleon-34B the sublayer output is normalized before addition:

$$
\begin{aligned}
h_{\mathrm{Llama}}&=x+A(N(x)),\\
y_{\mathrm{Llama}}&=h_{\mathrm{Llama}}+F(N(h_{\mathrm{Llama}})),\\
h_{\mathrm{Cham}}&=x+N_A(A(x)),\\
y_{\mathrm{Cham}}&=h_{\mathrm{Cham}}+N_F(F(h_{\mathrm{Cham}})).
\end{aligned}
$$

$A$ is attention, including QK-Norm where used, and $F$ is the feed-forward branch. For a toy branch $f(x)=100x$ and $x=(1,2)$, pre-norm gives a branch output with RMS 100. Normalizing that output gives RMS 1 with a unit gain. The total residual can still grow when several branches add vectors pointing in the same direction.

<details id="chameleon-code-norm"><summary>Normalizing a sublayer's input or output</summary>

~~~python
import numpy as np

x = np.array([1., 2.])

def rms_norm(vector):
    return vector / np.sqrt(np.mean(vector ** 2))

pre_branch = 100 * rms_norm(x)
post_branch = rms_norm(100 * x)

assert np.isclose(np.sqrt(np.mean(pre_branch ** 2)), 100.)
assert np.isclose(np.sqrt(np.mean(post_branch ** 2)), 1.)
assert np.linalg.norm(x + post_branch) > np.linalg.norm(x)
~~~

</details>

The reported 7B recipe used QK-Norm, z-loss, and dropout 0.1 after attention and the feed-forward layers. Dropout did not fix the 34B runs; moving the norms and dropping dropout stabilized that model. The authors also got 7B to train without dropout once they reordered norms, retaining QK-Norm.

![[00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-training-ablations.png|Three sets of loss curves show 7B and 34B training, a 7B ablation without image generation, and 34B dropout runs with two norm placements.]]

*Chameleon Team, [Figure 6, p. 7](https://arxiv.org/pdf/2405.09818v2#page=7), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Panel (b) trains on a different objective. Comparing its absolute loss with panel (a) would not compare model quality.*

An average loss can hide a rising image-token loss if the text part improves. Inspect each modality separately alongside norms and logits. A short run cannot establish that the late instability has been fixed.

<span id="7-данные-и-две-стадии-обучения"></span>

## Pre-training data and answer tuning

<span id="порядок-модальностей-задаёт-условную-задачу"></span>

### Sequence order changes the conditional task

Putting an image before its caption makes the caption targets conditional on the image, $p(T\mid I)$. Putting the caption first makes the image conditional on that caption, $p(I\mid T)$. Next-token training models the leading part too. Chameleon places the image first in about half the text–image pairs. Interleaved documents add longer dependencies, such as keeping a character consistent with both an earlier picture and the preceding paragraph.

The report lists about 2.9T text tokens, 1.5T tokens from 1.4B text–image pairs, and 400B tokens in interleaved documents. These rounded figures sum to 4.8T; the architecture table lists 4.4T for the dataset, 2.1 passes, and both variants reportedly saw about 9.2T tokens. The published numbers do not specify an exact reconstruction of the sampled mixture.

The first stage takes 80% of training. In the last 20%, the authors reduce the weight of the first stage's data by half, add higher-quality data and filtered instruction examples, and keep a similar proportion of image–text tokens. Stanford CS336's shorthand “50/50” is not a specification of all sampling weights.

<span id="почему-доля-примеров-не-равна-доле-токенов"></span>

### Example counts are not token counts

Ninety text answers of 100 tokens plus ten image-only answers of 1,026 tokens give images 10% of the examples and about 53.3% of the targets.

<details id="chameleon-code-mix"><summary>Images as a share of examples and tokens</summary>

~~~python
text_tokens = 90 * 100
image_tokens = 10 * 1026
image_fraction = image_tokens / (text_tokens + image_tokens)

assert 0.532 < image_fraction < 0.534
assert image_fraction > .1
~~~

</details>

Averaging over targets gives longer answers more weight. Averaging within each answer and then over answers does not. Sampling the datasets is yet another decision. This example does not assume an unspecified reduction in Chameleon's training.

Table 3 lists 1.6M text examples, 14.1K code, 15.6K visual chat, 64.3K image generation, 16.9K mixed generation, and 95.3K safety examples. Those counts do not give the probability of choosing a category at a step. The authors describe an imbalance that makes the model issue pictures even when not asked or rarely offer one even when requested.

In supervised fine-tuning, the prompt remains visible and the loss applies to answer positions. For loss values $[4,3,2,1]$ and an already shifted answer mask $[0,0,1,1]$, the answer average is 1.5, not the all-position average 2.5.

<details id="chameleon-code-sft"><summary>Averaging loss over answer targets</summary>

~~~python
import numpy as np

per_target_loss = np.array([4., 3., 2., 1.])
answer_mask = np.array([0., 0., 1., 1.])
answer_loss = (per_target_loss * answer_mask).sum() / answer_mask.sum()

assert answer_loss == 1.5
assert per_target_loss.mean() == 2.5
~~~

</details>

During fine-tuning, the authors padded input images to preserve visual evidence and used center crops for image outputs. For an $800\times400$ photo, fitting it into a square at size 512 leaves a $512\times256$ image with padding. Resizing for a center crop gives $1024\times512$ before removing 256 pixels from each side. The latter can remove an object near an edge. The pinned public ImageTokenizer uses center cropping; the inference implementation need not match the paper's preparation for training inputs.

<span id="8-что-даёт-единый-словарь-а-чего-он-не-гарантирует"></span>

## Evaluating a mixed response

<span id="реконструкция-генерация-и-проверка-фактов"></span>

### Reconstruction, generation, and accuracy

If two small inscriptions map to the same code grid, the Transformer cannot tell which was in the source. A decoder might draw plausible letters without recovering the original text. The authors report weak reconstruction on text-heavy images, relevant to [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|document reading]].

Generation starts without an original image to recover. The Transformer has to sample a suitable grid from the images the decoder can represent. Good reconstruction doesn't establish that code prediction models the right distribution, and a low next-token loss need not guarantee compliance with a detailed request.

![[00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-mixed-document.png|Chameleon describes three birds, placing a generated picture after each bird's description.]]

*Chameleon Team, [Figure 2, p. 3](https://arxiv.org/pdf/2405.09818v2#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is an example of the response format, not a verified account of the birds.*

The pictured species, the facts in the description, and the requested number of birds can each be correct or wrong. A convincing generated portrait is not evidence that the neighboring natural-history claim is true.

<a id="chameleon-mixed-evaluation"></a>

<span id="как-читались-результаты-статьи"></span>

### How the human preference scores were counted

The report gathered 1,048 prompts expecting a mixed-modal response: 441 contained images and 607 were text only. The unmodified GPT-4V and Gemini Pro responses used text. For a second comparison, those systems wrote captions for DALL-E 3; the resulting baselines are GPT-4V+ and Gemini+. The results describe those particular systems and task distribution.

![[00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-human-evaluation.png|Task-fulfillment results on the left; shares of Chameleon wins, ties, and losses against four baselines on the right.]]

*Chameleon Team, [Figure 9, p. 12](https://arxiv.org/pdf/2405.09818v2#page=12), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The plus sign marks the DALL-E 3 baselines. Absolute task fulfillment and pairwise preference use different questions.*

Against Gemini+, the figure shows 41.5% wins, 34.5% ties, and 24.0% losses. Giving half credit for a tie yields $41.5+34.5/2=58.75\%$ from the rounded bars. The exact counts in [Appendix B, Table 11](https://arxiv.org/pdf/2405.09818v2#page=26) give $(435+362/2)/1048\approx58.8\%$ overall. The 60.4% stated in the main text matches the row for the 441 image-containing prompts: $(194+145/2)/441\approx60.4\%$. The paper does not explain this change of denominator. For GPT-4V+, both the plotted shares and the appendix give 51.6% overall.

<details id="chameleon-code-wins"><summary>Two denominators for the Gemini+ comparison</summary>

~~~python
import math

gemini_plus_score_from_figure = 41.5 + .5 * 34.5
gpt4v_plus_score_from_figure = 35.8 + .5 * 31.6

assert gemini_plus_score_from_figure == 58.75
assert math.isclose(gpt4v_plus_score_from_figure, 51.6)
assert abs(gemini_plus_score_from_figure - 60.4) > 1.6
assert round(100 * (435 + .5 * 362) / (435 + 362 + 251), 1) == 58.8
assert round(100 * (194 + .5 * 145) / (194 + 145 + 102), 1) == 60.4
~~~

</details>

The human evaluation includes no OCR tasks, so these results cannot settle performance on a fine-print document. Evaluate the selected output modality, visual quality, factual accuracy, and consistency across images and text for the intended application.

<span id="архитектура-и-доступный-релиз"></span>

### The paper and the released interface

The paper describes image generation and the pinned inference code contains TextDecoder and ImageDecoder. [Transformers 5.20.2](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/chameleon) documents text output and states that its image generation module is not released. Check the particular checkpoint, tokenizer, and library rather than treating the paper as an API specification. The paper's CC BY license is separate from the research license that applies to the implementation and weights.

<span id="9-chameleon-llava-и-diffusion-три-разных-контракта"></span>

## When to use image codes

For a text answer about a scanned invoice, a high-resolution image encoder and an ordinary text head may be enough. For illustrations inside a story, an application can assemble a response from separate text and diffusion models, or a model such as Chameleon can predict where an image starts and continue the same sequence with its codes. In either case, evaluate the preprocessing and decoder as well as the Transformer. A system good at continuing an illustrated story may still fail to distinguish small digits on a scan.

<a id="chameleon-to-emu3"></a>

<span id="от-изображения-к-видео-emu3"></span>

### From images to video: Emu3

Emu3 represents text, images, and video in one causal sequence. Its tokenizer compresses the spatial axes by eight and the temporal axis by four. The number of resulting positions depends on both frame size and how many frames are encoded together.

![[00 Учебник/Assets/Figures/curated/chameleon-tokenizers-2026-11-10/emu3-figure1.png|At the bottom, text, images, and video become tokens. One Transformer predicts continuations, which decode into the corresponding modalities at the top.]]

*Emu3 Team, [Figure 1](https://arxiv.org/html/2409.18869v1#p1.g1), [original PNG](https://arxiv.org/html/2409.18869v1/emu3_arch5.png), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Green boxes mark token positions. The diagram does not show the row and frame boundaries.*

At $512\times512$, Chameleon's image grid is $32\times32=1024$ codes. The ratios in [Emu3 §2.2](https://arxiv.org/html/2409.18869v1#S2.SS2) give $64\times64=4096$ codes for one image. Four frames of the same size can also produce 4096 codes: temporal compression turns the group into one latent slice. The tokenizer processes those frames together. Its serialized sequence also marks rows and frames and encodes size and duration as text ([§2.4](https://arxiv.org/html/2409.18869v1#S2.SS4)).

More codes imply more positions for a fixed image size, not necessarily better decoding. The codebooks, decoders, and training data differ too.

<span id="источники-и-продолжение"></span>

## Sources and next chapter

- Chameleon Team, [Chameleon v2](https://arxiv.org/html/2405.09818v2), §§2.1–2.4, 3–4 and Appendix B: tokenization, stable training, decoding, and human evaluation.
- van den Oord, Vinyals, and Kavukcuoglu, [Neural Discrete Representation Learning](https://arxiv.org/html/1711.00937v2), §§3.1–3.3: the discrete bottleneck, straight-through training, and the prior.
- Samuel and Øvrelid, [Tokenization with Factorized Subword Encoding](https://aclanthology.org/2023.findings-acl.890/), §2: the codebook and gradient shortcut shown here.
- Gafni et al., [Make-A-Scene](https://arxiv.org/abs/2203.13131v1), §§3.2–3.5: reconstruction losses and training with face and object features.
- [Stanford CS336, Lecture 17](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_17.py#L267) compares Chameleon with VLMs that inject visual features. The Chameleon figures are from the paper cited in the lecture.
- [Pinned Meta implementation](https://github.com/facebookresearch/chameleon/tree/dbbe47fdbc7a3cc4982ae931fea75e227262e70a/chameleon/inference) and [Transformers 5.20.2 documentation](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/chameleon).
- Previous: [[en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving|Evaluating VLMs]]. Next: [[00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Tools and agents]].
