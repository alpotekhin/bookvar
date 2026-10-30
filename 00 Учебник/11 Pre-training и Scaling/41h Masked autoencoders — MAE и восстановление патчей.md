---
title: Masked autoencoders — MAE и восстановление патчей
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://arxiv.org/abs/2111.06377
  - https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html
  - https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8
  - https://cs231n.stanford.edu/slides/2025/lecture_12.pdf
---

# 41h. Masked autoencoders — MAE и восстановление патчей

A masked autoencoder (MAE) receives a subset of an image's patches and
learns to reconstruct the missing pixels. A label such as “fox” is
unnecessary for this task:
the complete image already contains the values against which predictions
can be compared. The learned encoder can later be adapted to classification
or another recognition task.

MAE uses a large missing fraction and an asymmetric architecture. At the
usual 75% masking ratio, the expensive encoder processes only one quarter
of the image's patch tokens. A smaller decoder receives the encoded visible
patches and placeholders for the missing ones. It predicts pixels at all
positions, but the reconstruction loss selects only missing patches.

The ordinary and denoising-autoencoder objectives are developed in
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|Autoencoder и VAE]].
Here the corruption, token ordering and computation through both networks
are specified explicitly. Unlike the preceding contrastive methods, this
objective does not compare an image with a set of negative images.

<a id="asymmetric-mae"></a>
## The encoder sees the visible patches

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mae-2026-10-28/architecture.png|Original MAE asymmetric encoder and decoder architecture]]

*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár and Ross
Girshick, [MAE project, architecture diagram](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/README.md).
The complete original PNG is retained under the project's CC BY-NC 4.0
license; [license copy](/licenses/mae-efb2a80-CC-BY-NC-4.0.txt).
It is the architecture illustrated in Figure 1 of the paper.*

In the diagram, selected image patches become a short sequence of visible
tokens. The encoder transforms
that sequence. Only afterward are learned mask tokens inserted at the
missing spatial positions. The decoder transforms the full sequence and
projects each patch token to its predicted pixel values.

The distinction between removing and zeroing a patch affects the encoder's
work. If a zero-filled patch or a learned mask token remains in its input
sequence, an ordinary dense Transformer still computes attention and MLP
operations for that position. MAE removes these positions before the
encoder's Transformer blocks. Its implementation can use dense operations
on the remaining shorter sequence; it does not require a sparse-attention
kernel to obtain this reduction.

For a 224×224 RGB image divided into 16×16 patches, there are
$L=14\cdot14=196$ patch tokens. Keeping 25% leaves 49. The pinned code
also appends a CLS token, giving 50 encoder tokens. Its decoder restores
all 196 patch positions and includes CLS during its own Transformer blocks,
so its sequence length is 197. The final pixel prediction removes CLS.

Both networks use bidirectional self-attention in this implementation.
The name “decoder” describes the reconstruction role; it does not imply
the causal mask of an autoregressive language decoder. All missing patch
outputs are predicted in one forward pass. The reconstruction network is
discarded when the pretrained encoder is used for recognition.

<a id="patch-layout"></a>
## From an image tensor to patch targets

Let a square RGB image have height and width $H$, and let the patch side
be $p$, with $H$ divisible by $p$. There are $(H/p)^2$ patches and
$P=3p^2$ pixel coordinates per patch. A batch changes shape from
$[B,3,H,H]$ to $[B,L,P]$. This pixel vector is the reconstruction target;
the encoder instead receives a learned projection of a patch into its
embedding dimension.

Take a 4×4 image and 2×2 patches. Number the first channel 0…15 in row-major
order, the second 16…31 and the third 32…47. The upper-left patch contains
four spatial pixels. Its flattened target starts with channels 0,16,32 of
the upper-left pixel, then 1,17,33 of its right neighbor, then 4,20,36 and
5,21,37 from the next row. The source's flattening order is spatial row,
spatial column, channel within each patch.

The following two helpers retain the original operations and comments from
`models_mae.py`. The small constructor supplies only a patch size and the
loss option. It does not instantiate a pretrained model. `patchify` and
`unpatchify` in this source assume square images and three channels.

```python
from types import SimpleNamespace
import torch

class PatchLayout:
    def __init__(self, patch_size=2, norm_pix_loss=False):
        self.patch_embed = SimpleNamespace(patch_size=(patch_size, patch_size))
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


ops = PatchLayout()
image = torch.arange(48, dtype=torch.float64).reshape(1, 3, 4, 4)
target = ops.patchify(image)
assert target.shape == (1, 4, 12)
assert target[0, 0].tolist() == [0., 16., 32., 1., 17., 33.,
                                     4., 20., 36., 5., 21., 37.]
torch.testing.assert_close(ops.unpatchify(target), image)
print("first patch", target[0, 0].tolist())
```

*Source methods `patchify` and `unpatchify`:
Meta Platforms and contributors,
[pinned models_mae.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py),
CC BY-NC 4.0. Only whitespace is normalized. The constructor and examples
are added here. The other source methods below use the same pin and license.*

Reversing the operation checks that each target coordinate returns to its
original image location.
A matching tensor shape alone would not detect a mistaken channel order.

<a id="mask-and-restore"></a>
## Selecting tokens and restoring their positions

The source generates one random number for each patch of each image,
sorts the patch indices by those numbers and keeps the first
$\lfloor L(1-r)\rfloor$ indices, where $r$ is the masking ratio.
This samples a fixed-size visible subset rather than making independent
keep/drop decisions with a variable token count. Each image has its own
shuffle. A new forward pass normally samples new noise.

For four positions, use noise $(0.4,0.9,0.7,0.1)$. Sorting gives patch
indices $(3,0,2,1)$. At $r=0.75$, only patch 3 is kept. The inverse
permutation is $(1,3,2,0)$: original position 0 occurs at shuffled index 1,
position 1 at index 3, position 2 at index 2, and position 3 at index 0.

| Original position | Noise | Kept? | Decoder input |
| --- | ---: | --- | --- |
| 0 | 0.4 | No | Shared mask vector |
| 1 | 0.9 | No | Shared mask vector |
| 2 | 0.7 | No | Shared mask vector |
| 3 | 0.1 | Yes | Projected encoder feature of patch 3 |

The decoder first has the compact list consisting of visible features
followed by enough copies of the mask vector. Applying the inverse
permutation to that list restores image order. Using the original shuffle
again would generally put features in the wrong positions.

The next program fixes the random numbers for this one trace, while
executing the original `random_masking` method. Markers 0,10,20,30 identify
input positions; −1 identifies the shared mask vector. A real decoder
receives vectors, not these scalar IDs.

```python
from unittest.mock import patch

class MaskedPatches(PatchLayout):
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

ops = MaskedPatches()

tokens = (10 * torch.arange(4, dtype=torch.float64)).reshape(1, 4, 1)
noise = torch.tensor([[.4, .9, .7, .1]])
with patch("torch.rand", return_value=noise):
    visible, binary_mask, ids_restore = ops.random_masking(tokens, .75)
assert visible.flatten().tolist() == [30.]
assert binary_mask.tolist() == [[1., 1., 1., 0.]]
assert ids_restore.tolist() == [[1, 3, 2, 0]]
mask_tokens = torch.full((1, 3, 1), -1., dtype=torch.float64)
packed = torch.cat([visible, mask_tokens], dim=1)
restored = torch.gather(packed, 1, ids_restore.unsqueeze(-1))
assert restored.flatten().tolist() == [-1., -1., -1., 30.]
positions = torch.tensor([[[100.], [200.], [300.], [400.]]])
assert (restored + positions).flatten().tolist() == [99., 199., 299., 430.]
print("mask", binary_mask.tolist(), "restored", restored.flatten().tolist())
```

*Original `random_masking`: Meta Platforms and contributors,
[models_mae.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py),
CC BY-NC 4.0; the wrapper class and controlled-noise trace are added here.*

The three mask vectors start with identical values. Decoder positional
embeddings distinguish the places at which those vectors must make
predictions. Without position information, identical masked tokens in
bidirectional attention cannot identify which missing spatial location
they are meant to reconstruct.

The encoder has its own positional embeddings. In the source, they are
added to all projected patches before the visible subset is gathered.
Patch 3 therefore keeps position 3 even when it becomes the first visible
patch in the compact sequence. CLS is appended separately and is excluded
from patch shuffling and inverse shuffling. Encoder and decoder positions
are fixed two-dimensional sine/cosine embeddings in the original model.

<a id="masked-pixel-loss"></a>
## Pixel error only at missing patches

For image $b$ and patch $i$, let $X_{b,i,c}$ be target pixel coordinate $c$
and $\hat X_{b,i,c}$ its prediction. The binary mask $m_{b,i}$ is one for
a removed patch and zero for a visible one. The source computes

$$
\begin{aligned}
 E_{b,i}&=\frac1P\sum_{c=1}^{P}
 (\hat X_{b,i,c}-X_{b,i,c})^2,\\
 \mathcal L&=
 \frac{\sum_{b,i}m_{b,i}E_{b,i}}{\sum_{b,i}m_{b,i}}.
\end{aligned}
$$

First average the squared errors over coordinates within a patch. Then
average these patch errors over removed positions. With a common image
size and masking ratio, every image contributes the same number of removed
patches. The global denominator would weight images differently if their
numbers of masked patches were allowed to vary.

For the four-patch example, predict every coordinate in the first three
patches two units too high, and every coordinate in the visible patch
100 units too high. Each missing patch has mean squared error 4. The
loss is $(4+4+4)/3=4$; the much larger visible-patch error has zero weight.
For each of the 36 missing pixel coordinates, the derivative is
$2\cdot2/(3\cdot12)=1/9$.

```python
class MAEOps(MaskedPatches):
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

ops = MAEOps()
loss_mask = torch.tensor([[1., 1., 1., 0.]], dtype=torch.float64)
prediction = target.detach().clone()
prediction[:, :3] += 2.
prediction[:, 3] += 100.
prediction.requires_grad_()
loss = ops.forward_loss(image, prediction, loss_mask)
loss.backward()
torch.testing.assert_close(loss, torch.tensor(4., dtype=torch.float64))
torch.testing.assert_close(prediction.grad[:, :3],
                           torch.full((1, 3, 12), 1/9, dtype=torch.float64))
assert prediction.grad[:, 3].count_nonzero() == 0
changed_visible = prediction.detach().clone()
changed_visible[:, 3] -= 500.
torch.testing.assert_close(ops.forward_loss(image, changed_visible, loss_mask), loss)
print("masked MSE", loss.item(), "one masked-coordinate gradient", prediction.grad[0, 0, 0].item())
```

*Original `forward_loss`: Meta Platforms and contributors,
[models_mae.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py),
CC BY-NC 4.0; the numerical prediction/gradient example is added here.*

Zero direct loss on visible *output pixels* does not freeze the encoder.
Its visible input features supply context to the decoder. Changing those
features changes predictions at missing positions, so their reconstruction
errors backpropagate through decoder attention and the visible-token
encoder. Both networks are optimized jointly. There is no MoCo-style
momentum teacher or stopped key branch in this computation.

A missing region can have several plausible contents. Under squared error,
averaging compatible targets can reduce expected loss while producing a
blurred reconstruction. A sharp-looking output and a useful representation
are different measurements; neither can replace recognition evaluation.

For one coordinate that is equally likely to be 0 or 1 given the visible
context, predicting 0.5 gives expected squared error 0.25. Predicting 0
gives expected error 0.5. The lower-error prediction averages the two
possibilities rather than choosing either one. This scalar case explains
why an MSE-trained image reconstruction can lose texture even when its
predicted object outline is plausible.

<a id="normalized-targets"></a>
## Normalizing the reconstruction target

The source optionally centers and scales the pixel vector of each patch
before computing loss. This changes what the decoder is asked to predict.
It is separate from the ImageNet channel normalization applied to an image
before entering the network. Per-patch target normalization happens inside
`forward_loss`; it is not an additional transformation of the encoder input.

For $P$ coordinates, the pinned PyTorch code uses the default sample
variance:

$$
\begin{aligned}
 \mu_i&=\frac1P\sum_c X_{i,c},\\
 s_i^2&=\frac1{P-1}\sum_c(X_{i,c}-\mu_i)^2,\\
 Z_{i,c}&=\frac{X_{i,c}-\mu_i}{\sqrt{s_i^2+10^{-6}}}.
\end{aligned}
$$

One mean and variance cover all RGB coordinates in a patch. This is not
three independent per-channel normalizations. A constant patch has zero
centered target and remains finite because of epsilon. For a nonconstant
12-coordinate patch, predicting zero gives a mean normalized squared
target close to $11/12$, rather than exactly one, because variance used
the denominator $P-1$ while the loss uses $P$.

```python
centered = target - target.mean(dim=-1, keepdim=True)
sample_variance = centered.square().sum(dim=-1, keepdim=True) / (target.shape[-1]-1)
normalized_target = centered / (sample_variance + 1e-6).sqrt()
ops.norm_pix_loss = True
zero_prediction = torch.zeros_like(target)
normalized_loss = ops.forward_loss(image, zero_prediction, loss_mask)
manual_loss = (normalized_target.square().mean(-1) * loss_mask).sum() / loss_mask.sum()
torch.testing.assert_close(normalized_loss, manual_loss)
constant_image = torch.full_like(image, 5.)
assert ops.forward_loss(constant_image, zero_prediction, loss_mask).item() == 0.
assert torch.isnan(ops.forward_loss(image, zero_prediction, torch.zeros_like(loss_mask)))
ops.norm_pix_loss = False
print("zero prediction against normalized targets", normalized_loss.item())
```

The last assertion exposes a boundary of the original helper. With no
masked patches, its denominator is zero. An application must require a
nonempty masked set or define a separate empty-objective behavior. A
masking ratio of one is a different extreme: the encoder has no visible
image patches and the decoder can only use CLS, positions and learned
parameters. Neither extreme describes the usual 75% training setup.

The project distinguishes representation training from its visualization
model. Its original pretraining instructions state:

> Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.

*Meta Platforms and contributors,
[PRETRAIN.md](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/PRETRAIN.md),
CC BY-NC 4.0. The excerpt is retained in English.*

<a id="mae-training-step"></a>
## A complete small training step

The four helpers above specify target layout, selection and loss. To
follow the gradient through the whole computation, the next model adds a
patch projection, encoder attention, decoder attention and a pixel head.
It retains the source's `forward_encoder`, `forward_decoder` and `forward`
operations. The source encoder comment about kept sequence length is
corrected to $L(1-r)$; the operations are unchanged.

This CPU example uses one PyTorch Transformer block in each network,
16-dimensional encoder features and 12-dimensional decoder features.
Fixed illustrative position vectors replace the original 2D sine/cosine
initialization. The source model instead uses timm blocks and much larger
networks; its ViT-L configuration has a 24-block, 1024-wide encoder and
an 8-block, 512-wide decoder. The small constructor below is not that
architecture and does not reproduce its initialization or accuracy.

```python
from torch import nn

class TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (2, 2)
        self.proj = nn.Conv2d(3, 16, kernel_size=2, stride=2)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)

class TinyMAE(nn.Module, MAEOps):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 16))
        self.pos_embed = nn.Parameter(torch.randn(1, 5, 16)*.02, requires_grad=False)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            16, 4, 64, dropout=0., activation="gelu", batch_first=True,
            norm_first=True, layer_norm_eps=1e-6)])
        self.norm = nn.LayerNorm(16, eps=1e-6)
        self.decoder_embed = nn.Linear(16, 12)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 12))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 5, 12)*.02, requires_grad=False)
        self.decoder_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            12, 3, 48, dropout=0., activation="gelu", batch_first=True,
            norm_first=True, layer_norm_eps=1e-6)])
        self.decoder_norm = nn.LayerNorm(12, eps=1e-6)
        self.decoder_pred = nn.Linear(12, 12)
        self.norm_pix_loss = False

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

torch.manual_seed(31)
model = TinyMAE().double()
optimizer = torch.optim.SGD(model.parameters(), lr=.001)
small_images = image / 47.
optimizer.zero_grad()
torch.manual_seed(9)
before, predicted_pixels, training_mask = model(small_images, mask_ratio=.75)
assert predicted_pixels.shape == (1, 4, 12)
assert training_mask.sum().item() == 3
before.backward()
parameters = dict(model.named_parameters())
gradient_names = ["patch_embed.proj.weight", "blocks.0.self_attn.in_proj_weight",
                  "decoder_blocks.0.self_attn.in_proj_weight", "decoder_pred.weight", "mask_token"]
gradient_norms = {name: parameters[name].grad.norm().item() for name in gradient_names}
assert all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms.values())
assert model.pos_embed.grad is None and model.decoder_pos_embed.grad is None
optimizer.step()
torch.manual_seed(9)  # Same mask for a controlled before/after comparison.
with torch.no_grad():
    after, _, repeated_mask = model(small_images, mask_ratio=.75)
torch.testing.assert_close(training_mask, repeated_mask)
assert after < before.detach()
print("fixed-mask loss", before.item(), "->", after.item())
print("gradient norms", gradient_norms)
```

The encoder first projects four image patches to $[1,4,16]$ and adds their
positions. Selection reduces this to $[1,1,16]$; CLS changes it to
$[1,2,16]$. The decoder projection changes the width to 12. Three mask
vectors restore the four patch positions, and CLS gives $[1,5,12]$ during
decoder attention. Removing CLS after the pixel head gives $[1,4,12]$,
the same layout as the reconstruction target.

The checked gradients reach the patch projection, encoder attention,
decoder attention, mask token and output projection. Positional parameters
remain fixed. Unlike the fixed noise used for the ordering trace, normal
training resamples a mask on each forward. The repeated seed here holds
the objective fixed solely to interpret one optimizer step. A lower loss
on this one small tensor establishes neither generalization nor useful
recognition features.

The original training loop also supports mixed precision and gradient
accumulation. Its documented ViT-L example uses an effective batch size
$64\cdot8\cdot8=4096$ and a base learning rate of $1.5\cdot10^{-4}$,
scaled by effective batch size divided by 256, giving 0.0024. These are
the archived recipe's settings, not requirements for the small program.
An accumulation step changes optimization batch size; MAE does not require
a contrastive dictionary of all those images' embeddings.

<a id="reading-reconstructions"></a>
## Raw predictions and reconstructions with visible pixels restored

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mae-2026-10-28/pixel-reconstruction.png|Original MAE pixel reconstruction with all four labeled panels]]

*Meta Platforms and the MAE authors,
[visualization notebook, pixel-reconstruction output](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/demo/mae_visualize.ipynb),
cell 10. Complete original PNG extracted unchanged from the saved output;
CC BY-NC 4.0. Fox image credited in cell 6 as ILSVRC2012_val_00046145.
These are the authors' saved results, not a new local model run.*

The four panels are the complete image, masked input, raw decoder
reconstruction and reconstruction with visible input pixels pasted back.
The third panel can reconstruct visible locations poorly because their
output pixels receive no direct loss. The final panel conceals that error
by copying the visible input. It should not be read as a raw prediction
of every image pixel.

The authors' notebook implements the final image as
`im_paste = x * (1 - mask) + y * mask`, where `mask` has been expanded
from patch flags to pixel coordinates. The same operation can be checked
without loading a checkpoint:

```python
pixel_mask = model.unpatchify(training_mask.unsqueeze(-1).repeat(1, 1, 12))
raw_reconstruction = model.unpatchify(predicted_pixels.detach())
pasted = small_images * (1-pixel_mask) + raw_reconstruction * pixel_mask
torch.testing.assert_close(pasted[pixel_mask == 0], small_images[pixel_mask == 0])
torch.testing.assert_close(pasted[pixel_mask == 1], raw_reconstruction[pixel_mask == 1])
print("visible coordinates copied", (pixel_mask == 0).sum().item())
```

For display, the source notebook also reverses ImageNet channel
normalization and clips the resulting color values. Those display steps
do not change the training loss. A model trained to predict per-patch
normalized targets needs its output interpreted in those target units;
the visualization checkpoint deliberately uses ordinary pixel targets.
The hidden patch's mean and scale are removed from a normalized target.
A predicted normalized vector alone cannot recover those absolute values;
they would need to be supplied or predicted separately. Copying the
visible input does not restore the hidden patches' missing statistics.

### The optional GAN model in the same demo

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mae-2026-10-28/gan-comparison.png|Original MAE demo model with extra GAN loss, separate from the pixel-only baseline]]

*Meta Platforms and the MAE authors,
[visualization notebook, extra-GAN-loss output](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/demo/mae_visualize.ipynb),
cell 14; CC BY-NC 4.0. The same image credit and all four original panel
labels are retained. This is a different checkpoint with an extra GAN loss.*

The authors' demo also loads a separately trained model with an adversarial
loss. Its reconstruction has sharper texture. That image is not evidence
that the pixel-only MSE objective produces the same result, or that the
encoder yields better classification features. The published demo supplies
these two different checkpoints; the source pretraining loop inspected
above implements the masked reconstruction objective, not this extra
adversarial-training recipe.

<a id="mae-cost-and-evaluation"></a>
## Computation and representation evaluation

Removing three quarters of the patch tokens reduces the encoder's
token-wise projection and MLP work roughly in proportion to retained
length. The attention score matrix scales quadratically with length.
Including CLS, 197 full-sequence tokens versus 50 retained-sequence tokens
give $197^2/50^2\approx15.52$ times as many score entries. This is an
attention-matrix comparison, not a prediction of whole-step speedup.
Patch projection, decoder work, memory movement and other operations remain.

The original paper's Table 2 measures 800-epoch pretraining with TensorFlow
on 128 TPU-v3 cores. Its ViT-L encoder with mask tokens and an eight-block
decoder takes 42.4 hours; removing mask tokens from the encoder gives
15.4 hours with the same decoder, a reported 2.8× speedup. A one-block
decoder gives 11.6 hours and 3.7×. The baseline includes mask tokens in
the encoder. These are historical source measurements, not local GPU
results or a comparison against supervised ViT training.

The objective should also be judged after transferring the encoder to
a recognition task. Linear probing freezes the encoder and trains a
linear classifier. End-to-end fine-tuning changes the encoder as well.
MAE's ablations show that these protocols respond differently to the
reconstruction architecture:

| Change in the paper's ViT-L ablation | Fine-tuning top-1, % | Linear-probe top-1, % |
| --- | ---: | ---: |
| One-block decoder | 84.8 | 65.5 |
| Eight-block decoder, default recipe | 84.9 | 73.5 |
| Predict normalized pixels, otherwise default | 85.4 | 73.9 |
| Put mask tokens in the encoder, otherwise default | 84.2 | 59.6 |

These entries use the Table 1 ablation protocol: 800 pretraining epochs,
75% random masking and the specified ViT-L setup, unless that row changes
a component. They are not the later 1600-epoch checkpoint results. A
deeper decoder can absorb more reconstruction-specific processing, while
fine-tuning can adapt the encoder's own layers. The near-equal fine-tuning
scores of one- and eight-block decoders coexist with a large linear-probe
gap. A frozen probe therefore answers a more restricted question than
fine-tuning the whole encoder.

### Other reconstruction targets

The pixel target can be replaced without changing the visible-patch
selection. In Table 1d, one variant projects patches into a PCA basis and
predicts only the 96 largest-variance components. The target then omits
the remaining components instead of asking the decoder to reconstruct all
RGB coordinates. The reported fine-tuning/linear-probe scores fall to
84.6/72.3, compared with 84.9/73.5 for the unnormalized-pixel baseline.
Compressing the target does not automatically preserve the information
that will help recognition.

Another variant uses a separately pretrained dVAE image tokenizer,
following BEiT. That tokenizer maps image content to discrete code indices.
The decoder predicts scores for those indices and uses cross-entropy,
rather than regressing RGB values with MSE. The additional tokenizer has
its own training and inference cost. The tokenizer used in this experiment
was pretrained on 250 million DALL-E images, so the complete pipeline's
data provenance extends beyond the MAE image-training set.

Its reported fine-tuning/linear-probe scores are 85.3/71.6. Predicting
normalized pixels gives 85.4/73.9 in the same ablation table. The discrete
target does not improve either score over normalized pixels here, although
it slightly improves fine-tuning over unnormalized pixels. This comparison
supports the paper's choice of a direct pixel target in that setup;
it does not rule out learned visual tokenizers for other tasks or models.

### Masks, augmentations and training length

At the same removed fraction, a contiguous missing block and scattered
missing patches leave different evidence. A large block can remove an
entire object part and all of its nearby context. A regular grid leaves
neighbors at predictable intervals, making interpolation easier. The
paper compares these cases in Table 1f and Figure 6: block-wise masking
at 75% has higher reconstruction loss, while grid masking at 75% has
lower loss and sharper reconstructions than random masking.

Their recognition results do not rank by reconstruction sharpness.
Random masking at 75% gives 84.9/73.5 for fine-tuning/linear probing;
block masking at 75% gives 82.8/63.9, and grid masking gives 84.0/66.0.
The block-masking experiment at 50% is a different removed fraction and
should not be described as a same-ratio comparison. The reported 75%
random-mask choice is an empirical result for the studied image setup,
not a guarantee for every resolution, patch size or data domain.

Random masking also changes the observed input each time the image is
used. The original MAE baseline combines it with random resized cropping
and horizontal flipping. Table 1e finds that adding color jitter lowers
both reported recognition scores in this recipe. A center crop without
flipping still produces useful features, but its linear-probe result is
65.7 rather than the baseline's 73.5. The contrastive methods' reliance
on strong view augmentation should not be transferred unchanged to this
reconstruction objective; nor does masking make all augmentation choices
equivalent.

Training length is another part of the comparison. In Figure 7, extending
the default unnormalized-pixel ViT-L recipe from 800 to 1600 epochs changes
fine-tuning from 84.9 to 85.1 and linear probing from 73.5 to 75.1. These
are complete training schedules, not successive evaluations of one
800-epoch run. They must also be distinguished from the normalized-pixel
1600-epoch checkpoint results reported elsewhere in the project.

An epoch counts visits to images, not equal computation across methods.
The MAE encoder sees a quarter of the patch tokens per image in this
recipe, while a two-view contrastive method encodes two image views.
The decoder, attention lengths and augmentation policies still contribute
to cost. Comparing epochs alone hides those differences. The
train/validation/test and frozen-probe selection rules from
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|the SimCLR chapter]]
still apply when evaluating these representations.

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder|Momentum contrast — MoCo, очередь и key encoder]]

К модулям: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]

**Источники:** He et al.,
[Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html),
§3 and §4.1, Figures 1–7 and Tables 1–2; Meta Platforms and contributors,
[pinned MAE implementation](https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8),
`models_mae.py`, `engine_pretrain.py`, `PRETRAIN.md`, `FINETUNE.md` and
the complete visualization notebook, CC BY-NC 4.0.
Ehsan Adeli, [Stanford CS231n 2025, Lecture 12](https://cs231n.stanford.edu/slides/2025/lecture_12.pdf#page=52),
slides 52–60, for the course sequence and references.
