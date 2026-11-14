import { mapMarkdownProse } from './markdown-prose.js';

const revision = 'e632aa89ca9e6638d52e1b686095e7442faffbb0';
const course = 'https://github.com/mryab/efficient-dl-systems/';
const source = course + 'blob/' + revision + '/';
const raw = course + 'raw/' + revision + '/';
const week02 = 'week02_fast_pipelines/seminar/images/';
const week06 = 'week06_dl_arithmetic/seminar/images/';
const snapshots = 'Assets/Sources/Efficient DL Systems/repository-figures/';

function figure(asset: string, alt: string, credit: string) {
  return '\n\n![[' + asset + '|' + alt + ']]\n\n*' + credit + '*\n\n';
}

function snapshot(file: string, alt: string, explanation: string) {
  return figure(snapshots + file, alt,
    'Efficient Deep Learning Systems course authors, [original snapshot](' + source + week06 + file +
    '), [MIT license](' + source + 'LICENSE). ' + explanation);
}

function imageOrLink(url: string, alt: string): string {
  switch (url) {
    case raw + week02 + 'FP8.png':
      return '\n\n<a id="edls-fp8-scaling"></a>' + figure(
        '00 Учебник/Assets/Figures/curated/precision-review-2026-09/te-mxfp8-blocks.png',
        'Left: one FP32 scale for a whole tensor. Right: a separate E8M0 scale for each block.',
        'NVIDIA Transformer Engine contributors, [original FP8 and MXFP8 comparison]' +
        '(https://github.com/NVIDIA/TransformerEngine/blob/eb56ba22fd2b3c1fa4c962a4a575d4ba12d1fcbc/docs/examples/MXFP8_FP8_comparison_1.png), ' +
        '[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The notebook splits the two panels; ' +
        'the full original is shown here. Each real MXFP8 block has 32 values; the drawing shows four.',
      ) + '> In [Transformer Engine’s hybrid recipe](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html#fp8-data-type), ' +
        'E4M3 is used for forward weights and activations, and E5M2 for the gradient operand in the backward pass. ' +
        'This is not a rule that stores weight gradients in E5M2 and activation gradients in E4M3.\n\n';
    case raw + week02 + 'MXFP8.png':
      return '[Compare the FP8 and MXFP8 panels above](#edls-fp8-scaling) · [MXFP8 panel in the original notebook](' + source + week02 + 'MXFP8.png)';
    case raw + week06 + 'vanilla_ce.png':
      return snapshot('vanilla_ce.png', 'Several allocations overlap and the combined trace rises above the 1.00G tick.',
        'Read the peak where the colored allocations overlap.');
    case raw + week06 + 'compiled_ce.png':
      return 'The course supplies [the same image for the compiled run](' + source + week06 + 'compiled_ce.png' +
        '). The command above names the snapshot rather than the script. ' +
        'Run [profile_compiled_ce.py](' + source + 'week06_dl_arithmetic/seminar/cross_entropy/profile_compiled_ce.py' +
        ') from the course checkout to obtain a separate trace.';
    case raw + week06 + 'liger_ce.png':
      return snapshot('liger_ce.png', 'A single allocation spans the timeline, just above the 260M tick.',
        'The [profiling script](' + source + 'week06_dl_arithmetic/seminar/cross_entropy/profile_liger_ce.py) ' +
        'allocates 1,024 × 128,000 BF16 logits: 262,144,000 bytes (250 MiB). ' +
        'The next variant projects one chunk at a time instead.');
    case 'https://ar5iv.labs.arxiv.org/html/2309.14509/assets/figs/image3.png':
      return figure('00 Учебник/Assets/Figures/curated/tensor-sequence-2026-10-23/ulysses-design.png',
        'The first all-to-all redistributes queries, keys and values by attention head. The second restores the sequence partition.',
        'Sam Ade Jacobs et al., [DeepSpeed Ulysses, Figure 2](https://arxiv.org/html/2309.14509v2#S3.F2), ' +
        '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure.');
    case raw + week02 + 'mixed_precision.jpeg':
      return '[Mixed-precision training cycle (NVIDIA)](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/#mixed-precision-training-iteration) · ' +
        '[Diagram as used in the notebook](' + source + week02 + 'mixed_precision.jpeg)';
    case raw + week02 + 'loss_scaling.png':
      return '[Loss-scaling diagram in the notebook](' + source + week02 + 'loss_scaling.png' + ')';
    case raw + week02 + 'mem_snapshot.jpeg':
      return '[Three training iterations in the PyTorch memory timeline](https://pytorch.org/blog/understanding-gpu-memory-1/#understanding-gpu-memory-1-visualizing-all-allocations-over-time) · ' +
        '[Screenshot in the notebook](' + source + week02 + 'mem_snapshot.jpeg)';
    case raw + week06 + 'mem_hierarchy.png':
      return '[H100 memory hierarchy, Figure 2 by Aleksa Gordić](https://www.aleksagordic.com/blog/matmul/mem_hierarchy.png)';
    case raw + week06 + 'liger_fused.avif':
      return figure('00 Учебник/Assets/Figures/curated/triton-review-2026-10-23/liger-flce-original.png',
        'The linear head projects one input chunk. The loss kernel reuses the chunked logits buffer for their gradients.',
        'Pin-Lun Hsu et al., [Liger Kernel, Figure 1](https://arxiv.org/html/2410.10989v3#S3.F1), ' +
        '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure from the paper.');
    case 'https://pytorch.org/tutorials/_images/megatron_lm.png':
      return '[Tensor-parallel MLP and attention, Figure 1 of the PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works)';
    case 'https://i.imgur.com/a1QhKF7.png':
      return '[Example generation (the generated code is blue)](' + url + ')';
    case 'https://i.imgur.com/3gZ2KPB.png':
      return '[Screenshot of the checkpoint shards in Files and versions](' + url + ')';
    case 'https://i.imgur.com/cv9WvYw.png':
      return '[Screenshot: copy the download link for one shard](' + url + ')';
    case 'https://habrastorage.org/webt/xq/r5/8a/xqr58aw0gd6tdm-j45yqad67w1a.png':
      return '[Quantization diagram: which axes are reduced to compute the scales](' + url + ')';
    default:
      return '[' + (alt.trim() || 'Original image') + '](' + (url.startsWith(raw) ? source + url.slice(raw.length) : url) + ')';
  }
}

const fusionPair = '<p float="left">\n' +
  '<img src="' + raw + week06 + 'fused_kernels1.png" width="400"/>\n' +
  '<img src="' + raw + week06 + 'fused_kernels2.png" width="400"/>\n</p>';
const colab = 'https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/';
const pinNotebook = (url: string) => url.replace(colab + 'main/', colab + revision + '/');

/** Adapt only the reader copy. The imported slides and notebook markup remain untouched. */
export function edlsReaderMedia(markdown: string): string {
  return mapMarkdownProse(markdown, prose => prose
    .replace('The embedded PDF is the primary visual version.', 'The original PDF contains the slide figures.')
    .replace(/<iframe class="source-pdf" src="(https:\/\/github\.com\/mryab\/efficient-dl-systems\/blob\/[a-f0-9]{40}\/[^"\s]+\.pdf)\?raw=1"[^>]*><\/iframe>/g,
      (_, url: string) => '[Open the original slides (PDF)](' + url + ')')
    .replace(fusionPair, '[Unfused and fused kernels (Horace He)](https://horace.io/brrr_intro.html) · ' +
      '[Unfused diagram](' + source + week06 + 'fused_kernels1.png) · [Fused diagram](' + source + week06 + 'fused_kernels2.png)')
    .replace(/\[!\[[^\]\n]*\]\(https:\/\/camo\.githubusercontent\.com\/[^)]+\)\]\((https:\/\/colab\.research\.google\.com\/github\/mryab\/efficient-dl-systems\/[^)]+)\)/g,
      (_, url: string) => '[Open in Colab](' + pinNotebook(url) + ')')
    .replace(/<a href="(https:\/\/colab\.research\.google\.com\/github\/mryab\/efficient-dl-systems\/[^"]+)"[^>]*><img src="https:\/\/colab\.research\.google\.com\/assets\/colab-badge\.svg"[^>]*\/><\/a>/g,
      (_, url: string) => '[Open in Colab](' + pinNotebook(url) + ')')
    .replace(/!\[([^\]\n]*)\]\((https:\/\/[^)\s]+)\)/g,
      (_, alt: string, url: string) => imageOrLink(url, alt)));
}
