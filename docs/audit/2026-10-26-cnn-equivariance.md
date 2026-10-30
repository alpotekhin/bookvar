# CNN: thematic rotation-equivariance extension

The Harvard Network Architectures review reached the end of its CNN section.
The accepted MLP and CNN explanations already cover dense layers, parameter
counts, universal-approximation limits, spatial kernels, channels, pooling,
receptive fields and im2col. They were read in full rather than judged from
headings. The missing symmetry-group extension was added to the existing CNN
chapter instead of creating another short parallel chapter.

## Current artifact and preservation

Canonical chapter:
`00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md`.
Current length: 1093 lines. SHA-256:
`d2fe8eaa5ceda18400dbbfae92d1fdcb13a61c07a35964adae0ef76cda65ee6e`.
Reader route:
`/textbook/neural-networks/cnn-lenet-resnet/#rotation-equivariant-feature-maps`.

The complete final chapter was reread in order in four bounded chunks after
the final explanatory edit. Removing the single marked insertion restores
the previously accepted 880-line chapter, SHA-256
`c6cceb20910882ae1408dc7013c286ce77c3e00b0ab32070505e7df6b6bf5355`.
Both the independent Python checker and the unit test enforce this guard.
No earlier explanation, source figure, code block or navigation anchor was
replaced. The MLP chapter was read for comparison and not edited.

## Sources and explanation review

Harvard source pin: `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`.
The current reading cursor covers the opening, principles and MLP/CNN text
through line 1990. The long numeric zebra path definitions were inspected
as the complete original SVG in the browser, not coordinate-by-coordinate.
The earlier selected decision-framework material at 4560–4993 remains read.
The next sequential cursor is line 1991, RNNs. The full chapter and native
image directory are not accepted as complete.

The missing topic was followed to primary sources:

- the complete escnn README and LICENSE, pin
  `9ad44cc37d694d9c805c4fef4d16722a254b3bf2`;
- Cohen and Welling's ICML 2016 publisher page and relevant arXiv v3 HTML
  material on input/output actions, group feature maps, correlation and
  transformed filters; no long paper excerpt or paper figure copied;
- the UvA regular-group-convolution and steerable-CNN tutorial introductions
  and scope, retained as further-reading links. Their full notebooks,
  training runs and full course transfer are not claimed complete.

Source archive and hashes:
`05 Источники/Originals/group-equivariance-2026-10-26/README.md`.

| Mechanism | Preserved explanation / worked check | Qualification |
| --- | --- | --- |
| Scalar versus vector fields | Full original PNG and unchanged English source paragraph; a temperature value moves, while an optical-flow vector also changes direction | Input and output actions need not be the same array operation |
| Equivariance versus invariance | Explicit `Phi(T_g x)=T'_g Phi(x)`; invariant classifier uses the identity output action | Intermediate vectors need not remain unchanged in the original coordinates |
| Quarter-turn lifting | One asymmetric 3×3 kernel produces four orientation planes from a 9×9 image, giving `[1,1,4,7,7]` | Valid, stride-one, square arrays, no padding/bias; first lifting layer only |
| Orientation permutation | Rotating the image requires a spatial turn and `roll(1)`; spatial rotation alone fails the printed counterexample | Orientation ordering determines the sign of the roll |
| Sharing versus execution cost | Nine independent weights versus 36 unrelated values, but four output maps and four filter applications | Parameter sharing does not promise unchanged activation memory or latency |
| Later group layers | Relative pose `g^-1 h`, including the origin-only orientation example `(3-1) mod 4=2` | Full later-layer G-correlation is explained, not implemented by the lifting helper |
| Nonlinearity and pooling | ReLU commutes with the permutation action used here; global orientation/spatial pooling is invariant in the CPU example | This is not a claim that arbitrary componentwise nonlinearities preserve every vector representation |
| Continuous-rotation demo | Two complete original GIFs show the authors' steerable versus conventional response; original English explanation retained | Grid interpolation/discretization causes fluctuations; animations are not local model runs |

The no-ai-slop review retained mechanisms, examples, equations and source
language. A closing integration-status disclaimer was moved out of the
reader explanation; source excerpt text was not rewritten. Necessary
contrasts between different transformation actions and execution stages
remain.

## Figures and permission evidence

The three complete source-original objects use BSD-3-Clause-Clear, not MIT:

| Asset | Size | SHA-256 |
| --- | --- | --- |
| escnn-feature-fields.png | 2082×547 | `2f6dea2bbf0cb3cb3982e248b2f3adada87a362435f944d34ab8c614cb617dee` |
| escnn-vectorfield.gif | 1024×366 | `c05794ecb738ec55a0f45bfea4491030273fe36d73bb344ece7e826c157441cb` |
| escnn-conventional-cnn.gif | 1024×366 | `2cf065c8179acea1156edd19af44f9cbff2e895e799ede040fc075ab722b5d27` |

All bytes, including GIF frames, are unchanged. The local curated manifest,
central registry, pinned source README and full license hashes are checked.
The complete notice is stored beside the assets and reproduced in the
chapter's expandable source-license section, verified on the built site.
No drawing, crop, relabeling, translation or recompression was performed.

GrouPy was investigated first. Its custom educational permission is followed
by a separate redistribution restriction. Its downloaded originals remain
research sources and are not embedded, copied to curated assets or called
MIT-licensed. escnn supplies the used illustrations. A failed direct PMLR
PDF download is not counted as a saved or reviewed local PDF.

## Executed evidence

The current `publishing/tools/check_cnn_examples.py`, SHA-256
`ca73ea065d759522f814b3adb81264ad2a6909728e77eb6e76af3d7ed24d211b`,
executes all 16 actual Python fences in reading order on CPU, with offline
cached PyTorch 2.14.1 and NumPy 2.3.5. It also checks:

- 720 quarter-turn configurations across 240 random square-input/kernel
  configurations, including multi-input/multi-output channels;
- the orientation/spatial action, invariant global pooling and a new
  finite-difference gradcheck through the lifting operation;
- both exact English escnn excerpts and the accepted-chapter preservation
  hash;
- all earlier D2L AST comparisons, 80 source/library correlations, four
  previous gradchecks, 396 indexed stride/padding/dilation cases and the
  prior group/depthwise/pooling/residual checks.

No escnn/GrouPy installation, trained network, image dataset, accelerator,
throughput benchmark or full continuous-rotation test was run. The report
does not assign the author's GIF outputs to the local CPU example.

Validation logs under `/var/tmp/`:

- `bookvar-cnn-equivariance-final-examples.log`: all current chapter
  examples and additional checks pass, with the final chapter hash;
- `bookvar-cnn-equivariance-all-unit.log`: 63 files / 513 unit tests pass
  on the complete extension before the final prose/notice layout changes;
- `bookvar-cnn-equivariance-final-unit.log`: five affected tests pass on
  the final chapter, source bindings and architecture practice;
- `bookvar-cnn-equivariance-build-final.log`: 1969-page final Astro build,
  22.55 seconds; search index and sitemap complete;
- `bookvar-cnn-equivariance-output.log`: 135 final built-output tests pass;
- `bookvar-cnn-equivariance-links.log`: zero broken internal routes,
  fragments or files;
- `bookvar-cnn-equivariance-coverage.log`: structural source-coverage pass,
  210 rows / 58 destinations. Not evidence of full semantic integration.

An intermediate figure test timed out because deep equality traversed large
GIF buffers byte-by-byte. It now uses native `Buffer.equals`, preserving
exact byte comparison and all hash checks; the timeout was not raised.
The first output test also expected 15 code blocks. It was updated to 16
and strengthened with the three new source images, mechanism headings,
code and full notice assertions; none of the prior checks was removed.

## Actual browser inspection

The final local route was loaded at 1280×900 and 360×800. The complete static
figure and both GIFs were viewed; the animations visibly change frames.
Both source comparisons retain all three panels, including their headings.
At mobile width, the page is 360/360 pixels, all ten display formulas fit
the 328-pixel column and there are no KaTeX errors. The new assets load at
their original dimensions. At 3× the GIF's input and stabilized panels are
accessible; the PNG's scalar/vector labels and arrow directions were read
through the viewer. Inline mobile images remain overviews rather than
full-size labels.

The full copyright notice opens on the actual page. The English route
shows the explicit fallback notice, the same new section, all three new
images and ten fitting formulas. No English translation body was edited.
The final tab is left on the new original-demo section; viewport overrides
were reset.

Screenshots:

- `screenshots/2026-10-26-cnn-equivariance-demo-desktop.png` (final build);
- `screenshots/2026-10-26-cnn-equivariance-mobile-right.png`;
- `screenshots/2026-10-26-cnn-conventional-mobile-right.png`.

## Broader source findings and next cursor

The Harvard MLP section's sum-of-sines example does not establish exponential
width. On a compact box, approximate each univariate sine within `epsilon/d`
and sum the branches: the uniform errors sum to at most epsilon. These
branches are themselves representable in an MLP. Piecewise-linear
interpolation on a fixed interval even gives a polynomial construction;
the source example is not an exponential lower bound. The accepted MLP's
qualified universal-approximation explanation was retained instead.

Likewise, FLOPs divided only by weight storage do not establish actual HBM
traffic or a hardware bottleneck. The source's parameter-count comparisons
do not establish MNIST accuracy, sample-complexity or speed ratios. Its
hard-coded illustrative frontier includes Swin-T at the same compute value
and lower accuracy than ConvNeXt-T. None was adopted as a controlled
benchmark or universal result.

Next: continue the source at line 1991 (RNNs), then attention, Transformer,
recommendation and computational primitives; compare against accepted
chapters before adding material. H-V1-06/H-F1-06 and the whole-book goal
remain open. Accepted RL, FinOps, MIG and prior systems material must not
be requeued merely because a course row is partial. No stage, commit,
push, deployment, translation or live infrastructure mutation occurred.
