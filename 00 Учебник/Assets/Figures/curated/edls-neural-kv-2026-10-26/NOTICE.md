# Original HF KV-cache diagram

Rémi Ouazan Reboul, Arthur Zucker and Luc Georges / Hugging Face,
*Continuous batching from first principles*, KV-cache section.

The complete original `kv_cache.png` is copied without visual modification
from image-dataset commit `c8c3cf72b3b724ce734259e4a086d30618a9198b`.
Source page: https://huggingface.co/blog/continuous_batching#kv-cache
Image and license evidence are recorded in `assets.json` and the global registry.

License: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/.
The source image dataset's explicit license declaration is preserved at
`05 Источники/Originals/scheduling-2026-10-25/hf-images-README.md`.
No crop, redraw, label translation, resampling or recompression was applied.

The diagram suppresses attention-head axes. It explains old versus new token
states; the reference code uses an actual four-dimensional GQA cache.
