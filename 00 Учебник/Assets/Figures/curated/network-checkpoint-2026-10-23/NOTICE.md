# Network and checkpoint source illustrations

Both PNGs are complete byte-for-byte source originals. No crop, redraw,
translation, recoloring or resampling was performed.

- jax-superpod-diagram.png: Jacob Austin, Swapnil Patil, Adam Paszke,
  Reiner Pope, How to Scale Your Model / How to Think About GPUs.
  Pin 3ae1dee9eef3430aa671162397afe5dd79f299ca. MIT; full license and
  source Markdown retained in Originals/network-checkpoint-2026-10-23.
  Public notice: licenses/jax-scaling-book-3ae1dee9-MIT.txt.
  The drawing is a conceptual node/NVSwitch/InfiniBand hierarchy, not a
  complete cable/NIC inventory. Full bisection assumes a properly sized
  nonblocking fabric, not every tree or every customer deployment.
- deepspeed-ucp-flowchart.png: Xinyu Lian et al., DeepSpeed Universal
  Checkpointing. Pin c2c81993948fc28385542196c8544fb442017987.
  Apache-2.0; full license, source README and tutorial retained in the
  same source folder. Public notice: licenses/deepspeed-c2c81993-Apache-2.0.txt.
  This shows DeepSpeed UCP conversion, not the PyTorch DCP implementation.
  Labels (including original wording) are unchanged. Atomic checkpoint in
  the article is a parameter-level representation, not whole-generation
  transactional durability. The denser Figure 1 was inspected but not
  selected as the embedded diagram; it remains available in the source.

The withdrawn Harvard five-level SVG and its historical provenance remain
on disk; it was not deleted or relicensed. The existing Open Hackathons
and PyTorch async diagrams keep their separate original notices.
