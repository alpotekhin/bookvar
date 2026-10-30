# Distributed training overview — editorial review, 2026-10-13

Scope: Russian chapter 44, its figure usage records, and
`publishing/tools/check_distributed_overview_examples.py`. The original and
the entire revised chapter were read; a second reviewer independently read
the complete revised chapter and test script. No English or deployment work.

## Substantive corrections

- A single training step introduces activations, gradients and optimizer
  state before comparing parallelism methods. The 7B memory example states
  decimal GB, dtypes, master-weight assumptions and excluded buffers.
- DDP replicas normally process different data and dropout masks. This is
  not a synchronization failure. Equality requires synchronized initial
  states, common reduced gradients and consistent optimizer updates.
- Token-weighted accumulation uses `(R/M) * local_loss_sum` before DDP's
  averaging, with no second division by the microbatch count. A two-rank
  example distinguishes the global token mean from a mean of local means.
- Small numerical examples distinguish all-gather, all-reduce and
  reduce-scatter, and distinguish column concatenation from row summation
  in tensor parallelism.
- Two weight all-gathers plus one gradient reduce-scatter cost
  `(R-1)/R * (2W+G)` outgoing bytes in the stated ring model. The shorter
  `3(R-1)W/R` expression requires equal weight and gradient byte sizes.
- GPipe idle fractions have explicit assumptions. Increasing microbatch
  count is not claimed to increase memory at fixed global batch size.
- The 64-GPU hybrid example distinguishes dense DP=4 from expert DP=2;
  EP is not multiplied into a DP count that already includes it. Its global
  batch uses the four independent dense-data streams, not TP or PP factors.
- AMP orders accumulation, unscale, clipping, optimizer step and scaler
  update. The fragment states equal-token and nonempty-microbatch assumptions
  and does not pretend to implement every FSDP/TP global-norm calculation.
- Network time below compute time does not establish overlap: all-gathers
  and reductions have dependencies. Memory estimates 66/34 GB are partial,
  conditional examples rather than measured peaks or universal savings.
- Checkpoint correctness means continuing with optimizer, RNG and data
  state, not merely loading weights. Strong and weak scaling are separated.

## Sources and figures

Checked the pinned CS336 lecture-07 distributed implementation, the local
original Harvard TP SVG, and current PyTorch 2.14 DDP/AMP/FSDP documentation.
Checked Megatron Bridge's group-size convention against the hybrid example.
Source URLs and precise figure references are adjacent to the explanations.

Seven existing source illustrations are used: PyTorch DDP Figure 4,
Hugging Face Ultra-Scale Playbook FSDP forward/backward and GPipe diagrams,
Harvard's TP split, and EDLS lecture-6 TP/EP diagrams. No illustration was
invented or redrawn. Registry `used_in` entries now include chapter 44.

The older Harvard pipeline illustration was not retained: its drawn schedule
has a previously documented dependency problem. The replacement is the
original HF GPipe schedule. Removed image files remain available to other
consumers; the detailed pipeline chapter still needs its own editorial pass.

## Executed numerical checks

`python publishing/tools/check_distributed_overview_examples.py` passes:

- weighted token losses and gradients, including microbatch splitting;
- collective examples and exact small-matrix TP equivalence;
- state memory, dtype-dependent traffic, conditional communication time;
- GPipe idle fractions and 64-GPU process-group arithmetic;
- AST and execution of the actual chapter's AMP fragment with mocks,
  checking every accumulation/unscale/clip/step event;
- all seven retained source-unit IDs, old explicit anchors and seven images.

These are CPU arithmetic and mocked sequencing checks. They do not execute
CUDA, NCCL, FSDP, a distributed process group, GPU memory measurement, or a
throughput benchmark. A separate acceptance record covers the final site
build, links and browser inspection after all current chapter edits settle.
