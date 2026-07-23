---
title: "LMCache MP mode — CUDA IPC, SHM, and pickle transfer paths"
type: source-note
status: active
locale: en
translation_of: "05 Источники/LMCache/LMCache MP mode — transfer paths.md"
last_verified: 2026-07-23
primary_sources:
  - https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/
  - https://docs.lmcache.ai/mp/architecture.html
---

# LMCache MP mode — CUDA IPC, SHM, and pickle transfer paths

Multiprocess mode separates the inference worker from the process that owns the
external cache. The separation removes fate sharing and lets several engines
use one host-memory pool, but it also removes the convenient assumption that a
pointer is meaningful on both sides. The transfer path is the concrete answer
to how a paged KV layout crosses that boundary.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/mp-mode-architecture.png]]

*LMCache MP architecture and storage paths. Original image from Jiayue Chen,
Tony Lin, and the LMCache team, [Understanding LMCache MP Mode Transfer
Paths](https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/).*

## Gathering comes before transport

vLLM does not keep one request in a contiguous allocation. Its block table
maps logical token ranges to physical pages that may be scattered across the
GPU pool. A store operation must gather selected pages into a contiguous
transfer buffer. Retrieval performs the inverse scatter into destination pages.
These copies exist independently of whether the process boundary is crossed by
CUDA IPC, shared memory, or a socket.

This gives every path two costs:

$$
T_{path}=T_{gather/scatter}+T_{process\ boundary}+T_{tier\ copy}.
$$

Counting only network or PCIe bandwidth misses the memory-layout work at each
end.

## CUDA IPC path

A device pointer is process-local. CUDA IPC lets the owner export a compact
handle that another process imports to obtain access to the same allocation.
The handle is metadata, not a serialization of the tensor. LMCache couples it
with shape, dtype, layout, and a CUDA event that establishes when it is safe to
read or overwrite the allocation.

For store, the cache process accesses the worker's paged GPU memory through the
imported handle, gathers blocks into a contiguous GPU staging buffer, and copies
that buffer into LMCache-managed host memory. Retrieve reverses the path. IPC
avoids shipping the entire tensor through the control socket, but it does not
make the gather or GPU-to-host copy disappear.

## Shared-memory path

On non-CUDA platforms, both processes can map the same host allocation from
`/dev/shm`. With the device-specific native operations, the worker gathers KV
directly into the shared L1 buffer. The blog counts this as one copy. A Python
fallback may add a staging copy to coalesce small chunks and obtain acceptable
throughput.

This path depends on deployment details that are easy to overlook in a
container: the requested L1 pool must fit in the mounted shared-memory volume.
If `/dev/shm` is left at a small default size, the intended zero-copy sharing
cannot be used.

## Pickle and socket fallback

When the cache pool lives in private server RAM, the worker cannot map it. The
portable fallback gathers into CPU chunks, serializes the chunks, transmits the
bytes through ZMQ, deserializes them, and writes them into L1. The original
article counts four data transformations/copies. It works without CUDA IPC or a
shared mapping, which makes it valuable for development and unsupported
accelerators, but it should not be mistaken for the high-throughput data path.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/mp-transfer-paths.png]]

*The three store paths and their copy boundaries. Original LMCache blog image;
the CUDA, SHM, and pickle columns should be read as physical data paths, not API
names alone.*

| Path | Cross-process mechanism | Main copies | Typical reason to choose it |
|---|---|---:|---|
| CUDA IPC | imported device allocation and synchronization event | gather plus device-to-host | CUDA GPU with a cache daemon on the same host |
| SHM | both processes map `/dev/shm` L1 | one with native ops; potentially two in fallback | non-CUDA device or CPU path with sufficient shared memory |
| Pickle | serialized bytes over ZMQ | roughly four | universal fallback and functional testing |

The article is particularly useful because it turns “KV offload” into a trace
of actual memory regions. Production reviews should use the same discipline:
name the source layout, staging buffers, ownership boundary, synchronization,
and final tier for every arrow.

## Original source

Jiayue Chen, Tony Lin, and LMCache Team,
[Understanding LMCache MP Mode Transfer Paths: A Beginner's Guide](https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/), 2026-06-15.
