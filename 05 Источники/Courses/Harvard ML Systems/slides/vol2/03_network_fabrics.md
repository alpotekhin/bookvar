---
title: "VOL2 Slides — Network Fabrics"
type: external-resource
status: imported-source
language: en
source_kind: beamer-slides
source_commit: 45ecc8d82fcae70c149cdce550d3b3d3411df913
---

> [!note] Complete original Harvard source
> Source: [`slides/vol2/03_network_fabrics/03_network_fabrics.tex`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/slides/vol2/03_network_fabrics/03_network_fabrics.tex) at commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`.
> License: [CC BY-NC-SA 4.0](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE).
> The readable layer below is mechanically extracted from the original;
> the complete unmodified source follows on the same page.

# VOL2 Slides — Network Fabrics

## Readable slide sequence

## Learning Objectives

{
[1 min]
% -- LINK: Ch2 built the physical infrastructure; this chapter wires it together
Students know accelerators, nodes, and racks. Now: the fabric that turns
isolated hardware into a unified training system.

% -- NARRATE: What to SAY while showing this slide
``This chapter is about the physics of wires and switches, not the algorithms
that run on them. The network determines whether expensive GPUs compute or
sit idle. By the end, you will diagnose network bottlenecks quantitatively.''

% -- ENGAGE: Specific question for THIS slide
``How long does it take to transfer 350 GB of gradients at 50 GB/s?''
Quick mental math: 7 seconds. That is 7 seconds of 1,000 GPUs idling.

% -- FLEX: [CORE] Sets expectations for the lecture.
IF SHORT: Read objectives quickly, do not elaborate.
}
- Model network cost using the {{} framework} and identify bandwidth- vs.\ latency-dominated regimes
- Compare **RDMA** transport protocols (**InfiniBand** and **RoCE**) in terms of latency and lossless guarantees
- Analyze topologies (**fat-tree**, **rail-optimized**, **dragonfly**) by computing **bisection bandwidth**
- Evaluate congestion control (**PFC**, **DCQCN**, **HPCC**) and their impact on tail latency
- Design network virtualization strategies using **SR-IOV** and traffic isolation
- Diagnose network bottlenecks using RDMA counters and bandwidth testing tools

## Visual Language

{
[0.5 min]
% -- NARRATE: Same color system as previous lectures. Point briefly and move on.
% -- FLEX: [OPTIONAL] Skip if students have seen this twice already.
}

Throughout this course, colors carry meaning:

{0.3cm}
[T]
  
    {mlsyscard}{computestroke}
    **Blue** --- Compute / Processing\\
    { GPU ops, forward/backward pass, inference}
    {mlsyscard}
    {0.15cm}
    {mlsyscard}{datastroke}
    **Green** --- Data / Memory\\
    { Data flow, caches, healthy paths}
    {mlsyscard}
  
  
    {mlsyscard}{routingstroke}
    **Orange** --- Routing / Scheduling\\
    { Load balancers, batch windows}
    {mlsyscard}
    {0.15cm}
    {mlsyscard}{errorstroke}
    **Red** --- Error / Cost / Bottleneck\\
    { Loss, decode phase, waste}
    {mlsyscard}

## Why the Network Dominates at Scale

{
[2 min]
% -- LINK: Ch2 showed the 18x bandwidth cliff; this slide shows its operational consequence
Students know the cliff exists. Now: what happens when 1,000 GPUs hit it simultaneously.

% -- NARRATE: What to SAY while showing this slide
``A single H100 does 989 TFLOPS. But 1,000 GPUs must synchronize 350 GB of
gradients per step. If the network cannot keep pace, GPUs idle. The network
is not auxiliary infrastructure --- it determines whether a \$300M cluster
trains efficiently or wastes millions in idle compute.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``How long does 350 GB take at 50 GB/s?'' Expected: 7 seconds.
``That is 7 seconds of 1,000 GPUs doing nothing. Every step.''

% -- WARN: What students will get wrong
Students treat the network as plumbing that ``just works.'' Correct: a single
slow link idles 999 GPUs due to BSP barrier synchronization.

% -- FLEX: [CORE] This slide establishes the central thesis of the chapter.
}

[T]
  
    The network is the **synchronization backbone**:

    {0.15cm}
    {1pt}
- 175B model $$ 350 GB gradients per step
- 1,000 GPUs must all exchange before next step
- **One slow link** idles 999 GPUs
    

    {0.15cm}
    {mlsyscard}{crimson}
    The network is not auxiliary infrastructure. It determines whether a \$300M cluster
    trains efficiently or wastes millions in idle compute.
    {mlsyscard}
  
  
    {five-level-model.pdf}

## Frame 4

/

## The Law of Distributed Efficiency: Implications

{
[1.5 min]
% -- LINK: Why-the-network-dominates made the qualitative case; this slide gives the
% formal equation that quantifies how much of each training step is exposed to the fabric.
Students now know the network is the bottleneck. This slide gives them the
formal equation they will use all semester to reason about that bottleneck.

% -- NARRATE: What to SAY while showing this slide
``Every distributed training step has three components: compute, communication, and
whatever communication we managed to hide by overlapping it with compute. The equation
is T_step = T_compute + T_comm(N) - T_overlap. The term that actually hurts is the
gap: T_comm(N) - T_overlap. That gap is what the fabric determines. A better fabric
does not change T_compute; it shrinks the gap by enabling more overlap. This is why
network design is an efficiency problem, not just a speed problem.''

Point to the equation and say: ``N is the number of GPUs. Notice T_comm grows with N ---
as we add GPUs, we exchange more gradient data. The fabric's job is to keep T_overlap
close to T_comm so the gap stays small.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``If T_comm doubles because we scaled from 256 to 512 GPUs, what needs to
happen to T_overlap to keep scaling efficiency above 90 percent?''
Expected: T_overlap must also roughly double --- meaning the fabric must support
twice as much gradient communication hidden behind computation.

% -- WARN: What students will get wrong
Students assume more GPUs = proportionally more bandwidth needed. Correct: the key
metric is overlap, not raw bandwidth. A fabric with lower latency can start overlap
earlier, achieving the same efficiency at lower bandwidth cost.

% -- FLEX: [CORE] This equation is the formal anchor for the entire chapter.
IF SHORT: Write the equation, define T_overlap, state the punchline (fabric controls
the gap), move on.
}

{0.2cm}
[T]
  
    The **nonoverlapped communication** term determines scaling efficiency:
    {1pt}
- $T_{{compute}}$: fixed by model and hardware
- $T_{{comm}}(N)$: grows with GPU count $N$
- $T_{{overlap}}$: communication hidden behind compute
    
    {0.1cm}
    {mlsyscard}{crimson}
    The fabric determines $T_{{overlap}}$: lower latency enables earlier pipelining;
    higher bandwidth shrinks the gap directly.
    {mlsyscard}
  
  
    {mlsyscard}{computestroke}
    { **Engineering consequence**: A perfectly efficient fabric achieves
    $T_{{overlap}} = T_{{comm}}(N)$, reducing $T_{{step}}$ to
    $T_{{compute}}$ alone. Real fabrics fall short of this ideal --- the gap
    is what we measure and optimize.}
    {mlsyscard}

## The Bandwidth Cliff

{
[1.5 min]
% -- LINK: Previous slide showed the network matters; now the 18x cliff quantifies why
Students saw the synchronization cost. Now: the specific bandwidth numbers.

% -- NARRATE: What to SAY while showing this slide
Point to each column: ``NVLink: 900 GB/s within a node. InfiniBand: 50 GB/s
across nodes. 18x cliff. This ratio has persisted across four GPU generations.
It is physics --- electrical signaling over copper vs optical over distance ---
not engineering laziness.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``What changes when you cross a node boundary?''
Expected: bandwidth drops 18x, latency increases 5--10x.

% -- WARN: What students will get wrong
Students expect the 18x cliff to shrink with newer hardware. It has been
stable for four generations because the underlying physics has not changed.

% -- FLEX: [CORE] The 18x cliff is the central number of this chapter.
}

% --- Layout: FULL-WIDTH IMAGE + annotation ---

{bandwidth-hierarchy.pdf}

{0.1cm}

[T]
  
    
    {datastroke}{**NVLink**}\\
    { 900 GB/s (intra-node)}
  
  
    
    {errorstroke}{**18$×$ cliff**}\\
    { Node boundary}
  
  
    
    {routingstroke}{**InfiniBand NDR**}\\
    { 50 GB/s (inter-node)}

## The Bandwidth Cliff in Action

{
[1.5 min]
% -- LINK: Bandwidth cliff showed the ratio; this table shows the dollar impact
Students saw 18x. Now: what that costs in utilization and money.

% -- NARRATE: What to SAY while showing this slide
``8 GPUs on NVLink: AllReduce takes 1 ms, 99.5% utilization. 64 GPUs across
IB: AllReduce balloons to 30 ms, utilization drops to 87%. That 12-point gap,
compounded over months, costs millions of dollars.''

% -- WARN: What students will get wrong
Students think 87% utilization is good. Correct: on a \$300M cluster,
13% idle = \$39M wasted per year.

% -- FLEX: [CORE] Quantifies the economic cost of the bandwidth cliff.
IF SHORT: State the two numbers (99.5% vs 87%) and the dollar consequence.
}

{}{1.15}
{
{tabular}{@{}lrrr@{}}
  
  **Scenario** & **AllReduce Time** & **Utilization** & **Bottleneck** \\
  
  8 GPUs, NVLink (900 GB/s) & 1 ms & {datastroke}{**99.5%**} & Compute \\
  64 GPUs, IB NDR (50 GB/s) & 30 ms & {errorstroke}{**87%**} & Network \\
  
{tabular}
}

{0.2cm}
{mlsyscard}{errorstroke}
The 12-point utilization gap represents **millions of dollars** in wasted compute
over a months-long training run. Network fabric design is the central engineering
challenge of distributed training.
{mlsyscard}

## ML Inverts Datacenter Assumptions

{
[2 min]
% -- LINK: Bandwidth cliff showed the quantitative gap; workload inversion explains WHY
Standard datacenter networks were designed for web traffic. ML inverts every assumption.

% -- NARRATE: What to SAY while showing this slide
``Traditional datacenter: millions of small flows, asynchronous, loss-tolerant.
ML training: few massive flows, synchronous, loss-INtolerant. Every single
assumption is inverted. This is why standard enterprise Ethernet fails for ML.''
Point to BSP callout: ``All 1,024 GPUs must finish before ANY can proceed.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why can you not just run training over your existing datacenter network?''
Expected: existing networks are designed for mice flows with loss tolerance.

% -- WARN: What students will get wrong
Students with web-scale experience assume statistical multiplexing helps.
Correct: BSP training creates simultaneous elephant flows --- no multiplexing.

% -- FLEX: [CORE] Workload inversion is the conceptual foundation for RDMA and lossless.
}

% --- Layout: FULL-WIDTH IMAGE ---

{workload-inversion.pdf}

{0.1cm}
{BSP (Bulk Synchronous Parallel):}{All 1,024 GPUs must finish before *any* can proceed. The slowest link is the bottleneck.}

## Predict: What Happens to One Dropped Packet?

{
[2 min]
% -- LINK: Workload inversion said loss-intolerant; this prediction shows WHY
Students heard lossless is required. This makes them discover the consequence.

% -- NARRATE: What to SAY while showing this slide
``60 seconds. What happens when a single packet drops 900 MB into a 1 GB
RDMA transfer? Write your prediction.''
Most will guess ``just retransmit that one packet.'' The reveal: RDMA uses
Go-Back-N, so one drop retransmits the entire 100 MB tail.

% -- ENGAGE: This IS the active learning moment
Do NOT reveal yet. Let students sit with the discomfort of not knowing.
The surprise makes the lossless requirement visceral, not abstract.

% -- FLEX: [CORE] This prediction creates the emotional case for lossless networking.
}

{0.7cm}
{ Think--Write--Share}

{0.4cm}
{ In a 1,024-GPU synchronous training job using RDMA,\\
what happens when a *single packet* is dropped\\
900 MB into a 1 GB gradient transfer?}

{0.4cm}
{ Write your prediction. {midgray}{(60 seconds)}}

{0.3cm}
{{lightgray}{Hint: RDMA does NOT use TCP's Selective Acknowledgement.}}

## Level 1: Wire and Link Physics

{
[2 min]
% -- LINK: Workload inversion showed what ML needs; this slide shows the wire physics
Students know the requirements. Now: what the physical medium can deliver.

% -- NARRATE: What to SAY while showing this slide
``PAM4 doubles data rate by encoding 2 bits per symbol, but voltage margins
shrink 3x, requiring FEC at every hop. FEC adds 100--200 ns per hop ---
irreducible physics tax. A 3-hop fat-tree: 600--1,200 ns of pure latency floor.''
Point to the medium table: ``Copper under 3 meters intra-rack. Fiber for
inter-rack. Each has different cost, power, and reach trade-offs.''
Mention: ``SerDes power: 3,000 links at 25W each = 75 kW --- over 10% of
cluster power just to move bits.''

% -- WARN: What students will get wrong
Students assume latency is dominated by distance (speed of light). Correct:
FEC processing dominates at datacenter distances. 100 meters at light speed
is 0.5 us; FEC adds 0.6--1.2 us.

% -- FLEX: [OPTIONAL] Wire physics is important context but not directly examinable.
IF SHORT: State the FEC latency floor and SerDes power, skip medium details.
}

[T]
  
    **PAM4 Signaling** (4 voltage levels, 2 bits/symbol):
    {0pt}
- Doubles data rate without higher symbol rate
- Voltage margin shrinks 3$×$ $$ **FEC required**
- FEC adds 100--200 ns per hop (irreducible)
    

    {0.1cm}
    **Medium trade-offs:**
    {0pt}
- {computestroke}{Copper (DAC)}: $<$3 m, intra-rack
- {routingstroke}{Active Optical}: 100+ m, inter-rack
    

    {0.1cm}
    {Physics tax:}{3-hop fat-tree $$ 600--1,200 ns FEC floor.}
  
  
    {}{1.05}
    {
    {tabular}{@{}lll@{}}
      
      **Medium** & **Reach** & **Use** \\
      
      DAC (copper) & $<$3 m & In-rack \\
      AOC (fiber) & 30 m & Inter-rack \\
      SMF & 10+ km & Building \\
      
    {tabular}
    }

    {0.15cm}
    {
    {tabular}{@{}lr@{}}
      
      **Component** & **Power** \\
      
      SerDes + xcvr (400G) & $$25 W \\
      3,000 links total & 75 kW \\
      
    {tabular}
    }
    {{midgray}{$>$10% of cluster power}}

## Level 2: RDMA and GPUDirect

{
[2 min]
% -- LINK: Wire physics showed the physical medium; RDMA shows the transport layer
Students saw the physical constraints. Now: how RDMA bypasses the kernel
to achieve 1--2 us latency vs TCP's 50--100 us.

% -- NARRATE: What to SAY while showing this slide
Point to the diagram: ``Traditional TCP: 3 memory copies, kernel involvement,
50--100 us. RDMA: zero copies, NIC reads GPU memory directly via PCIe DMA,
1--2 us. For 350 GB of gradients, RDMA eliminates 700 GB of redundant copies
per step.'' Emphasize: ``The NIC talks directly to GPU memory. The CPU is
not involved at all.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why cannot we just use faster TCP?'' Expected: TCP's kernel involvement
and memory copies are architectural, not just slow software.

% -- WARN: What students will get wrong
Students think RDMA is just ``optimized TCP.'' Correct: RDMA fundamentally
changes the data path --- it bypasses the kernel entirely.

% -- FLEX: [CORE] RDMA is the foundational transport for all distributed training.
}

% --- Layout: FULL-WIDTH IMAGE ---

{rdma-data-path.pdf}

{0.1cm}
{Key:}{RDMA bypasses the kernel and CPU entirely --- NIC reads GPU memory directly via PCIe DMA.}

## InfiniBand vs.\ RoCE

{
[2 min]
% -- LINK: RDMA showed the transport; IB vs RoCE shows two implementations
Students know RDMA bypasses the kernel. Now: two ways to deliver it.

% -- NARRATE: What to SAY while showing this slide
``Both IB and RoCE expose the same Verbs API. The difference is reliability.
IB: hardware credit-based flow control, natively lossless. RoCE: relies on
Ethernet PFC to approximate losslessness --- fragile under scale.''
Point to PFC storm risk row: ``A misconfigured RoCE switch can freeze the
entire fabric in 200 ms. IB is immune.''

% -- WARN: What students will get wrong
Common error: ``RoCE is just IB over Ethernet.'' No --- the lossless
guarantees are weaker. PFC approximates losslessness but can fail catastrophically.

% -- FLEX: [CORE] The IB vs RoCE choice is a real production decision.
IF SHORT: Focus on the lossless row and PFC storm risk row only.
}

{}{1.15}
{tabular}{@{}lll@{}}
  
  **Property** & **InfiniBand** & **RoCE (Ethernet)** \\
  
  Lossless & {datastroke}{**Native**} (credit-based) & Approximated (PFC) \\
  Latency & 1--2 $$s & 3--5 $$s \\
  Congestion & Hardware adaptive routing & DCQCN / HPCC \\
  Switch cost & Higher (single vendor) & {datastroke}{**20--40% lower**} \\
  Vendor lock-in & {errorstroke}{Single} (NVIDIA/Mellanox) & {datastroke}{**Multi-vendor**} \\
  PFC storm risk & {datastroke}{**Immune**} & {errorstroke}{Vulnerable} \\
  
{tabular}

{0.15cm}
{mlsyscard}{routingstroke}
Both deliver 95%+ line-rate for large transfers. IB wins on tail latency (30--50% lower).
RoCE wins on operational flexibility and cost.
{mlsyscard}

## PFC Deadlock: When Lossless Goes Wrong

{
[2 min]
% -- LINK: IB vs RoCE mentioned PFC storm risk; this slide unpacks the failure mode

% -- NARRATE: What to SAY while showing this slide
``PFC (Priority Flow Control) approximates losslessness on Ethernet by sending
PAUSE frames when buffers fill. The problem: PAUSE propagates backwards, hop by hop.
A single congested port can freeze an entire subnet in under 200 ms. This is called
a PFC storm or head-of-line blocking cascade.''
Walk through the scenario: ``Switch A congests. Sends PAUSE to Switch B. Switch B
fills its buffer, sends PAUSE to Switch C. Within 200 ms, the entire spine is frozen.
All traffic --- not just the congested flow --- is blocked.''

% -- ENGAGE: ``Why does InfiniBand not have this problem?''
Expected: IB uses credit-based flow control --- each link grants credits independently.
No PAUSE propagation. Congestion stays local.

% -- WARN: Students assume PFC problems are rare. At Meta and Microsoft scale,
PFC storms were a leading cause of training job failures before mitigations.

% -- FLEX: [OPTIONAL] Important for RoCE environments but not universal.
IF SHORT: State the 200 ms cascade number and the IB immunity, skip the diagram.
}

[T]
  
    **PFC Storm Cascade:**

    {0.1cm}
    {1pt}
- { Port congests $$ buffer fills}
- { PAUSE frame sent upstream}
- { Upstream buffer fills $$ PAUSE propagates}
- { **Entire subnet frozen in $<$200 ms**}
    

    {0.1cm}
    {mlsyscard}{errorstroke}
    { **Head-of-line blocking**: ALL traffic through the congested switch is stopped, not just the offending flow. One bad actor freezes the fabric.}
    {mlsyscard}
  
  
    **Mitigation strategies:**
    {1pt}
- { {datastroke}{**Watchdog**}: Detect PAUSE duration $>$ threshold, disable PFC on that port}
- { {routingstroke}{**DCQCN**}: ECN marking before buffers fill}
- { {computestroke}{**Separate VLANs**}: Isolate training traffic from storage traffic}
    

    {0.1cm}
    { InfiniBand: **immune** (credit-based, no PAUSE propagation).}

## Frame 14

{{} Model: Two Regimes}
{
[2 min]
% -- LINK: IB vs RoCE compared transports; alpha-beta formalizes cost modeling
Students know the transport options. Now: the quantitative model for communication cost.

% -- NARRATE: What to SAY while showing this slide
``alpha is startup latency: switch hops, FEC, software overhead. beta is sustained
bandwidth. The crossover $n^* =  × $ determines which term dominates.
For IB NDR ($=1.5\,$s, $=50$ GB/s): $n^*=1.5×10^{-6}×50×10^{9}=75{,}000$ bytes $$ 75 KB.
Below 75 KB: latency-bound. Above: bandwidth-bound.
Topology choice shifts both parameters.''

% -- ENGAGE: Specific question for THIS slide
``Is a 350 GB AllReduce latency-bound or bandwidth-bound?''
Expected: bandwidth-bound --- 350 GB is millions of times above $n^*$.

% -- WARN: What students will get wrong
Students assume all network transfers are bandwidth-limited. Correct: control
messages (heartbeats, barriers) are tiny and latency-dominated.

% -- FLEX: [CORE] The alpha-beta model is the diagnostic tool for network cost.
IF AHEAD: ``What is the alpha-beta crossover for a 3-hop dragonfly vs 6-hop fat-tree?''
}

% --- Layout: FULL-WIDTH IMAGE ---

{alpha-beta-model.pdf}

{0.1cm}

{Diagnostic:}{Is your workload above or below $n^*$? That determines whether you optimize for hops (topology) or link speed (bandwidth).}

## Frame 15

{{} Model: Worked Example}
{
[2 min]
% -- LINK: Alpha-beta model introduced the two regimes; this worked example demonstrates
Students saw the model. Now: a concrete calculation that surprises.

% -- NARRATE: What to SAY while showing this slide
``10 KB control message. IB HDR: 1.91 us. IB NDR (2x bandwidth): 1.70 us.
Despite doubling bandwidth, only 11% improvement. Why? Because alpha dominates.
You added 40% to transceiver cost for negligible benefit.''
This is the quantitative proof that more bandwidth is not always the answer.

% -- WARN: What students will get wrong
Students assume faster links always help. This example proves that for small
messages, reducing hop count (topology) matters more than link speed.

% -- FLEX: [CORE] This worked example grounds the alpha-beta model concretely.
IF SHORT: State the 11% number and the punchline, skip the calculation walkthrough.
}

**Small message: 10 KB control synchronization**

{0.15cm}
{}{1.2}
{
{tabular}{@{}llll@{}}
  
  **Link** & **$$ ($$s)** & **$n/$ ($$s)** & **$T(n)$ ($$s)** \\
  
  IB HDR (200G) & 1.5 & 0.41 & **1.91** \\
  IB NDR (400G) & 1.5 & 0.20 & **1.70** \\
  
  {3}{r}{**Speedup:**} & {errorstroke}{**only 11%**} \\
  
{tabular}
}

{0.2cm}
{mlsyscard}{errorstroke}
**Diagnosis:** Doubling bandwidth yields only 11% improvement because
$$ dominates for small messages. Upgrading to NDR may add 40% to transceiver cost
for negligible benefit in the latency-dominated regime.
{mlsyscard}

## Quick Check

{
[1 min]
% -- LINK: Alpha-beta worked example showed the 10 KB case; this micro-retrieval
asks students to apply the same crossover logic independently.
% -- NARRATE: Quick retrieval. ``50 KB message, alpha=1.5 us, beta=50 GB/s.
$n^* = 1.5 × 10^{-6} × 50 × 10^9 = 75$ KB. Since 50 KB $<$ 75 KB: latency-bound.''
% -- ENGAGE: Cold-call after 15 seconds: ``Is the 50 KB message latency- or bandwidth-bound?
What is the crossover?'' Expected: n*=75 KB; 50 KB < 75 KB, so latency-bound.
% -- WARN: Students may forget to compute n* first and just compare 50 KB to 50 GB/s.
They need to compute n* = alpha*beta in bytes first.
% -- FLEX: [CORE] Reinforces the alpha-beta crossover calculation.
}

{1.0cm}
{ Quick Check}

{0.6cm}
{ A message is 50 KB on IB NDR ($=1.5\,$s, $=50$ GB/s).}\\
{ Latency-bound or bandwidth-bound?}

{0.5cm}
{{midgray}{15 seconds --- then I will cold-call.}}

## Your Turn: Bandwidth- or Latency-Dominated?

{
[3 min]
% -- LINK: The 10 KB example showed latency-dominated; this exercise shows the opposite
Students saw alpha dominate for small messages. Now: bandwidth dominates for AllReduce.

% -- NARRATE: What to SAY while showing this slide
``90 seconds. 350 GB AllReduce on IB NDR. Calculate T(n) and identify the regime.''
After pause: ``$n/$ = 350/50 = 7.0 seconds. $$ = 0.0000015 seconds.
Clearly bandwidth-dominated. Doubling bandwidth gives nearly 2x speedup ---
the opposite of the 10 KB case.''

% -- ENGAGE: This IS the active learning moment
The contrast with the 10 KB example is the teaching point: same model,
opposite conclusions. The crossover $n^*$ determines which lever matters.

% -- WARN: What students will get wrong
Students may add alpha and n/beta incorrectly or confuse units (us vs seconds).
Remind: convert everything to the same unit before adding.

% -- FLEX: [CORE] This exercise and the 10 KB example together prove the alpha-beta model.
}

[T]
  
    { Classify the Regime}

    {0.2cm}
    A 175B model AllReduce transfers 350 GB across IB NDR:
- $$ = 1.5 $$s, $$ = 50 GB/s
- $n$ = 350 GB
    

    {0.1cm}
    **1.** Calculate $T(n)$ and identify the dominant term.\\
    **2.** Would doubling bandwidth help here?

    {{midgray}{(90 seconds --- then compare)}}
  
  
    
    {mlsyscard}{datastroke}
    **Solution:**\\[0.1cm]
    {
    $n/$ = 350 / 50 = **7.0 s**\\
    $$ = 0.0000015 s\\[0.1cm]
    {computestroke}{**Bandwidth-dominated!**}\\[0.1cm]
    Yes --- 2$×$ BW $$ $$2$×$ speedup.\\
    (Opposite of the 10 KB case.)
    }
    {mlsyscard}

## Fat-Tree (Clos): The Industry Standard

{
[2 min]
% -- LINK: Alpha-beta quantified transfer cost; topology determines the alpha and beta values
Students can model cost per transfer. Now: how topology shapes those parameters.

% -- NARRATE: What to SAY while showing this slide
``Fat-tree: hierarchical switch tiers providing full bisection bandwidth.
$N = k^3/4$ hosts. Radix 64 gives 65,536 GPUs. Non-blocking means every
AllReduce gets full line-rate. Cost: O(N log N) switches, \$20--100M for 4,096 GPUs.''

% -- ENGAGE: Specific question for THIS slide
``Why is full bisection bandwidth critical for AllReduce?''
Expected: BSP means all GPUs inject simultaneously --- no statistical multiplexing.

% -- WARN: What students will get wrong
Students assume any tree topology provides full bisection BW. Correct: only
non-blocking (1:1 subscription) fat-trees do. Oversubscribed trees lose bandwidth.

% -- FLEX: [CORE] Fat-tree is the baseline topology for all comparisons.
}

% --- Layout: FULL-WIDTH IMAGE ---

{fat-tree-topology.pdf}

{0.1cm}

{Key formula:}{$N = k^3/4$ hosts. Radix $k$=64 $$ 65,536 GPUs with full bisection bandwidth.}

## Rail-Optimized Topology

{
[2 min]
% -- LINK: Fat-tree is the general-purpose standard; rail-optimized is the
specialized alternative designed specifically for tensor parallelism.

% -- NARRATE: What to SAY while showing this slide
``NVIDIA SuperPOD uses a rail-optimized design. Each GPU in a node connects to a
different leaf switch --- 8 GPUs, 8 rails. Tensor parallelism happens within the
node via NVLink (900 GB/s). Data parallelism goes across rails via IB (50 GB/s per
rail). Key advantage: TP traffic never touches the fabric. Key constraint: jobs
MUST be topology-aware --- a job placed across two pods pays a 2--4x penalty.''

% -- ENGAGE: ``Why does each GPU connect to a DIFFERENT leaf switch?''
Expected: to distribute DP AllReduce traffic across 8 independent paths.

% -- WARN: Students assume any topology works for TP. Correct: rail-optimized
REQUIRES that TP stays intra-node. Violating this constraint collapses performance.

% -- FLEX: [CORE] Rail-optimized is the dominant production topology for LLM training.
IF SHORT: State the design principle and the topology-aware constraint.
}

[T]
  
    **Rail-Optimized Design (SuperPOD):**

    {0.1cm}
    {1pt}
- { 8 GPUs per node $$ 8 independent ``rails''}
- { Each GPU connects to a *different* leaf switch}
- { TP: intra-node via NVLink (900 GB/s)}
- { DP: across rails via IB (50 GB/s $×$ 8 = 400 GB/s aggregate)}
    

    {0.1cm}
    {mlsyscard}{datastroke}
    { **Advantage**: TP traffic never touches the fabric. 8 independent DP paths maximize aggregate bandwidth.}
    {mlsyscard}
  
  
    {mlsyscard}{errorstroke}
    { **Constraint**: Jobs must be *topology-aware*. Cross-pod placement: 2--4$×$ slower.\\[0.1cm]
    Requires: topology-aware scheduler, contiguous GPU allocation, NVLink for TP.}
    {mlsyscard}

    {0.1cm}
    { Used by: NVIDIA DGX SuperPOD, Meta Grand Teton (variant).}

## Lighthouse: Archetype A --- Two Traffic Classes (1/2)

{
[1.5 min]
% -- LINK: Rail-Optimized Topology showed the mechanism; this slide explains WHY
% that topology exists by naming the workload archetype that drove it.
Students now know how rail-optimized topology works. This slide names the model
class that made it the dominant production design.

% -- NARRATE: What to SAY while showing this slide
``GPT-4 and Llama-3 scale training uses 3D Parallelism: data parallelism across
nodes, tensor parallelism within nodes, pipeline parallelism across pipeline stages.
This creates two distinct traffic patterns: (1) Data-parallel AllReduce --- bandwidth-
hungry, 350 GB+ of gradients per step, tolerates tens of milliseconds. (2) Tensor-
parallel activation exchange --- latency-sensitive, happens on every forward pass
microbatch, must complete in microseconds.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why is TP latency-sensitive but DP is not?''
Expected: TP activations are on the forward-pass critical path --- every microbatch
waits for TP completion before continuing. DP AllReduce happens after the backward
pass and can often be overlapped with the next microbatch. Different criticality,
different fabric requirements.

% -- WARN: What students will get wrong
Students assume all communication in distributed training has the same requirements.
Correct: 3D parallelism creates heterogeneous traffic classes.

% -- FLEX: [CORE] This is the chapter's motivating real-world example.
IF SHORT: Name the two traffic classes, state that TP stays on NVLink and DP
uses one hop per rail.
}

**Archetype A (GPT-4/Llama-3): 3D Parallelism generates two traffic classes:**

{0.2cm}
[T]
  
    {mlsyscard}{computestroke}
    { **Traffic Class 1 --- DP AllReduce**\\
    Bandwidth-hungry gradient averaging across nodes.\\
    350 GB+ per step; tolerates tens of ms.\\
    {datastroke}{Rail design: 1 hop, 50 GB/s $×$ 8 rails.}}
    {mlsyscard}
  
  
    {mlsyscard}{routingstroke}
    { **Traffic Class 2 --- TP Activation Exchange**\\
    Latency-sensitive; on the forward-pass critical path.\\
    Microsecond deadline; 900 GB/s required.\\
    {datastroke}{Rail design: stays on NVLink, never hits the fabric.}}
    {mlsyscard}

## Lighthouse: Archetype A --- Why Rails Exist (2/2)

{
[1 min]
% -- LINK: Previous slide showed the two traffic classes; this slide explains why
% the rail topology is the direct consequence of those classes.
Students saw the two traffic classes. Now: why rails are the logical consequence.

% -- NARRATE: What to SAY while showing this slide
``The rail design maps cleanly: TP activations stay intra-node on NVLink (900 GB/s),
never touching the fabric at all. DP AllReduce crosses a single hop per rail --- exactly
what the single-hop rail design provides. The topology is not arbitrary; it was
reverse-engineered from the communication pattern of this specific model archetype.''

% -- ENGAGE: ``What topology would Archetype B (batch inference) drive instead?''
Expected: different fabric priorities --- lower latency, more independent paths.

% -- WARN: A fabric optimized for DP AllReduce (bandwidth-optimized) may be wrong
for TP activations (latency-optimized), which is exactly why the rail architecture
keeps them on separate networks.

% -- FLEX: [CORE] The key insight: topology is workload-driven.
IF SHORT: State the design principle and move on.
}

[T]
  
    {mlsyscard}{crimson}
    { **Why rails exist:** The two-class traffic pattern of
    Archetype A (GPT-4/Llama-3) required a topology that could simultaneously
    saturate inter-node bandwidth *and* keep TP on a zero-hop NVLink path.
    The rail-optimized design is not a general-purpose improvement ---
    it is the topology shaped by this specific workload.}
    {mlsyscard}
  
  
    {Design principle:}{Topology is workload-driven. Archetype A's 3D parallelism drove the rail topology; a different parallelism strategy would drive a different fabric.}

## Topology Trade-offs

{
[2 min]
% -- LINK: Fat-tree showed the standard; this compares three topology choices
Students know fat-tree. Now: rail-optimized and dragonfly alternatives.

% -- NARRATE: What to SAY while showing this slide
``Fat-tree: flexible, expensive, full bisection BW. Rail-optimized: 1-hop tensor
parallelism but requires topology-aware job scheduling. Dragonfly: 50% less cabling,
but 2--4x slowdown for jobs that cross groups. No single winner.''

% -- ENGAGE: Specific question for THIS slide
``If your entire training job fits in one dragonfly group, which topology wins?''
Expected: dragonfly --- within-group performance matches fat-tree at lower cost.

% -- WARN: What students will get wrong
Students pick one topology as universally best. Correct: the choice depends on
workload mix (single large job vs many small jobs) and scale.

% -- FLEX: [CORE] Topology trade-offs are a real production design decision.
IF SHORT: State the three options and the ``no free lunch'' conclusion.
}

% --- Layout: FULL-WIDTH IMAGE ---

{topology-comparison.pdf}

{0.1cm}
{No free lunch:}{Topology choice is workload-dependent. Fat-trees provide flexibility; dragonflies save cost; rails minimize latency.}

## Bisection Bandwidth: Why It Matters

{
[2 min]
% -- LINK: Topology comparison named bisection BW; this slide quantifies its impact
Students heard ``full bisection bandwidth.'' Now: what happens without it.

% -- NARRATE: What to SAY while showing this slide
``Bisection bandwidth: minimum BW across any equal partition. At 1:1 (non-blocking):
25.6 TB/s, AllReduce in 13.7 seconds. At 4:1 (oversubscribed): 6.4 TB/s,
AllReduce in 54.7 seconds --- 4x slower.''
Point to the cost callout: ``For a \$300M cluster where sync is 30% of time,
4:1 oversubscription wastes over \$142M in idle GPU-hours.''

% -- WARN: What students will get wrong
Students assume oversubscription is fine because web traffic handles it.
Correct: BSP training has no statistical multiplexing --- oversubscription
directly multiplies sync time.

% -- FLEX: [CORE] Bisection bandwidth is the quantitative test for topology adequacy.
IF SHORT: State the 4x slowdown and \$142M waste number.
}

[T]
  
    **Bisection bandwidth**: minimum BW across any equal partition.

    {0.15cm}
    {}{1.15}
    {
    {tabular}{@{}lrr@{}}
      
      **Subscription** & **Bisection BW** & **AllReduce** \\
      
      1:1 (non-blocking) & 25.6 TB/s & 13.7 s \\
      4:1 (oversubscribed) & 6.4 TB/s & 54.7 s \\
      
    {tabular}
    }

    {0.15cm}
    {Cost of 4:1:}{4$×$ slower AllReduce $$ \$142M+ wasted GPU-hours on a \$300M cluster.}
  
  
    {mlsyscard}{crimson}
    { BSP training means *all* GPUs inject simultaneously. Unlike web
    traffic, there is no statistical multiplexing --- the oversubscription ratio directly
    multiplies sync time.}
    {mlsyscard}

## Congestion Control: PFC $$ DCQCN $$ HPCC

{
[2 min]
% -- LINK: Topology determined the paths; congestion control manages traffic on those paths
Students know the topology. Now: what happens when traffic exceeds link capacity.

% -- NARRATE: What to SAY while showing this slide
``Three levels, increasing in precision. PFC: reactive PAUSE frames, can cascade
and freeze the fabric. DCQCN: proactive ECN marking, 85--90% utilization, but
blind to severity. HPCC: in-network telemetry, 95%+ utilization, requires
programmable switches.''

% -- ENGAGE: Specific question for THIS slide
``Why is 85% utilization not good enough for training?''
Expected: on a \$300M cluster, 15% idle = \$45M wasted. Every percentage point matters.

% -- WARN: What students will get wrong
Students assume PFC is sufficient because it prevents loss. Correct: PFC storms
can cascade and freeze the entire fabric --- worse than dropping packets.

% -- FLEX: [CORE] Congestion control directly impacts training efficiency.
IF SHORT: Name the three levels, state the utilization numbers, skip details.
}

% --- Layout: FULL-WIDTH IMAGE ---

{congestion-control.pdf}

{0.1cm}
{Progression:}{More precise feedback $$ higher link utilization $$ less wasted GPU time.}

## The Incast Problem

{
[2 min]
% -- LINK: Congestion control handled general congestion; incast is the specific ML pattern
Students saw general congestion mechanisms. Incast is the dominant ML-specific
pattern: many-to-one at every AllReduce completion.

% -- NARRATE: What to SAY while showing this slide
``Incast: hundreds of senders target the same receiver port simultaneously.
This is not rare --- it happens hundreds of times per training step. Buffer
overflow even when the fabric is uncongested, because the destination port
is the bottleneck, not the path.''
Walk through mitigations: ``Layer-staggered AllReduce, tree-based collectives,
deeper switch buffers. Adaptive routing helps but cannot solve incast ---
the destination is the constraint.''

% -- WARN: What students will get wrong
Students think adaptive routing solves incast. Correct: routing distributes
paths but cannot create bandwidth at the destination port.

% -- FLEX: [OPTIONAL] Incast is important for practitioners but less examinable.
IF SHORT: State the pattern (many-to-one), the frequency (hundreds per step),
and the key insight (destination bottleneck, not path bottleneck).
}

[T]
  
    **Incast**: many-to-one traffic pattern.

    {0.1cm}
    {0pt}
- AllReduce: hundreds of senders $$ one receiver
- Buffer overflow even when fabric is uncongested
- **Deterministic**: happens every training step
    

    {0.1cm}
    **Mitigation:**
    {0pt}
- Layer-staggered AllReduce (overlap)
- Tree-based collective algorithms
- Deeper switch buffers (32--64 MB)
    
  
  
    {mlsyscard}{errorstroke}
    { In a 1,024-GPU AllReduce, incast occurs **hundreds of times per step**.
    A congested port stalls *all* GPUs for 100+ $$s per event.}
    {mlsyscard}

    {0.1cm}
    {mlsyscard}{datastroke}
    { Adaptive routing *helps* but cannot *solve* incast --- the destination
    port is the bottleneck, not the path.}
    {mlsyscard}

## Discussion: InfiniBand or Ethernet?

{
[3 min]
% -- LINK: IB vs RoCE table provided specs; now students make the actual decision
Students have the data. This forces them to weigh trade-offs.

% -- NARRATE: ``Turn to your neighbor. 10,000 GPUs, \$200M budget. IB or RoCE? 90 seconds.''
Cold-call 2--3 pairs. No single right answer. IB: better tail latency, simpler
lossless. Ethernet: 20--40% cheaper switches, multi-vendor, shared with serving.

% -- ENGAGE: This IS the active learning moment
Push for reasoning, not just a label. ``What was the deciding factor for your choice?''
Emphasize: the industry is converging (Spectrum-4 adds IB features to Ethernet).

% -- FLEX: [CORE] The IB/Ethernet debate is the chapter's signature discussion.
IF SHORT: Pose to whole class, take 2 answers, state the convergence trend.
}

{0.6cm}
{ Turn and Talk {midgray}{(90 seconds)}}

{0.5cm}
{ You are building a 10,000-GPU training cluster.\\
Your budget is \$200M.\\[0.3cm]
**Do you choose InfiniBand or RoCE (Ethernet)?**}

{0.5cm}
[c]
  
    {computestroke}{**InfiniBand**}\\
    { 30--50% lower tail latency\ lossless\ vendor}
  
  
    {routingstroke}{**RoCE / Ethernet**}\\
    { 20--40% lower switch cost\-vendor\ infra with serving}

## Production Clusters: SuperPOD vs.\ Grand Teton

{
[2 min]
% -- LINK: IB vs RoCE discussion was theoretical; these are real production implementations
Students debated the choice. Now: how NVIDIA and Meta actually built their clusters.

% -- NARRATE: What to SAY while showing this slide
``Two philosophies. SuperPOD: IB-native, 256 GPUs per unit, rail-optimized.
Grand Teton: RoCE on Ethernet, 16,000+ GPUs, aggressive DCQCN tuning.
Both achieve 95%+ line-rate for large transfers. The choice is operational
philosophy, not performance superiority.''

% -- ENGAGE: ``If you were building a new 10,000-GPU training cluster from scratch
with a clean-slate budget and no existing infrastructure, which implementation
philosophy would you choose and why?'' Let 2--3 students answer before moving
to the Discussion frame that follows.

% -- WARN: What students will get wrong
Students assume one approach dominates. Correct: both work at scale.
The real variable is organizational expertise and existing infrastructure.

% -- FLEX: [OPTIONAL] Production examples are illustrative but not examinable.
IF SHORT: Name the two approaches, state that both achieve 95%+ line-rate, move on.
}

% --- Layout: FULL-WIDTH IMAGE ---

{superpod-vs-teton.pdf}

{0.1cm}
{Trend:}{NVIDIA Spectrum-4 adds IB-style adaptive routing to Ethernet. The two ecosystems are converging.}

## Virtualization: SR-IOV and Multi-Tenancy

{
[2 min]
% -- LINK: Production clusters showed single-tenant use; SR-IOV enables multi-tenancy
Students saw dedicated clusters. But 30% idle = \$90M wasted. Virtualization shares the fabric.

% -- NARRATE: What to SAY while showing this slide
``SR-IOV lets a physical NIC present as multiple virtual functions, each with
hardware-isolated DMA queues. Less than 2% latency overhead vs bare metal.
But bandwidth is strictly partitioned: 8 VFs on a 400G NIC = 50G each.
No bursting beyond your slice. Three isolation dimensions: bandwidth guarantees,
latency determinism, and security (memory snooping prevention).''

% -- WARN: What students will get wrong
Students assume virtualization means sharing = interference. Correct: SR-IOV
provides hardware isolation --- separate DMA queues, not software multiplexing.

% -- FLEX: [OPTIONAL] Important for cloud/multi-tenant environments but less critical for
single-organization clusters.
IF SHORT: State the 2% overhead number and the bandwidth partitioning constraint.
}

**SR-IOV**: NIC presents multiple Virtual Functions (VFs)

{0.1cm}
[T]
  
    {0pt}
- Hardware-isolated DMA queues per VM/container
- Bypasses hypervisor $$ $<$2% latency overhead
- Strict BW partitioning: 8 VFs on 400G = 50G each
    

    {0.1cm}
    **Three isolation dimensions:**
    {0pt}
- Bandwidth guarantees (min throughput)
- Latency determinism (no HoL blocking)
- Security (memory snooping prevention)
    
  
  
    {mlsyscard}{routingstroke}
    { A \$300M cluster at 30% idle = **\$90M waste**. Virtualization shares
    fabric among training, inference, and preprocessing.}
    {mlsyscard}

## Monitoring and Debugging the Fabric

{
[2 min]
% -- LINK: Fabric behavior showed what can go wrong; monitoring detects it before it is too late
Students know congestion and incast. Now: how to detect degradation in production.

% -- NARRATE: What to SAY while showing this slide
``Network issues are SILENT. They manifest as subtle training slowdowns, not errors.
A 10% throughput drop can waste thousands of GPU-hours before anyone notices.''
Walk through three layers: ``Link-level: BER predicts transceiver failure 24--48 hrs
ahead. Transport: PFC counters are the canary. Bandwidth: baseline with ib_write_bw,
NEVER iperf (iperf measures TCP, not RDMA).''

% -- WARN: What students will get wrong
Students will use iperf to test RDMA networks. A link showing 30 Gbps via
iperf may deliver 390 Gbps via ib_write_bw. Always use RDMA-native tools.

% -- FLEX: [CORE] Monitoring prevents millions in wasted compute.
IF SHORT: State the ``silent degradation'' problem and the iperf warning.
}

[T]
  
    **Three telemetry layers:**

    {0.1cm}
    {computestroke}{**1. Link-level:**}
    { Bit Error Rate, FEC corrections, symbol errors.
    A rising BER predicts transceiver failure 24--48 hrs ahead.}

    {0.1cm}
    {routingstroke}{**2. Transport-level:**}
    { PFC PAUSE frame rates, PortXmitDiscards, RDMA retransmits.
    Rising PFC counters = ``canary in the coal mine.''}

    {0.1cm}
    {datastroke}{**3. Bandwidth validation:**}
    { Use {ib_write_bw} / {ib_read_lat} from perftest.
    **Never** use {iperf} (measures TCP, not RDMA).}
  
  
    {mlsyscard}{errorstroke}
    { Network issues are *silent*. A 10% throughput drop can waste
    **thousands of GPU-hours** before detection. Alert on PFC counters and
    track bandwidth baselines weekly.}
    {mlsyscard}

    {0.1cm}
    {**Debug workflow:**\\
    1. Measure (ib_write_bw)\\
    2. Correlate (PFC + BER)\\
    3. Isolate (single link test)\\
    4. Repair + re-baseline}

## Your Turn: Ring AllReduce Latency Crossover

{
[3 min]
% -- LINK: Alpha-beta model showed two regimes; this exercise applies it to
Ring AllReduce at real fleet scale to find where latency dominates.

% -- NARRATE: ``A 70B model on 256 GPUs. 90 seconds. Compute the alpha-beta
time for Ring AllReduce over IB NDR. At what GPU count does latency dominate?''
After pause: ``Ring AllReduce time: $T = 2(N-1)/N × M/ + (N-1) × $.
M = 280 GB (70B $×$ 4 bytes mixed precision gradients). $$ = 50 GB/s (IB NDR).
$$ = 5 $$s per hop. At N=256: bandwidth term = 2 $×$ 255/256 $×$ 280/50
= 11.2 s. Latency term = 255 $×$ 5 $$s = 1.3 ms. Bandwidth dominates by 8,600$×$.
Latency dominates when $M < N ×  × $ = 256 $×$ 5$$s $×$ 50 GB/s
= 64 KB. For small control messages ($<$64 KB), topology (hops) matters more than link speed.''

% -- ENGAGE: This IS the active learning moment.
After: ``At 4,096 GPUs, the latency term grows to 20 ms. Still small vs 11.2 s.
At what message size does latency = bandwidth?'' This is the crossover $n^*$.

% -- WARN: Students forget the $2(N-1)/N$ factor in Ring AllReduce.
ALSO NOTE: This exercise uses $ = 5\,$s, which is the end-to-end hardware
link latency (IB_NDR_LATENCY_US = 5 $$s from the registry). The $$-$$
model slides used $ = 1.5\,$s (FABRIC_ALPHA_NDR), which is the software
startup parameter. The two measure different things: 1.5 $$s is the per-hop
overhead at the transport layer; 5 $$s captures full end-to-end hardware latency.
The crossover point shifts accordingly (75 KB vs 64 KB here). Flag this explicitly
so students are not confused: ``Note that this exercise uses the hardware one-way
latency of 5 $$s, not the software startup alpha of 1.5 $$s used in the model.''

% -- FLEX: [CORE] This is the chapter's quantitative exercise.
IF SHORT: Give 60 seconds, show the solution, emphasize the 8,600x ratio.
}

[T]
  
    { Ring AllReduce: $$-$$ Analysis}

    {0.1cm}
    { $T_{{Ring}} = {2{N\!-\!1}{N}  {M}{}}_{{bandwidth}} + {(N\!-\!1)  }_{{latency}}$}

    {0.15cm}
    **Given:** 70B model, 256 GPUs
    {0pt}
- { $M$ = 280 GB (mixed precision gradients)}
- { $$ = 50 GB/s (IB NDR)}
- { $$ = 5 $$s per hop}
    

    {0.1cm}
    **Calculate:** Total time and crossover point.

    {{midgray}{(90 seconds)}}
  
  
    
    {mlsyscard}{computestroke}
    { **Solution:**\\
    BW term: $2 × 255/256 × 280/50$\\
    = **11.2 s**\\
    Latency term: $255 × 5$s\\
    = **1.3 ms**\\[0.1cm]
    BW dominates by 8,600$×$!\\[0.1cm]
    Crossover at $M < 64$ KB:\\
    Below 64 KB: optimize **hops**\\
    Above 64 KB: optimize **link speed**}
    {mlsyscard}

## Fabric Diagnostic Flowchart

{
[1 min]
% -- LINK: Monitoring showed the three telemetry layers; this flowchart
synthesizes them into a diagnostic procedure.

% -- NARRATE: ``When training slows down mysteriously, follow this flowchart.
Step 1: Check ib_write_bw baseline --- has bandwidth degraded? Step 2: Check PFC
counters --- is the fabric under congestion? Step 3: Check BER --- is a transceiver
failing? Step 4: Correlate with training throughput timeline. The root cause is
usually: (a) bad transceiver (replace), (b) PFC storm (isolate offending flow),
or (c) topology mismatch (topology-aware placement).''

% -- FLEX: [OPTIONAL] Reference slide for operational debugging.
IF SHORT: Show the flowchart, name the three root causes, move on.
}

{fabric-diagnostic-flowchart.pdf}

{0.1cm}

{Diagnostic order:}{Bandwidth baseline $$ PFC counters $$ BER trends $$ correlate with training throughput.}

## Fallacies

{
[2 min]
% -- LINK: Each fallacy points back to where it was disproved in this lecture:
(1) bandwidth fallacy --- disproved by the 10 KB $$-$$ worked example;
(2) oversubscription fallacy --- disproved by the bisection bandwidth calculation;
(3) adaptive routing fallacy --- disproved by the incast slide;
(4) IB=fast-Ethernet fallacy --- disproved by the IB vs RoCE comparison table.

% -- NARRATE: Four claims proven wrong with numbers from this lecture.
Spend extra time on the first: ``We showed the 10 KB message only improves 11%
with 2x bandwidth. The alpha-beta model proves this quantitatively.''
On the oversubscription fallacy: ``Enterprise IT staff are trained to accept 4:1
oversubscription because web traffic is statistical. BSP training is not ---
every node injects simultaneously.''

% -- ENGAGE: Ask students which fallacy they found hardest to unlearn.
``Turn to your neighbor: which of these four fallacies is most likely to survive
in your intuition? Why?'' Take 2--3 responses. The oversubscription fallacy
typically persists because enterprise IT default is 4:1 oversubscription.

% -- WARN: The oversubscription fallacy is particularly persistent because
enterprise IT staff are trained to accept it as a cost-optimization. The IB=fast-Ethernet
fallacy persists because procurement compares spec-sheets, not system behavior.

% -- FLEX: [CORE] Fallacies correct misconceptions before they solidify.
IF SHORT: Cover the first fallacy (bandwidth) and the IB fallacy, skip the other two.
}

**Fallacy:** *More bandwidth always means faster training.*\\
{ For a 10 KB message, upgrading from 200G to 400G yields only 11% improvement.
The $$ term dominates small messages. Doubling bandwidth adds 40% to cost for negligible gain.}

{0.1cm}
**Fallacy:** *Network oversubscription is acceptable if ``most'' traffic is local.*\\
{ BSP training injects simultaneously from *all* nodes. 4:1 oversubscription $$ 4$×$ slower AllReduce. On a \$300M cluster (30% sync), this wastes \$142M+ in idle GPU-hours.}

{0.1cm}
**Fallacy:** *Adaptive routing eliminates topology-aware placement.*\\
{ Adaptive routing balances traffic but cannot create bandwidth that does not exist. A 1,024-GPU job across oversubscribed spine groups will still be throttled by cross-group links.}

{0.1cm}
**Fallacy:** *InfiniBand is just fast Ethernet.*\\
{ Procurement teams compare on link rate alone. IB provides kernel-bypass RDMA, hardware credit-based flow control, and bounded failure modes by design. Ethernet requires PFC/ECN approximations to behave losslessly --- a software-configuration discipline that fails catastrophically in 200 ms when misconfigured. The choice is a system architecture decision, not a bandwidth selection.}

## Pitfalls

{
[1.5 min]
% -- LINK: Pitfall 1 links to the PFC Deadlock slide; Pitfall 2 links to the Monitoring
slide (ib_write_bw vs iperf); Pitfall 3 links to the Monitoring slide's canary discussion.

% -- NARRATE: Three operational mistakes. Spend extra time on the PFC storm pitfall ---
``A misconfigured switch can freeze the entire fabric in 200 ms. Monitor PFC counters.''
The iperf pitfall directly connects to the monitoring slide. The counter-monitoring
pitfall is the most financially costly: silent degradation can waste thousands of
GPU-hours before anyone notices.

% -- ENGAGE: Ask which pitfall the class thinks is most common in production.
``Quick show of hands: which of these three pitfalls do you think is most common
in real production deployments?'' Take 2--3 responses. Answer: PFC storm + iperf
testing co-occur frequently --- teams discover their network is degraded because
training slows and they test with iperf, see full speed, conclude the network is fine.

% -- WARN: The iperf pitfall is a trap because iperf DOES return good numbers even
when RDMA is broken. A team that tests with iperf and sees 30 Gbps may conclude the
network is fine, while ib_write_bw shows 390 Gbps --- meaning the kernel network
is working but the RDMA path is misconfigured.

% -- FLEX: [CORE] Pitfalls prevent expensive operational mistakes.
IF SHORT: Cover the lossless Ethernet pitfall only.
}

**Pitfall:** *Assuming lossless Ethernet is as reliable as InfiniBand.*\\
RoCE approximates losslessness via PFC. A misconfigured switch can trigger a PFC storm, freezing the entire fabric in 200 ms. InfiniBand's credit-based flow control is immune.

{0.15cm}
**Pitfall:** {Testing network performance with {iperf} instead of RDMA tools.}\\
{iperf} measures TCP (kernel-based). A link showing 30 Gbps via {iperf} may deliver 390 Gbps via {ib_write_bw}. Always validate with RDMA-native tools from {perftest}.

{0.15cm}
**Pitfall:** *Neglecting PFC and ECN counter monitoring in production.*\\
A gradual increase in PFC PAUSE frames or PortXmitDiscards is the canary for a failing transceiver or routing imbalance. A 10% throughput drop can waste thousands of GPU-hours before detection.

## Muddiest Point

{
[1 min]
% -- NARRATE: ``Write the one concept you found most confusing. Anonymous. One sentence.''
% -- FLEX: [CORE] Address top 2--3 confusions in next lecture's opening.
}

{1.0cm}
{ What was the **muddiest point** today?}

{0.8cm}
{ Write down the concept you found **most confusing**.}

{0.5cm}
{{midgray}{Anonymous. One sentence. Submit before you leave.}}

## What Were the Key Ideas?

{
[2 min]
% -- NARRATE: ``Close your notes. Write four key concepts. 90 seconds.''
Walk around while students write. Do NOT show Key Takeaways yet.
% -- FLEX: [CORE] Retrieval practice.
}

{1.5cm}
{ Close your notes.}

{0.8cm}
{ Write down the **4 most important concepts** from today.}

{0.8cm}
{{midgray}{90 seconds --- no peeking.}}

## Key Takeaways

{
[2 min]
% -- LINK: Students wrote their own list; now compare.
% -- NARRATE: ``Check your list. Did you capture: 18x cliff, 75 KB crossover,
95% HPCC utilization, 4x oversubscription penalty?''
% -- FLEX: [CORE] Official summary. Read every bullet.
}

{1pt}
- **Network as Computer**: At scale, the interconnect determines performance. The 18$×$ NVLink-to-IB cliff is the central design challenge.
- **$$-$$ Framework**: $T(n) =  + n/$. Messages $<$75 KB are latency-bound (IB NDR crossover); gradients ($>$MB) are bandwidth-bound.
- **Lossless is Non-Negotiable**: RDMA requires zero packet loss. IB provides natively; Ethernet must approximate via PFC/ECN.
- **Topology Choice**: Fat-trees provide full bisection BW; rail-optimized minimizes latency; dragonflies reduce cabling at scale.
- **Congestion Control**: PFC $$ DCQCN (85%) $$ HPCC (95%+). More precise feedback = higher utilization.
- **Monitor or Waste**: PFC counters, BER trends, and RDMA bandwidth baselines detect silent network degradation.

## References

{
[0.5 min]
% -- NARRATE: ``Start with Leiserson for fat-trees and DCQCN for congestion control.''
% -- FLEX: [OPTIONAL] Skip if running short.
}

{Leiserson85}{C. Leiserson. ``Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing.'' IEEE Trans.\ Comp., 1985.}
{Kim+08}{J. Kim, W. Dally et al.\ ``Technology-Driven, Highly-Scalable Dragonfly Topology.'' ISCA 2008.}
{DCQCN15}{Y. Zhu et al.\ ``Congestion Control for Large-Scale RDMA Deployments.'' SIGCOMM 2015.}
{HPCC19}{Y. Li et al.\ ``HPCC: High Precision Congestion Control.'' SIGCOMM 2019.}
{GrandTeton}{Meta. ``Grand Teton: Training Infrastructure at Scale.'' 2023.}

## Next Lecture: Data Storage

{
[1 min]
% -- LINK: Network is wired. Next: the data that feeds it.
% -- NARRATE: ``The fleet is wired. But compute and network are useless without data.
Next chapter: parallel storage, data-loading pipelines, and checkpoint strategies
that keep the fleet supplied. Central question: how do you feed PetaFLOPS of compute
from storage that is 500x slower?''
% -- FLEX: [CORE] Forward hook.
IF SHORT: State the central question and dismiss.
}

[c]
  
    
    {{computestroke}{Storage}}\\[0.3cm]
    { Parallel file systems\, SSDs\}
  
  
    
    {{routingstroke}{Data Loading}}\\[0.3cm]
    { Prefetching pipelines\ locality\/O bottlenecks}
  
  
    
    {{datastroke}{Checkpoints}}\\[0.3cm]
    { State preservation\ recovery\ hierarchy}
  

{0.3cm}

{ The fleet is wired. How do we keep it fed with data?}\\[0.1cm]
{**Massive datasets and enormous checkpoints need parallel storage.**}

## Backup: $$-$$ Quick Reference

{
[1 min]
% -- NARRATE: Backup reference for alpha-beta calculations.
% -- FLEX: [OPTIONAL] Only show if requested.
}

**Communication Cost Models:**

{0.15cm}
{}{1.3}
{tabular}{@{}ll@{}}
  
  **Model** & **Formula** \\
  
  Point-to-point & $T(n) =  + n/$ \\
  Ring AllReduce & $T = 2{N-1}{N}  {M}{} + (N-1)  $ \\
  Tree AllReduce & $T = 2_2(N)  ( + M/(2))$ \\
  Crossover ($n^*$) & $n^* =   $ \\
  
{tabular}

{0.15cm}
**Canonical $$ values:** IB HDR/NDR: 1.5 $$s. RoCE: 3--5 $$s.\\
**Canonical $$ values:** IB HDR: 25 GB/s. IB NDR: 50 GB/s. NVLink 4.0: 900 GB/s.

## Backup: PFC Deadlock and Congestion Control Reference

{
[1 min]
% -- NARRATE: Extended reference for congestion control mechanisms.
% -- FLEX: [OPTIONAL] Use if students want to understand DCQCN vs HPCC deeply.
}

**Congestion Control Progression:**

{0.15cm}
{}{1.2}
{tabular}{@{}llrl@{}}
  
  **Mechanism** & **Signal** & **Utilization** & **Risk** \\
  
  PFC (reactive) & PAUSE frames & 70--80% & PFC storm cascade \\
  DCQCN (proactive) & ECN marking & 85--90% & Blind to severity \\
  HPCC (telemetry) & In-network INT & 95%+ & Requires programmable switches \\
  
{tabular}

{0.15cm}
{ **PFC Storm Prevention:** Watchdog timers (disable PFC on stuck port), separate VLANs for training vs storage, DCQCN to prevent buffer filling.}

## Complete original Beamer source

```tex
% =============================================================================
% Chapter 3: Network Fabrics — ML Systems Lecture Slides (Volume II)
% =============================================================================
% IMAGES NEEDED (SVGs converted to PDF):
%   - five-level-model.pdf       - bandwidth-hierarchy.pdf
%   - workload-inversion.pdf     - rdma-data-path.pdf
%   - alpha-beta-model.pdf       - fat-tree-topology.pdf
%   - topology-comparison.pdf    - congestion-control.pdf
%   - superpod-vs-teton.pdf
% =============================================================================
\documentclass[aspectratio=169, 12pt]{beamer}
\usepackage{../../assets/beamerthememlsys}

\mlsyssetup{
  volume       = {Volume II},
  chapter      = {Chapter 3},
  logo         = {../../assets/img/logo-mlsysbook.png},
  instlogo     = {../../assets/img/logo-harvard.png},
  chaptertitle = {Network Fabrics},
}

% --- Fonts ---
\usepackage[T1]{fontenc}
\usepackage[scaled=0.9]{helvet}
\usepackage{courier}
\renewcommand{\familydefault}{\sfdefault}

% --- Packages ---
\usepackage{booktabs}
\usepackage{amsmath}

% --- Image paths ---
\graphicspath{
  {images/}
}

% --- Chapter-specific macros ---
\newcommand{\ab}{$\alpha$-$\beta$}

% --- Helper: safe image include ---
\newcommand{\safeimg}[2][width=\textwidth,keepaspectratio]{%
  \IfFileExists{images/#2}{\includegraphics[#1]{#2}}{%
    \IfFileExists{#2}{\includegraphics[#1]{#2}}{%
      \fbox{\parbox[c][2.5cm][c]{0.85\linewidth}{\centering\footnotesize\textcolor{midgray}{[Missing image]}}}%
    }%
  }%
}

% --- Section count for navigation (must match actual \section{} count) ---
\setcounter{mlsystotalsections}{9}

\title{Network Fabrics}
\author{Vijay Janapa Reddi}
\institute{Harvard University}
\date{}

\begin{document}

% =============================================================================
% TITLE SLIDE
% =============================================================================
\mlsystitle{Network Fabrics}{The Synchronization Backbone}{cover_network_fabrics.png}

% =============================================================================
% LEARNING OBJECTIVES
% =============================================================================
\begin{frame}{Learning Objectives}
\note{
[1 min]
% -- LINK: Ch2 built the physical infrastructure; this chapter wires it together
Students know accelerators, nodes, and racks. Now: the fabric that turns
isolated hardware into a unified training system.

% -- NARRATE: What to SAY while showing this slide
``This chapter is about the physics of wires and switches, not the algorithms
that run on them. The network determines whether expensive GPUs compute or
sit idle. By the end, you will diagnose network bottlenecks quantitatively.''

% -- ENGAGE: Specific question for THIS slide
``How long does it take to transfer 350 GB of gradients at 50 GB/s?''
Quick mental math: 7 seconds. That is 7 seconds of 1,000 GPUs idling.

% -- FLEX: [CORE] Sets expectations for the lecture.
IF SHORT: Read objectives quickly, do not elaborate.
}

\small
\begin{enumerate}
  \item Model network cost using the \textbf{\ab{} framework} and identify bandwidth- vs.\ latency-dominated regimes
  \item Compare \textbf{RDMA} transport protocols (\textbf{InfiniBand} and \textbf{RoCE}) in terms of latency and lossless guarantees
  \item Analyze topologies (\textbf{fat-tree}, \textbf{rail-optimized}, \textbf{dragonfly}) by computing \textbf{bisection bandwidth}
  \item Evaluate congestion control (\textbf{PFC}, \textbf{DCQCN}, \textbf{HPCC}) and their impact on tail latency
  \item Design network virtualization strategies using \textbf{SR-IOV} and traffic isolation
  \item Diagnose network bottlenecks using RDMA counters and bandwidth testing tools
\end{enumerate}

\end{frame}

\begin{frame}{Visual Language}
\note{
[0.5 min]
% -- NARRATE: Same color system as previous lectures. Point briefly and move on.
% -- FLEX: [OPTIONAL] Skip if students have seen this twice already.
}

\small
Throughout this course, colors carry meaning:

\vspace{0.3cm}
\begin{columns}[T]
  \begin{column}{0.45\textwidth}
    \begin{mlsyscard}{computestroke}
    \textbf{Blue} --- Compute / Processing\\
    {\footnotesize GPU ops, forward/backward pass, inference}
    \end{mlsyscard}
    \vspace{0.15cm}
    \begin{mlsyscard}{datastroke}
    \textbf{Green} --- Data / Memory\\
    {\footnotesize Data flow, caches, healthy paths}
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.45\textwidth}
    \begin{mlsyscard}{routingstroke}
    \textbf{Orange} --- Routing / Scheduling\\
    {\footnotesize Load balancers, batch windows}
    \end{mlsyscard}
    \vspace{0.15cm}
    \begin{mlsyscard}{errorstroke}
    \textbf{Red} --- Error / Cost / Bottleneck\\
    {\footnotesize Loss, decode phase, waste}
    \end{mlsyscard}
  \end{column}
\end{columns}
\end{frame}



% =============================================================================
\section{The Gradient Bus}
% =============================================================================

\begin{frame}{Why the Network Dominates at Scale}
\note{
[2 min]
% -- LINK: Ch2 showed the 18x bandwidth cliff; this slide shows its operational consequence
Students know the cliff exists. Now: what happens when 1,000 GPUs hit it simultaneously.

% -- NARRATE: What to SAY while showing this slide
``A single H100 does 989 TFLOPS. But 1,000 GPUs must synchronize 350 GB of
gradients per step. If the network cannot keep pace, GPUs idle. The network
is not auxiliary infrastructure --- it determines whether a \$300M cluster
trains efficiently or wastes millions in idle compute.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``How long does 350 GB take at 50 GB/s?'' Expected: 7 seconds.
``That is 7 seconds of 1,000 GPUs doing nothing. Every step.''

% -- WARN: What students will get wrong
Students treat the network as plumbing that ``just works.'' Correct: a single
slow link idles 999 GPUs due to BSP barrier synchronization.

% -- FLEX: [CORE] This slide establishes the central thesis of the chapter.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    The network is the \alert{synchronization backbone}:

    \vspace{0.15cm}
    \begin{itemize}\setlength\itemsep{1pt}
      \item 175B model $\to$ 350 GB gradients per step
      \item 1,000 GPUs must all exchange before next step
      \item \textbf{One slow link} idles 999 GPUs
    \end{itemize}

    \vspace{0.15cm}
    \begin{mlsyscard}{crimson}
    The network is not auxiliary infrastructure. It determines whether a \$300M cluster
    trains efficiently or wastes millions in idle compute.
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.42\textwidth}
    \safeimg[width=\textwidth,height=5.5cm,keepaspectratio]{five-level-model.pdf}
  \end{column}
\end{columns}

\end{frame}

% --- P1: Law of Distributed Efficiency (Iron Law of Scale) ---
% \mlsysfocus is a STANDALONE frame macro — it generates its own \begin{frame}/\end{frame}.
\mlsysfocus{The Law of Distributed Efficiency}{%
$T_{\text{step}} = \underbrace{T_{\text{compute}}}_{\text{fixed}} +
\underbrace{T_{\text{comm}}(N) - T_{\text{overlap}}}_{\text{exposed to fabric}}$%
}

\begin{frame}{The Law of Distributed Efficiency: Implications}
\note{
[1.5 min]
% -- LINK: Why-the-network-dominates made the qualitative case; this slide gives the
% formal equation that quantifies how much of each training step is exposed to the fabric.
Students now know the network is the bottleneck. This slide gives them the
formal equation they will use all semester to reason about that bottleneck.

% -- NARRATE: What to SAY while showing this slide
``Every distributed training step has three components: compute, communication, and
whatever communication we managed to hide by overlapping it with compute. The equation
is T\_step = T\_compute + T\_comm(N) - T\_overlap. The term that actually hurts is the
gap: T\_comm(N) - T\_overlap. That gap is what the fabric determines. A better fabric
does not change T\_compute; it shrinks the gap by enabling more overlap. This is why
network design is an efficiency problem, not just a speed problem.''

Point to the equation and say: ``N is the number of GPUs. Notice T\_comm grows with N ---
as we add GPUs, we exchange more gradient data. The fabric's job is to keep T\_overlap
close to T\_comm so the gap stays small.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``If T\_comm doubles because we scaled from 256 to 512 GPUs, what needs to
happen to T\_overlap to keep scaling efficiency above 90 percent?''
Expected: T\_overlap must also roughly double --- meaning the fabric must support
twice as much gradient communication hidden behind computation.

% -- WARN: What students will get wrong
Students assume more GPUs = proportionally more bandwidth needed. Correct: the key
metric is overlap, not raw bandwidth. A fabric with lower latency can start overlap
earlier, achieving the same efficiency at lower bandwidth cost.

% -- FLEX: [CORE] This equation is the formal anchor for the entire chapter.
IF SHORT: Write the equation, define T\_overlap, state the punchline (fabric controls
the gap), move on.
}

\small
\vspace{0.2cm}
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    The \textbf{nonoverlapped communication} term determines scaling efficiency:
    \begin{itemize}\setlength\itemsep{1pt}
      \item $T_{\text{compute}}$: fixed by model and hardware
      \item $T_{\text{comm}}(N)$: grows with GPU count $N$
      \item $T_{\text{overlap}}$: communication hidden behind compute
    \end{itemize}
    \vspace{0.1cm}
    \begin{mlsyscard}{crimson}
    The fabric determines $T_{\text{overlap}}$: lower latency enables earlier pipelining;
    higher bandwidth shrinks the gap directly.
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{computestroke}
    {\footnotesize \textbf{Engineering consequence}: A perfectly efficient fabric achieves
    $T_{\text{overlap}} = T_{\text{comm}}(N)$, reducing $T_{\text{step}}$ to
    $T_{\text{compute}}$ alone. Real fabrics fall short of this ideal --- the gap
    is what we measure and optimize.}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{The Bandwidth Cliff}
\note{
[1.5 min]
% -- LINK: Previous slide showed the network matters; now the 18x cliff quantifies why
Students saw the synchronization cost. Now: the specific bandwidth numbers.

% -- NARRATE: What to SAY while showing this slide
Point to each column: ``NVLink: 900 GB/s within a node. InfiniBand: 50 GB/s
across nodes. 18x cliff. This ratio has persisted across four GPU generations.
It is physics --- electrical signaling over copper vs optical over distance ---
not engineering laziness.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``What changes when you cross a node boundary?''
Expected: bandwidth drops 18x, latency increases 5--10x.

% -- WARN: What students will get wrong
Students expect the 18x cliff to shrink with newer hardware. It has been
stable for four generations because the underlying physics has not changed.

% -- FLEX: [CORE] The 18x cliff is the central number of this chapter.
}

% --- Layout: FULL-WIDTH IMAGE + annotation ---
\centering
\safeimg[width=0.88\textwidth,height=4.2cm,keepaspectratio]{bandwidth-hierarchy.pdf}

\vspace{0.1cm}
\small
\begin{columns}[T]
  \begin{column}{0.30\textwidth}
    \centering
    \textcolor{datastroke}{\textbf{NVLink}}\\
    {\footnotesize 900 GB/s (intra-node)}
  \end{column}
  \begin{column}{0.30\textwidth}
    \centering
    \textcolor{errorstroke}{\textbf{18$\times$ cliff}}\\
    {\footnotesize Node boundary}
  \end{column}
  \begin{column}{0.30\textwidth}
    \centering
    \textcolor{routingstroke}{\textbf{InfiniBand NDR}}\\
    {\footnotesize 50 GB/s (inter-node)}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{The Bandwidth Cliff in Action}
\note{
[1.5 min]
% -- LINK: Bandwidth cliff showed the ratio; this table shows the dollar impact
Students saw 18x. Now: what that costs in utilization and money.

% -- NARRATE: What to SAY while showing this slide
``8 GPUs on NVLink: AllReduce takes 1 ms, 99.5\% utilization. 64 GPUs across
IB: AllReduce balloons to 30 ms, utilization drops to 87\%. That 12-point gap,
compounded over months, costs millions of dollars.''

% -- WARN: What students will get wrong
Students think 87\% utilization is good. Correct: on a \$300M cluster,
13\% idle = \$39M wasted per year.

% -- FLEX: [CORE] Quantifies the economic cost of the bandwidth cliff.
IF SHORT: State the two numbers (99.5\% vs 87\%) and the dollar consequence.
}

\small
\renewcommand{\arraystretch}{1.15}
{\footnotesize
\begin{tabular}{@{}lrrr@{}}
  \toprule
  \textbf{Scenario} & \textbf{AllReduce Time} & \textbf{Utilization} & \textbf{Bottleneck} \\
  \midrule
  8 GPUs, NVLink (900 GB/s) & 1 ms & \textcolor{datastroke}{\textbf{99.5\%}} & Compute \\
  64 GPUs, IB NDR (50 GB/s) & 30 ms & \textcolor{errorstroke}{\textbf{87\%}} & Network \\
  \bottomrule
\end{tabular}
}

\vspace{0.2cm}
\begin{mlsyscard}{errorstroke}
The 12-point utilization gap represents \textbf{millions of dollars} in wasted compute
over a months-long training run. Network fabric design is the central engineering
challenge of distributed training.
\end{mlsyscard}

\end{frame}

% =============================================================================
\section{Workload Inversion}
% =============================================================================

\begin{frame}{ML Inverts Datacenter Assumptions}
\note{
[2 min]
% -- LINK: Bandwidth cliff showed the quantitative gap; workload inversion explains WHY
Standard datacenter networks were designed for web traffic. ML inverts every assumption.

% -- NARRATE: What to SAY while showing this slide
``Traditional datacenter: millions of small flows, asynchronous, loss-tolerant.
ML training: few massive flows, synchronous, loss-INtolerant. Every single
assumption is inverted. This is why standard enterprise Ethernet fails for ML.''
Point to BSP callout: ``All 1,024 GPUs must finish before ANY can proceed.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why can you not just run training over your existing datacenter network?''
Expected: existing networks are designed for mice flows with loss tolerance.

% -- WARN: What students will get wrong
Students with web-scale experience assume statistical multiplexing helps.
Correct: BSP training creates simultaneous elephant flows --- no multiplexing.

% -- FLEX: [CORE] Workload inversion is the conceptual foundation for RDMA and lossless.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.92\textwidth,height=4.8cm,keepaspectratio]{workload-inversion.pdf}

\vspace{0.1cm}
\mlsysinsight{BSP (Bulk Synchronous Parallel):}{All 1,024 GPUs must finish before \emph{any} can proceed. The slowest link is the bottleneck.}

\end{frame}

% --- ACTIVE LEARNING 1: Predict ---
\begin{frame}{Predict: What Happens to One Dropped Packet?}
\note{
[2 min]
% -- LINK: Workload inversion said loss-intolerant; this prediction shows WHY
Students heard lossless is required. This makes them discover the consequence.

% -- NARRATE: What to SAY while showing this slide
``60 seconds. What happens when a single packet drops 900 MB into a 1 GB
RDMA transfer? Write your prediction.''
Most will guess ``just retransmit that one packet.'' The reveal: RDMA uses
Go-Back-N, so one drop retransmits the entire 100 MB tail.

% -- ENGAGE: This IS the active learning moment
Do NOT reveal yet. Let students sit with the discomfort of not knowing.
The surprise makes the lossless requirement visceral, not abstract.

% -- FLEX: [CORE] This prediction creates the emotional case for lossless networking.
}

\centering
\vspace{0.7cm}
{\Large\bfseries Think--Write--Share}

\vspace{0.4cm}
{\large In a 1,024-GPU synchronous training job using RDMA,\\
what happens when a \emph{single packet} is dropped\\
900 MB into a 1 GB gradient transfer?}

\vspace{0.4cm}
{\normalsize Write your prediction. \textcolor{midgray}{(60 seconds)}}

\vspace{0.3cm}
{\small\textcolor{lightgray}{Hint: RDMA does NOT use TCP's Selective Acknowledgement.}}

\end{frame}

% =============================================================================
\section{Wire \& Transport}
% =============================================================================

\begin{frame}{Level 1: Wire and Link Physics}
\note{
[2 min]
% -- LINK: Workload inversion showed what ML needs; this slide shows the wire physics
Students know the requirements. Now: what the physical medium can deliver.

% -- NARRATE: What to SAY while showing this slide
``PAM4 doubles data rate by encoding 2 bits per symbol, but voltage margins
shrink 3x, requiring FEC at every hop. FEC adds 100--200 ns per hop ---
irreducible physics tax. A 3-hop fat-tree: 600--1,200 ns of pure latency floor.''
Point to the medium table: ``Copper under 3 meters intra-rack. Fiber for
inter-rack. Each has different cost, power, and reach trade-offs.''
Mention: ``SerDes power: 3,000 links at 25W each = 75 kW --- over 10\% of
cluster power just to move bits.''

% -- WARN: What students will get wrong
Students assume latency is dominated by distance (speed of light). Correct:
FEC processing dominates at datacenter distances. 100 meters at light speed
is 0.5 us; FEC adds 0.6--1.2 us.

% -- FLEX: [OPTIONAL] Wire physics is important context but not directly examinable.
IF SHORT: State the FEC latency floor and SerDes power, skip medium details.
}

\footnotesize
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{PAM4 Signaling} (4 voltage levels, 2 bits/symbol):
    \begin{itemize}\setlength\itemsep{0pt}
      \item Doubles data rate without higher symbol rate
      \item Voltage margin shrinks 3$\times$ $\to$ \textbf{FEC required}
      \item FEC adds 100--200 ns per hop (irreducible)
    \end{itemize}

    \vspace{0.1cm}
    \textbf{Medium trade-offs:}
    \begin{itemize}\setlength\itemsep{0pt}
      \item \textcolor{computestroke}{Copper (DAC)}: $<$3 m, intra-rack
      \item \textcolor{routingstroke}{Active Optical}: 100+ m, inter-rack
    \end{itemize}

    \vspace{0.1cm}
    \mlsysalert{Physics tax:}{3-hop fat-tree $\to$ 600--1,200 ns FEC floor.}
  \end{column}
  \begin{column}{0.42\textwidth}
    \renewcommand{\arraystretch}{1.05}
    {\scriptsize
    \begin{tabular}{@{}lll@{}}
      \toprule
      \textbf{Medium} & \textbf{Reach} & \textbf{Use} \\
      \midrule
      DAC (copper) & $<$3 m & In-rack \\
      AOC (fiber) & 30 m & Inter-rack \\
      SMF & 10+ km & Building \\
      \bottomrule
    \end{tabular}
    }

    \vspace{0.15cm}
    {\scriptsize
    \begin{tabular}{@{}lr@{}}
      \toprule
      \textbf{Component} & \textbf{Power} \\
      \midrule
      SerDes + xcvr (400G) & $\sim$25 W \\
      3,000 links total & 75 kW \\
      \bottomrule
    \end{tabular}
    }
    {\scriptsize\textcolor{midgray}{$>$10\% of cluster power}}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{Level 2: RDMA and GPUDirect}
\note{
[2 min]
% -- LINK: Wire physics showed the physical medium; RDMA shows the transport layer
Students saw the physical constraints. Now: how RDMA bypasses the kernel
to achieve 1--2 us latency vs TCP's 50--100 us.

% -- NARRATE: What to SAY while showing this slide
Point to the diagram: ``Traditional TCP: 3 memory copies, kernel involvement,
50--100 us. RDMA: zero copies, NIC reads GPU memory directly via PCIe DMA,
1--2 us. For 350 GB of gradients, RDMA eliminates 700 GB of redundant copies
per step.'' Emphasize: ``The NIC talks directly to GPU memory. The CPU is
not involved at all.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why cannot we just use faster TCP?'' Expected: TCP's kernel involvement
and memory copies are architectural, not just slow software.

% -- WARN: What students will get wrong
Students think RDMA is just ``optimized TCP.'' Correct: RDMA fundamentally
changes the data path --- it bypasses the kernel entirely.

% -- FLEX: [CORE] RDMA is the foundational transport for all distributed training.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.92\textwidth,height=4.5cm,keepaspectratio]{rdma-data-path.pdf}

\vspace{0.1cm}
\mlsysconcept{Key:}{RDMA bypasses the kernel and CPU entirely --- NIC reads GPU memory directly via PCIe DMA.}

\end{frame}

\begin{frame}{InfiniBand vs.\ RoCE}
\note{
[2 min]
% -- LINK: RDMA showed the transport; IB vs RoCE shows two implementations
Students know RDMA bypasses the kernel. Now: two ways to deliver it.

% -- NARRATE: What to SAY while showing this slide
``Both IB and RoCE expose the same Verbs API. The difference is reliability.
IB: hardware credit-based flow control, natively lossless. RoCE: relies on
Ethernet PFC to approximate losslessness --- fragile under scale.''
Point to PFC storm risk row: ``A misconfigured RoCE switch can freeze the
entire fabric in 200 ms. IB is immune.''

% -- WARN: What students will get wrong
Common error: ``RoCE is just IB over Ethernet.'' No --- the lossless
guarantees are weaker. PFC approximates losslessness but can fail catastrophically.

% -- FLEX: [CORE] The IB vs RoCE choice is a real production decision.
IF SHORT: Focus on the lossless row and PFC storm risk row only.
}

\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lll@{}}
  \toprule
  \textbf{Property} & \textbf{InfiniBand} & \textbf{RoCE (Ethernet)} \\
  \midrule
  Lossless & \textcolor{datastroke}{\textbf{Native}} (credit-based) & Approximated (PFC) \\
  Latency & 1--2 $\mu$s & 3--5 $\mu$s \\
  Congestion & Hardware adaptive routing & DCQCN / HPCC \\
  Switch cost & Higher (single vendor) & \textcolor{datastroke}{\textbf{20--40\% lower}} \\
  Vendor lock-in & \textcolor{errorstroke}{Single} (NVIDIA/Mellanox) & \textcolor{datastroke}{\textbf{Multi-vendor}} \\
  PFC storm risk & \textcolor{datastroke}{\textbf{Immune}} & \textcolor{errorstroke}{Vulnerable} \\
  \bottomrule
\end{tabular}

\vspace{0.15cm}
\begin{mlsyscard}{routingstroke}
Both deliver 95\%+ line-rate for large transfers. IB wins on tail latency (30--50\% lower).
RoCE wins on operational flexibility and cost.
\end{mlsyscard}

\end{frame}

\begin{frame}{PFC Deadlock: When Lossless Goes Wrong}
\note{
[2 min]
% -- LINK: IB vs RoCE mentioned PFC storm risk; this slide unpacks the failure mode

% -- NARRATE: What to SAY while showing this slide
``PFC (Priority Flow Control) approximates losslessness on Ethernet by sending
PAUSE frames when buffers fill. The problem: PAUSE propagates backwards, hop by hop.
A single congested port can freeze an entire subnet in under 200 ms. This is called
a PFC storm or head-of-line blocking cascade.''
Walk through the scenario: ``Switch A congests. Sends PAUSE to Switch B. Switch B
fills its buffer, sends PAUSE to Switch C. Within 200 ms, the entire spine is frozen.
All traffic --- not just the congested flow --- is blocked.''

% -- ENGAGE: ``Why does InfiniBand not have this problem?''
Expected: IB uses credit-based flow control --- each link grants credits independently.
No PAUSE propagation. Congestion stays local.

% -- WARN: Students assume PFC problems are rare. At Meta and Microsoft scale,
PFC storms were a leading cause of training job failures before mitigations.

% -- FLEX: [OPTIONAL] Important for RoCE environments but not universal.
IF SHORT: State the 200 ms cascade number and the IB immunity, skip the diagram.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{PFC Storm Cascade:}

    \vspace{0.1cm}
    \begin{enumerate}\setlength\itemsep{1pt}
      \item {\footnotesize Port congests $\to$ buffer fills}
      \item {\footnotesize PAUSE frame sent upstream}
      \item {\footnotesize Upstream buffer fills $\to$ PAUSE propagates}
      \item {\footnotesize \textbf{Entire subnet frozen in $<$200 ms}}
    \end{enumerate}

    \vspace{0.1cm}
    \begin{mlsyscard}{errorstroke}
    {\scriptsize \textbf{Head-of-line blocking}: ALL traffic through the congested switch is stopped, not just the offending flow. One bad actor freezes the fabric.}
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.42\textwidth}
    \textbf{Mitigation strategies:}
    \begin{itemize}\setlength\itemsep{1pt}
      \item {\footnotesize \textcolor{datastroke}{\textbf{Watchdog}}: Detect PAUSE duration $>$ threshold, disable PFC on that port}
      \item {\footnotesize \textcolor{routingstroke}{\textbf{DCQCN}}: ECN marking before buffers fill}
      \item {\footnotesize \textcolor{computestroke}{\textbf{Separate VLANs}}: Isolate training traffic from storage traffic}
    \end{itemize}

    \vspace{0.1cm}
    {\scriptsize InfiniBand: \textbf{immune} (credit-based, no PAUSE propagation).}
  \end{column}
\end{columns}

\end{frame}

% =============================================================================
\section{The $\alpha$-$\beta$ Model}
% =============================================================================

\mlsysfocus{The Communication Cost Model}{%
$T(n) = \underbrace{\alpha}_{\text{Startup}} + \underbrace{\dfrac{n}{\beta}}_{\text{Transfer}}$%
}

\begin{frame}{\ab{} Model: Two Regimes}
\note{
[2 min]
% -- LINK: IB vs RoCE compared transports; alpha-beta formalizes cost modeling
Students know the transport options. Now: the quantitative model for communication cost.

% -- NARRATE: What to SAY while showing this slide
``alpha is startup latency: switch hops, FEC, software overhead. beta is sustained
bandwidth. The crossover $n^* = \alpha \times \beta$ determines which term dominates.
For IB NDR ($\alpha=1.5\,\mu$s, $\beta=50$ GB/s): $n^*=1.5\times10^{-6}\times50\times10^{9}=75{,}000$ bytes~$\approx$~75 KB.
Below 75 KB: latency-bound. Above: bandwidth-bound.
Topology choice shifts both parameters.''

% -- ENGAGE: Specific question for THIS slide
``Is a 350 GB AllReduce latency-bound or bandwidth-bound?''
Expected: bandwidth-bound --- 350 GB is millions of times above $n^*$.

% -- WARN: What students will get wrong
Students assume all network transfers are bandwidth-limited. Correct: control
messages (heartbeats, barriers) are tiny and latency-dominated.

% -- FLEX: [CORE] The alpha-beta model is the diagnostic tool for network cost.
IF AHEAD: ``What is the alpha-beta crossover for a 3-hop dragonfly vs 6-hop fat-tree?''
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.88\textwidth,height=4.2cm,keepaspectratio]{alpha-beta-model.pdf}

\vspace{0.1cm}
\small
\mlsysconcept{Diagnostic:}{Is your workload above or below $n^*$? That determines whether you optimize for hops (topology) or link speed (bandwidth).}

\end{frame}

\begin{frame}{\ab{} Model: Worked Example}
\note{
[2 min]
% -- LINK: Alpha-beta model introduced the two regimes; this worked example demonstrates
Students saw the model. Now: a concrete calculation that surprises.

% -- NARRATE: What to SAY while showing this slide
``10 KB control message. IB HDR: 1.91 us. IB NDR (2x bandwidth): 1.70 us.
Despite doubling bandwidth, only 11\% improvement. Why? Because alpha dominates.
You added 40\% to transceiver cost for negligible benefit.''
This is the quantitative proof that more bandwidth is not always the answer.

% -- WARN: What students will get wrong
Students assume faster links always help. This example proves that for small
messages, reducing hop count (topology) matters more than link speed.

% -- FLEX: [CORE] This worked example grounds the alpha-beta model concretely.
IF SHORT: State the 11\% number and the punchline, skip the calculation walkthrough.
}

\small
\textbf{Small message: 10 KB control synchronization}

\vspace{0.15cm}
\renewcommand{\arraystretch}{1.2}
{\footnotesize
\begin{tabular}{@{}llll@{}}
  \toprule
  \textbf{Link} & \textbf{$\alpha$ ($\mu$s)} & \textbf{$n/\beta$ ($\mu$s)} & \textbf{$T(n)$ ($\mu$s)} \\
  \midrule
  IB HDR (200G) & 1.5 & 0.41 & \textbf{1.91} \\
  IB NDR (400G) & 1.5 & 0.20 & \textbf{1.70} \\
  \midrule
  \multicolumn{3}{r}{\textbf{Speedup:}} & \textcolor{errorstroke}{\textbf{only 11\%}} \\
  \bottomrule
\end{tabular}
}

\vspace{0.2cm}
\begin{mlsyscard}{errorstroke}
\textbf{Diagnosis:} Doubling bandwidth yields only 11\% improvement because
$\alpha$ dominates for small messages. Upgrading to NDR may add 40\% to transceiver cost
for negligible benefit in the latency-dominated regime.
\end{mlsyscard}

\end{frame}

% --- ACTIVE LEARNING: Micro-Retrieval Cue ---
\begin{frame}{Quick Check}
\note{
[1 min]
% -- LINK: Alpha-beta worked example showed the 10 KB case; this micro-retrieval
asks students to apply the same crossover logic independently.
% -- NARRATE: Quick retrieval. ``50 KB message, alpha=1.5 us, beta=50 GB/s.
$n^* = 1.5 \times 10^{-6} \times 50 \times 10^9 = 75$ KB. Since 50 KB $<$ 75 KB: latency-bound.''
% -- ENGAGE: Cold-call after 15 seconds: ``Is the 50 KB message latency- or bandwidth-bound?
What is the crossover?'' Expected: n*=75 KB; 50 KB < 75 KB, so latency-bound.
% -- WARN: Students may forget to compute n* first and just compare 50 KB to 50 GB/s.
They need to compute n* = alpha*beta in bytes first.
% -- FLEX: [CORE] Reinforces the alpha-beta crossover calculation.
}

\centering
\vspace{1.0cm}
{\Large\bfseries Quick Check}

\vspace{0.6cm}
{\large A message is 50 KB on IB NDR ($\alpha=1.5\,\mu$s, $\beta=50$ GB/s).}\\
{\large Latency-bound or bandwidth-bound?}

\vspace{0.5cm}
{\normalsize\textcolor{midgray}{15 seconds --- then I will cold-call.}}

\end{frame}



% --- ACTIVE LEARNING 2: Exercise ---
\begin{frame}{Your Turn: Bandwidth- or Latency-Dominated?}
\note{
[3 min]
% -- LINK: The 10 KB example showed latency-dominated; this exercise shows the opposite
Students saw alpha dominate for small messages. Now: bandwidth dominates for AllReduce.

% -- NARRATE: What to SAY while showing this slide
``90 seconds. 350 GB AllReduce on IB NDR. Calculate T(n) and identify the regime.''
After pause: ``$n/\beta$ = 350/50 = 7.0 seconds. $\alpha$ = 0.0000015 seconds.
Clearly bandwidth-dominated. Doubling bandwidth gives nearly 2x speedup ---
the opposite of the 10 KB case.''

% -- ENGAGE: This IS the active learning moment
The contrast with the 10 KB example is the teaching point: same model,
opposite conclusions. The crossover $n^*$ determines which lever matters.

% -- WARN: What students will get wrong
Students may add alpha and n/beta incorrectly or confuse units (us vs seconds).
Remind: convert everything to the same unit before adding.

% -- FLEX: [CORE] This exercise and the 10 KB example together prove the alpha-beta model.
}

\small
\begin{columns}[T]
  \begin{column}{0.58\textwidth}
    {\normalsize\bfseries Classify the Regime}

    \vspace{0.2cm}
    A 175B model AllReduce transfers 350 GB across IB NDR:
    \begin{itemize}
      \item $\alpha$ = 1.5 $\mu$s, $\beta$ = 50 GB/s
      \item $n$ = 350 GB
    \end{itemize}

    \vspace{0.1cm}
    \textbf{1.} Calculate $T(n)$ and identify the dominant term.\\
    \textbf{2.} Would doubling bandwidth help here?

    {\footnotesize\textcolor{midgray}{(90 seconds --- then compare)}}
  \end{column}
  \begin{column}{0.38\textwidth}
    \pause
    \begin{mlsyscard}{datastroke}
    \textbf{Solution:}\\[0.1cm]
    {\footnotesize
    $n/\beta$ = 350 / 50 = \textbf{7.0 s}\\
    $\alpha$ = 0.0000015 s\\[0.1cm]
    \textcolor{computestroke}{\textbf{Bandwidth-dominated!}}\\[0.1cm]
    Yes --- 2$\times$ BW $\to$ $\sim$2$\times$ speedup.\\
    (Opposite of the 10 KB case.)
    }
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

% =============================================================================
\section{Topology}
% =============================================================================

\begin{frame}{Fat-Tree (Clos): The Industry Standard}
\note{
[2 min]
% -- LINK: Alpha-beta quantified transfer cost; topology determines the alpha and beta values
Students can model cost per transfer. Now: how topology shapes those parameters.

% -- NARRATE: What to SAY while showing this slide
``Fat-tree: hierarchical switch tiers providing full bisection bandwidth.
$N = k^3/4$ hosts. Radix 64 gives 65,536 GPUs. Non-blocking means every
AllReduce gets full line-rate. Cost: O(N log N) switches, \$20--100M for 4,096 GPUs.''

% -- ENGAGE: Specific question for THIS slide
``Why is full bisection bandwidth critical for AllReduce?''
Expected: BSP means all GPUs inject simultaneously --- no statistical multiplexing.

% -- WARN: What students will get wrong
Students assume any tree topology provides full bisection BW. Correct: only
non-blocking (1:1 subscription) fat-trees do. Oversubscribed trees lose bandwidth.

% -- FLEX: [CORE] Fat-tree is the baseline topology for all comparisons.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.88\textwidth,height=4.0cm,keepaspectratio]{fat-tree-topology.pdf}

\vspace{0.1cm}
\small
\mlsysconcept{Key formula:}{$N = k^3/4$ hosts. Radix $k$=64 $\to$ 65,536 GPUs with full bisection bandwidth.}

\end{frame}

\begin{frame}{Rail-Optimized Topology}
\note{
[2 min]
% -- LINK: Fat-tree is the general-purpose standard; rail-optimized is the
specialized alternative designed specifically for tensor parallelism.

% -- NARRATE: What to SAY while showing this slide
``NVIDIA SuperPOD uses a rail-optimized design. Each GPU in a node connects to a
different leaf switch --- 8 GPUs, 8 rails. Tensor parallelism happens within the
node via NVLink (900 GB/s). Data parallelism goes across rails via IB (50 GB/s per
rail). Key advantage: TP traffic never touches the fabric. Key constraint: jobs
MUST be topology-aware --- a job placed across two pods pays a 2--4x penalty.''

% -- ENGAGE: ``Why does each GPU connect to a DIFFERENT leaf switch?''
Expected: to distribute DP AllReduce traffic across 8 independent paths.

% -- WARN: Students assume any topology works for TP. Correct: rail-optimized
REQUIRES that TP stays intra-node. Violating this constraint collapses performance.

% -- FLEX: [CORE] Rail-optimized is the dominant production topology for LLM training.
IF SHORT: State the design principle and the topology-aware constraint.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{Rail-Optimized Design (SuperPOD):}

    \vspace{0.1cm}
    \begin{itemize}\setlength\itemsep{1pt}
      \item {\footnotesize 8 GPUs per node $\to$ 8 independent ``rails''}
      \item {\footnotesize Each GPU connects to a \emph{different} leaf switch}
      \item {\footnotesize TP: intra-node via NVLink (900 GB/s)}
      \item {\footnotesize DP: across rails via IB (50 GB/s $\times$ 8 = 400 GB/s aggregate)}
    \end{itemize}

    \vspace{0.1cm}
    \begin{mlsyscard}{datastroke}
    {\scriptsize \textbf{Advantage}: TP traffic never touches the fabric. 8 independent DP paths maximize aggregate bandwidth.}
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{errorstroke}
    {\footnotesize \textbf{Constraint}: Jobs must be \emph{topology-aware}. Cross-pod placement: 2--4$\times$ slower.\\[0.1cm]
    Requires: topology-aware scheduler, contiguous GPU allocation, NVLink for TP.}
    \end{mlsyscard}

    \vspace{0.1cm}
    {\scriptsize Used by: NVIDIA DGX SuperPOD, Meta Grand Teton (variant).}
  \end{column}
\end{columns}

\end{frame}

% --- P1: Archetype A Lighthouse (GPT-4/Llama-3 rail-optimized fleet) ---
\begin{frame}{Lighthouse: Archetype A --- Two Traffic Classes (1/2)}
\note{
[1.5 min]
% -- LINK: Rail-Optimized Topology showed the mechanism; this slide explains WHY
% that topology exists by naming the workload archetype that drove it.
Students now know how rail-optimized topology works. This slide names the model
class that made it the dominant production design.

% -- NARRATE: What to SAY while showing this slide
``GPT-4 and Llama-3 scale training uses 3D Parallelism: data parallelism across
nodes, tensor parallelism within nodes, pipeline parallelism across pipeline stages.
This creates two distinct traffic patterns: (1) Data-parallel AllReduce --- bandwidth-
hungry, 350 GB+ of gradients per step, tolerates tens of milliseconds. (2) Tensor-
parallel activation exchange --- latency-sensitive, happens on every forward pass
microbatch, must complete in microseconds.''

% -- ENGAGE: Specific question for THIS slide
Ask: ``Why is TP latency-sensitive but DP is not?''
Expected: TP activations are on the forward-pass critical path --- every microbatch
waits for TP completion before continuing. DP AllReduce happens after the backward
pass and can often be overlapped with the next microbatch. Different criticality,
different fabric requirements.

% -- WARN: What students will get wrong
Students assume all communication in distributed training has the same requirements.
Correct: 3D parallelism creates heterogeneous traffic classes.

% -- FLEX: [CORE] This is the chapter's motivating real-world example.
IF SHORT: Name the two traffic classes, state that TP stays on NVLink and DP
uses one hop per rail.
}

\small
\textbf{Archetype A (GPT-4/Llama-3): 3D Parallelism generates two traffic classes:}

\vspace{0.2cm}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \begin{mlsyscard}{computestroke}
    {\footnotesize \textbf{Traffic Class 1 --- DP AllReduce}\\
    Bandwidth-hungry gradient averaging across nodes.\\
    350 GB+ per step; tolerates tens of ms.\\
    \textcolor{datastroke}{Rail design: 1 hop, 50 GB/s $\times$ 8 rails.}}
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.48\textwidth}
    \begin{mlsyscard}{routingstroke}
    {\footnotesize \textbf{Traffic Class 2 --- TP Activation Exchange}\\
    Latency-sensitive; on the forward-pass critical path.\\
    Microsecond deadline; 900 GB/s required.\\
    \textcolor{datastroke}{Rail design: stays on NVLink, never hits the fabric.}}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{Lighthouse: Archetype A --- Why Rails Exist (2/2)}
\note{
[1 min]
% -- LINK: Previous slide showed the two traffic classes; this slide explains why
% the rail topology is the direct consequence of those classes.
Students saw the two traffic classes. Now: why rails are the logical consequence.

% -- NARRATE: What to SAY while showing this slide
``The rail design maps cleanly: TP activations stay intra-node on NVLink (900 GB/s),
never touching the fabric at all. DP AllReduce crosses a single hop per rail --- exactly
what the single-hop rail design provides. The topology is not arbitrary; it was
reverse-engineered from the communication pattern of this specific model archetype.''

% -- ENGAGE: ``What topology would Archetype B (batch inference) drive instead?''
Expected: different fabric priorities --- lower latency, more independent paths.

% -- WARN: A fabric optimized for DP AllReduce (bandwidth-optimized) may be wrong
for TP activations (latency-optimized), which is exactly why the rail architecture
keeps them on separate networks.

% -- FLEX: [CORE] The key insight: topology is workload-driven.
IF SHORT: State the design principle and move on.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \begin{mlsyscard}{crimson}
    {\footnotesize \textbf{Why rails exist:} The two-class traffic pattern of
    Archetype A (GPT-4/Llama-3) required a topology that could simultaneously
    saturate inter-node bandwidth \emph{and} keep TP on a zero-hop NVLink path.
    The rail-optimized design is not a general-purpose improvement ---
    it is the topology shaped by this specific workload.}
    \end{mlsyscard}
  \end{column}
  \begin{column}{0.42\textwidth}
    \mlsysinsight{Design principle:}{Topology is workload-driven. Archetype A's 3D parallelism drove the rail topology; a different parallelism strategy would drive a different fabric.}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{Topology Trade-offs}
\note{
[2 min]
% -- LINK: Fat-tree showed the standard; this compares three topology choices
Students know fat-tree. Now: rail-optimized and dragonfly alternatives.

% -- NARRATE: What to SAY while showing this slide
``Fat-tree: flexible, expensive, full bisection BW. Rail-optimized: 1-hop tensor
parallelism but requires topology-aware job scheduling. Dragonfly: 50\% less cabling,
but 2--4x slowdown for jobs that cross groups. No single winner.''

% -- ENGAGE: Specific question for THIS slide
``If your entire training job fits in one dragonfly group, which topology wins?''
Expected: dragonfly --- within-group performance matches fat-tree at lower cost.

% -- WARN: What students will get wrong
Students pick one topology as universally best. Correct: the choice depends on
workload mix (single large job vs many small jobs) and scale.

% -- FLEX: [CORE] Topology trade-offs are a real production design decision.
IF SHORT: State the three options and the ``no free lunch'' conclusion.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.92\textwidth,height=4.8cm,keepaspectratio]{topology-comparison.pdf}

\vspace{0.1cm}
\mlsysalert{No free lunch:}{Topology choice is workload-dependent. Fat-trees provide flexibility; dragonflies save cost; rails minimize latency.}

\end{frame}

\begin{frame}{Bisection Bandwidth: Why It Matters}
\note{
[2 min]
% -- LINK: Topology comparison named bisection BW; this slide quantifies its impact
Students heard ``full bisection bandwidth.'' Now: what happens without it.

% -- NARRATE: What to SAY while showing this slide
``Bisection bandwidth: minimum BW across any equal partition. At 1:1 (non-blocking):
25.6 TB/s, AllReduce in 13.7 seconds. At 4:1 (oversubscribed): 6.4 TB/s,
AllReduce in 54.7 seconds --- 4x slower.''
Point to the cost callout: ``For a \$300M cluster where sync is 30\% of time,
4:1 oversubscription wastes over \$142M in idle GPU-hours.''

% -- WARN: What students will get wrong
Students assume oversubscription is fine because web traffic handles it.
Correct: BSP training has no statistical multiplexing --- oversubscription
directly multiplies sync time.

% -- FLEX: [CORE] Bisection bandwidth is the quantitative test for topology adequacy.
IF SHORT: State the 4x slowdown and \$142M waste number.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{Bisection bandwidth}: minimum BW across any equal partition.

    \vspace{0.15cm}
    \renewcommand{\arraystretch}{1.15}
    {\footnotesize
    \begin{tabular}{@{}lrr@{}}
      \toprule
      \textbf{Subscription} & \textbf{Bisection BW} & \textbf{AllReduce} \\
      \midrule
      1:1 (non-blocking) & 25.6 TB/s & 13.7 s \\
      4:1 (oversubscribed) & 6.4 TB/s & 54.7 s \\
      \bottomrule
    \end{tabular}
    }

    \vspace{0.15cm}
    \mlsysalert{Cost of 4:1:}{4$\times$ slower AllReduce $\to$ \$142M+ wasted GPU-hours on a \$300M cluster.}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{crimson}
    {\footnotesize BSP training means \emph{all} GPUs inject simultaneously. Unlike web
    traffic, there is no statistical multiplexing --- the oversubscription ratio directly
    multiplies sync time.}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

% =============================================================================
\section{Fabric Behavior}
% =============================================================================

\begin{frame}{Congestion Control: PFC $\to$ DCQCN $\to$ HPCC}
\note{
[2 min]
% -- LINK: Topology determined the paths; congestion control manages traffic on those paths
Students know the topology. Now: what happens when traffic exceeds link capacity.

% -- NARRATE: What to SAY while showing this slide
``Three levels, increasing in precision. PFC: reactive PAUSE frames, can cascade
and freeze the fabric. DCQCN: proactive ECN marking, 85--90\% utilization, but
blind to severity. HPCC: in-network telemetry, 95\%+ utilization, requires
programmable switches.''

% -- ENGAGE: Specific question for THIS slide
``Why is 85\% utilization not good enough for training?''
Expected: on a \$300M cluster, 15\% idle = \$45M wasted. Every percentage point matters.

% -- WARN: What students will get wrong
Students assume PFC is sufficient because it prevents loss. Correct: PFC storms
can cascade and freeze the entire fabric --- worse than dropping packets.

% -- FLEX: [CORE] Congestion control directly impacts training efficiency.
IF SHORT: Name the three levels, state the utilization numbers, skip details.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.92\textwidth,height=4.8cm,keepaspectratio]{congestion-control.pdf}

\vspace{0.1cm}
\mlsysconcept{Progression:}{More precise feedback $\to$ higher link utilization $\to$ less wasted GPU time.}

\end{frame}

\begin{frame}{The Incast Problem}
\note{
[2 min]
% -- LINK: Congestion control handled general congestion; incast is the specific ML pattern
Students saw general congestion mechanisms. Incast is the dominant ML-specific
pattern: many-to-one at every AllReduce completion.

% -- NARRATE: What to SAY while showing this slide
``Incast: hundreds of senders target the same receiver port simultaneously.
This is not rare --- it happens hundreds of times per training step. Buffer
overflow even when the fabric is uncongested, because the destination port
is the bottleneck, not the path.''
Walk through mitigations: ``Layer-staggered AllReduce, tree-based collectives,
deeper switch buffers. Adaptive routing helps but cannot solve incast ---
the destination is the constraint.''

% -- WARN: What students will get wrong
Students think adaptive routing solves incast. Correct: routing distributes
paths but cannot create bandwidth at the destination port.

% -- FLEX: [OPTIONAL] Incast is important for practitioners but less examinable.
IF SHORT: State the pattern (many-to-one), the frequency (hundreds per step),
and the key insight (destination bottleneck, not path bottleneck).
}

\footnotesize
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{Incast}: many-to-one traffic pattern.

    \vspace{0.1cm}
    \begin{itemize}\setlength\itemsep{0pt}
      \item AllReduce: hundreds of senders $\to$ one receiver
      \item Buffer overflow even when fabric is uncongested
      \item \textbf{Deterministic}: happens every training step
    \end{itemize}

    \vspace{0.1cm}
    \textbf{Mitigation:}
    \begin{itemize}\setlength\itemsep{0pt}
      \item Layer-staggered AllReduce (overlap)
      \item Tree-based collective algorithms
      \item Deeper switch buffers (32--64 MB)
    \end{itemize}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{errorstroke}
    {\scriptsize In a 1,024-GPU AllReduce, incast occurs \textbf{hundreds of times per step}.
    A congested port stalls \emph{all} GPUs for 100+ $\mu$s per event.}
    \end{mlsyscard}

    \vspace{0.1cm}
    \begin{mlsyscard}{datastroke}
    {\scriptsize Adaptive routing \emph{helps} but cannot \emph{solve} incast --- the destination
    port is the bottleneck, not the path.}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

% --- ACTIVE LEARNING 3: Discussion ---
\begin{frame}{Discussion: InfiniBand or Ethernet?}
\note{
[3 min]
% -- LINK: IB vs RoCE table provided specs; now students make the actual decision
Students have the data. This forces them to weigh trade-offs.

% -- NARRATE: ``Turn to your neighbor. 10,000 GPUs, \$200M budget. IB or RoCE? 90 seconds.''
Cold-call 2--3 pairs. No single right answer. IB: better tail latency, simpler
lossless. Ethernet: 20--40\% cheaper switches, multi-vendor, shared with serving.

% -- ENGAGE: This IS the active learning moment
Push for reasoning, not just a label. ``What was the deciding factor for your choice?''
Emphasize: the industry is converging (Spectrum-4 adds IB features to Ethernet).

% -- FLEX: [CORE] The IB/Ethernet debate is the chapter's signature discussion.
IF SHORT: Pose to whole class, take 2 answers, state the convergence trend.
}

\centering
\vspace{0.6cm}
{\large\bfseries Turn and Talk \textcolor{midgray}{(90 seconds)}}

\vspace{0.5cm}
{\large You are building a 10,000-GPU training cluster.\\
Your budget is \$200M.\\[0.3cm]
\alert{Do you choose InfiniBand or RoCE (Ethernet)?}}

\vspace{0.5cm}
\begin{columns}[c]
  \begin{column}{0.45\textwidth}\centering\small
    \textcolor{computestroke}{\textbf{InfiniBand}}\\
    {\footnotesize 30--50\% lower tail latency\\Native lossless\\Single vendor}
  \end{column}
  \begin{column}{0.45\textwidth}\centering\small
    \textcolor{routingstroke}{\textbf{RoCE / Ethernet}}\\
    {\footnotesize 20--40\% lower switch cost\\Multi-vendor\\Shared infra with serving}
  \end{column}
\end{columns}

\end{frame}

% =============================================================================
\section{Cluster Design}
% =============================================================================

\begin{frame}{Production Clusters: SuperPOD vs.\ Grand Teton}
\note{
[2 min]
% -- LINK: IB vs RoCE discussion was theoretical; these are real production implementations
Students debated the choice. Now: how NVIDIA and Meta actually built their clusters.

% -- NARRATE: What to SAY while showing this slide
``Two philosophies. SuperPOD: IB-native, 256 GPUs per unit, rail-optimized.
Grand Teton: RoCE on Ethernet, 16,000+ GPUs, aggressive DCQCN tuning.
Both achieve 95\%+ line-rate for large transfers. The choice is operational
philosophy, not performance superiority.''

% -- ENGAGE: ``If you were building a new 10,000-GPU training cluster from scratch
with a clean-slate budget and no existing infrastructure, which implementation
philosophy would you choose and why?'' Let 2--3 students answer before moving
to the Discussion frame that follows.

% -- WARN: What students will get wrong
Students assume one approach dominates. Correct: both work at scale.
The real variable is organizational expertise and existing infrastructure.

% -- FLEX: [OPTIONAL] Production examples are illustrative but not examinable.
IF SHORT: Name the two approaches, state that both achieve 95\%+ line-rate, move on.
}

% --- Layout: FULL-WIDTH IMAGE ---
\centering
\safeimg[width=0.92\textwidth,height=4.8cm,keepaspectratio]{superpod-vs-teton.pdf}

\vspace{0.1cm}
\mlsysinsight{Trend:}{NVIDIA Spectrum-4 adds IB-style adaptive routing to Ethernet. The two ecosystems are converging.}

\end{frame}

\begin{frame}{Virtualization: SR-IOV and Multi-Tenancy}
\note{
[2 min]
% -- LINK: Production clusters showed single-tenant use; SR-IOV enables multi-tenancy
Students saw dedicated clusters. But 30\% idle = \$90M wasted. Virtualization shares the fabric.

% -- NARRATE: What to SAY while showing this slide
``SR-IOV lets a physical NIC present as multiple virtual functions, each with
hardware-isolated DMA queues. Less than 2\% latency overhead vs bare metal.
But bandwidth is strictly partitioned: 8 VFs on a 400G NIC = 50G each.
No bursting beyond your slice. Three isolation dimensions: bandwidth guarantees,
latency determinism, and security (memory snooping prevention).''

% -- WARN: What students will get wrong
Students assume virtualization means sharing = interference. Correct: SR-IOV
provides hardware isolation --- separate DMA queues, not software multiplexing.

% -- FLEX: [OPTIONAL] Important for cloud/multi-tenant environments but less critical for
single-organization clusters.
IF SHORT: State the 2\% overhead number and the bandwidth partitioning constraint.
}

\footnotesize
\textbf{SR-IOV}: NIC presents multiple Virtual Functions (VFs)

\vspace{0.1cm}
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \begin{itemize}\setlength\itemsep{0pt}
      \item Hardware-isolated DMA queues per VM/container
      \item Bypasses hypervisor $\to$ $<$2\% latency overhead
      \item Strict BW partitioning: 8 VFs on 400G = 50G each
    \end{itemize}

    \vspace{0.1cm}
    \textbf{Three isolation dimensions:}
    \begin{itemize}\setlength\itemsep{0pt}
      \item Bandwidth guarantees (min throughput)
      \item Latency determinism (no HoL blocking)
      \item Security (memory snooping prevention)
    \end{itemize}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{routingstroke}
    {\scriptsize A \$300M cluster at 30\% idle = \textbf{\$90M waste}. Virtualization shares
    fabric among training, inference, and preprocessing.}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

% =============================================================================
\section{Monitoring}
% =============================================================================

\begin{frame}{Monitoring and Debugging the Fabric}
\note{
[2 min]
% -- LINK: Fabric behavior showed what can go wrong; monitoring detects it before it is too late
Students know congestion and incast. Now: how to detect degradation in production.

% -- NARRATE: What to SAY while showing this slide
``Network issues are SILENT. They manifest as subtle training slowdowns, not errors.
A 10\% throughput drop can waste thousands of GPU-hours before anyone notices.''
Walk through three layers: ``Link-level: BER predicts transceiver failure 24--48 hrs
ahead. Transport: PFC counters are the canary. Bandwidth: baseline with ib\_write\_bw,
NEVER iperf (iperf measures TCP, not RDMA).''

% -- WARN: What students will get wrong
Students will use iperf to test RDMA networks. A link showing 30 Gbps via
iperf may deliver 390 Gbps via ib\_write\_bw. Always use RDMA-native tools.

% -- FLEX: [CORE] Monitoring prevents millions in wasted compute.
IF SHORT: State the ``silent degradation'' problem and the iperf warning.
}

\footnotesize
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    \textbf{Three telemetry layers:}

    \vspace{0.1cm}
    \textcolor{computestroke}{\textbf{1. Link-level:}}
    {\scriptsize Bit Error Rate, FEC corrections, symbol errors.
    A rising BER predicts transceiver failure 24--48 hrs ahead.}

    \vspace{0.1cm}
    \textcolor{routingstroke}{\textbf{2. Transport-level:}}
    {\scriptsize PFC PAUSE frame rates, PortXmitDiscards, RDMA retransmits.
    Rising PFC counters = ``canary in the coal mine.''}

    \vspace{0.1cm}
    \textcolor{datastroke}{\textbf{3. Bandwidth validation:}}
    {\scriptsize Use \texttt{ib\_write\_bw} / \texttt{ib\_read\_lat} from perftest.
    \alert{Never} use \texttt{iperf} (measures TCP, not RDMA).}
  \end{column}
  \begin{column}{0.42\textwidth}
    \begin{mlsyscard}{errorstroke}
    {\footnotesize Network issues are \emph{silent}. A 10\% throughput drop can waste
    \textbf{thousands of GPU-hours} before detection. Alert on PFC counters and
    track bandwidth baselines weekly.}
    \end{mlsyscard}

    \vspace{0.1cm}
    {\scriptsize\textbf{Debug workflow:}\\
    1. Measure (ib\_write\_bw)\\
    2. Correlate (PFC + BER)\\
    3. Isolate (single link test)\\
    4. Repair + re-baseline}
  \end{column}
\end{columns}

\end{frame}

% --- ACTIVE LEARNING: Ring AllReduce Crossover Exercise ---
\begin{frame}{Your Turn: Ring AllReduce Latency Crossover}
\note{
[3 min]
% -- LINK: Alpha-beta model showed two regimes; this exercise applies it to
Ring AllReduce at real fleet scale to find where latency dominates.

% -- NARRATE: ``A 70B model on 256 GPUs. 90 seconds. Compute the alpha-beta
time for Ring AllReduce over IB NDR. At what GPU count does latency dominate?''
After pause: ``Ring AllReduce time: $T = 2(N-1)/N \times M/\beta + (N-1) \times \alpha$.
M = 280 GB (70B $\times$ 4 bytes mixed precision gradients). $\beta$ = 50 GB/s (IB NDR).
$\alpha$ = 5 $\mu$s per hop. At N=256: bandwidth term = 2 $\times$ 255/256 $\times$ 280/50
= 11.2 s. Latency term = 255 $\times$ 5 $\mu$s = 1.3 ms. Bandwidth dominates by 8,600$\times$.
Latency dominates when $M < N \times \alpha \times \beta$ = 256 $\times$ 5$\mu$s $\times$ 50 GB/s
= 64 KB. For small control messages ($<$64 KB), topology (hops) matters more than link speed.''

% -- ENGAGE: This IS the active learning moment.
After: ``At 4,096 GPUs, the latency term grows to 20 ms. Still small vs 11.2 s.
At what message size does latency = bandwidth?'' This is the crossover $n^*$.

% -- WARN: Students forget the $2(N-1)/N$ factor in Ring AllReduce.
ALSO NOTE: This exercise uses $\alpha = 5\,\mu$s, which is the end-to-end hardware
link latency (IB\_NDR\_LATENCY\_US = 5 $\mu$s from the registry). The $\alpha$-$\beta$
model slides used $\alpha = 1.5\,\mu$s (FABRIC\_ALPHA\_NDR), which is the software
startup parameter. The two measure different things: 1.5 $\mu$s is the per-hop
overhead at the transport layer; 5 $\mu$s captures full end-to-end hardware latency.
The crossover point shifts accordingly (75 KB vs 64 KB here). Flag this explicitly
so students are not confused: ``Note that this exercise uses the hardware one-way
latency of 5 $\mu$s, not the software startup alpha of 1.5 $\mu$s used in the model.''

% -- FLEX: [CORE] This is the chapter's quantitative exercise.
IF SHORT: Give 60 seconds, show the solution, emphasize the 8,600x ratio.
}

\small
\begin{columns}[T]
  \begin{column}{0.55\textwidth}
    {\normalsize\bfseries Ring AllReduce: $\alpha$-$\beta$ Analysis}

    \vspace{0.1cm}
    {\footnotesize $T_{\text{Ring}} = \underbrace{2\frac{N\!-\!1}{N} \cdot \frac{M}{\beta}}_{\text{bandwidth}} + \underbrace{(N\!-\!1) \cdot \alpha}_{\text{latency}}$}

    \vspace{0.15cm}
    \textbf{Given:} 70B model, 256 GPUs
    \begin{itemize}\setlength\itemsep{0pt}
      \item {\footnotesize $M$ = 280 GB (mixed precision gradients)}
      \item {\footnotesize $\beta$ = 50 GB/s (IB NDR)}
      \item {\footnotesize $\alpha$ = 5 $\mu$s per hop}
    \end{itemize}

    \vspace{0.1cm}
    \textbf{Calculate:} Total time and crossover point.

    {\footnotesize\textcolor{midgray}{(90 seconds)}}
  \end{column}
  \begin{column}{0.42\textwidth}
    \pause
    \begin{mlsyscard}{computestroke}
    {\scriptsize \textbf{Solution:}\\
    BW term: $2 \times 255/256 \times 280/50$\\
    = \textbf{11.2 s}\\
    Latency term: $255 \times 5\mu$s\\
    = \textbf{1.3 ms}\\[0.1cm]
    BW dominates by 8,600$\times$!\\[0.1cm]
    Crossover at $M < 64$ KB:\\
    Below 64 KB: optimize \textbf{hops}\\
    Above 64 KB: optimize \textbf{link speed}}
    \end{mlsyscard}
  \end{column}
\end{columns}

\end{frame}

\begin{frame}{Fabric Diagnostic Flowchart}
\note{
[1 min]
% -- LINK: Monitoring showed the three telemetry layers; this flowchart
synthesizes them into a diagnostic procedure.

% -- NARRATE: ``When training slows down mysteriously, follow this flowchart.
Step 1: Check ib\_write\_bw baseline --- has bandwidth degraded? Step 2: Check PFC
counters --- is the fabric under congestion? Step 3: Check BER --- is a transceiver
failing? Step 4: Correlate with training throughput timeline. The root cause is
usually: (a) bad transceiver (replace), (b) PFC storm (isolate offending flow),
or (c) topology mismatch (topology-aware placement).''

% -- FLEX: [OPTIONAL] Reference slide for operational debugging.
IF SHORT: Show the flowchart, name the three root causes, move on.
}

\centering
\safeimg[width=0.88\textwidth,height=4.2cm,keepaspectratio]{fabric-diagnostic-flowchart.pdf}

\vspace{0.1cm}
\small
\mlsysconcept{Diagnostic order:}{Bandwidth baseline $\to$ PFC counters $\to$ BER trends $\to$ correlate with training throughput.}

\end{frame}

% =============================================================================
\section{Wrap-Up}
% =============================================================================

\begin{frame}{Fallacies}
\note{
[2 min]
% -- LINK: Each fallacy points back to where it was disproved in this lecture:
(1) bandwidth fallacy --- disproved by the 10 KB $\alpha$-$\beta$ worked example;
(2) oversubscription fallacy --- disproved by the bisection bandwidth calculation;
(3) adaptive routing fallacy --- disproved by the incast slide;
(4) IB=fast-Ethernet fallacy --- disproved by the IB vs RoCE comparison table.

% -- NARRATE: Four claims proven wrong with numbers from this lecture.
Spend extra time on the first: ``We showed the 10 KB message only improves 11\%
with 2x bandwidth. The alpha-beta model proves this quantitatively.''
On the oversubscription fallacy: ``Enterprise IT staff are trained to accept 4:1
oversubscription because web traffic is statistical. BSP training is not ---
every node injects simultaneously.''

% -- ENGAGE: Ask students which fallacy they found hardest to unlearn.
``Turn to your neighbor: which of these four fallacies is most likely to survive
in your intuition? Why?'' Take 2--3 responses. The oversubscription fallacy
typically persists because enterprise IT default is 4:1 oversubscription.

% -- WARN: The oversubscription fallacy is particularly persistent because
enterprise IT staff are trained to accept it as a cost-optimization. The IB=fast-Ethernet
fallacy persists because procurement compares spec-sheets, not system behavior.

% -- FLEX: [CORE] Fallacies correct misconceptions before they solidify.
IF SHORT: Cover the first fallacy (bandwidth) and the IB fallacy, skip the other two.
}

\scriptsize
\textbf{Fallacy:} \textit{More bandwidth always means faster training.}\\
{\scriptsize For a 10 KB message, upgrading from 200G to 400G yields only 11\% improvement.
The $\alpha$ term dominates small messages. Doubling bandwidth adds 40\% to cost for negligible gain.}

\vspace{0.1cm}
\textbf{Fallacy:} \textit{Network oversubscription is acceptable if ``most'' traffic is local.}\\
{\scriptsize BSP training injects simultaneously from \emph{all} nodes. 4:1 oversubscription $\to$ 4$\times$ slower AllReduce. On a \$300M cluster (30\% sync), this wastes \$142M+ in idle GPU-hours.}

\vspace{0.1cm}
\textbf{Fallacy:} \textit{Adaptive routing eliminates topology-aware placement.}\\
{\scriptsize Adaptive routing balances traffic but cannot create bandwidth that does not exist. A 1,024-GPU job across oversubscribed spine groups will still be throttled by cross-group links.}

\vspace{0.1cm}
\textbf{Fallacy:} \textit{InfiniBand is just fast Ethernet.}\\
{\scriptsize Procurement teams compare on link rate alone. IB provides kernel-bypass RDMA, hardware credit-based flow control, and bounded failure modes by design. Ethernet requires PFC/ECN approximations to behave losslessly --- a software-configuration discipline that fails catastrophically in 200 ms when misconfigured. The choice is a system architecture decision, not a bandwidth selection.}

\end{frame}

\begin{frame}{Pitfalls}
\note{
[1.5 min]
% -- LINK: Pitfall 1 links to the PFC Deadlock slide; Pitfall 2 links to the Monitoring
slide (ib\_write\_bw vs iperf); Pitfall 3 links to the Monitoring slide's canary discussion.

% -- NARRATE: Three operational mistakes. Spend extra time on the PFC storm pitfall ---
``A misconfigured switch can freeze the entire fabric in 200 ms. Monitor PFC counters.''
The iperf pitfall directly connects to the monitoring slide. The counter-monitoring
pitfall is the most financially costly: silent degradation can waste thousands of
GPU-hours before anyone notices.

% -- ENGAGE: Ask which pitfall the class thinks is most common in production.
``Quick show of hands: which of these three pitfalls do you think is most common
in real production deployments?'' Take 2--3 responses. Answer: PFC storm + iperf
testing co-occur frequently --- teams discover their network is degraded because
training slows and they test with iperf, see full speed, conclude the network is fine.

% -- WARN: The iperf pitfall is a trap because iperf DOES return good numbers even
when RDMA is broken. A team that tests with iperf and sees 30 Gbps may conclude the
network is fine, while ib\_write\_bw shows 390 Gbps --- meaning the kernel network
is working but the RDMA path is misconfigured.

% -- FLEX: [CORE] Pitfalls prevent expensive operational mistakes.
IF SHORT: Cover the lossless Ethernet pitfall only.
}

\footnotesize
\textbf{Pitfall:} \textit{Assuming lossless Ethernet is as reliable as InfiniBand.}\\
RoCE approximates losslessness via PFC. A misconfigured switch can trigger a PFC storm, freezing the entire fabric in 200 ms. InfiniBand's credit-based flow control is immune.

\vspace{0.15cm}
\textbf{Pitfall:} \textit{Testing network performance with \texttt{iperf} instead of RDMA tools.}\\
\texttt{iperf} measures TCP (kernel-based). A link showing 30 Gbps via \texttt{iperf} may deliver 390 Gbps via \texttt{ib\_write\_bw}. Always validate with RDMA-native tools from \texttt{perftest}.

\vspace{0.15cm}
\textbf{Pitfall:} \textit{Neglecting PFC and ECN counter monitoring in production.}\\
A gradual increase in PFC PAUSE frames or PortXmitDiscards is the canary for a failing transceiver or routing imbalance. A 10\% throughput drop can waste thousands of GPU-hours before detection.

\end{frame}

% --- RETRIEVAL PRACTICE ---

% --- MUDDIEST POINT ---
\begin{frame}{Muddiest Point}
\note{
[1 min]
% -- NARRATE: ``Write the one concept you found most confusing. Anonymous. One sentence.''
% -- FLEX: [CORE] Address top 2--3 confusions in next lecture's opening.
}

\centering
\vspace{1.0cm}
{\Large\bfseries What was the \alert{muddiest point} today?}

\vspace{0.8cm}
{\normalsize Write down the concept you found \textbf{most confusing}.}

\vspace{0.5cm}
{\small\textcolor{midgray}{Anonymous. One sentence. Submit before you leave.}}

\end{frame}

\begin{frame}{What Were the Key Ideas?}
\note{
[2 min]
% -- NARRATE: ``Close your notes. Write four key concepts. 90 seconds.''
Walk around while students write. Do NOT show Key Takeaways yet.
% -- FLEX: [CORE] Retrieval practice.
}

\centering
\vspace{1.5cm}
{\Large\bfseries Close your notes.}

\vspace{0.8cm}
{\large Write down the \textbf{4 most important concepts} from today.}

\vspace{0.8cm}
{\normalsize\textcolor{midgray}{90 seconds --- no peeking.}}

\end{frame}

\begin{frame}{Key Takeaways}
\note{
[2 min]
% -- LINK: Students wrote their own list; now compare.
% -- NARRATE: ``Check your list. Did you capture: 18x cliff, 75 KB crossover,
95\% HPCC utilization, 4x oversubscription penalty?''
% -- FLEX: [CORE] Official summary. Read every bullet.
}

\footnotesize
\begin{itemize}\setlength\itemsep{1pt}
  \item \textbf{Network as Computer}: At scale, the interconnect determines performance. The 18$\times$ NVLink-to-IB cliff is the central design challenge.
  \item \textbf{$\alpha$-$\beta$ Framework}: $T(n) = \alpha + n/\beta$. Messages $<$75 KB are latency-bound (IB NDR crossover); gradients ($>$MB) are bandwidth-bound.
  \item \textbf{Lossless is Non-Negotiable}: RDMA requires zero packet loss. IB provides natively; Ethernet must approximate via PFC/ECN.
  \item \textbf{Topology Choice}: Fat-trees provide full bisection BW; rail-optimized minimizes latency; dragonflies reduce cabling at scale.
  \item \textbf{Congestion Control}: PFC $\to$ DCQCN (85\%) $\to$ HPCC (95\%+). More precise feedback = higher utilization.
  \item \textbf{Monitor or Waste}: PFC counters, BER trends, and RDMA bandwidth baselines detect silent network degradation.
\end{itemize}

\end{frame}

\begin{frame}{References}
\note{
[0.5 min]
% -- NARRATE: ``Start with Leiserson for fat-trees and DCQCN for congestion control.''
% -- FLEX: [OPTIONAL] Skip if running short.
}

\small
\mlsysref{Leiserson85}{C. Leiserson. ``Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing.'' IEEE Trans.\ Comp., 1985.}
\mlsysref{Kim+08}{J. Kim, W. Dally et al.\ ``Technology-Driven, Highly-Scalable Dragonfly Topology.'' ISCA 2008.}
\mlsysref{DCQCN15}{Y. Zhu et al.\ ``Congestion Control for Large-Scale RDMA Deployments.'' SIGCOMM 2015.}
\mlsysref{HPCC19}{Y. Li et al.\ ``HPCC: High Precision Congestion Control.'' SIGCOMM 2019.}
\mlsysref{GrandTeton}{Meta. ``Grand Teton: Training Infrastructure at Scale.'' 2023.}

\end{frame}

\begin{frame}{Next Lecture: Data Storage}
\note{
[1 min]
% -- LINK: Network is wired. Next: the data that feeds it.
% -- NARRATE: ``The fleet is wired. But compute and network are useless without data.
Next chapter: parallel storage, data-loading pipelines, and checkpoint strategies
that keep the fleet supplied. Central question: how do you feed PetaFLOPS of compute
from storage that is 500x slower?''
% -- FLEX: [CORE] Forward hook.
IF SHORT: State the central question and dismiss.
}

\footnotesize
\begin{columns}[c]
  \begin{column}{0.30\textwidth}
    \centering
    {\large\bfseries\textcolor{computestroke}{Storage}}\\[0.3cm]
    {\footnotesize Parallel file systems\\NVMe, SSDs\\Checkpointing}
  \end{column}
  \begin{column}{0.30\textwidth}
    \centering
    {\large\bfseries\textcolor{routingstroke}{Data Loading}}\\[0.3cm]
    {\footnotesize Prefetching pipelines\\Data locality\\I/O bottlenecks}
  \end{column}
  \begin{column}{0.30\textwidth}
    \centering
    {\large\bfseries\textcolor{datastroke}{Checkpoints}}\\[0.3cm]
    {\footnotesize State preservation\\Fault recovery\\Storage hierarchy}
  \end{column}
\end{columns}

\vspace{0.3cm}
\centering
{\normalsize The fleet is wired. How do we keep it fed with data?}\\[0.1cm]
{\small\textbf{Massive datasets and enormous checkpoints need parallel storage.}}

\end{frame}



\appendix

\begin{frame}{Backup: $\alpha$-$\beta$ Quick Reference}
\note{
[1 min]
% -- NARRATE: Backup reference for alpha-beta calculations.
% -- FLEX: [OPTIONAL] Only show if requested.
}

\footnotesize
\textbf{Communication Cost Models:}

\vspace{0.15cm}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{}ll@{}}
  \toprule
  \textbf{Model} & \textbf{Formula} \\
  \midrule
  Point-to-point & $T(n) = \alpha + n/\beta$ \\
  Ring AllReduce & $T = 2\frac{N-1}{N} \cdot \frac{M}{\beta} + (N-1) \cdot \alpha$ \\
  Tree AllReduce & $T = 2\log_2(N) \cdot (\alpha + M/(2\beta))$ \\
  Crossover ($n^*$) & $n^* = \alpha \cdot \beta$ \\
  \bottomrule
\end{tabular}

\vspace{0.15cm}
\textbf{Canonical $\alpha$ values:} IB HDR/NDR: 1.5 $\mu$s. RoCE: 3--5 $\mu$s.\\
\textbf{Canonical $\beta$ values:} IB HDR: 25 GB/s. IB NDR: 50 GB/s. NVLink 4.0: 900 GB/s.

\end{frame}

\begin{frame}{Backup: PFC Deadlock and Congestion Control Reference}
\note{
[1 min]
% -- NARRATE: Extended reference for congestion control mechanisms.
% -- FLEX: [OPTIONAL] Use if students want to understand DCQCN vs HPCC deeply.
}

\footnotesize
\textbf{Congestion Control Progression:}

\vspace{0.15cm}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}llrl@{}}
  \toprule
  \textbf{Mechanism} & \textbf{Signal} & \textbf{Utilization} & \textbf{Risk} \\
  \midrule
  PFC (reactive) & PAUSE frames & 70--80\% & PFC storm cascade \\
  DCQCN (proactive) & ECN marking & 85--90\% & Blind to severity \\
  HPCC (telemetry) & In-network INT & 95\%+ & Requires programmable switches \\
  \bottomrule
\end{tabular}

\vspace{0.15cm}
{\footnotesize \textbf{PFC Storm Prevention:} Watchdog timers (disable PFC on stuck port), separate VLANs for training vs storage, DCQCN to prevent buffer filling.}

\end{frame}


\end{document}
```
