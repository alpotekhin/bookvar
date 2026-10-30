# MLOps — resource scheduling and tenant isolation

Canonical chapter: `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md`.
Preview: `/textbook/deployment/mlops/#resource-scheduling-and-tenant-isolation`.
Final 1656 lines, SHA-256
`0f76754dfcd1c28d515724662fac537ad978f7acfa47f0138a3b8cd63aa5f251`.

## Preservation and selected explanation review

The preceding 1312-line chapter had been read in full and its accepted
SHA was freshly rechecked before this extension. The final added section
and its transitions into the surrounding platform/economics explanation
were read in full. The preservation check removes exactly one marked
addition and restores the accepted SHA
`57ee342a14d420a5c97a7a48511dfe1efa430bcb29d3c207c8e3e676dc51739e`.
No earlier prose, formulas, figures or RL chapter was rewritten.

Full source scopes, including their listings and related diagrams, are
recorded in `05 Источники/Originals/platform-resources-2026-10-26/README.md`.
Harvard resource-management/isolation and infrastructure-management sections
were compared against the accepted fleet-orchestration and measurement
chapters. The latter already explain fragmentation, gang allocation,
checkpoint recovery and weighted fleet metrics; they are cross-linked.
Full Kueue Concepts, Cluster Queue and Cohort pages and the complete
Kubernetes multi-tenancy page were read. NVIDIA readings are the complete
selected sharing sections, not a claim to have reviewed the entire SDK,
every GPU or the full operational deployment procedure.

| Mechanism | Source and retained reasoning | Checked case / qualification |
|---|---|---|
| Training versus serving resource policy | Two full original Harvard paragraphs | Completed checkpoint versus process restart; a scaling decision is not ready serving capacity. Existing detailed recovery chapter retained. |
| Workload admission versus placement | Original Kueue component SVG, quota distinction and full basic YAML | 8 CPUs / 40 GiB / 4 pods fails the 9 CPU / 36 GiB / 5 pod quota in memory. Additional checks and placement still apply. Empty namespace selector is identified as permissive. No YAML is applied. |
| Borrowing/lending and priority | Full Kueue examples and original cohort/flavor SVG | Nominal 9+12 CPU pool, A borrowing cap 1, B lending cap 1 and additional/used-capacity cases. Full resource/flavor/hierarchy limits are distinguished from the one-resource two-queue toy calculation. Workload and pod priority are separated; FIFO ordering/defaults are pinned. |
| GPU sharing | Original NVIDIA time-slicing and MPS excerpts, comparison table | Eight physical GPUs × ten references = 80 advertised units, not 80 physical devices or equal HBM partitions. 25+25 GiB exceeds a declared 40 GiB budget; 8+2.5 GiB exceeds a nominal 10 GiB client limit. Plugin mode/version restrictions and software versus hardware isolation remain explicit. |
| Tenant boundary | Original Kubernetes diagram and namespace paragraph | Team-level ownership differs from customers inside one process. RBAC, network enforcement, storage, quotas, host/GPU memory and virtual-control-plane boundaries have distinct mechanisms. |
| Request admission | Editorial application-level extension | Same 10 RPS with 100 versus 10,000 input tokens gives 1000 versus 100,000 input tokens/s; no corresponding latency ratio is asserted. Batch quota admission is not HTTP scheduling; authenticated identity, queue/concurrency/token/cache limits are separate. |

Eight excerpts retain source language and words; cross-link/emphasis markup
and the named footnote marker are removed where noted. The original basic
YAML is byte-identical. A source typo in the quoted quota-reservation
sentence is retained rather than silently rewritten. Editorial examples
and the limits of their models are identified beside the text.

The no-ai-slop review was separate from source/logic checks. Technical
contrasts, terms, caveats and worked cases are retained. No claim that the
entire MLOps source chapters/labs/decks are finished follows from this unit.

## Source figures and rights

Two original Kueue SVGs and one original Kubernetes PNG were added unchanged.
Their full source viewBoxes, raster resolution, source/page hashes, author
attribution, pinned revisions and actual licenses are retained in the
manifest/registry. Kueue is Apache-2.0; the Kubernetes image is CC BY 4.0.
Harvard-derived narrative/examples explicitly use CC BY-NC-SA 4.0. Public
copies of the source licenses are included. No blanket license replaces
the previous Google/SRE/Feast objects' licenses.

The three NVIDIA images were inspected and not used: two small 415px A100
profile images are poor candidates for enlargement, and the third is a
dropdown screenshot. The surrounding source's imprecise compute geometry
and Harvard's universal utilization/physical-isolation claims were not
transferred as architectural facts. An accurate, high-resolution MIG
diagram remains a visual follow-up; no homemade replacement is inserted.

## Executed evidence on final text

- The printed quota Python fence executed. An independent enumeration of
  own/borrowed allocations checks 500 generated pool states plus invalid
  inputs. This is arithmetic, not an execution of Kueue.
- All ten chapter Python fences and the previous 1400 mechanism cases pass.
  The earlier feature-store checker still reconstructs its accepted base;
  both its Python fences / 300 cases and three actual SQL blocks in DuckDB
  1.4.1 were rerun, including 800 oracle prediction rows.
- Final unit suite: 504 passed in 58 files. Final build: 1967 pages,
  23.15 seconds. Final output tests: 135 passed.
- Built internal links: zero broken routes, fragments or files. Source
  coverage passes 210 matrix rows / 56 destinations; this does not establish
  semantic completeness. Scoped git diff --check passes.

Logs: `/var/tmp/bookvar-platform-resources-{unit-final,build-final,
output-final,links-final,coverage-final,examples-final,
existing-examples-final,feature-examples-final}-2026-10-26.log`.

## Actual rendered inspection

The latest build was explicitly reloaded before UI inspection. Desktop
1280×900 and phone 360×800 were inspected. All 13 chapter images load; the
three new originals were visually compared with their source previews.
SVG text remains crisp in the viewer. Phone inline diagrams are overviews;
their small labels require zoom. At 3×, labels were readable and horizontal
panning exposed the first job and final ClusterQueue boxes. The GPU table
has 480px of content inside 313px and keyboard scrolling reached 167/167.
Document width was 345px at a 360px viewport; no page-wide overflow or
KaTeX errors were observed. The other new diagram/PNG were also opened
at 3× on phone.

EN was checked as an explicitly labelled fallback containing the new
section and all three images, not an authored translation. No EN-body
files changed. RU preview was left open and viewport overrides reset.
Screenshots: `docs/audit/screenshots/2026-10-26-platform-resources-*.png`.

## Remaining work and scope

Full H-V1-14/H-V2-12/labs/slides integration, high-quality MIG illustration
and the wider source/rights audit remain open. FinOps, cost attribution and
the remaining lab/deck explanations need a thematic gap pass rather than
being closed by archive presence. Practice 17 still lacks the separately
reported container/RPC/load experiment. No live Kubernetes/Helm, GPU,
production scheduling, translation, stage, commit, push or deployment was
performed. The global textbook goal remains active.
