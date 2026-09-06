---
title: Stanford CS336 — Language Modeling from Scratch, Spring 2026
type: source-note
status: verified
last_verified: 2026-09-04
---

# Stanford CS336 — Language Modeling from Scratch, Spring 2026

Это закреплённый source layer официального курса, а не основной учебный маршрут.
Последовательное изучение идёт по каноническим главам учебника; здесь сохранены
оригинальные англоязычные материалы, их dependency closure и аудит границ.

Статус: **inventory complete; editorial integration active**. Semantic extraction
остаётся воспроизводимой из immutable archive; редакционные решения наложены
отдельным проверяемым overlay и не переписывают source semantics.

## Реестры аудита

- [source-manifest.yml](source-manifest.yml) — объекты и pinned revisions;
- [semantic-review.json](semantic-review.json) — reviewed per-source semantic boundaries;
- [editorial-map.yml](editorial-map.yml) — persistent reviewed destinations and explicit deferrals;
- [source-units.yml](source-units.yml) — semantic extraction index;
- [coverage.yml](coverage.yml) — одна строка покрытия на каждый source unit;
- [visuals.yml](visuals.yml) — semantic figure/table/code-trace/derivation sequences;
- [artifact-inventory.json](artifact-inventory.json) — SHA-256 каждого файла;
- [snapshot-lock.json](snapshot-lock.json) — SHA репозиториев и архивов.

Извлечение executable lectures идёт из архивированных edtrace renderings;
PDF/handout spans проверяются Poppler. Версии инструментов, extractor SHA,
ledger SHA/counts и parent checksums записаны в `snapshot-lock.json`/visual ledger.
Raw page/raster detections служат evidence и не становятся отдельными teaching visuals.

## Pinned revisions

- `lectures`: [`8b59b50730766695c2ffedd1a79c50cd09b9eb91`](https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91)
- `assignment1-basics`: [`a158843b20107949f1a8d7df1b05cd33b9166712`](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712)
- `assignment2-systems`: [`ca8bc81a59b70516f7ebb2da4808daade877c736`](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)
- `assignment3-scaling`: [`03e9372992e913061b9e78b5cfcb62ad8a87de35`](https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35)
- `assignment4-data`: [`0555bea66369872d912652debf10b115ca0688c8`](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8)
- `assignment5-alignment`: [`c2734a26308710949fe13226960a1e8cece94b7e`](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e)
- `stanford-cs336.github.io`: [`25d740fd9060cc6613163b5b88ca88a5f64138ff`](https://github.com/stanford-cs336/stanford-cs336.github.io/tree/25d740fd9060cc6613163b5b88ca88a5f64138ff)

## 19 встреч курса

| № | Дата | Тема | Артефакт |
|---:|---|---|---|
| 1 | 2026-03-30 | Overview, tokenization — Percy Liang | [`lecture_01.py`](Lectures/repository/lecture_01.py) |
| 2 | 2026-04-01 | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) — Percy Liang | [`lecture_02.py`](Lectures/repository/lecture_02.py) |
| 3 | 2026-04-06 | Architectures, hyperparameters — Tatsunori Hashimoto | [`lecture_03.pdf`](Lectures/repository/lecture_03.pdf) |
| 4 | 2026-04-08 | Attention alternatives and mixture of experts — Tatsunori Hashimoto | [`lecture_04.pdf`](Lectures/repository/lecture_04.pdf) |
| 5 | 2026-04-13 | GPUs, TPUs — Tatsunori Hashimoto | [`lecture_05.pdf`](Lectures/repository/lecture_05.pdf) |
| 6 | 2026-04-15 | Kernels, Triton — Percy Liang | [`lecture_06.py`](Lectures/repository/lecture_06.py) |
| 7 | 2026-04-20 | Parallelism — Percy Liang | [`lecture_07.py`](Lectures/repository/lecture_07.py) |
| 8 | 2026-04-22 | Parallelism — Tatsunori Hashimoto | [`lecture_08.pdf`](Lectures/repository/lecture_08.pdf) |
| 9 | 2026-04-27 | Scaling laws — Tatsunori Hashimoto | [`lecture_09.pdf`](Lectures/repository/lecture_09.pdf) |
| 10 | 2026-04-29 | Inference — Percy Liang | [`lecture_10.py`](Lectures/repository/lecture_10.py) |
| 11 | 2026-05-04 | Scaling laws — Tatsunori Hashimoto | [`lecture_11.pdf`](Lectures/repository/lecture_11.pdf) |
| 12 | 2026-05-06 | Evaluation — Percy Liang | [`lecture_12.py`](Lectures/repository/lecture_12.py) |
| 13 | 2026-05-11 | Data (sources, datasets) — Percy Liang | [`lecture_13.py`](Lectures/repository/lecture_13.py) |
| 14 | 2026-05-13 | Data (filtering, deduplication, mixing, synthetic data) — Percy Liang | [`lecture_14.py`](Lectures/repository/lecture_14.py) |
| 15 | 2026-05-18 | Mid/post-training (SFT/RLHF) — Tatsunori Hashimoto | [`lecture_15.pdf`](Lectures/repository/lecture_15.pdf) |
| 16 | 2026-05-20 | Post-training - RLVR — Tatsunori Hashimoto | [`lecture_16.pdf`](Lectures/repository/lecture_16.pdf) |
| 17 | 2026-05-27 | Alignment - multimodality — Percy Liang | [`lecture_17.py`](Lectures/repository/lecture_17.py) |
| 18 | 2026-06-01 | Guest lecture: Daniel Selsam — Daniel Selsam | официальный артефакт не опубликован; слот зафиксирован явно |
| 19 | 2026-06-03 | Guest lecture: Dan Fu — Dan Fu | официальный артефакт не опубликован; слот зафиксирован явно |

## Assignments 1–5

- Assignment 1: [Basics](Assignments/assignment1-basics/README.md), pinned at `a158843b20107949f1a8d7df1b05cd33b9166712`.
- Assignment 2: [Systems](Assignments/assignment2-systems/README.md), pinned at `ca8bc81a59b70516f7ebb2da4808daade877c736`.
- Assignment 3: [Scaling](Assignments/assignment3-scaling/README.md), pinned at `03e9372992e913061b9e78b5cfcb62ad8a87de35`.
- Assignment 4: [Data](Assignments/assignment4-data/README.md), pinned at `0555bea66369872d912652debf10b115ca0688c8`.
- Assignment 5: [Alignment and Reasoning RL](Assignments/assignment5-alignment/README.md), pinned at `c2734a26308710949fe13226960a1e8cece94b7e`.
- Assignment 5 optional branch: [safety, instruction tuning, and RLHF supplement](Assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf). Это объект внутри pinned `assignment5-alignment`, а не шестой репозиторий.

## Объекты source layer

<a id="lecture-01"></a>
### lecture-01: Overview, tokenization

- local path: `Lectures/repository/lecture_01.py`;
- extracted units: 54;
- semantic visual rows: 15;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-02"></a>
### lecture-02: PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)

- local path: `Lectures/repository/lecture_02.py`;
- extracted units: 24;
- semantic visual rows: 8;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-03"></a>
### lecture-03: Architectures, hyperparameters

- local path: `Lectures/repository/lecture_03.pdf`;
- extracted units: 11;
- semantic visual rows: 4;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-04"></a>
### lecture-04: Attention alternatives and mixture of experts

- local path: `Lectures/repository/lecture_04.pdf`;
- extracted units: 17;
- semantic visual rows: 11;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-05"></a>
### lecture-05: GPUs, TPUs

- local path: `Lectures/repository/lecture_05.pdf`;
- extracted units: 12;
- semantic visual rows: 6;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-06"></a>
### lecture-06: Kernels, Triton

- local path: `Lectures/repository/lecture_06.py`;
- extracted units: 21;
- semantic visual rows: 9;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-07"></a>
### lecture-07: Parallelism

- local path: `Lectures/repository/lecture_07.py`;
- extracted units: 12;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-08"></a>
### lecture-08: Parallelism

- local path: `Lectures/repository/lecture_08.pdf`;
- extracted units: 11;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-09"></a>
### lecture-09: Scaling laws

- local path: `Lectures/repository/lecture_09.pdf`;
- extracted units: 10;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-10"></a>
### lecture-10: Inference

- local path: `Lectures/repository/lecture_10.py`;
- extracted units: 31;
- semantic visual rows: 14;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-11"></a>
### lecture-11: Scaling laws

- local path: `Lectures/repository/lecture_11.pdf`;
- extracted units: 8;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-12"></a>
### lecture-12: Evaluation

- local path: `Lectures/repository/lecture_12.py`;
- extracted units: 36;
- semantic visual rows: 29;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-13"></a>
### lecture-13: Data (sources, datasets)

- local path: `Lectures/repository/lecture_13.py`;
- extracted units: 16;
- semantic visual rows: 8;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-14"></a>
### lecture-14: Data (filtering, deduplication, mixing, synthetic data)

- local path: `Lectures/repository/lecture_14.py`;
- extracted units: 15;
- semantic visual rows: 10;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-15"></a>
### lecture-15: Mid/post-training (SFT/RLHF)

- local path: `Lectures/repository/lecture_15.pdf`;
- extracted units: 9;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-16"></a>
### lecture-16: Post-training - RLVR

- local path: `Lectures/repository/lecture_16.pdf`;
- extracted units: 9;
- semantic visual rows: 5;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="lecture-17"></a>
### lecture-17: Alignment - multimodality

- local path: `Lectures/repository/lecture_17.py`;
- extracted units: 10;
- semantic visual rows: 7;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-01"></a>
### assignment-01: Assignment 1: Basics

- local path: `Assignments/assignment1-basics`;
- extracted units: 149;
- semantic visual rows: 1;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-02"></a>
### assignment-02: Assignment 2: Systems

- local path: `Assignments/assignment2-systems`;
- extracted units: 101;
- semantic visual rows: 3;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-03"></a>
### assignment-03: Assignment 3: Scaling

- local path: `Assignments/assignment3-scaling`;
- extracted units: 16;
- semantic visual rows: 1;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-04"></a>
### assignment-04: Assignment 4: Data

- local path: `Assignments/assignment4-data`;
- extracted units: 70;
- semantic visual rows: 1;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-05"></a>
### assignment-05: Assignment 5: Alignment and Reasoning RL

- local path: `Assignments/assignment5-alignment`;
- extracted units: 87;
- semantic visual rows: 1;
- disposition is recorded per unit/visual in the generated coverage ledgers.

<a id="assignment-05-safety-supplement"></a>
### assignment-05-safety-supplement: Assignment 5 optional supplement: safety, instruction tuning, and RLHF

- local path: `Assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`;
- extracted units: 78;
- semantic visual rows: 1;
- disposition is recorded per unit/visual in the generated coverage ledgers.

## Права и атрибуция

Пользователь подтвердил открытое образовательное переиспользование для локального
сохранения и последующего атрибутированного переноса. Это зафиксировано как
`rights_status: permission-recorded`, а не как название лицензии. Явные
LICENSE-файлы репозиториев сохранены в оригинальных snapshot directories;
права на заимствованные upstream figures всё равно проверяются пообъектно.
