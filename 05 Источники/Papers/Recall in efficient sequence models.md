---
title: "Recall в эффективных sequence models: Zoology, Repeat After Me и Mamba-based LMs"
type: source-note
status: reviewed
source_type: paper-cluster
last_verified: 2026-08-31
primary_sources:
  - https://arxiv.org/abs/2312.04927
  - https://arxiv.org/abs/2402.01032
  - https://arxiv.org/abs/2406.07887
concepts: [associative recall, MQAR, copying, phonebook, hybrid models]
---

# Recall в эффективных sequence models

## Общий вопрос

Что происходит, когда контекст содержит много пар ключ–значение, а конкретный
запрос появляется только после них? Архитектура с адресуемым attention может
отложить выбор записи. Fixed-state model должна заранее сжать все потенциально
полезные пары.

## Три взаимодополняющих источника

**Zoology** вводит MQAR и связывает значительную часть разрыва языкового качества
между attention и gated-convolution models с associative recall. В контролируемых
экспериментах sparse attention почти закрывает этот разрыв, но результат относится
к исследованным архитектурам и данным.

**Repeat After Me** даёт теоретические и экспериментальные ограничения fixed-size
latent state на copying и retrieval. Его вывод не равен утверждению «SSM не умеет
длинный контекст»: ограничение относится к точному воспроизведению при высокой
информационной плотности.

**An Empirical Study of Mamba-based Language Models** сравнивает 8B Transformer,
Mamba, Mamba-2 и hybrid при общих данных. Чистые SSM отстают на copying,
in-context learning и phonebook; гибрид с небольшой долей attention заметно
улучшает recall.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/phonebook-pure-figure3.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/phonebook-hybrid-figure7.png]]

## Методологический вывод

Formal context length, perplexity и Needle-in-a-Haystack не заменяют copying,
MQAR, state tracking и multi-hop retrieval. Проверка должна повторять операцию
памяти целевого приложения. Нельзя переносить один synthetic result на любое
длинноконтекстное рассуждение.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|ограничения сжатого состояния]] ·
[[02 Areas/ML & DL/03 Исследовательские линии/Альтернативы полному attention и stateful sequence models|обзор направления]]
