---
title: "CS336 — Tokenization"
type: source-note
status: legacy
source_only: true
course: "Stanford CS336"
last_updated: 2026-09-07
canonical_target: "[[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]]"
robots: noindex
search_exclude: true
---

# CS336 — Tokenization

> [!note] Историческая карточка источника
> Эта страница больше не является отдельной учебной главой. Материал о BPE,
> byte-level vocabulary, pre-tokenization, Unigram и выборе словаря перенесён в
> канонический раздел учебника и сверен с закреплённой версией Stanford CS336.

**Основная глава:** [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]].

В ней BPE разобран на одном корпусе шаг за шагом; отдельно показаны обучение и
применение токенизатора, переход от Unicode к UTF-8 bytes, регулярное
предварительное разбиение GPT, WordPiece, Unigram, SentencePiece, специальные
токены, многоязычность и способы сравнения токенизаторов.

**Архив курса:** [[02 Areas/ML & DL/05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-01|Lecture 1 — Overview, tokenization]].

**Практика:** [[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля#Этап 1. Байты, BPE и токенизатор|реализация byte-level BPE и проверка round-trip]].

Старая заметка сохранена как адрес для Obsidian-ссылок. Учебный текст следует
читать только по канонической главе: там устранены неточные универсальные
утверждения о размере словаря и стоимости attention, а каждое сравнение связано
с конкретным корпусом и метрикой.
