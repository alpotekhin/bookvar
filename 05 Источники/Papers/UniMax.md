---
title: "UniMax: Fairer and more Effective Language Sampling for Large-Scale Multilingual Pretraining"
type: source-note
status: canonical
last_verified: 2026-09-22
source_url: https://arxiv.org/abs/2304.09151v1
year: 2023
---

# UniMax

Метод распределяет обучающий бюджет между языками с ограничением числа
повторений каждого языкового корпуса. Это позволяет стремиться к более
равномерному покрытию, не повторяя малые корпуса неограниченно.

В [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|главе о синтетических данных и учебных программах]]
используется принцип ограничения повторений из
[статьи Chung et al.](https://arxiv.org/abs/2304.09151v1) и его разбор
в Stanford CS336 Lecture 14. Ограничение на долю источника выводится
из отношения предъявленных токенов к доступному объёму; это не полная
реализация алгоритма UniMax.
