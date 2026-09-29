---
title: "s1: Simple test-time scaling"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2501.19393v3
authors: [Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, Tatsunori Hashimoto]
year: 2025
---

# s1: Simple test-time scaling

В работе объединены две разные процедуры: SFT на небольшом наборе s1K
и управление продолжительностью генерации при применении модели. Первая
разобрана в [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#s1-59-тысяч-1-тысяча|главе о дистилляции]].
Отбор по формату, трудности и предметной области нельзя приравнивать к проверке
правильности каждой цепочки рассуждений.

Проверены §2.2 и приложения A, C.3, C.4 [версии v3](https://arxiv.org/abs/2501.19393v3).
Существенные уточнения: отдельно отбирались 384 примера; оставшиеся выбирались
по предметным областям с предпочтением длинных решений. Claude 3.5 использовался
при отборе, а Claude 3.7 — при проверке корректности финальных наборов.
s1K-1.1 содержит те же вопросы с заново полученными ответами DeepSeek-R1.

Код и данные: [simplescaling/s1, проверенная ревизия](https://github.com/simplescaling/s1/tree/77272c6e925d610257a50b520bad15330b513389).

Budget forcing подробно разобран в
[[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#budget-forcing-заставить-траекторию-продолжиться|главе о вычислениях во время ответа]].
Проверены §3.1, §4.2 и генерация двух стадий в авторском evaluation harness:
режим игнорирования завершающих маркеров не задаёт точного минимального
числа токенов. Строка `Wait` — обычный текст; финальный ответ получает
отдельный этап генерации.

В эту главу перенесён полный авторский пример с подсчётом букв, без
перерисовки и изменения текста. Источник — файл
[`visuals/raspberry_single.pdf`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/visuals/raspberry_single.pdf)
из репозитория, не PDF статьи. Лицензия этого файла — Apache-2.0;
её копия включена в сайт как `licenses/s1-Apache-2.0.txt`.
Страница, разрешение рендера и контрольные суммы сохранены в asset registry.
