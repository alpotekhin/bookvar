---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2305.18290v3
authors: [Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn]
year: 2023
license: CC BY 4.0
---

# Direct Preference Optimization

Статья выводит попарную функцию потерь через аналитическое решение задачи
максимизации награды с KL-регуляризацией. Сокращение нормировочной константы
позволяет выразить разность наград через вероятности выбранного и отвергнутого
ответов. Вывод, градиент, численные примеры и ограничения изложены в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|главе DPO]].

Figure 1 сравнивает RLHF с отдельной моделью награды и DPO. Figure 2 показывает
компромисс награда–KL в эксперименте с продолжением отзывов IMDb; его нельзя
трактовать как универсальный рейтинг алгоритмов. В главу включена левая панель
с целыми осями и легендой, без перерисовки.

Для воспроизводимого извлечения сохранён
[конференционный PDF NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)
в `Source PDFs/dpo-neurips-2023.pdf`. Это не файл arXiv v3. Право на
использование той же авторской иллюстрации подтверждается
[лицензией версии v3](https://arxiv.org/abs/2305.18290v3), CC BY 4.0.
Координаты извлечения физической страницы 7 при 640 DPI и контрольные суммы
сохранены в `05 Источники/asset-registry.yml`.

Официальный [код](https://github.com/eric-mitchell/direct-preference-optimization)
содержит расчёт суммарных логарифмов вероятностей и `preference_loss`.
Короткие функции главы являются учебными реализациями этих операций, а не
копией полного тренера.
