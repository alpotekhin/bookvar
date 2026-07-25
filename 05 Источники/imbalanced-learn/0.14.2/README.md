---
title: "imbalanced-learn 0.14.2 — полный оригинальный корпус"
type: source-note
status: pinned
last_verified: 2026-07-24
source_url: https://github.com/scikit-learn-contrib/imbalanced-learn
license: MIT
---

# imbalanced-learn 0.14.2 — полный оригинальный корпус

Это локальная неизменённая копия частей официального репозитория
`scikit-learn-contrib/imbalanced-learn`, использованных для главы Bookvar
[[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация]].

Зафиксированный snapshot:

- версия: `0.14.2`;
- commit: [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/8504e95f0160f61d1b617ca66f779646d2ee609e);
- дата импорта: 2026-07-24;
- лицензия: [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE);
- целостность: [[02 Areas/ML & DL/05 Источники/imbalanced-learn/0.14.2/manifest.yml|manifest.yml]] содержит SHA-256 каждого файла.

## Читаемая публикация

Все 20 файлов snapshot доступны не только как исходные RST, Python, BibTeX и
LICENSE, но и как отдельные английские страницы:
[[02 Areas/ML & DL/05 Источники/imbalanced-learn/0.14.2/published/index|открыть полный указатель]].
В User Guide сохранены формулы, примеры и официальные иллюстрации; gallery
разобрана на чередующиеся пояснения и исполняемые блоки кода; BibTeX показан
как библиография с отдельной записью для каждой работы.

Публикационный слой воспроизводимо пересобирается с проверкой зафиксированных
SHA-256:

```bash
python publishing/tools/import_imbalanced_learn.py --root . --update-navigation
```

## Полный User Guide в оригинальном английском

Файлы сохранены в исходном формате reStructuredText без перевода и
редакторских изменений:

1. [`original/user-guide/introduction.rst`](original/user-guide/introduction.rst)
2. [`original/user-guide/over_sampling.rst`](original/user-guide/over_sampling.rst)
3. [`original/user-guide/under_sampling.rst`](original/user-guide/under_sampling.rst)
4. [`original/user-guide/combine.rst`](original/user-guide/combine.rst)
5. [`original/user-guide/ensemble.rst`](original/user-guide/ensemble.rst)
6. [`original/user-guide/miscellaneous.rst`](original/user-guide/miscellaneous.rst)
7. [`original/user-guide/metrics.rst`](original/user-guide/metrics.rst)
8. [`original/user-guide/model_selection.rst`](original/user-guide/model_selection.rst)
9. [`original/user-guide/common_pitfalls.rst`](original/user-guide/common_pitfalls.rst)
10. [`original/user-guide/zzz_references.rst`](original/user-guide/zzz_references.rst)
11. [`original/user-guide/refs.bib`](original/user-guide/refs.bib)

Собранная официальная HTML-версия:
[User Guide 0.14.2](https://imbalanced-learn.org/stable/user_guide.html).

## Релевантная gallery в оригинальном английском

Сохранены полные исходные Python-файлы, включая narrative docstrings:

- [`plot_impact_imbalanced_classes.py`](original/gallery/plot_impact_imbalanced_classes.py)
- [`plot_comparison_over_sampling.py`](original/gallery/plot_comparison_over_sampling.py)
- [`plot_comparison_under_sampling.py`](original/gallery/plot_comparison_under_sampling.py)
- [`plot_illustration_tomek_links.py`](original/gallery/plot_illustration_tomek_links.py)
- [`plot_bagging_classifier.py`](original/gallery/plot_bagging_classifier.py)
- [`plot_comparison_ensemble_classifier.py`](original/gallery/plot_comparison_ensemble_classifier.py)
- [`plot_metrics.py`](original/gallery/plot_metrics.py)
- [`plot_pipeline_classification.py`](original/gallery/plot_pipeline_classification.py)

Полная официальная gallery:
[Examples 0.14.2](https://imbalanced-learn.org/stable/auto_examples/index.html).

## Граница между источником и редакцией

Каталог `original/` — source-native слой: его файлы не локализованы и не
переписаны. Русская глава Bookvar — самостоятельное каноническое изложение,
которое связывает материал User Guide с выбором метрики, порога и протокола
оценивания. Локальные PNG скопированы из официально собранной gallery и
зарегистрированы отдельно в `05 Источники/asset-registry.yml`.
