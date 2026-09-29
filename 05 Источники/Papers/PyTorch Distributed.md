---
title: "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2006.15704v1
authors: [Shen Li et al.]
year: 2020
license: CC BY-NC-SA 4.0
---

# PyTorch Distributed

Статья объясняет организацию DDP: объединение градиентов в буферы, запуск
all-reduce по готовности и поддержание одинакового порядка операций на
разных процессах. Figure 3 показывает две ошибки — обмен разными буферами
и ожидание невычисленного градиента. Figure 4 связывает обработчики autograd,
градиенты параметров и буферы обмена.

Обе схемы включены в [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|главу DDP]].
Они описывают реализацию 2020 года; текущие параметры API и правила перестройки
буферов проверяются отдельно по документации PyTorch, а не выводятся из рисунка.

Неизменённый PDF v1 сохранён в `Source PDFs/pytorch-ddp-2006.15704v1.pdf`.
Рисунки на физических страницах 5 и 6 извлечены при 400 DPI без перерисовки.
Ссылка `view license` в [карточке версии](https://arxiv.org/abs/2006.15704v1)
подтверждает [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
SHA-256 и прямоугольники извлечения записаны в `05 Источники/asset-registry.yml`.
