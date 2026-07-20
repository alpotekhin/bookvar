---
title: "BERT"
type: model-family
organization: Google
first_release: 2018
latest_verified_release: BERT and descendants
last_verified: 2026-07-20
architecture_base: encoder-only Transformer
modalities: [text]
status: foundational
---

# BERT

До BERT перенос обучения обычно означал готовые word embeddings или однонаправленную языковую модель. BERT предобучает весь Transformer-encoder так, чтобы представление каждого токена зависело от контекста слева и справа, а затем дообучает эту же сеть на размеченной задаче с минимальной новой «головой».

## Архитектура и вход

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/bert-input-output.png]]

*Рисунок: Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/), CC BY-NC-SA 4.0; локальная копия. Каждый входной токен получает контекстуализированный выход; в отличие от GPT encoder видит обе стороны позиции.*

BERT-base содержит 12 encoder-блоков, hidden size 768 и 12 attention heads (110M параметров); BERT-large — 24 блока, 1024 и 16 heads (340M). Во вход суммируются WordPiece embedding, embedding позиции и embedding сегмента A/B. Специальный `[CLS]` используется как агрегат для классификации, `[SEP]` разделяет последовательности. Максимальная длина исходных моделей — 512 токенов.

## Почему нужен masked language modeling

Обычное предсказание следующего токена вынуждает causal mask. BERT случайно выбирает 15% позиций и просит восстановить исходный токен. Из выбранных позиций 80% заменяются `[MASK]`, 10% — случайным токеном, 10% остаются неизменными. Такая смесь уменьшает несовпадение: `[MASK]` не встречается на fine-tuning. В исходной версии второе задание, next sentence prediction, различает настоящую следующую фразу и случайную; RoBERTa позднее показала, что NSP не обязательно при улучшенном обучении.

## Данные и семейство потомков

Оригинальный BERT обучался на BooksCorpus и английской Wikipedia с WordPiece-словарём 30,522. Multilingual BERT расширил словарь и языки, но не изменил принцип encoder-only. RoBERTa оставила архитектуру почти той же, увеличила данные и batch, применила dynamic masking и убрала NSP. ALBERT разделила параметры между слоями и факторизовала embedding. DeBERTa разнесла представления содержания и относительной позиции. Эти модели следует описывать как архитектурные или training-diffs, а не как «новые BERT по размеру».

| Релиз | Дата | Что изменилось относительно BERT |
|---|---|---|
| BERT | 2018-10 | Двунаправленный encoder, masked language modeling и next sentence prediction. |
| multilingual BERT | 2018-11 | Один WordPiece-словарь и один encoder для 104 языков; базовый блок не менялся. |
| RoBERTa | 2019-07 | Больше данных и шагов, dynamic masking, крупнее batch, без NSP. |
| ALBERT | 2019-09 | Факторизация embedding и разделение параметров между слоями уменьшают число весов. |
| DeBERTa | 2020-06 | Раздельные представления содержания и относительной позиции в attention. |

Эта хронология не означает, что каждый следующий релиз заменил предыдущий.
RoBERTa прежде всего исправляет режим обучения, ALBERT меняет параметризацию, а
DeBERTa — способ учитывать позицию. Поэтому сравнивать их нужно по типу
изменения, а не только по дате или итоговому benchmark.

## Fine-tuning и inference

Для классификации берут вектор `[CLS]`; для разметки последовательности классифицируют каждый токен; для extractive QA две головы предсказывают начало и конец ответа. Все encoder-токены вычисляются одновременно, поэтому BERT удобен для embeddings, reranking, NER и понимания документа, но сам по себе не является авторегрессионным генератором. Стоимость self-attention квадратична по длине, а KV-cache для пошагового decode не нужен.

## Опубликовано и ограничения

Архитектура, код, веса и корпуса исходного BERT опубликованы. Важно не приписывать BERT современный chat, tool use или long context: эти свойства принадлежат другим моделям и training recipes. WordPiece также создаёт отдельную проблему согласования субтокенов с метками слов.

## Источники

- [BERT paper](https://arxiv.org/abs/1810.04805).
- [google-research/bert](https://github.com/google-research/bert) — официальный код и checkpoints.
- [RoBERTa](https://arxiv.org/abs/1907.11692) и [DeBERTa](https://arxiv.org/abs/2006.03654) — важные diffs.
- [ALBERT](https://arxiv.org/abs/1909.11942) — факторизация embedding и разделение параметров.
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) — официальный список языков и release note.
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — учебная визуализация.
