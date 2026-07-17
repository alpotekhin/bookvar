---
title: Вопросы по NLP
type: question-index
status: active
last_updated: 2026-07-16
last_verified: 2026-07-16
---

# Вопросы по NLP

## Представление текста

- Что считается задачей NLP и почему текст нельзя напрямую подать в нейросеть?
  → [[02 Areas/ML & DL/Concepts/NLP/NLP|NLP]]
- Что такое токен и чем word-, subword- и character-level токенизация отличаются?
  → [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]]
- Как работает BPE и почему частые пары объединяются в новые токены?
  → [[02 Areas/ML & DL/Concepts/NLP/BPE|BPE]]
- Что хранит word embedding?
  → [[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]]
- Чем GloVe отличается от predictive embeddings?
  → [[02 Areas/ML & DL/Concepts/NLP/GloVe|GloVe]]
- Как решается проблема одного вектора для многозначного слова?
  → [[02 Areas/ML & DL/Concepts/NLP/ELMo|ELMo]]

## Языковое моделирование

- Что именно предсказывает языковая модель?
  → [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]]
- Как устроена n-gram-модель и где возникает sparsity?
  → [[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]]
- Что измеряет perplexity и почему её нельзя бездумно сравнивать между
  tokenizer-ами? → [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]]
- Чем causal language modeling отличается от masked language modeling?
  → [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal LM]] ·
  [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked LM]]

## Последовательности и перевод

- Почему обычной нейросети неудобно работать с последовательностью переменной
  длины? → [[02 Areas/ML & DL/Concepts/Architectures/RNN|RNN]]
- Откуда берутся vanishing и exploding gradients?
  → [[02 Areas/ML & DL/Concepts/Training/Vanishing Gradient|Vanishing Gradient]] ·
  [[02 Areas/ML & DL/Concepts/Training/Gradient Clipping|Gradient Clipping]]
- Как LSTM хранит информацию дольше обычной RNN?
  → [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]]
- Как работает encoder-decoder Seq2Seq?
  → [[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]]
- Почему fixed-size context vector стал bottleneck?
  → [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|От Seq2Seq к Transformer]]
- Как attention помог машинному переводу?
  → [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]

## Задачи и оценка

- Как строится text classification pipeline?
  → [[02 Areas/ML & DL/Concepts/NLP/Text Classification|Text Classification]]
- Что измеряет BLEU и где эта метрика ошибается?
  → [[02 Areas/ML & DL/Concepts/Evaluation/BLEU Score|BLEU]]
- Зачем нужны GLUE и transfer learning?
  → [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]] ·
  [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]
- Чем dense retrieval отличается от генерации ответа?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]
