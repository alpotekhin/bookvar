---
title: Учебник по NLP и LLM
type: textbook-chapter
status: active
last_updated: 2026-07-17
---

# Учебник по NLP и LLM

Подробный визуальный маршрут от представления текста до современных LLM,
reasoning, retrieval и agents.

> [!info] Как устроена база
> Учебник задаёт порядок чтения. [[02 Areas/ML & DL/01 Справочник/_index|Справочник]]
> хранит точные определения, [[02 Areas/ML & DL/02 Атлас моделей/_index|Атлас]]
> сравнивает модели, а [[02 Areas/ML & DL/03 Исследовательские линии/_index|исследовательские линии]]
> отслеживают свежие papers.

![[02 Areas/ML & DL/00 Учебник/Assets/Учебная карта]]

## Полный маршрут

### I. Основания

1. [[02 Areas/ML & DL/00 Учебник/00 Как пользоваться учебником]]
2. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation]]
3. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения]]
4. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings]]
5. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram]]
6. [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction]]
7. [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/02 Sampling и генерация]]
8. [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN, LSTM и Seq2Seq]]

### II. Transformer и современный LLM-блок

9. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]
10. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
11. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]
12. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]
13. [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура]]
14. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]]
15. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache]]
16. [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/00 Карта модуля и источники|Mixture of Experts — карта модуля]]
17. [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 От SwiGLU к MoE]]

### III. Архитектуры и жизненный цикл

18. [[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока]]
19. [[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/02 Альтернативы Transformer и multimodality]]
20. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training]]
21. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/02 Distributed training и precision]]
22. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники]]
23. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]
24. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data]]
25. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling]]
26. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]
27. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO]]
28. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers]]
29. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1]]
30. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation]]
31. [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute]]
32. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/01 KV-cache, batching и FlashAttention]]
33. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/02 Quantization и deployment]]

### IV. Системы вокруг модели

34. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/00 Карта модуля и источники|Retrieval и RAG — карта модуля]]
35. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/01 Embeddings и retrieval]]
36. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/02 RAG как система]]
37. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/01 Vision-language и omni models]]
38. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля]]
39. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/01 Tool use и agents]]
40. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/02 Evaluation и воспроизводимость]]

## Другие режимы

- [[02 Areas/ML & DL/00 Учебник/Карта крупных учебных модулей|Карта крупных учебных модулей]]
- [[02 Areas/ML & DL/00 Учебник/Покрытие программы|Покрытие всех 40 тем]]
- [[02 Areas/ML & DL/01 Справочник/_index|Справочник]]
- [[02 Areas/ML & DL/02 Атлас моделей/_index|Атлас моделей]]
- [[02 Areas/ML & DL/03 Исследовательские линии/_index|Исследовательские линии]]
- [[02 Areas/ML & DL/04 Вопросы/_index|Вопросы]]
- [[02 Areas/ML & DL/05 Источники/_index|Источники]]
- [[02 Areas/ML & DL/06 Практика/_index|Практика]]
