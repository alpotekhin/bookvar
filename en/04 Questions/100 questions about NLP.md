---
title: 100 questions about NLP
type: question-index
status: active
locale: en
translation_of: "04 Вопросы/100 вопросов по NLP.md"
last_updated: 2026-07-23
last_verified: 2026-07-23
---

# 100 questions about NLP

This list was imported from the [public Notion page](https://dynamic-epoch-4bb.notion.site/100-questions-NLP-english-337ac246920c4afd9c54af825f5076f1). “Original answer” links to the preserved source notes. The English source currently contains substantive answers to 22 of the 100 questions; it trails the newer Russian page, so missing translations are marked explicitly. Links to canonical Bookvar chapters are maintained separately: a source answer is not a substitute for a textbook chapter.

## Classical methods and TF–IDF

**Bookvar chapters:** [[00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Representing text numerically]]

- **1. Write TF-IDF from scratch.** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#1. Write TF-IDF from scratch.|original answer]]
- **2. What is normalization in TF-IDF?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#2. What is normalization in TF-IDF?|original answer]]
- **3. Why do you need to know about TF-IDF in our time, and how can you use it in complex models?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#3. Why do you need to know about TF-IDF in our time, and how can you use it in complex models?|original answer]]
- **4. Explain how Naive Bayes works. What can you use it for?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#4. Explain how Naive Bayes works. What can you use it for?|original answer]]
- **5. How can SVM be prone to overfitting?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#5. How can SVM be prone to overfitting?|original answer]]
- **6. Explain possible methods for text preprocessing (lemmatization and stemming). What algorithms do you know for this, and in what cases would you use them?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#6. Explain possible methods for text preprocessing (lemmatization and stemming). What algorithms do you know for this, and in what cases would you use them?|original answer]]
- **7. What metrics for text similarity do you know?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#7. What metrics for text similarity do you know?|original answer]]
- **8. Explain the difference between cosine similarity and cosine distance. Which of these values can be negative? How would you use them?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#8. Explain the difference between cosine similarity and cosine distance. Which of these values can be negative? How would you use them?|original answer]]

## Metrics

**Bookvar chapters:** [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Model evaluation]]

- **9. Explain precision and recall in simple words and what you would look at in the absence of F1 score?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#9. Explain precision and recall in simple words and what you would look at in the absence of F1 score?|original answer]]
- **10. In what case would you observe changes in specificity?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#10. In what case would you observe changes in specificity?|original answer]]
- **11. When would you look at macro, and when at micro metrics? Why does the weighted metric exist?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#11. When would you look at macro, and when at micro metrics? Why does the weighted metric exist?|original answer]]
- **12. What is perplexity? What can we consider it with?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#12. What is perplexity? What can we consider it with?|original answer]]
- **13. What is the BLEU metric?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#13. What is the BLEU metric?|original answer]]
- **14. Explain the difference between different types of ROUGE metrics?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#14. Explain the difference between different types of ROUGE metrics?|original answer]]
- **15. What is the difference between BLUE and ROUGE?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#15. What is the difference between BLUE and ROUGE?|original answer]]

## Word2Vec and embeddings

**Bookvar chapters:** [[00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|From words to embeddings]]

- **16. Explain how Word2Vec learns? What is the loss function? What is maximized?** — answer not yet supplied
- **17. What methods of obtaining embeddings do you know? When will each be better?** — answer not yet supplied
- **18. What is the difference between static and contextual embeddings?** — answer not yet supplied
- **19. What are the two main architectures you know, and which one learns faster?** — answer not yet supplied
- **20. What is the difference between Glove, ELMO, FastText, and Word2Vec?** — answer not yet supplied
- **21. What is negative sampling and why is it needed? What other tricks for Word2Vec do you know, and how can you apply them?** — answer not yet supplied
- **22. What are dense and sparse embeddings? Provide examples.** — answer not yet supplied
- **23. Why might the dimensionality of embeddings be important?** — answer not yet supplied
- **24. What problems can arise when training Word2Vec on short textual data, and how can you deal with them?** — answer not yet supplied

## RNN and CNN

**Bookvar chapters:** [[00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNN and BPTT]] · [[00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|Convolutional networks]]

- **25. How many training parameters are there in a simple 1-layer RNN?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#25. How many training parameters are there in a simple 1-layer RNN?|original answer]]
- **26. How does RNN training occur?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#26. How does RNN training occur?|original answer]]
- **27. What problems exist in RNN?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#27. What problems exist in RNN?|original answer]]
- **28. What types of RNN networks do you know? Explain the difference between GRU and LSTM?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#28. What types of RNN networks do you know? Explain the difference between GRU and LSTM?|original answer]]
- **29. What parameters can we tune in such networks? (Stacking, number of layers)** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#29. What parameters can we tune in such networks? (Stacking, number of layers)|original answer]]
- **30. What are vanishing gradients for RNN? How do you solve this problem?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#30. What are vanishing gradients for RNN? How do you solve this problem?|original answer]]
- **31. Why use a Convolutional Neural Network in NLP, and how can you use it? How can you compare CNN within the attention paradigm?** — [[04 Вопросы/100 вопросов по NLP — исходные ответы#31. Why use a Convolutional Neural Network in NLP, and how can you use it? How can you compare CNN within the attention paradigm?|original answer]]

## Attention and Transformer

**Bookvar chapters:** [[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention]] · [[00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete Transformer]]

- **32. How do you compute attention? (additional: for what task was it proposed, and why?)** — answer not yet supplied
- **33. Complexity of attention? Compare it with the complexity in RNN.** — answer not yet supplied
- **34. Compare RNN and attention. In what cases would you use attention, and when RNN?** — answer not yet supplied
- **35. Write attention from scratch.** — answer not yet supplied
- **36. Explain masking in attention.** — answer not yet supplied
- **37. What is the dimensionality of the self-attention matrix?** — answer not yet supplied
- **38. What is the difference between BERT and GPT in terms of attention calculation?** — answer not yet supplied
- **39. What is the dimensionality of the embedding layer in the transformer?** — answer not yet supplied
- **40. Why are embeddings called contextual? How does it work?** — answer not yet supplied
- **41. What is used in transformers, layer norm or batch norm, and why?** — answer not yet supplied
- **42. Why do transformers have PreNorm and PostNorm?** — answer not yet supplied
- **43. Explain the difference between soft and hard (local/global) attention?** — answer not yet supplied
- **44. Explain multihead attention.** — answer not yet supplied
- **45. What other types of attention mechanisms do you know? What are the purposes of these modifications?** — answer not yet supplied
- **46. How does self-attention become more complex with an increase in the number of heads?** — answer not yet supplied

## Transformer model families

**Bookvar chapters:** [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Encoder, decoder, and encoder–decoder]] · [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT]]

- **47. Why does BERT largely lag behind RoBERTa, and what can you take from RoBERTa?** — answer not yet supplied
- **48. What are T5 and BART models? How do they differ?** — answer not yet supplied
- **49. What are task-agnostic models? Provide examples.** — answer not yet supplied
- **50. Explain transformer models by comparing BERT, GPT, and T5.** — answer not yet supplied
- **51. What major problem exists in BERT, GPT, etc., regarding model knowledge? How can this be addressed?** — answer not yet supplied
- **52. How does a decoder-like GPT work during training and inference? What is the difference?** — answer not yet supplied
- **53. Explain the difference between heads and layers in transformer models.** — answer not yet supplied

## Positional information

**Bookvar chapters:** [[00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]] · [[00 Учебник/07 Анатомия современной LLM/04 RoPE.md|RoPE]]

- **54. Why is information about positions lost in embeddings of transformer models with attention?** — answer not yet supplied
- **55. Explain approaches to positional embeddings and their pros and cons.** — answer not yet supplied
- **56. Why can't we simply add an embedding with the token index?** — answer not yet supplied
- **57. Why don't we train positional embeddings?** — answer not yet supplied
- **58. What is relative and absolute positional encoding?** — answer not yet supplied
- **59. Explain in detail the working principle of rotary positional embeddings.** — answer not yet supplied

## Pre-training

**Bookvar chapters:** [[00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Next-token prediction]] · [[00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Pre-training data]]

- **60. How does causal language modeling work?** — answer not yet supplied
- **61. When do we use a pretrained model?** — answer not yet supplied
- **62. How to train a transformer from scratch? Explain your pipeline, and in what cases would you do this?** — answer not yet supplied
- **63. What models, besides BERT and GPT, do you know for various pretraining tasks?** — answer not yet supplied

## Tokenization

**Bookvar chapters:** [[00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece, and Unigram]]

- **64. What types of tokenizers do you know? Compare them.** — answer not yet supplied
- **65. Can you extend a tokenizer? If yes, in what case would you do this? When would you retrain a tokenizer? What needs to be done when adding new tokens?** — answer not yet supplied
- **66. How do regular tokens differ from special tokens?** — answer not yet supplied
- **67. Why is lemmatization not used in transformers? And why do we need tokens?** — answer not yet supplied
- **68. How is a tokenizer trained? Explain with examples of WordPiece and BPE.** — answer not yet supplied
- **69. What position does the CLS vector occupy? Why?** — answer not yet supplied
- **70. What tokenizer is used in BERT, and which one in GPT?** — answer not yet supplied
- **71. Explain how modern tokenizers handle out-of-vocabulary words?** — answer not yet supplied
- **72. What does the tokenizer vocab size affect? How will you choose it in the case of new training?** — answer not yet supplied

## Training and adaptation

**Bookvar chapters:** [[00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|Optimization and training stability]] · [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT and instruction data]]

- **73. What is class imbalance? How can it be identified? Name all approaches to solving this problem.** — answer not yet supplied
- **74. Can dropout be used during inference, and why?** — answer not yet supplied
- **75. What is the difference between the Adam optimizer and AdamW?** — answer not yet supplied
- **76. How do consumed resources change with gradient accumulation?** — answer not yet supplied
- **77. How to optimize resource consumption during training?** — answer not yet supplied
- **78. What ways of distributed training do you know?** — answer not yet supplied
- **79. What is textual augmentation? Name all methods you know.** — answer not yet supplied
- **80. Why is padding less frequently used? What is done instead?** — answer not yet supplied
- **81. Explain how warm-up works.** — answer not yet supplied
- **82. Explain the concept of gradient clipping?** — answer not yet supplied
- **83. How does teacher forcing work, provide examples?** — answer not yet supplied
- **84. Why and how should skip connections be used?** — answer not yet supplied
- **85. What are adapters? Where and how can we use them?** — answer not yet supplied
- **86. Explain the concepts of metric learning. What approaches do you know?** — answer not yet supplied
- **87. What does the temperature in softmax control? What value would you set?** — answer not yet supplied

## Decoding

**Bookvar chapters:** [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and next-token selection]]

- **88. Explain types of sampling in generation? top-k, top-p, nucleus sampling?** — answer not yet supplied
- **89. What is the complexity of beam search, and how does it work?** — answer not yet supplied

## Large language models

**Bookvar chapters:** [[00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG]] · [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Quantization]] · [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts]]

- **90. What is sentence embedding? What are the ways you can obtain it?** — answer not yet supplied
- **91. How does LoRA work? How would you choose parameters? Imagine that we want to fine-tune a large language model, apply LORA with a small R, but the model still doesn't fit in memory. What else can be done?** — answer not yet supplied
- **92. What is the difference between prefix tuning, p-tuning, and prompt tuning?** — answer not yet supplied
- **93. Explain the scaling law.** — answer not yet supplied
- **94. Explain all stages of LLM training. From which stages can we abstain, and in what cases?** — answer not yet supplied
- **95. How does RAG work? How does it differ from few-shot KNN?** — answer not yet supplied
- **96. What quantization methods do you know? Can we fine-tune quantized models?** — answer not yet supplied
- **97. How can you prevent catastrophic forgetting in LLM?** — answer not yet supplied
- **98. Explain the working principle of KV cache, Grouped-Query Attention, and MultiQuery Attention.** — answer not yet supplied
- **99. Explain the technology behind MixTral, what are its pros and cons?** — answer not yet supplied
- **100. How are you? How are things going?** — answer not yet supplied

