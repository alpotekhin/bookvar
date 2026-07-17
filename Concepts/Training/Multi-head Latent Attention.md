---
title: "Multi-head Latent Attention"
aliases: [MLA, Multi-head Latent Attention, Latent Attention]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-V2|DeepSeek-V2]]"
  - "[[02 Areas/ML & DL/Papers/DeepSeek-V3 Technical Report|DeepSeek-V3]]"
courses: []
sources:
  - "[DeepSeek-V2: A Strong, Economical and Efficient MoE LLM (2024)](https://arxiv.org/abs/2405.04434)"
  - "[Sebastian Raschka — Multi-Head Latent Attention (MLA)](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)"
  - "[Vizuara — Decoding Multi-Head Latent Attention](https://vizuara.substack.com/p/decoding-multi-head-latent-attention)"
  - "[planetbanatt — Understanding Multi-Head Latent Attention](https://planetbanatt.net/articles/mla.html)"
  - "[Lior Sinai — DeepSeek's Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)"
---

# Multi-head Latent Attention (MLA)

## Зачем это нужно: KV-cache как bottleneck

При авторегрессивном инференсе LLM хранит **KV-cache** — ранее вычисленные Key и Value для каждого токена, чтобы не пересчитывать их на каждом шаге. Размер кэша на токен:

$$\text{KV-cache per token} = 2 \times n_h \times d_h \times n_{layers}$$

Для модели с 128 головами, $d_h = 128$, 60 слоёв:
$$2 \times 128 \times 128 \times 60 = 1{,}966{,}080 \text{ элементов на токен}$$

При batch size = 32 и контексте 100K токенов — это **десятки гигабайт** только на KV-cache. Это главный bottleneck для serving LLM: GPU memory определяет максимальный batch size → определяет throughput → определяет стоимость inference.

## Существующие решения и их недостатки

### Multi-Query Attention (MQA)

Все query-головы разделяют **одну** пару KV. Экономия: $n_h \times$ по памяти.

**Проблема:** одна KV-пара не может отразить всё разнообразие информации, нужное разным query-головам. Качество падает.

### Grouped-Query Attention (GQA)

$n_h$ query-голов разделяются на $g$ групп, каждая со своей KV-парой. Компромисс между MHA и MQA.

**Проблема:** всё ещё теряет представительную мощность. GQA с 8 группами — это всего 8 разных «точек зрения» на информацию для KV, при 64 query-головах.

### Общий недостаток MQA/GQA

Оба метода **уменьшают число KV-голов**, ограничивая что модель может «запомнить» о каждом токене. Это фундаментальный trade-off: меньше памяти = меньше информации.

## MLA: другой подход — сжатие вместо сокращения

### Ключевой insight

Вопрос DeepSeek: **зачем хранить полные Key и Value, если они содержат избыточную информацию?**

Высокоразмерные K и V для разных голов сильно коррелированы — «суть» токена можно описать гораздо компактнее. MLA сжимает KV в **латентный вектор** низкой размерности, из которого при необходимости восстанавливаются полные K и V.

### Формальное описание

**Стандартный MHA:**

$$k_t^{(i)} = W_K^{(i)} h_t, \quad v_t^{(i)} = W_V^{(i)} h_t$$

Каждая голова $i$ имеет свои проекции. Все $k_t^{(i)}, v_t^{(i)}$ хранятся в кэше.

**MLA (шаг 1 — сжатие):**

$$c_t^{KV} = W^{DKV} h_t, \quad c_t^{KV} \in \mathbb{R}^{d_c}$$

Один латентный вектор $c_t^{KV}$ размерности $d_c$ (например, 512) вместо всех KV-голов.

**MLA (шаг 2 — восстановление):**

$$k_t^{(i)} = W_K^{U(i)} c_t^{KV}, \quad v_t^{(i)} = W_V^{U(i)} c_t^{KV}$$

При вычислении attention каждая голова восстанавливает свои K и V из общего $c_t^{KV}$.

**Что хранится в кэше:** только $c_t^{KV}$ (размер $d_c$), а не все $n_h \times d_h$ для K и V.

### Визуализация сжатия

```
Standard MHA:
  h_t → [W_K^1] → k_t^1  }
  h_t → [W_K^2] → k_t^2  }  Всё хранится в кэше: 2 × n_h × d_h
  ...                       }
  h_t → [W_V^1] → v_t^1  }
  h_t → [W_V^2] → v_t^2  }
  ...

MLA:
  h_t → [W^{DKV}] → c_t^{KV}  ← Только это хранится (d_c)
                        ↓
            [W_K^{U(1)}] → k_t^1  }
            [W_K^{U(2)}] → k_t^2  }  Восстанавливается on-the-fly
            ...                     }
            [W_V^{U(1)}] → v_t^1  }
            [W_V^{U(2)}] → v_t^2  }
```

### Сжатие Query (бонус)

MLA также сжимает Query аналогичным способом:

$$c_t^Q = W^{DQ} h_t, \quad q_t^{(i)} = W_Q^{U(i)} c_t^Q$$

Это не влияет на KV-cache (Query не кэшируются), но уменьшает число параметров и ускоряет computation.

## Decoupled Rotary Position Embedding

Стандартный [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|RoPE]] применяется к Key напрямую. Но в MLA Keys восстанавливаются из $c_t^{KV}$ — применение RoPE **до** сжатия невозможно (позиционная информация теряется при сжатии).

Решение — **decoupled RoPE**: отдельный маленький вектор $k_t^{rope}$ вычисляется параллельно и конкатенируется с Key после восстановления:

$$k_t^{final} = [W_K^U c_t^{KV} \; || \; \text{RoPE}(W^{KR} h_t)]$$

В кэше хранится $[c_t^{KV} \; || \; k_t^{rope}]$ — маленькая добавка к латентному вектору.

## Экономия памяти

| Метод | KV-cache per token | Относительно MHA |
|-------|-------------------|------------------|
| MHA | $2 \times n_h \times d_h$ | 1.0x |
| MQA | $2 \times d_h$ | $1/n_h$ |
| GQA ($g$ групп) | $2 \times g \times d_h$ | $g/n_h$ |
| **MLA** | $d_c + d_{rope}$ | **~$1/n_h$ или меньше** |

Для DeepSeek-V2: **93.3% сокращение** KV-cache по сравнению с MHA, при этом качество **выше** чем у MHA.

## Почему MLA не теряет качество

**Low-rank hypothesis:** информация, которую модель извлекает из токена для attention, имеет **эффективную размерность** гораздо ниже, чем $n_h \times d_h$. Латентный вектор $c_t^{KV}$ размерности $d_c$ захватывает основное «содержание», а up-projection матрицы $W_K^U, W_V^U$ восстанавливают head-specific детали.

Эмпирически: при $d_c \approx 4 \times d_h$ (вместо $2 \times n_h \times d_h$) MLA показывает **лучшие** результаты, чем MHA. Причина — low-rank compression действует как **implicit regularization**, предотвращая переобучение в attention patterns.

## Influence: кто уже использует MLA

- **DeepSeek-V2** (2024) — первая модель с MLA
- **DeepSeek-V3** (2024) — 671B MoE с MLA → $5.5M training
- **DeepSeek-R1** (2025) — reasoning поверх V3 с MLA
- Активные исследования по адаптации MLA для других архитектур

MLA — одна из самых значимых архитектурных инноваций 2024-2025: сопоставимый эффект на inference cost, как Flash Attention на training speed.

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — базовый механизм
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — проблема, которую MLA решает
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-V3|DeepSeek-V3]] — модель, использующая MLA
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — RoPE и decoupled вариант
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — комплементарная оптимизация (compute vs memory)
