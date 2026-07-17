---
title: "Tree of Thoughts"
aliases: [Tree of Thoughts, ToT, Deliberate Problem Solving]
type: concept
category: Reasoning
papers:
  - "[[02 Areas/ML & DL/Papers/Tree of Thoughts|Tree of Thoughts]]"
courses: []
sources:
  - "[Yao et al. (2023) — Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)"
  - "[GitHub — princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm)"
---

# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

![[02 Areas/ML & DL/raw/papers/tree-of-thoughts/images/tot-fig1.png]]
*IO, CoT, CoT-SC и Tree of Thoughts как частные случаи дерева рассуждений: ToT добавляет branching, evaluation и backtracking (источник: Yao et al., 2023)*

## Зачем это нужно: LLM как «System 1» мышление

Все современные LLM работают одинаково: генерируют текст **токен за токеном, слева направо**. Даже [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] -- это линейная цепочка рассуждений, где каждый шаг необратим. Если модель пошла не в ту сторону на первом шаге -- она не может **отступить назад** и попробовать другой путь.

Это напоминает **System 1** мышление (Kahneman, 2011) -- быстрое, автоматическое, интуитивное. Но задачи, требующие планирования и поиска (математические головоломки, кроссворды, стратегические игры), требуют **System 2** -- медленного, осознанного мышления с exploration и backtracking.

Авторы обращаются к истокам AI: Newell, Shaw и Simon в 1950-х годах характеризовали problem solving как **поиск по комбинаторному пространству, представленному как дерево**. Tree of Thoughts переносит эту идею на LLM.

## Формализация: IO, CoT, CoT-SC как частные случаи деревьев

Гениальность работы -- в унифицированном взгляде на методы инференса LLM:

| Метод | Глубина | Ширина | Описание |
|-------|---------|--------|----------|
| **IO** | 1 | 1 | Один input -> один output |
| **CoT** | >1 | 1 | Линейная цепочка мыслей |
| **CoT-SC** | >1 | k | k параллельных цепочек, majority vote |
| **ToT** | >1 | b | Дерево с BFS/DFS, evaluation, backtracking |

Все методы -- **частные случаи ToT** с разной depth и breadth. ToT -- наиболее общий фреймворк.

Формально: состояние $s = [x, z_{1\ldots i}]$ -- это вход $x$ плюс последовательность мыслей (thoughts) $z_1, \ldots, z_i$. Каждая мысль -- когерентная языковая единица, служащая промежуточным шагом к решению.

## Четыре компонента ToT

### 1. Thought Decomposition (декомпозиция мыслей)

Ключевой вопрос: какого «размера» должна быть одна мысль? Слишком маленькая (один токен) -- нельзя оценить перспективность. Слишком большая (целая книга) -- нельзя сгенерировать когерентно.

| Задача | Размер мысли | Пример |
|--------|-------------|--------|
| Game of 24 | Одно уравнение | *"13 - 9 = 4 (left: 4, 4, 10)"* |
| Creative Writing | Один абзац плана | *"1. Introduce a book that connects..."* |
| Mini Crosswords | Одно слово | *"h1. shown"* |

### 2. Thought Generator $G(p_\theta, s, k)$

Два подхода к генерации $k$ кандидатных мыслей:

**(a) i.i.d. sampling** из CoT промпта: $z^{(j)} \sim p_\theta^{\text{CoT}}(z_{i+1} \mid s)$. Работает лучше, когда пространство мыслей **богатое** (каждая мысль -- абзац). Diversity через sampling.

**(b) Propose prompt**: $[z^{(1)}, \ldots, z^{(k)}] \sim p_\theta^{\text{propose}}(z_{i+1}^{(1\ldots k)} \mid s)$. Работает лучше, когда пространство **ограничено** (каждая мысль -- слово или уравнение). Одна генерация, разные варианты -- избегает дупликатов.

### 3. State Evaluator $V(p_\theta, S)$

Оценка перспективности состояний -- эвристика для поиска. **Революционная идея**: вместо programmed heuristics (DeepBlue) или learned heuristics (AlphaGo), LLM **сама рассуждает** о перспективности состояния.

Два подхода:

**(a) Value-based** (независимая оценка): для каждого состояния $s$ LLM генерирует оценку (sure / likely / impossible). Основана на few lookahead simulations (быстро проверить, что 5 + 5 + 14 = 24) и commonsense (1, 2, 3 слишком малы для 24).

**(b) Vote-based** (сравнительная оценка): LLM сравнивает несколько состояний и голосует за лучшее. Подходит, когда success трудно оценить напрямую (coherency текста).

### 4. Search Algorithm

**(a) BFS** (Algorithm 1): на каждом шаге сохраняем $b$ лучших состояний. Подходит для неглубоких деревьев ($T \leq 3$). Используется для Game of 24 (depth 3, breadth 5) и Creative Writing (depth 1).

**(b) DFS** (Algorithm 2): исследуем самое перспективное состояние first, с pruning и backtracking. Подходит для глубоких деревьев. Используется для Mini Crosswords (depth до 10).

Авторы оставляют **A*** и **MCTS** для будущей работы.

## Эксперименты

### Game of 24: главный showcase

![[02 Areas/ML & DL/raw/papers/tree-of-thoughts/images/tot-game-of-24.png]]
*ToT в Game of 24: (a) thought generation — генерация кандидатов на каждом шаге, (b) valuation — оценка перспективности каждого состояния (источник: Yao et al., 2023)*

Задача: используя 4 числа и арифметические операции (+-*/), получить 24. Пример: (10 - 4) * (13 - 9) = 24.

| Метод | Success Rate |
|-------|-------------|
| IO prompt | 7.3% |
| **CoT prompt** | **4.0%** |
| CoT-SC (k=100) | 9.0% |
| IO + Refine (k=10) | 27% |
| IO (best of 100) | 33% |
| CoT (best of 100) | 49% |
| **ToT (b=1)** | **45%** |
| **ToT (b=5)** | **74%** |

*Таблица: Результаты GPT-4 на Game of 24 (Table 2 из статьи)*

**74% vs 4%** -- ToT в 18.5 раз лучше CoT! Даже CoT-SC с 100 сэмплами достигает лишь 9%.

**Анализ ошибок (Figure 3b)**: ~60% CoT сэмплов уже ошибаются после **первого шага** (первые 3 слова, напр. "4 + 9"). Это иллюстрирует фундаментальную проблему left-to-right decoding.

**Scale analysis (Figure 3a)**: ToT с b=1 (45%) уже лучше CoT best-of-100 (49%) при гораздо меньшем количестве visited nodes.

### Creative Writing

Задача: 4 случайных предложения, нужно написать связный текст из 4 абзацев, заканчивающихся этими предложениями.

| Метод | GPT-4 Coherency (1-10) | Human Preference |
|-------|----------------------|------------------|
| IO | 6.19 | - |
| CoT | 6.93 | - |
| **ToT** | **7.56** | **Выбирается в 41% vs 21% для CoT** |

ToT генерирует plan (breadth 5 планов, vote для лучшего), затем пишет текст по плану.

### Mini Crosswords (5x5)

| Метод | Word Success | Letter Success | Game Success |
|-------|-------------|----------------|--------------|
| CoT | 15.6% | 59.9% | 0% |
| **ToT (DFS)** | **60%** | **84%** | **4/20** |

CoT не решает **ни одного** кроссворда. ToT с DFS и pruning решает 20%.

## Стоимость: trade-off quality vs compute

ToT значительно дороже одного CoT вызова. Для Game of 24:
- CoT: 1 вызов LLM
- ToT (b=5): ~5 генераций + ~15 оценок = **~100 вызовов LLM на задачу**

Это стоимость **deliberate reasoning** -- System 2 мышление требует больше ресурсов.

## Концептуальные достоинства

1. **Generality**: IO, CoT, CoT-SC, self-refinement -- всё частные случаи ToT
2. **Modularity**: base LM, thought decomposition, generation, evaluation, search -- всё варьируется независимо
3. **Adaptability**: разные задачи -> разные конфигурации
4. **Convenience**: не нужен дополнительный training, работает с pre-trained LM

## Направления развития

- **MCTS (Monte Carlo Tree Search)** вместо BFS/DFS -- более эффективная exploration
- **Learned value functions** вместо LLM-based evaluation
- **Training**: fine-tuning LLM на successful ToT trajectories
- **Composition с ReAct**: tree search + external actions
- **LATS (Language Agent Tree Search, 2023)** -- ToT + ReAct + environment feedback

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] -- линейный частный случай ToT (depth>1, breadth=1)
- [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]] -- CoT-SC: частный случай (depth>1, breadth=k, majority vote)
- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] -- acting в среде, ортогонально к tree search
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] -- ToT как продвинутый фреймворк инференса

## Дополнительные ресурсы

- [GitHub — princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm) -- все промпты и код
- [Tree of Thoughts paper (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
