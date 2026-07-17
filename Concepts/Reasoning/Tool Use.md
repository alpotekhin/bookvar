---
title: "Tool Use"
aliases: [Tool Use, Tool Calling, Function Calling, использование инструментов]
type: concept
status: legacy
category: Reasoning
papers:
  - "[[02 Areas/ML & DL/Papers/Toolformer|Toolformer]]"
  - "[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]"
  - "[[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]"
  - "[[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]]"
courses: []
sources:
  - "[Schick et al. — Toolformer: Language Models Can Teach Themselves to Use Tools (2023)](https://arxiv.org/abs/2302.04761)"
  - "[OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)"
  - "[Anthropic — Model Context Protocol](https://modelcontextprotocol.io/)"
  - "[Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)"
---

# Tool Use — LLM использует внешние инструменты

## Зачем это нужно: ограничения closed-box LLM

LLM обучена на статическом корпусе и работает только со своими параметрами. Это порождает фундаментальные ограничения:

- **Устаревшие знания** — модель ничего не знает после knowledge cutoff
- **Математика и код** — точные вычисления плохо удаются autoregressive сэмплированию
- **Факты** — галлюцинации по специфичным вопросам (актуальные цены, API, документация)
- **Действия** — нельзя изменить состояние мира (отправить email, записать в БД, задеплоить код)

**Tool use** — парадигма, где LLM вызывает **внешние инструменты** (функции, API, интерпретаторы кода, поиск), получает результат и продолжает рассуждение. LLM становится **оркестратором**, а не оракулом.

## Как работает вызов инструментов

Базовый цикл (reasoning-acting loop, близкий к [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]]):

```
1. User: "Какая погода в Москве и Нью-Йорке?"
2. LLM выводит: call(get_weather, {"city": "Moscow"})
3. Tool выполняется: returns "-5°C, снег"
4. LLM получает результат, выводит: call(get_weather, {"city": "New York"})
5. Tool returns "+8°C, дождь"
6. LLM финализирует: "В Москве -5°C снег, в Нью-Йорке +8°C дождь."
```

Каждый tool описан **схемой** (name, description, JSON schema параметров). Модель должна выбрать инструмент, сгенерировать валидные аргументы, распарсить результат и решить, нужен ли ещё вызов.

## Эволюция: три волны

### 1. Toolformer (Meta, 2023) — self-supervised обучение

[[02 Areas/ML & DL/Papers/Toolformer|Toolformer]] предложил обучать модель вызывать инструменты **без разметки**:

1. Взять обычный текст
2. LLM сэмплирует потенциальные места для tool call (calculator, QA, translate, calendar, search)
3. Сравнить perplexity continuation с и без результата tool — оставить только полезные вызовы
4. Fine-tune модель на тексте с **вставленными tool calls**

Результат: GPT-J 6.7B с tools обходит GPT-3 175B на arithmetic, QA, temporal tasks. Но подход ограничен: 5 инструментов, один вызов за раз, нет multi-turn планирования.

### 2. Function calling (OpenAI, 2023) — инструменты в API

Июнь 2023: OpenAI добавил **function calling** в GPT-4/GPT-3.5 API. Разработчик передаёт список функций с JSON Schema; модель возвращает структурированный `tool_calls` ответ вместо текста:

```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_abc",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Moscow\"}"
    }
  }]
}
```

Стало стандартом API. Anthropic Claude, Google Gemini, open-source модели (Mistral, LLaMA, Qwen) приняли эту схему. Появились бенчмарки: **Berkeley Function Calling Leaderboard** (BFCL), **ToolBench**, **τ-bench** (multi-turn).

### 3. MCP (Anthropic, 2024) — стандарт для серверов

**Model Context Protocol** (ноябрь 2024) — открытый протокол, стандартизирующий подключение LLM к внешним системам. Вместо того чтобы каждое приложение писало свою интеграцию, MCP-сервер экспортирует инструменты, ресурсы, prompts через stdio/SSE.

Преимущества:
- **Переиспользуемость** — один MCP server (filesystem, github, postgres) работает с любым MCP-клиентом (Claude Desktop, Cursor, Zed)
- **Discovery** — клиент узнаёт доступные tools динамически
- **Композиция** — агент может подключать сотни MCP-серверов одновременно

С 2025 MCP стал индустриальным стандартом; крупные вендоры (OpenAI, Google) добавили совместимость.

## Обучение tool-use через RL

Supervised fine-tuning на tool trajectories даёт базовые навыки, но плох в exploration, error recovery, multi-turn планировании. Решение — **reinforcement learning** с verifiable rewards.

### RLVR как фундамент

[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] (Reinforcement Learning with Verifiable Rewards) — парадигма, где reward определяется **детерминированной проверкой** выхода: прошёл ли тест, вернул ли tool успешный результат, достигнута ли цель. Tool-use идеально ложится на RLVR: у нас есть execution trace, можно проверить, решена ли задача.

Стандартный алгоритм — [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] (DeepSeek, 2024): group-relative advantage без отдельной value model. Модель генерирует $G$ trajectories на промпт, награды нормализуются внутри группы.

### ASTRA (Beike, 2026)

[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]] — training recipe для tool-use агентов, обученных GRPO без KL-penalty. Reward — trajectory-level F1: $r = \text{solved}/\text{total}$, $p = \text{solved}/\text{calls}$, итоговый reward $2rp/(r+p)$. Штраф за **избыточные** вызовы (high recall, low precision) — модель учится не спамить tool calls.

Показал: cold-start GRPO (без SFT warm-start) даёт более гибкие политики; adaptive batch filling решает проблему zero-advantage батчей.

### ToolRL (2025)

[[02 Areas/ML & DL/Papers/ToolRL|ToolRL]] изучил влияние reward design и SFT-warm-start на tool-use. Ключевые находки на Qwen-2.5-7B, бенчмарк BFCL:

- **GRPO cold-start: 58.38%** — без SFT pre-training
- **SFT → GRPO: 39.25%** — SFT "memorizes" формат и мешает exploration

Дизайн reward: нужны и **format reward** (валидный JSON), и **correctness reward** (правильный выбор tool + аргументы), и **trajectory reward** (задача решена).

### Tool-N1 (Nemotron, 2025)

[[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]] — NVIDIA's recipe для reasoning + tool-use. Combining CoT с tool calls: модель сначала планирует через [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|CoT]], затем вызывает tool, анализирует результат и продолжает. Показал, что **small verifier-based reward** (бинарная правильность) работает лучше сложных learned reward models.

## Challenges

### Hallucination в tool calls

Модель может **выдумывать** tool'ы, несуществующие API, неправильные имена параметров. Частая проблема для weak моделей и при большом числе tools (>50).

**Митигации:**
- Strict JSON schema validation (constrained decoding, Outlines, Guidance)
- Retrieval tools по описанию перед вызовом (tool RAG) — показать модели только top-k релевантных
- Penalize hallucinated tools в training reward

### Schema adherence

Генерировать валидный JSON с правильными типами, обязательными полями, enum-значениями. Сложно на long-context и глубоко вложенных schemas.

**Митигации:**
- **Constrained decoding** — логиты фильтруются grammar'ом (JSON schema → finite state machine). Outlines, LMFE, vLLM guided decoding.
- Training на парах (schema, valid call)
- Retry с error message в промпт

### Error recovery

Реальные tools падают: timeout, 500 ошибки, неправильные аргументы. Агент должен **читать error message**, корректировать вызов и повторять, а не зацикливаться.

Хорошие агенты демонстрируют reasoning traces вроде: *"Tool returned 404 — значит ресурс не существует. Попробую другой endpoint."* Появляется только с правильным training data и RL на реалистичных error scenarios.

### Long-horizon planning

Задачи из τ-bench (airline, retail): нужны 10+ вызовов, поддержание state, откаты. Frontier модели (Claude 4.6, GPT-4.1) справляются частично — SOTA на τ-bench airline ~60% в 2025. Open-source модели 30-45%.

### Безопасность

Tool-использующие агенты способны **менять мир**: отправлять email, делать транзакции, запускать код. Ошибки дороги.

Практики:
- Confirm before destructive actions (удаление, деньги, deploy)
- Sandbox для code execution (Docker, Firecracker, E2B)
- Rate limiting, budget limits на запросы
- Prompt injection defense — tool result не должен давать модели новые instructions

## Бенчмарки

| Бенчмарк | Фокус |
|---------|-------|
| **BFCL** (Berkeley) | Single/multi-turn function calling, 2000+ tests |
| **ToolBench** | 16k tools из RapidAPI, multi-tool chains |
| **τ-bench** | Interactive multi-turn agent tasks |
| **SWE-bench** | Tool-use для codebase editing |
| **GAIA** | General assistant tasks с web browsing |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] — reasoning+acting loop, предшественник tool calling
- [[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]] — первое self-supervised обучение tool use
- [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] — парадигма обучения, где tool-use естественно укладывается
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — стандартный алгоритм для обучения tool-use агентов
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — reasoning между tool calls
- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]] — tool calls = форма test-time compute
