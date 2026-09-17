---
title: Хронология моделей и технологий LLM
type: concept
last_updated: 2026-09-07
last_verified: 2026-09-07
status: active
---

# Хронология моделей и технологий LLM

Хронология отмечает момент публикации архитектурной идеи, системной технологии
или первого официального релиза, который сделал линию заметной. Она не
устанавливает абсолютный приоритет: многие решения развивались параллельно, а
новый inference engine или продуктовый релиз может менять исполнение и
постобучение, не меняя базовый вычислительный блок модели.

В календарных таблицах тип события указан рядом с датой: «статья» означает публикацию работы, «анонс/карточка» — публичное представление, «линия репозитория» — датированный ориентир проекта. Для Mixtral, Gemma и Qwen2 отдельно отмечено, когда веса появились раньше технического отчёта. Это устраняет смешение даты релиза и даты статьи.

## 2017–2020: три интерфейса Transformer

| Дата | Релиз | Что вошло в историю архитектур | Официальный источник |
|---|---|---|---|
| 2017-06 (статья) | Transformer | encoder-decoder с self-attention и cross-attention вместо рекуррентной сети | [Vaswani et al.](https://arxiv.org/abs/1706.03762) |
| 2018-06 (анонс/карточка) | GPT-1 | decoder-only Transformer как переносимое авторегрессионное предобучение с последующим fine-tuning | [OpenAI, 11 июня 2018](https://openai.com/index/language-unsupervised/) |
| 2018-10 (статья) | BERT | encoder-only Transformer, masked language modeling и двунаправленный контекст | [Devlin et al., 11 октября 2018](https://arxiv.org/abs/1810.04805) |
| 2019-02 (анонс/карточка) | GPT-2 | масштабирование decoder-only предобучения и постановка задач как продолжения текста без отдельной головы | [OpenAI, 14 февраля 2019](https://openai.com/index/better-language-models/) |
| 2019-10 (статья) | T5 | единый text-to-text интерфейс для encoder-decoder и восстановление удалённых фрагментов | [Raffel et al.](https://arxiv.org/abs/1910.10683) |
| 2020-05 (анонс/карточка) | GPT-3 | 175B decoder-only модель и систематическое исследование few-shot/in-context learning | [OpenAI, 28 мая 2020](https://openai.com/index/language-models-are-few-shot-learners/) |

Эти ветви нельзя расположить на одной шкале «старее — новее». GPT оптимизирует
причинную генерацию, BERT — представление всего входа, T5 — преобразование
источника в отдельную выходную последовательность. Они остаются опорными точками
для сравнения современных моделей.

## 2021–2022: новые цели, состояния и мультимодальные соединители

| Дата | Линия | Главный сдвиг | Официальный источник |
|---|---|---|---|
| 2021-01 (анонс/карточка) | CLIP | контрастивное согласование изображений и текста; основа многих последующих vision encoders | [OpenAI, 5 января 2021](https://openai.com/index/clip/) |
| 2021-03 (статья) | GLM | авторегрессионное заполнение пропусков как сочетание понимания и генерации | [GLM paper](https://arxiv.org/abs/2103.10360) |
| 2021 (анонс/карточка) | RWKV | рекуррентное временное смешивание с параллельным обучением и постоянным состоянием при выводе | [официальная RWKV Wiki](https://wiki.rwkv.com/) |
| 2022-03 (статья) | InstructGPT | SFT, модель предпочтений и PPO как документированная линия постобучения диалогу | [Ouyang et al.](https://arxiv.org/abs/2203.02155) |
| 2022-04 (анонс/карточка) | Flamingo | замороженные vision/LM-компоненты, Perceiver Resampler и вставленные gated cross-attention слои | [Google DeepMind, 28 апреля 2022](https://deepmind.google/blog/tackling-multiple-tasks-with-a-single-visual-language-model/) |
| 2022-10 (статья) | GLM-130B | двуязычное масштабирование GLM и открытая инженерная документация обучения большой модели | [GLM-130B](https://arxiv.org/abs/2210.02414) |

Flamingo важен не просто как «модель с картинками». Он сделал явными три
независимых компонента мультимодальной системы: кодировщик, сжатие визуальных
признаков и место, где они вводятся в языковую модель.

## 2023: открытые decoder-only семейства и альтернативы полному attention

| Дата | Релиз или линия | Архитектурный ориентир | Официальный источник |
|---|---|---|---|
| 2023-01 (статья) | BLIP-2 | Q-Former как обучаемый bottleneck между замороженными vision encoder и LLM | [Li et al., 30 января 2023](https://arxiv.org/abs/2301.12597) |
| 2023-02 (статья) | LLaMA | компактная открытая decoder-only база с RMSNorm, SwiGLU и RoPE | [Touvron et al.](https://arxiv.org/abs/2302.13971) |
| 2023-04 (анонс/карточка) | LLaVA | проекция CLIP-признаков в пространство токенов LLM и visual instruction tuning | [Microsoft Research, LLaVA](https://www.microsoft.com/en-us/research/project/llava-large-language-and-vision-assistant/overview/) |
| 2023 (статья) | Falcon | MQA и RefinedWeb как совместный акцент на архитектуре вывода и открытом веб-корпусе | [Falcon report](https://arxiv.org/abs/2311.16867), [RefinedWeb](https://arxiv.org/abs/2306.01116) |
| 2023-07 (статья) | Llama 2 | открытая pre-trained/chat ветка; GQA в модели 70B | [Llama 2](https://arxiv.org/abs/2307.09288) |
| 2023-07 (статья) | RetNet | parallel, recurrent и chunkwise-recurrent формы одного retention-оператора | [Retentive Network](https://arxiv.org/abs/2307.08621) |
| 2023-09 (статья) | Qwen, Baichuan 2 | многоязычные открытые decoder-only экосистемы | [Qwen report](https://arxiv.org/abs/2309.16609), [Baichuan 2](https://arxiv.org/abs/2309.10305) |
| 2023-09 (анонс/карточка) | Mistral 7B | GQA и sliding-window attention в компактной открытой модели | [Mistral AI, 27 сентября 2023](https://mistral.ai/news/announcing-mistral-7b/) |
| 2023 (линия репозитория) | InternLM, Yi | отдельные открытые семейства с собственными training/serving экосистемами | [InternLM](https://github.com/InternLM/InternLM), [Yi](https://github.com/01-ai/Yi) |
| 2023-12 (статья) | Mamba | selective state space model с input-dependent состоянием и аппаратно-ориентированным scan | [Gu, Dao](https://arxiv.org/abs/2312.00752) |
| 2023-12 (статья) | Gemini 1.0 | нативно мультимодальная закрытая модель; точная внутренняя архитектура раскрыта лишь частично | [Google, Gemini report](https://arxiv.org/abs/2312.11805) |

## 2024: KV-кэш, MoE и модель–система как единый проект

| Дата | Релиз | Главный diff | Официальный источник |
|---|---|---|---|
| 2024-01 (статья; веса — 2023-12) | Mixtral 8×7B | открытый sparse MoE с top-2 маршрутизацией | [Mixtral of Experts](https://arxiv.org/abs/2401.04088), [анонс 11 декабря](https://mistral.ai/news/mixtral-of-experts/) |
| 2024-03 (статья; релиз — 2024-02) | Gemma | компактная открытая decoder-only ветка Google | [Gemma technical report](https://arxiv.org/abs/2403.08295), [релиз 21 февраля](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-open-models/) |
| 2024-03 (статья) | Jamba | чередование Attention и Mamba внутри sparse-MoE модели | [Jamba](https://arxiv.org/abs/2403.19887) |
| 2024-03 (анонс/карточка) | DBRX | fine-grained top-4 MoE и dropless serving stack | [Databricks release](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) |
| 2024-03 (анонс/карточка) | Command R | модель, постобучение которой ориентировано на retrieval, citations и tool use | [официальный model card](https://huggingface.co/CohereForAI/c4ai-command-r-v01) |
| 2024-05 (статья) | DeepSeek-V2 | MLA с латентным KV-кэшем и fine-grained DeepSeekMoE | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| 2024-07 (статья) | Llama 3.1 | плотная 405B модель, GQA и опубликованный контекст 128K | [Llama 3 report](https://arxiv.org/abs/2407.21783) |
| 2024-07 (статья; релиз — 2024-06) | Qwen2 | расширение многоязычной decoder-only линии и dense/MoE варианты | [Qwen2 report](https://arxiv.org/abs/2407.10671), [релиз 7 июня](https://qwenlm.github.io/blog/qwen2/) |
| 2024-07-31 (arXiv; релиз — 2024-06) | Gemma 2 | чередование локального и глобального внимания, logit soft-capping и distillation recipe | [статья Gemma 2](https://arxiv.org/abs/2408.00118), [релиз 27 июня](https://blog.google/innovation-and-ai/technology/developers-tools/google-gemma-2/) |
| 2024-09 (статья) | Qwen2-VL | dynamic resolution и multimodal RoPE | [Qwen2-VL](https://arxiv.org/abs/2409.12191) |
| 2024-12 (статья) | Phi-4 | data-centric малая модель с большим объёмом синтетических данных | [Phi-4 technical report](https://arxiv.org/abs/2412.08905) |
| 2024-12 (статья) | DeepSeek-V3 | MLA+MoE, multi-token prediction, FP8 и балансировка без вспомогательной функции потерь | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) |

## 2025–2026: рассуждение, нативная мультимодальность и гибридное внимание

| Дата | Релиз | Что действительно изменилось | Официальный источник |
|---|---|---|---|
| 2025-01 (статья) | DeepSeek-R1 | масштабная линия RL с проверяемыми наградами и дистилляцией рассуждения; backbone унаследован от V3 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| 2025-03 (статья) | Gemma 3 | мультимодальная ветка, чередование локального/глобального attention и длинный контекст | [Gemma 3 report](https://arxiv.org/abs/2503.19786) |
| 2025-04 (анонс/карточка) | Llama 4 | early-fusion multimodality и sparse MoE в Scout/Maverick | [Meta, Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| 2025-04 (анонс/карточка) | Qwen3 | dense/MoE семейство и единый переключаемый режим thinking; режим относится прежде всего к постобучению | [Qwen3 release](https://qwenlm.github.io/blog/qwen3/) |
| 2025-07 (линия репозитория) | Kimi K2 | 1T/32B-active MLA-MoE и MuonClip | [официальный Kimi K2 repository/report](https://github.com/MoonshotAI/Kimi-K2) |
| 2025 (линия репозитория) | GLM-4.5 | MoE и совмещённые thinking/non-thinking режимы | [официальный GLM-4.5 repository](https://github.com/zai-org/GLM-4.5) |
| 2025 (статья) | Kimi Linear | гибрид Kimi Delta Attention и полного attention как отдельная исследовательская линия | [Kimi Linear](https://arxiv.org/abs/2510.26692) |
| 2025-12 (статья) | DeepSeek-V3.2 | DeepSeek Sparse Attention поверх MLA | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| 2025-12 (статья о семействе; Nano — 15.12.2025) | Nemotron 3 | гибрид Mamba-2 и attention с MoE; последующий Super развивает LatentMoE, MTP и низкоточное предобучение | [Nemotron 3 overview](https://arxiv.org/abs/2512.20856), [[02 Атлас моделей/Семейства/Nemotron|карточка семейства]] |
| 2026 (линия репозитория) | Qwen3.5/3.6 | гибрид Gated Delta Networks и attention, sparse MoE и мультимодальность | [официальный Qwen3.6 repository](https://github.com/QwenLM/Qwen3.6) |
| 2026 (линия репозитория) | Kimi K2.5 | нативная мультимодальная agentic-ветка K2 | [официальный Kimi K2.5 repository](https://github.com/MoonshotAI/Kimi-K2.5) |
| 2026-02 (линия репозитория) | GLM-5 | 744B/40B-active, DSA и 28.5T training tokens; последующие 5.1-релизы не следует автоматически считать новым backbone | [GLM-5 report/repository](https://github.com/zai-org/GLM-5) |
| 2026-03 (статья) | Mamba-3 | следующее поколение state-space архитектуры | [Mamba-3](https://arxiv.org/abs/2603.15569) |
| 2026-04 (анонс/карточка) | Mistral Medium 3.5 | мультимодальная модель с опубликованными весами под Modified MIT; доступность весов отделяется от полного рецепта обучения | [официальная карточка v26.04](https://docs.mistral.ai/models/mistral-medium-3-5-26-04) |
| 2026 (статья) | Baichuan-M3 | медицинская ветка; специализированный домен не означает новую универсальную архитектуру | [Baichuan-M3 report](https://arxiv.org/abs/2602.06570) |

## Агенты, рассуждение и проверяемые траектории

Эта линия отмечает не новые backbone-модели, а изменения в устройстве системы
вокруг модели: формальную проверку, воспроизводимую среду, обучение на
траекториях, управление правами и защиту памяти. Поэтому одна и та же базовая
LLM может находиться по разные стороны этих переходов в зависимости от harness
и способа обучения.

| Дата | Работа или система | Что изменилось | Подробный разбор и первичный источник |
|---|---|---|---|
| 2023-06 | LeanDojo | закреплённая версия Lean-библиотеки, retrieval доступных посылок и проверка каждого доказательства ядром | [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты#Нейросетевой prover: от поиска посылок до проверки ядром|разбор]], [Yang et al.](https://arxiv.org/abs/2306.15626) |
| 2023-10 | COPRA | proof-state feedback, история неудачных шагов и backtracking превратили одиночную генерацию tactic в управляемый поиск | [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты#COPRA: поиск с состоянием, возвратом и обратной связью prover|разбор]], [Thakur et al.](https://arxiv.org/abs/2310.04353) |
| 2024-04 | OSWorld | benchmark computer-use agent стал контрактом из начального состояния, action space, reset и execution-based evaluator | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#OSWorld: восстанавливаемая VM, исполняемые действия и проверка результата|разбор]], [Xie et al.](https://arxiv.org/abs/2404.07972) |
| 2024-07 | AgentPoison | retrieval-триггер связал малую долю отравленных записей с целевым вредоносным поведением агента | [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем#Прямые и косвенные инъекции, отравление retrieval и памяти|разбор]], [Chen et al.](https://arxiv.org/abs/2407.12784) |
| 2024-09 | LaSR | найденные формулы обновляют библиотеку понятий, а библиотека направляет mutation и crossover следующего поколения | [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery#LaSR: совместная эволюция формул и библиотеки понятий|разбор]], [Grayeli et al.](https://arxiv.org/abs/2409.09359) |
| 2024-12 | AgentTrek | руководства из web превращаются в задачи и подсказки, которые заново исполняются и проверяются в актуальной GUI-среде | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#AgentTrek: от руководств к траекториям через управляемое воспроизведение|разбор]], [Xu et al.](https://arxiv.org/abs/2412.09605) |
| 2024-12 | TACO | мультимодальные обучающие траектории чередуют reasoning tokens и исполняемые actions; action-useless примеры отфильтровываются | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#TACO: совместное обучение токенов рассуждения и действий|разбор]], [TACO v1](https://arxiv.org/abs/2412.05479v1) |
| 2024-12 | Aguvis | единое визуальное пространство действий связало web, desktop и mobile, а обучение разделило grounding и длинные траектории | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#Aguvis: единое визуальное пространство наблюдений и действий|разбор]], [Xu et al.](https://arxiv.org/abs/2412.04454) |
| 2025-02 | InSTA | pipeline формулирует задачи на работающих сайтах, исполняет их и оставляет только траектории, прошедшие автоматическую проверку | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#InSTA: синтетические задачи из работающего web|разбор]], [Liu et al.](https://arxiv.org/abs/2502.06776) |
| 2025-04 | Progent | права на tool calls задаются исполнимой политикой с детерминированным enforcement и явным fallback | [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем#Progent: наименьшие привилегии как исполнимая политика вызова|разбор]], [Progent](https://arxiv.org/abs/2504.11703) |
| 2025-04 | DataSentinel | detector prompt injection обучается против адаптивного атакующего в minimax-постановке | [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем#Наблюдение за выполнением, потоки данных и формальная проверка политики|разбор]], [DataSentinel](https://arxiv.org/abs/2504.11358) |

## Inference и serving: от GPU-ядер к разделению prefill и decode

Эта линия описывает не новые backbone-модели, а способы исполнять уже обученную
модель. Поэтому её события вынесены отдельно от календаря релизов: оптимизация
ядра, планировщик запросов и управление KV-кэшем отвечают за разные уровни
системы и могут сочетаться в одном inference engine.

| Дата | Технология | Системный сдвиг | Первичный источник |
|---|---|---|---|
| 2019-06 | Triton | язык и компилятор для плиточных GPU-вычислений позволили описывать специализированные ядра выше уровня CUDA | [Tillet, Kung, Cox, MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) |
| 2022-05 | FlashAttention | точный attention стал IO-aware: tiling уменьшает обмен между HBM и SRAM без аппроксимации результата | [Dao et al., NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) |
| 2022-07 | Orca | iteration-level scheduling разрешил менять состав batch между шагами авторегрессионной генерации; это опорная работа для continuous batching | [Yu et al., OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) |
| 2023-06/10 | PagedAttention и vLLM | блочное управление KV-кэшем по аналогии с виртуальной памятью уменьшило фрагментацию и позволило держать больше запросов в batch | [первый релиз vLLM](https://vllm-project.github.io/2023/06/20/vllm.html), [Kwon et al., SOSP 2023](https://doi.org/10.1145/3600006.3613165) |
| 2023-12 / 2024-01 | SGLang и RadixAttention | radix tree сделал общие prompt-префиксы переиспользуемым KV-кэшем для ветвящихся и многошаговых программ | [Zheng et al.](https://arxiv.org/abs/2312.07104), [официальный анонс SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) |
| 2024-01 | Disaggregated prefill/decode | DistServe разместил compute-bound prefill и memory-bound decode на разных GPU и раздельно оптимизировал TTFT и TPOT | [Zhong et al., OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) |

## Линии, которые пересекают календарь

Календарный порядок скрывает причинные связи. Для архитектурного сравнения
полезнее читать события как несколько параллельных траекторий:

| Линия | Последовательность ориентиров | Что измерять |
|---|---|---|
| Топология задачи | Transformer → GPT/BERT/T5 | доступ к контексту, способ декодирования, функция предобучения |
| Память внимания | MHA → MQA/GQA → MLA → sparse attention | байты KV-кэша, объём чтения при decode, потеря качества и поддержка ядрами |
| Условная ёмкость | dense FFN → Mixtral/DBRX → DeepSeekMoE и современные sparse MoE | всего/активных параметров, top-k, балансировка и обмен между устройствами |
| Стоимость длинной последовательности | full attention → local attention → RetNet/RWKV/Mamba → гибриды Jamba/Kimi | асимптотика, состояние, качество дальних зависимостей и параллельность обучения |
| Постобучение | fine-tuning → RLHF → preference optimization → RLVR/reasoning | источник сигнала, online/offline режим и отделимость от backbone |
| Мультимодальность | CLIP → Flamingo/BLIP-2/LLaVA → native multimodal families | кодировщик, connector, место fusion и число токенов модальности |
| Inference и serving | Triton/FlashAttention → Orca continuous batching → PagedAttention/vLLM → SGLang/RadixAttention → disaggregated prefill/decode | пропускная способность, TTFT, TPOT, занятость GPU, фрагментация и переиспользование KV-кэша |

## Границы датированных утверждений

- дата релиза не доказывает архитектурный приоритет;
- опубликованная длина контекста не гарантирует одинаковое качество на всех
  позициях;
- новый thinking-, agentic- или tool-use-релиз может менять только постобучение;
- для закрытых GPT, Gemini и коммерческих Mistral фиксируются официальные
  возможности и model IDs, но не вымышленные размеры и устройство блока;
- `last_updated` означает дату последней сверки этой страницы. Для живых
  каталогов используются датированные релизы, а не alias `latest`.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К сравнительным маршрутам атласа]]
