# Questions zero-finding pages: дополнительное полное чтение

Дата: 2026-09-15. Scope: ровно шесть RU страниц, у которых в исходном `questions.json` нет findings и которые не имели собственного completion full-read evidence. Статус: **полное чтение 6/6 выполнено; source-level link review PASS; final built-link/render gate остаётся за root**.

## Граница результата

Изменён только этот новый отчёт. Ни один вопрос, исходный архив, канонический ответ, `questions.json`, `finding-ledger.json`, navigation, registry или publisher не редактировался. Не выполнялись git-команды, build, импорт, публикация или создание новых агентов. Новые P1/P2 в шести прочитанных страницах и проверенных контекстах ответов не обнаружены; это не новый полный аудит всех 62 целевых глав.

Шесть страниц прочитаны полностью, от frontmatter до EOF, а не по grep. Их текущие SHA-256 совпали с исходным audit read SHA **6/6** и повторно проверены в конце. Поэтому unchanged question body подтверждается всем файлом, включая ссылки и формулировки, а не только количеством вопросов.

| Страница | Полное чтение | Строки без пустого EOF | Bytes | Audit = current SHA-256 |
|---|---|---:|---:|---|
| `04 Вопросы/_index.md` | 1–EOF | 23 | 1345 | `e5fedebd0cce25eb46a039fa8d497be469f4bbe4074d70ba472c821817ee6c00` |
| `04 Вопросы/Вопросы по агентам.md` | 1–EOF | 125 | 18464 | `946e51f7b52c00f049bde194d58e1de0c4ff9500229821348e4fc6bd78bad51d` |
| `04 Вопросы/Вопросы по архитектурам.md` | 1–EOF | 80 | 9654 | `74603334ee4bf38554d3cd97824708dbeb4894cd0ef95ab5897d96b4c526ff6b` |
| `04 Вопросы/Вопросы по мультимодальным моделям.md` | 1–EOF | 26 | 5715 | `628a01f42e795ff316d0f91d983a66b2464e9c760c9c5d8c402148c98344f141` |
| `04 Вопросы/Вопросы по NLP.md` | 1–EOF | 162 | 20264 | `ec11eff759e7f855e0643d46a69692a5e00c02b92d9e6b81863556cbbd038f2e` |
| `04 Вопросы/Вопросы по Retrieval и RAG.md` | 1–EOF | 56 | 7046 | `b8a3be91ee04704fb2d61adfddc69afe236e4ccc999582a66a98242c2f0b47e3` |

Счётчик строк рассчитан как `text.trimEnd().split("\n").length`; он исключает только завершающую пустоту, а не внутренние пустые строки. По объёму это 472 строки. Новые файловые snapshots не создавались, поскольку source edits отсутствуют; для каждого файла сохранена точная связь с исходным audit SHA.

Этот отчёт добавляет фактическое чтение недостающих шести страниц к ранее **сообщённому в отдельных completion reports** чтению 274 страниц. Он не переписывает прошлый ledger, не приписывает новое чтение остальным 274 страницам и не превращает нулевые findings в автоматическую приёмку.

## Содержательное чтение

### 04 Вопросы/_index.md

Различает исходный список, сохранённые авторские ответы и тематические указатели; не обещает готового ключа на все вопросы. Все восемь переходов ведут в канонические маршруты текущего публикационного manifest.

### 04 Вопросы/Вопросы по агентам.md

Последовательность проходит от tool interface и состояния через planning/evaluation к web/computer-use, formal proof и безопасности. Вопросы различают результат, траекторию, разрешение и обученный verifier; InSTA 14,6% не выдаётся за accuracy.

### 04 Вопросы/Вопросы по архитектурам.md

Разделены граф внимания, задача обучения, механизмы блока, MoE/SSM и семейства моделей. Вопросы о KV-cache, sparse computation и полном наборе весов сохраняют разные величины; ответы доступны через актуальные headings или их явные aliases.

### 04 Вопросы/Вопросы по мультимодальным моделям.md

Вопросы разделяют patch features, contrastive objectives, connector, token budget, generation и duplex. Арифметика (224/14)² / (2×2) = 64 верна; связанная глава объясняет её отдельно от MRoPE.

### 04 Вопросы/Вопросы по NLP.md

Последовательный маршрут от Unicode/lookup и distributional representations к tokenization, n-grams, RNN/LSTM и Seq2Seq attention. Список явно не подменяет главы сокращёнными ответами; SentencePiece отделён от алгоритма, teacher forcing — от decoding.

### 04 Вопросы/Вопросы по Retrieval и RAG.md

Разделены обучение embedding space, candidate retrieval/reranking и grounding/evaluation. Вопросы правильно требуют отличать citation support, retrieval recall и конечное качество ответа.


Общие свойства списков: страницы действительно являются question indexes, поэтому ссылочная форма соответствует их роли; отсутствие развёрнутого ответа непосредственно под вопросом не является дефектом. Вопросы ведут к механизму, границе применимости или проверяемому примеру, а не только к названию модели.

## Проверка текущих ссылок

- Все **172** wikilinks проверены на существование source file и route в текущем `publishing/navigation.yml`.
- **156** ссылок содержат heading/fragment; каждая разрешается в один текущий heading либо явно сохранённый anchor alias.
- **62** уникальные целевые страницы. Отсутствующих source files, manifest routes и отсутствующих/неоднозначных fragments: **0**.
- Использованы настоящие функции publisher: `loadManifest`, `parseMarkdownHeadings(..., true)`, `normalizeObsidianHeading`, `wikiHeadingSlug`, `publicationHref`. Импорт `build.ts` выполнялся только ради read-only helpers; защищённый main/build entrypoint не запускался.
- Проверялась текущая Markdown/source привязка. HTML output, responsive view и итоговая проверка `site/dist` здесь **не выполнялись**; они не заменены этим source-level PASS.
- Исходный запуск через tsx CLI получил sandbox IPC `EPERM`; успешная проверка использовала Node с `--import ./publishing/node_modules/tsx/dist/loader.mjs --input-type=module`, без IPC CLI и без ослабления sandbox.
- Избыточный диагностический вывод со snippets однажды был усечён инструментом; он не учитывается как чтение. Полные шесть страниц прочитаны отдельными неусечёнными выводами; итоговый machine result со всеми 172 ссылками получен и разобран полностью.

По страницам:

| Страница | Все ссылки | С fragment |
|---|---:|---:|
| `04 Вопросы/_index.md` | 8 | 0 |
| `04 Вопросы/Вопросы по агентам.md` | 46 | 46 |
| `04 Вопросы/Вопросы по архитектурам.md` | 28 | 22 |
| `04 Вопросы/Вопросы по мультимодальным моделям.md` | 16 | 14 |
| `04 Вопросы/Вопросы по NLP.md` | 55 | 55 |
| `04 Вопросы/Вопросы по Retrieval и RAG.md` | 19 | 19 |

### Семантическая проверка source answers

Кроме полного чтения шести индексов прочитаны **19 выбранных контекстов целевых разделов**, по 25–30 строк от указанного heading (для страницы о переносе encoder — вводный контекст). Это выборочная проверка соответствия вопроса ответу, **не утверждение о полном повторном чтении 19 разделов или 62 глав**.

| Проверенный контекст | Source location |
|---|---|
| WebDreamer и необратимые действия | `00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md`, строка 218 |
| Action log | `01 Справочник/Agents/Agent mechanisms.md`, строка 87 |
| Verifier синтетических траекторий | `00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md`, строка 531 |
| Граница formal verifier | `00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md`, строка 231 |
| MLA и decoupled RoPE | `00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md`, строка 205 |
| Serving MoE | `00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md`, строка 490 |
| RetNet | `00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md`, строка 266 |
| CLIP и SigLIP | `00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md`, строка 122 |
| Перенос vision encoder | `02 Атлас моделей/Сравнения/Перенос vision encoder между VLM.md`, строка 1 |
| Qwen2-VL | `00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md`, строка 74 |
| Moshi и full duplex | `00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md`, строка 208 |
| Выборка отрицательных примеров | `00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md`, строка 146 |
| Byte-level BPE | `00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md`, строка 216 |
| SentencePiece | `00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md`, строка 346 |
| Kneser–Ney | `00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель.md`, строка 116 |
| Численный пример LSTM | `00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md`, строка 104 |
| Гибридная выдача | `00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid.md`, строка 141 |
| Генерация и цитаты | `00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md`, строка 224 |
| Оценивание RAG по этапам | `00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md`, строка 266 |

В этих контекстах подтверждены, в частности: предсказание world model не даёт разрешения на действие; action log не равен chain of thought; InSTA verifier selection не равен accuracy; formal proof не гарантирует соответствия исходной постановке; MLA разделяет content/RoPE; MoE FLOP savings не обещают latency; SigLIP сохраняет negative pairs; перенос projector требует совместимого обученного интерфейса; full duplex отличается от streaming; SentencePiece является toolkit; Kneser–Ney использует continuation counts; LSTM derivative обозначена как прямой путь; RAG citation correctness отделена от completeness и stage metrics.

Новые внешние первичные утверждения не добавлялись. Это проверка актуальной связности с существующими каноническими ответами; live web re-audit всех исходящих источников 62 целевых страниц не выполнялся и не заявляется.

## Сохранность исходных авторских ответов

Дополнительно сверена граница архива `04 Вопросы/100 вопросов по NLP — исходные ответы.md`, на который ссылается общий индекс. Архив не редактировался.

- Task-5 baseline snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/task-5-before/100 вопросов по NLP — исходные ответы.md`.
- Baseline bytes: 109511; SHA-256 `2b27cae33d29f35bb6ea95ad4e35891b327464078550f6ac9b8a944f06748f34`.
- Current full file bytes: 162977; SHA-256 `9a480d1f272c933e4107f4b1db14a0c679979d7f67d2deae68ef647b988e210b`.
- Из текущего текста только для сравнения в памяти удалены двухстрочное вставленное warning перед авторским текстом и приложение после heading `Редакционные исправления — 15 сентября 2026`. После `trimEnd()` оставшееся тело **равно baseline byte-for-byte в UTF-8**.
- В сохранённом теле остаются все 37 исходных figure embeds в прежнем порядке. Редакционные дополнения имеют явную границу; это не скрытая правка авторских ответов.
- Весь архив заново целиком не читался в этой шестистраничной подзадаче. Его сохранность подтверждена сравнением; существующая содержательная приёмка архивных errata описана отдельно в `questions-review.md`.
- Не закрывается оставшийся `questions:007:2` linked-render finding: root должен принять его по финальной сборке/HTML links.

## Полная source-level карта проверенных ссылок

Целевые идентификаторы ниже служат только компактной записью этого отчёта. SHA — текущие bytes на момент проверки; позднейшие root deltas могут иметь отдельные superseding hashes.

| ID | Целевой source | Route | SHA-256 |
|---|---|---|---|
| T01 | `04 Вопросы/100 вопросов по NLP.md` | `/questions/100-nlp/` | `99752501fd2c7f4fbc4df1311323741617ae084bc4549a7e5154711eecdb6f5c` |
| T02 | `04 Вопросы/100 вопросов по NLP — исходные ответы.md` | `/questions/100-nlp/source-answers/` | `9a480d1f272c933e4107f4b1db14a0c679979d7f67d2deae68ef647b988e210b` |
| T03 | `04 Вопросы/Вопросы по NLP.md` | `/questions/voprosy-po-nlp/` | `ec11eff759e7f855e0643d46a69692a5e00c02b92d9e6b81863556cbbd038f2e` |
| T04 | `04 Вопросы/Вопросы по LLM.md` | `/questions/llm/` | `fb2087b7d53e5334302ac039b1793c77fc0bfe9d1b94b74f0cca632a9137149a` |
| T05 | `04 Вопросы/Вопросы по архитектурам.md` | `/questions/voprosy-po-arhitekturam/` | `74603334ee4bf38554d3cd97824708dbeb4894cd0ef95ab5897d96b4c526ff6b` |
| T06 | `04 Вопросы/Вопросы по Retrieval и RAG.md` | `/questions/voprosy-po-retrieval-i-rag/` | `b8a3be91ee04704fb2d61adfddc69afe236e4ccc999582a66a98242c2f0b47e3` |
| T07 | `04 Вопросы/Вопросы по агентам.md` | `/questions/agents/` | `946e51f7b52c00f049bde194d58e1de0c4ff9500229821348e4fc6bd78bad51d` |
| T08 | `04 Вопросы/Вопросы по мультимодальным моделям.md` | `/questions/multimodal-models/` | `628a01f42e795ff316d0f91d983a66b2464e9c760c9c5d8c402148c98344f141` |
| T09 | `00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md` | `/textbook/agents/tool-use/` | `a449a857d60eae1e2055a22d104d276e91136d9d2d2b16672666b2a1079d4965` |
| T10 | `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md` | `/textbook/agents/harness-context-engineering/` | `124c4950849dc3f4c176a8b01ee600e5b0b16b127a627b2ff325b8c5901882f0` |
| T11 | `00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md` | `/textbook/agents/memory-planning-orchestration/` | `0608e7b9ee12888863954c9366d20e247c53e90584c32e4932df4dfd65fb3c35` |
| T12 | `00 Учебник/17 Tools и Agents/68 Оценивание агентных систем.md` | `/textbook/agents/evaluation/` | `cb3c55299c141e5e2b92ced2409e07c6e1e1c8ffb94563a89e094b30dcbe635d` |
| T13 | `01 Справочник/Agents/Agent mechanisms.md` | `/reference/agents/mechanisms/` | `5b8a8cd16793f3f811076a7c2ee28c9e6374e07cb22ea9ce7e84f6f701f16df6` |
| T14 | `06 Практика/25 Воспроизводимо оценить и red-team компьютерного агента.md` | `/practice/25-computer-agent-evaluation-red-team/` | `15481b90f2ab6b77629d24e7984207c1b53d2c8e53d769938e1f94f0309267bb` |
| T15 | `00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md` | `/textbook/agents/coding-web-computer-use/` | `bdab71660c7c4718cbf82ba36c1f008bb4fa21c819a9d97f745deac0e599abe3` |
| T16 | `00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md` | `/textbook/agents/formal-math/` | `dc885c1546d7d8328563d1f542033592c97cc8e2812549823f9a5e9c2dc108a3` |
| T17 | `00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md` | `/textbook/agents/scientific-discovery/` | `670e7741669b1da2d88ef7a763ed7877a6f43cf51645d4abbf0cf34e4b361e22` |
| T18 | `06 Практика/27 Проверяемый научный поиск на символьной регрессии.md` | `/practice/27-verifiable-scientific-discovery/` | `f176c7fe5c8ce20dafb1a65c7d4b022999a5327e47a7a8eea02e4df06225e3c3` |
| T19 | `00 Учебник/17 Tools и Agents/72 Безопасность агентных систем.md` | `/textbook/agents/security/` | `e374ab17ac76aa05114defa897f89e22e60283435464f3ca0ee807498bb77f21` |
| T20 | `00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md` | `/textbook/transformer/self-attention/` | `7f6fa6fcf5e9cd1b451395712c36849b0488461bfcb0f7222fb6c27f5cc07796` |
| T21 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md` | `/textbook/transformer/encoder-decoder-patterns/` | `df74468c152bba2498a23f446ebe257e62f6bc3f573c8265280bf6971e58d909` |
| T22 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md` | `/textbook/models/bert-roberta-deberta/` | `287efca47e3ce89e0b4c751556ff5c9e011660c16a3b78a47cdd8fd1ff8d249e` |
| T23 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md` | `/textbook/models/t5/` | `c96dce53be2521aa573181be175e0f979c9cb52677a5ade31ff04c3f96ca0df8` |
| T24 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md` | `/textbook/models/gpt-1/` | `f10f0a3e21d7ea7b207420c0ea0a00210af1d6bdc1da9475a656939421ee38f9` |
| T25 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md` | `/textbook/models/gpt-2/` | `ba02e4131bb09eb334254eb90a7affa6c761ea4508701c27400c42f8846dfa34` |
| T26 | `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md` | `/textbook/models/gpt-3/` | `64b8fc86a86e69fbac77f85eb9c096f0f44a2466ace9e7cca0604cd2e62403ad` |
| T27 | `00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md` | `/textbook/modern-llm/llama-architecture/` | `ea7413af665ed3a042c5952582fa6e46b7856019e7c1f8e62e8cda3b2ddd1a9f` |
| T28 | `00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md` | `/textbook/modern-llm/norm-gating-residual/` | `2fa4869dd10c01fde99c27dd5b01e32e5af89ba20d7409c6f9292c27c12c922f` |
| T29 | `00 Учебник/07 Анатомия современной LLM/04 RoPE.md` | `/textbook/modern-llm/rope/` | `86fb1f58f6ddaa72f6424c55ee810359d05749cd4f284d67ba4bf0c92e635bbc` |
| T30 | `00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md` | `/textbook/modern-llm/mha-mqa-gqa/` | `60402496763f3a3c4f1464d6ae354770805dbadb0566830001a1d9a0a6196f53` |
| T31 | `00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md` | `/textbook/modern-llm/mla-kv-compression/` | `03f760bc4321e59bed4119152399d3e3487aaff6a6763e785a31b7a27dd7b528` |
| T32 | `00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md` | `/textbook/modern-llm/long-context/` | `a905656cb7446ed983d0dce75ce80a8173eb0f0346a2aa7960efe054c0dcb433` |
| T33 | `00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md` | `/textbook/modern-llm/dense-ffn/` | `5647f04ffc0a1526e57a2ef2f84a17cb1f78362185e671e465b386a5854249fb` |
| T34 | `00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md` | `/textbook/moe/mixture-of-experts/` | `749852e844bc2cf6de647cf715d66cc42a4a9d669d830ee5f0e07943db898d31` |
| T35 | `00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md` | `/textbook/modern-llm/sequence-model-alternatives/` | `6d2f115d509a8e8e1897562af4f33818a7ebd3887f7f76b4017d7c07cf6df0d9` |
| T36 | `02 Атлас моделей/Семейства/Llama.md` | `/models/semeystva/llama/` | `8d5ae02e6f5b818e4a6eb6507458edaa30d97fcfb54614ef6ec3c84d7a8069b0` |
| T37 | `02 Атлас моделей/Семейства/Qwen.md` | `/models/semeystva/qwen/` | `f2d18cb7da6a131c84d4925f6f559561c78a6323b575e8a9eeff2f77203c2204` |
| T38 | `02 Атлас моделей/Семейства/DeepSeek.md` | `/models/families/deepseek/` | `922432e4f44a0ec5360475bb561bcc15df5438ef78be2c0651cb08e7a1d42206` |
| T39 | `02 Атлас моделей/Семейства/GLM.md` | `/models/semeystva/glm/` | `153c71b1ddd2d4f23578ee6550c9d246c48691cc362b6320d83fdc63b17ac8f9` |
| T40 | `02 Атлас моделей/Семейства/Kimi.md` | `/models/semeystva/kimi/` | `df641246623d8079e327145c37dc799cff5a4025c64c0019c457d1bcb215b951` |
| T41 | `02 Атлас моделей/_index.md` | `/models/` | `1a08660ffc2fdada88d6f0f77bbcba14658e4c55919710e6d1777f264d68f189` |
| T42 | `00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md` | `/textbook/multimodal/models/` | `5e354860bd007d0f2f58dfdb81825c552e150f8dea791035925b05eeb23d9163` |
| T43 | `00 Учебник/16 Multimodal Models/64a Connectors и fusion.md` | `/textbook/multimodal/connectors-fusion/` | `be7de7ba68c2c0e1af4a566873e1e9cd07654349bbc552abba06a81005c65b47` |
| T44 | `02 Атлас моделей/Сравнения/Перенос vision encoder между VLM.md` | `/models/sravneniya/perenos-vision-encoder-mezhdu-vlm/` | `0e1bacdc9a5c466e73b5396db928f5e006fa3db60a3ae002bb840ec87d6e4f33` |
| T45 | `00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md` | `/textbook/multimodal/visual-tokenization-resolution/` | `3a1f7c9087f22dad604205758212c6d7fd5cdd9116c98aec600d8e32537e79ad` |
| T46 | `02 Атлас моделей/Семейства/Мультимодальные семейства.md` | `/models/families/multimodal/` | `fc9420308d6dcc9d9d7dfbef3b6bae6c62946107c046f3eb12ec79f964b79e3c` |
| T47 | `00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md` | `/textbook/multimodal/unified-sequence-chameleon/` | `8524b8cf70ac004e191060ce8db8deee0f6eae51ddb2b2c746b4d0bed419e6c9` |
| T48 | `00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений.md` | `/textbook/multimodal/diffusion-flow-images/` | `9cfeb822081048c5f7e46e6ab75b69029ef09521f1b8de58c51c718d6e8a969c` |
| T49 | `00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md` | `/textbook/multimodal/video-audio-omni/` | `5fa2da00592cdf652c45368a90bd8434778e0f1ff0fb241cd98396343fcb9b63` |
| T50 | `06 Практика/19 Найти автомобильный номер в видеопотоке.md` | `/practice/19-nayti-avtomobilnyy-nomer-v-videopotoke/` | `a64efba17e40ed80e36d4686b6f4db818e59412e416e079aa195f542254989c8` |
| T51 | `00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md` | `/textbook/text-representation/numerical-representation/` | `990e2bf4db0245edc2a4cde974d7462f485bc349d3274c31ddecf96b07e483f2` |
| T52 | `00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md` | `/textbook/text-representation/word2vec-glove/` | `08eb673df9123ac39d3aa702505791068bf510d2764e7772e58e455bed018aab` |
| T53 | `00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md` | `/textbook/text-representation/tokenization/` | `e56be63c263f8b13212592aa24c75158d457cc4b00e1d9247ef4a3888536a59c` |
| T54 | `00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель.md` | `/textbook/language-modeling/ngram/` | `91357ea247a1316c9b3d68bcabc4d184d9ed6be3d698d3ace620fb0b213a2024` |
| T55 | `00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md` | `/textbook/recurrent-networks/rnn-bptt/` | `7d5a399b7264f41015ef6be3262a30c7611cb1853e832a60bb26e722fd3f9bf5` |
| T56 | `00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md` | `/textbook/recurrent-networks/lstm-gru/` | `f977e07887b104b327e3ea8e60c5a815e0c172767983b6c1e4965671c5c82b74` |
| T57 | `00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md` | `/textbook/recurrent-networks/seq2seq-bottleneck/` | `d656fe42e038122df03bb8c16c9f2031e761d8d4ab51c53d9b72b34fcd7b8990` |
| T58 | `00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md` | `/textbook/transformer/from-seq2seq-to-transformer/` | `0332b8584e2d3bded9f082d7a7ebfd3f62d61b8fa095fd4958cc6f318881c476` |
| T59 | `00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning.md` | `/textbook/retrieval/embeddings-metric-learning/` | `955689acfc58fc4df225f3b3c6cb46e96879617710d1f1440441e3d422417931` |
| T60 | `00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid.md` | `/textbook/retrieval/sparse-dense-hybrid/` | `4b6fdfa92ac5e7da70a99910cc651d539941432080b217679768bf7e389d64ba` |
| T61 | `00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction.md` | `/textbook/retrieval/reranking/` | `7fb98ef230b112506fcd930350804caa502a1a4b98653095c31215ca2d0f719c` |
| T62 | `00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md` | `/textbook/rag/full-pipeline/` | `b29009604b9b9a4cbe018a6c04c67f7114d067fd19b3ee88ce30e405e645b41e` |

### Ссылки: 04 Вопросы/_index.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 11 | T01 | — page link |
| 12 | T02 | — page link |
| 13 | T03 | — page link |
| 14 | T04 | — page link |
| 15 | T05 | — page link |
| 16 | T06 | — page link |
| 17 | T07 | — page link |
| 18 | T08 | — page link |

### Ссылки: 04 Вопросы/Вопросы по агентам.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 14 | T09 | `1-что-добавляет-инструмент` |
| 16 | T09 | `2-интерфейс-инструмента-является-частью-задачи` |
| 18 | T09 | `3-вызов-функции-генерация-кода-и-текстовые-действия` |
| 20 | T09 | `5-ошибка--часть-протокола` |
| 22 | T09 | `7-когда-workflow-лучше-свободного-агента` |
| 27 | T10 | `1-из-чего-состоит-один-шаг` |
| 29 | T10 | `4-состояние-не-равно-стенограмме` |
| 31 | T10 | `3-стабильный-префикс-и-динамический-суффикс` |
| 33 | T10 | `6-compaction-как-преобразование-состояния` |
| 35 | T10 | `7-reminder-tokens--не-особый-вид-токена` |
| 37 | T10 | `9-повторы-и-частичные-побочные-эффекты` |
| 42 | T11 | `3-какие-виды-памяти-действительно-нужны` |
| 44 | T11 | `4-конвейер-записи` |
| 46 | T11 | `5-извлечение-не-только-nearest-neighbors` |
| 48 | T11 | `6-план-как-проверяемая-гипотеза` |
| 50 | T11 | `8-когда-нужен-второй-агент` |
| 52 | T11 | `10-контракт-между-агентами` |
| 57 | T12 | `1-сначала-определяется-успешное-состояние` |
| 59 | T12 | `2-исход-траектория-и-процесс` |
| 61 | T12 | `7-несколько-запусков-вместо-одной-траектории` |
| 63 | T12 | `8-парное-сравнение-модели-и-исполняющей-оболочки` |
| 65 | T12 | `10-безопасность-оценивается-как-инвариант` |
| 67 | T12 | `14-минимальный-отчёт` |
| 72 | T11 | `реактивный-агент-поиск-в-реальной-среде-и-model-based-planning` |
| 74 | T11 | `webdreamer-вообразить-переход-до-необратимого-действия` |
| 76 | T13 | `environment-reset` |
| 78 | T13 | `action-log` |
| 80 | T14 | `4-разделить-outcome-trajectory-и-process-grading` |
| 85 | T15 | `где-находится-управление-свободный-цикл-aci-и-процедурный-workflow` |
| 87 | T15 | `почему-генерация-патча-и-поиск-уязвимости--разные-задачи` |
| 89 | T15 | `как-представить-страницу-html-accessibility-tree-screenshot-и-set-of-marks` |
| 91 | T15 | `best-first-search-в-реальной-web-среде` |
| 93 | T15 | `когда-дополнительный-поиск-окупается` |
| 95 | T15 | `верификатор-как-фильтр-синтетической-траектории` |
| 97 | T15 | `osworld-восстанавливаемая-vm-исполняемые-действия-и-проверка-результата` |
| 99 | T15 | `taco-совместное-обучение-токенов-рассуждения-и-действий` |
| 104 | T16 | `три-разрыва-постановка-шаги-доказательства-и-смысловая-эквивалентность` |
| 106 | T13 | `premise-selection` |
| 108 | T16 | `граница-автоматической-проверки` |
| 110 | T16 | `copra-поиск-с-состоянием-возвратом-и-обратной-связью-prover` |
| 112 | T17 | `когда-абстракция-ускоряет-поиск-а-когда-закрепляет-ошибку` |
| 114 | T18 | `6-разделить-подгонку-структурное-восстановление-и-перенос` |
| 119 | T19 | `инструкция-внутри-страницы--данные-а-не-полномочие` |
| 121 | T13 | `least-privilege-и-policy-enforcement` |
| 123 | T14 | `5-построить-недоверенный-retrieval-корпус` |
| 125 | T14 | `7-провести-повторные-испытания` |

### Ссылки: 04 Вопросы/Вопросы по архитектурам.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 14 | T20 | — page link |
| 16 | T21 | `главный-переключатель-матрица-видимости` |
| 18 | T21 | `сравнение-без-рекламных-упрощений` |
| 20 | T22 | `один-encoder--разные-способы-чтения-выхода` |
| 22 | T23 | `перенос-задача-становится-данными-а-не-новой-головой` |
| 27 | T24 | `второй-этап-задача-становится-последовательностью` |
| 29 | T25 | `что-показали-эксперименты` |
| 31 | T26 | `где-находится-обучение` |
| 33 | T27 | `что-унаследовано-от-gpt` |
| 35 | T28 | `как-четыре-механизма-складываются-в-один-блок` |
| 37 | T29 | `почему-абсолютные-углы-дают-относительное-смещение` |
| 42 | T30 | `размер-kv-кэша-выводится-из-форм-тензоров` |
| 44 | T31 | `5-decoupled-rope-содержание-сжимается-позиция-хранится-отдельно` |
| 46 | T32 | `sliding-и-sparse-attention-граф-связей-вместо-полного-квадрата` |
| 48 | T33 | `position-wise-не-значит-без-контекста` |
| 50 | T33 | `gating-значение-и-пропуск-в-разных-проекциях` |
| 52 | T34 | `router-и-top-k` |
| 54 | T34 | `serving-flops-экономятся-latency--не-автоматически` |
| 59 | T35 | `mamba-избирательная-модель-пространства-состояний` |
| 61 | T35 | `rwkv-attention-подобное-смешивание-как-recurrence` |
| 63 | T35 | `retnet-три-эквивалентных-формы-retention` |
| 65 | T35 | `почему-гибриды-возвращают-attention` |
| 70 | T36 | — page link |
| 72 | T37 | — page link |
| 74 | T38 | — page link |
| 76 | T39 | — page link |
| 78 | T40 | — page link |
| 80 | T41 | `как-читать-карточку` |

### Ссылки: 04 Вопросы/Вопросы по мультимодальным моделям.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 11 | T42 | `patch--это-позиция-в-последовательности-а-не-слово-из-словаря` |
| 12 | T42 | `5-что-именно-выучивает-контрастивная-модель` |
| 13 | T42 | `clip-и-siglip-две-формы-контрастивного-обучения` |
| 14 | T43 | `что-именно-вычисляет-resampler` |
| 15 | T43 | `7-что-обучать-а-что-замораживать` |
| 16 | T44 | — page link |
| 17 | T45 | `anyres-глобальный-кадр-и-локальные-плитки` |
| 18 | T45 | `qwen2-vl-native-resolution-merger-и-mrope` |
| 19 | T46 | `qwen-vl--qwen2-vl--qwen3-vl-что-меняется-в-интерфейсе` |
| 20 | T47 | `1-две-независимые-оси-представление-и-место-объединения` |
| 21 | T47 | `3-как-изображение-становится-дискретными-кодами` |
| 22 | T48 | `1-как-создаётся-учебный-пример` |
| 23 | T48 | `5-u-net-dit-и-objective--разные-оси` |
| 24 | T49 | `почему-аудиоголова--неоднозначное-название` |
| 25 | T49 | `moshi-временная-и-глубинная-авторегрессия` |
| 26 | T50 | — page link |

### Ссылки: 04 Вопросы/Вопросы по NLP.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 17 | T51 | `символ-байт-слово-и-токен` |
| 19 | T51 | `словарь-и-идентификаторы` |
| 21 | T51 | `one-hot-как-точная-но-неудобная-запись-адреса` |
| 23 | T51 | `выбор-вектора-из-таблицы-embedding-lookup` |
| 25 | T51 | `bag-of-words-порядок-отброшен-намеренно` |
| 28 | T51 | `матрица-токены--признаки-не-является-контекстом` |
| 34 | T52 | `2-значение-видно-по-окружению` |
| 37 | T52 | `3-самый-прямой-способ--посчитать-соседей` |
| 39 | T52 | `4-word2vec-выучим-векторы-через-простую-задачу` |
| 41 | T52 | `5-почему-нужна-выборка-отрицательных-примеров` |
| 43 | T52 | `6-glove-аппроксимация-логарифма-совместных-частот` |
| 45 | T52 | `7-что-появилось-в-пространстве` |
| 47 | T52 | `8-один-вектор-не-справляется-с-многозначностью` |
| 50 | T52 | `11-векторные-представления-для-поиска--отдельная-задача` |
| 55 | T53 | `1-у-текста-нет-единственного-правильного-разбиения` |
| 57 | T53 | `2-токенизатор--часть-модели` |
| 59 | T53 | `3-построим-bpe-вручную` |
| 61 | T53 | `4-почему-современные-варианты-bpe-начинаются-с-байтов` |
| 63 | T53 | `5-зачем-gpt-использует-предварительное-разбиение-регулярным-выражением` |
| 65 | T53 | `7-wordpiece` |
| 68 | T53 | `8-unigram` |
| 71 | T53 | `9-sentencepiece--библиотека-а-не-один-алгоритм` |
| 74 | T53 | `10-размер-словаря-две-противоположные-цены` |
| 77 | T53 | `11-почему-токенизация-влияет-на-языки` |
| 79 | T53 | `13-специальные-токены-и-шаблон-диалога` |
| 85 | T54 | `цепное-правило` |
| 87 | T54 | `цепное-правило` |
| 89 | T54 | `оценка-по-счётчикам` |
| 91 | T54 | `почему-нулевая-частота-разрушает-произведение` |
| 93 | T54 | `backoff-и-interpolation` |
| 96 | T54 | `идея-kneserney` |
| 99 | T54 | `логарифмы-и-perplexity` |
| 101 | T54 | `зачем-изучать-n-граммы-перед-нейронной-lm` |
| 106 | T55 | `состояние-вместо-окна` |
| 108 | T55 | `состояние-вместо-окна` |
| 110 | T55 | `bptt-backpropagation-по-развёрнутой-сети` |
| 112 | T55 | `bptt-backpropagation-по-развёрнутой-сети` |
| 115 | T55 | `усечение-и-ограничение-нормы` |
| 117 | T55 | `что-rnn-умеет-и-где-ломается` |
| 123 | T56 | `lstm-память-и-три-решения` |
| 125 | T56 | `lstm-память-и-три-решения` |
| 128 | T56 | `вычислимый-пример-одной-координаты` |
| 130 | T56 | `gru-объединить-память-и-выход` |
| 131 | T56 | `что-выбирать` |
| 136 | T57 | `условная-языковая-модель` |
| 138 | T57 | `формы-на-одном-примере` |
| 141 | T57 | `декодирование--не-обучение` |
| 144 | T57 | `почему-фиксированный-контекст-становится-узким-местом` |
| 147 | T58 | `2-оставим-не-резюме-а-память` |
| 150 | T58 | `как-получить-веса` |
| 153 | T58 | `3-проследим-один-перевод` |
| 155 | T58 | `4-bahdanau-и-luong--похожий-принцип-разные-сборки` |
| 158 | T58 | `5-что-именно-улучшил-attention` |
| 159 | T58 | `6-почему-этого-всё-ещё-недостаточно` |
| 162 | T58 | `7-три-термина-которые-нельзя-смешивать` |

### Ссылки: 04 Вопросы/Вопросы по Retrieval и RAG.md

| Source line | Target | Разрешённый fragment |
|---:|---|---|
| 14 | T59 | `1-от-токенных-представлений-к-представлению-последовательности` |
| 16 | T59 | `2-почему-обычный-bert-не-является-готовой-моделью-предложений` |
| 18 | T59 | `4-пары-определяют-геометрию` |
| 20 | T59 | `7-как-проверять-эмбеддинг` |
| 25 | T60 | `6-гибридная-выдача` |
| 27 | T60 | `4-что-именно-индексировать` |
| 29 | T60 | `5-query-processing` |
| 31 | T60 | `8-диагностика-retrieval` |
| 33 | T61 | `1-bi-encoder-и-cross-encoder-решают-разные-вычислительные-задачи` |
| 35 | T61 | `4-late-interaction-промежуточная-точка-colbert` |
| 37 | T61 | `5-cascade-и-бюджет-вычислений` |
| 42 | T62 | `1-что-добавила-исходная-архитектура-rag` |
| 44 | T62 | `4-chunking--гипотеза-проверяемая-на-запросах` |
| 46 | T62 | `8-сборка-контекста` |
| 48 | T62 | `9-генерация-и-цитаты` |
| 50 | T62 | `10-оценивание-по-этапам` |
| 52 | T62 | `11-failure-modes` |
| 54 | T62 | `12-эксплуатация` |
| 56 | T62 | `13-когда-rag-не-нужен` |

## Итог и ограничения

Full-read 6/6; source-level links 172/172; fragments 156/156; unchanged question pages 6/6. Новых original findings не создано; существующие 442 findings не пересчитаны и не изменены. Пробел чтения закрыт фактическим чтением, но итоговая приёмка rendered links остаётся отдельной работой root.

Навык `llm-wiki` использован для разделения механической проверки ссылок, смыслового соответствия и неизменяемого source layer. Широкий vault lint, создание новых определений, правки логов/индексов и автоматическое заполнение ответов не выполнялись: они вне ownership этого задания.
