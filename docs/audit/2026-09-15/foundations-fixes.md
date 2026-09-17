# Foundations — подтверждённые точечные исправления

Дата: 2026-09-15. После полного чтения 55/55 страниц исправлены только пять отдельно согласованных глав. Исходный постраничный аудит `foundations.json`/`foundations.md` остаётся снимком **до** исправлений: его `read_sha256` не заменяются хешами новой версии. Остальные P2/P3 из аудита не объявляются устранёнными. Коммитов и push не было, чужие изменения не откатывались.

## 1. GMM: согласованный E-step и M-step

[Глава Gaussian mixture и EM](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md>).

До: для начальных μ=(0,4), π=(0.5,0.5), σ²=(1,1) таблица объявляла γ₁=(0.99,0.95,0.05,0.01), N=(2,2), μ_new=(0.6,4.4). Эти responsibilities не следовали из заданных параметров и противоречили собственному расчёту γ₁(1)≈0.982.

После: γ₁(x)=1/(1+exp(4x−8)); таблица, взвешенные суммы, N, μ и π пересчитаны из неокруглённых значений. Дисперсии явно фиксированы; это ограниченный вариант EM, а не пропущенное обновление covariance. Получены μ_new≈(0.496153,4.468138), π_new≈(0.495505,0.504495). Observed log-likelihood вырос −7.429516 → −6.944253. Удалён только конфликтующий ручной расчёт; причинная линия и иллюстрации сохранены.

Источник механики: [официальное изложение GMM/EM scikit-learn](https://scikit-learn.org/stable/modules/mixture.html). Числа рассчитаны локально независимо; это не заимствованные результаты статьи.

## 2. BPE: исполнимая последовательность merges

[Глава BPE, WordPiece и Unigram](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md>).

До: после (a,a)→Z предлагалась отсутствующая пара (Z,b). После: (a,a)→Z, (Z,a)→Y, (Y,b)→X. Полные состояния: `aaabdaaabac → ZabdZabac → YbdYbac → XdXac`. Добавлены частоты, правило tie-breaking по первому появлению, различие перекрывающегося подсчёта и неперекрывающейся замены, раскрытие новых токенов и round-trip. Исходные английские CS336 frames и код `get_stats/merge` не изменены.

Первичный код: [Karpathy minbpe/base.py](https://raw.githubusercontent.com/karpathy/minbpe/master/minbpe/base.py). Регрессионная проверка исполняет именно функции из главы и проверяет, что каждая объявленная пара наиболее частая при указанном tie-break.

## 3. Ranking: held-out positives не управляют training sampler

[Глава Ranking и нейронные модели](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md>).

До: рекомендовалось «исключение будущих positives». После: negative pool строится только из каталога на train cutoff и positives обучающего snapshot. Для streaming явно оговорён отдельный point-in-time cutoff каждого шага. Пример {A,B,C}, observed={A} сохраняет pool={B,C}, даже если B окажется положительным после cutoff. Добавлен простой helper с двумя входами; held-out labels в его контракт не входят. Остальной ranking material не переписывался.

Опоры: [BPR, §3–4](https://arxiv.org/abs/1205.2618), [Ji et al. — temporal leakage](https://arxiv.org/abs/2010.11060). Разделение train cutoff и будущих labels — применение этих постановок к явно заданному temporal contract, не цитата авторов.

## 4. Evaluation: candidates не восстанавливаются из ground truth

[Глава Оценивание и эксплуатация](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md>).

До: `candidates |= positives` стояло после availability/forbidden filters; текст требовал принудительно возвращать test positive. Карточка одновременно называла retrieval top-1000 и full-catalog evaluation.

После: раздельно названы sampled ranker benchmark, full-catalog ranking и end-to-end replay. В replay candidates приходят из retrieval и пересекаются с допустимым каталогом; ground truth используется лишь оценщиком. Пропущенный eligible positive остаётся в denominator и даёт промах. Для этой учебной реализации явно выбрана оценка Recall/NDCG по eligible relevant items; excluded targets и запросы без eligible target обязательно подсчитываются отдельно. Это не молчаливое удаление неудобных users. Другой продуктовый denominator допустим только как отдельный заранее объявленный контракт. Карточка теперь соответствует replay.

Опоры: [Krichene & Rendle — sampled metrics](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/), [Ji et al. — global timeline](https://arxiv.org/abs/2010.11060). В локальном контрпримере доступны A/B/C, B запрещён, D ещё отсутствует, retrieval пропускает relevant C. Ranker получает только A, candidate recall=0. Замена test labels не меняет scored candidates. Случай без eligible target отражается отдельным счётчиком.

## 5. Encoder–Decoder: источник cross-attention memory

[Legacy-глава Encoder–Decoder](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/02 Encoder-Decoder Transformer.md>).

До: подпись передавала выход каждого encoder-блока каждому decoder-блоку. После: общей памятью служит выход последнего encoder-блока, то есть всего стека; промежуточные encoder-слои отдельно не передаются. Исправлена подпись, изображение не редактировалось и не рендерилось.

Первичный источник: [Attention Is All You Need, §3.1](https://arxiv.org/html/1706.03762v7). Проверка здесь текстовая и source-based, не визуальная.

## Проверки до и после

Навык verification-before-completion использован для порядка «воспроизвести → исправить → повторить ту же проверку». Первые восемь regression checks до изменений дали 8 FAIL (exit 1), после четырёх правок — 8 OK (exit 0). После добавления девятого теста на ещё не исправленную подпись получилось 1 FAIL; после пятой правки — **9/9 OK, exit 0**. Это девять проверок пяти дефектов и уточнённых контрактов, а не девять независимых содержательных ошибок.

Проверки используют стандартную библиотеку Python. Из глав исполняются только целевые учебные `get_stats/merge`, negative-pool helper и evaluation loop с локальными mock objects; сеть и реальные модели не вызываются. Для GMM проверяются отображённые значения таблицы с допуском округления 5.1e−7, четыре обновлённых параметра и рост likelihood. Это не проверка всей математики/всего production pipeline.

Команда из корня worktree, исполняющая сохранённый ниже тест без зависимости от временного файла:

```bash
awk '/^```python$/{copy=1;next} copy && /^```$/{exit} copy{print}' docs/audit/2026-09-15/foundations-fixes.md | python3 - .
```

Вывод успешного запуска:

```text
test_bpe_declared_merges_are_greedy_and_roundtrip (__main__.FoundationsRegression.test_bpe_declared_merges_are_greedy_and_roundtrip) ... ok
test_encoder_memory_is_final_stack_output (__main__.FoundationsRegression.test_encoder_memory_is_final_stack_output) ... ok
test_gmm_e_step_table (__main__.FoundationsRegression.test_gmm_e_step_table) ... ok
test_gmm_m_step_and_likelihood (__main__.FoundationsRegression.test_gmm_m_step_and_likelihood) ... ok
test_negative_pool_uses_training_snapshot_only (__main__.FoundationsRegression.test_negative_pool_uses_training_snapshot_only) ... ok
test_replay_candidates_do_not_depend_on_test_labels (__main__.FoundationsRegression.test_replay_candidates_do_not_depend_on_test_labels) ... ok
test_replay_card_labels_its_actual_universe (__main__.FoundationsRegression.test_replay_card_labels_its_actual_universe) ... ok
test_replay_counts_missing_eligible_positive_as_miss (__main__.FoundationsRegression.test_replay_counts_missing_eligible_positive_as_miss) ... ok
test_replay_keeps_forbidden_and_future_items_out (__main__.FoundationsRegression.test_replay_keeps_forbidden_and_future_items_out) ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.002s

OK
BPE: ['ZabdZabac', 'YbdYbac', 'XdXac'] roundtrip OK
GMM: gamma= [0.9996646498695336, 0.9820137900379085, 0.0003353501304664781, 6.144174602214718e-06] N= [1.9820199342125109, 2.0179800657874893] mu= [0.4961533910220236, 4.468138333689933] pi= [0.4955049835531277, 0.5044950164468723] loglik= -7.429515970201393 -> -6.9442527585648985
```

Также выполнен `git diff --check --` с пятью путями выше: exit 0, вывод пуст. Полный diff пяти глав перечитан. Из 55 файлов после исправлений изменились ровно пять согласованных; изображения и английские оригиналы не менялись. Публикационный build и визуальный рендер не запускались в этом пакете.

## Хеши версий

```json
[
  {
    "path": "00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md",
    "before_sha256": "b00433a609a57470c0d6fb0c06e6eaaff0c66ec6300e496fbf5c83a3dc07674a",
    "after_sha256": "ff0615a6f8110c1b798f46ee18b1b46cc0f2c79d9ff14299fa28582307b82334"
  },
  {
    "path": "00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md",
    "before_sha256": "9a18818c5428cd84be3b5ece14609c57d59ae17116d7d999ad100441178708d9",
    "after_sha256": "f6fe241fe2a1e28ce5e1331bff91a56b2cafdb938b5227da9cc98ba692e75213"
  },
  {
    "path": "00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md",
    "before_sha256": "df278808f710dfaa6389d9be0c97c7f380184130655b112e7408195b6032883f",
    "after_sha256": "fd9632eb3b0f24fcfa094ca6cafe1d473e726abbbbb73c3c8fd449edd6775af6"
  },
  {
    "path": "00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md",
    "before_sha256": "9559164ee9419d0474fdcceb07277db7983246e6a4799d6c71555f693d51cb9d",
    "after_sha256": "b3e4fdc4a4e740f643ab75b28e62ec14732903f60318096f2095a6ded1c946a8"
  },
  {
    "path": "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/02 Encoder-Decoder Transformer.md",
    "before_sha256": "35491537818b5547a2761c4b87add342e3fde1f0722aef7da13f34ef5fbcf103",
    "after_sha256": "faf71baf44a0d7310c55b15f0621abff8d70647d03d914817079d142715c311d"
  }
]
```

## Полный локальный regression script

```python
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(sys.argv.pop(1)).resolve()
GMM = "00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md"
BPE = "00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md"
RANK = "00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md"
EVAL = "00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md"
ENCDEC = "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/02 Encoder-Decoder Transformer.md"

def read(path):
    return (ROOT / path).read_text()

def blocks(path):
    return re.findall(r"```python\n(.*?)\n```", read(path), re.S)

def replay(ground_truth):
    source = next(b for b in blocks(EVAL) if "for user in eligible_test_users:" in b)
    scored, metrics, candidate_recalls, eligibility, empty, calls = [], [], [], [], [], []
    class Catalog:
        def available_at(self, time):
            return {"A", "B", "C"}
    class Model:
        def score(self, user, candidates):
            scored.append(set(candidates))
            return {item: {"A": 1, "B": 8, "C": 9, "D": 10}[item] for item in candidates}
    def retrieve(user, time, eligible):
        calls.append(set(eligible))
        return {"A", "B", "D"}  # misses eligible C; attempts to return forbidden/future items
    env = dict(
        eligible_test_users=["u"], test_relevant={"u": set(ground_truth)},
        test_time={"u": 10}, catalog=Catalog(), model=Model(), k=1,
        forbidden_items=lambda user, time=None: {"B"}, retrieve=retrieve,
        take_top_k=lambda scores, k: sorted(scores, key=scores.get, reverse=True)[:k],
        accumulate_metrics=lambda top, positives: metrics.append((set(top), set(positives))),
        accumulate_candidate_recall=lambda candidates, positives: candidate_recalls.append(
            len(set(candidates) & set(positives)) / len(positives)),
        report_target_eligibility=lambda user, total, eligible: eligibility.append((total, eligible)),
        report_no_eligible_target=lambda user: empty.append(user),
    )
    exec(source, env)
    return scored, metrics, candidate_recalls, eligibility, empty, calls

class FoundationsRegression(unittest.TestCase):
    def test_gmm_e_step_table(self):
        rows = {}
        for line in read(GMM).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[0] in {"0", "1", "4", "5"}:
                rows[int(cells[0])] = [float(c.replace(",", ".")) for c in cells[1:]]
        self.assertEqual(set(rows), {0, 1, 4, 5})
        for x, displayed in rows.items():
            g = 1 / (1 + math.exp(4 * x - 8))
            for shown, expected in zip(displayed, [g, g*x, 1-g, (1-g)*x]):
                self.assertAlmostEqual(shown, expected, delta=0.00000051)

    def test_gmm_m_step_and_likelihood(self):
        xs = [0, 1, 4, 5]
        gs = [1 / (1 + math.exp(4*x - 8)) for x in xs]
        ns = [sum(gs), sum(1-g for g in gs)]
        sums = [sum(x*g for x, g in zip(xs, gs)), sum(x*(1-g) for x, g in zip(xs, gs))]
        mus = [s/n for s, n in zip(sums, ns)]
        pis = [n/4 for n in ns]
        for value in mus + pis:
            displayed = f"{value:.6f}".replace(".", "{,}")
            self.assertTrue(displayed in read(GMM), f"missing computed parameter {displayed}")
        def ll(m, p):
            return sum(math.log(sum(w*math.exp(-(x-u)**2/2)/math.sqrt(2*math.pi)
                                    for u, w in zip(m, p))) for x in xs)
        self.assertGreaterEqual(ll(mus, pis), ll([0, 4], [.5, .5]))
        print("GMM:", "gamma=", gs, "N=", ns, "mu=", mus, "pi=", pis,
              "loglik=", ll([0, 4], [.5, .5]), "->", ll(mus, pis))

    def test_bpe_declared_merges_are_greedy_and_roundtrip(self):
        env = {}
        for block in blocks(BPE):
            if block.startswith("def get_stats(") or block.startswith("def merge("):
                exec(block, env)
        rules = re.findall(r"^\(([a-zA-Z]), ([a-zA-Z])\) → ([XYZ])$", read(BPE), re.M)
        self.assertEqual(len(rules), 3)
        ids = list("aaabdaaabac")
        vocab = {c: c for c in ids}
        states = []
        for left, right, new in rules:
            counts = env["get_stats"](ids)
            self.assertEqual((left, right), max(counts, key=counts.get))
            ids = env["merge"](ids, (left, right), new)
            vocab[new] = vocab[left] + vocab[right]
            states.append("".join(ids))
        self.assertEqual(states, ["ZabdZabac", "YbdYbac", "XdXac"])
        self.assertEqual("".join(vocab[i] for i in ids), "aaabdaaabac")
        print("BPE:", states, "roundtrip OK")

    def test_replay_keeps_forbidden_and_future_items_out(self):
        scored, _, _, eligibility, _, calls = replay({"B", "C", "D"})
        self.assertEqual(scored, [{"A"}])
        self.assertEqual(calls, [{"A", "C"}])
        self.assertEqual(eligibility, [(3, 1)])

    def test_replay_counts_missing_eligible_positive_as_miss(self):
        _, metrics, recall, _, _, _ = replay({"C"})
        self.assertEqual(metrics, [({"A"}, {"C"})])
        self.assertEqual(recall, [0.0])

    def test_replay_candidates_do_not_depend_on_test_labels(self):
        self.assertEqual(replay({"C"})[0], replay({"D"})[0])
        _, metrics, recall, eligibility, empty, _ = replay({"B", "D"})
        self.assertEqual(metrics, [])
        self.assertEqual(recall, [])
        self.assertEqual(eligibility, [(2, 0)])
        self.assertEqual(empty, ["u"])

    def test_negative_pool_uses_training_snapshot_only(self):
        helpers = [b for b in blocks(RANK) if b.startswith("def training_negative_pool(")]
        self.assertEqual(len(helpers), 1, "explicit train-only sampler contract missing")
        env = {}
        exec(helpers[0], env)
        pool = env["training_negative_pool"]
        catalog, positives = {"A", "B", "C"}, {"A"}
        self.assertEqual(pool(catalog, positives), {"B", "C"})
        env["test_positives"] = {"B"}
        self.assertEqual(pool(catalog, positives), {"B", "C"})
        self.assertEqual(catalog, {"A", "B", "C"})
        self.assertEqual(positives, {"A"})

    def test_replay_card_labels_its_actual_universe(self):
        self.assertTrue("evaluation: end_to_end_replay" in read(EVAL), "card is not labelled end_to_end_replay")
        self.assertTrue("candidate_universe: eligible_catalog_at_request_time" in read(EVAL), "card omits eligible universe")

    def test_encoder_memory_is_final_stack_output(self):
        text = " ".join(read(ENCDEC).split())
        self.assertTrue("выход последнего encoder-блока" in text,
                        "caption must identify final encoder output")
        self.assertFalse("выход каждого encoder-блока" in text)

if __name__ == "__main__":
    unittest.main(verbosity=2)
```
