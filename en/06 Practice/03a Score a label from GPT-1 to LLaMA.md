---
title: "GPT-1 to LLaMA: from class scores to text labels"
type: practice
status: active
locale: en
translation_status: current
canonical_revision: "3c122c97ea3534cd2547f8a755cace24f68dde4499bff1f5720bbbbe8e9c7e22"
translation_reviewed: "2026-11-13"
translation_of: "06 Практика/03a Оценить текстовую метку от GPT-1 до LLaMA.md"
last_updated: 2026-11-13
---

# GPT-1 to LLaMA: from class scores to text labels

## Work it out by hand

“A cat sleeps” entails “An animal rests.” The correct relation is
`entailment`. The numbers in this exercise are supplied, so the candidate
that wins by score need not be the correct answer.

In [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]],
`Delim` separates the two sentences and `Extract` comes after both.
A linear head takes the state at `Extract` and produces one logit for
each of `entailment`, `neutral`, and `contradiction`. With logits
$(1,0,-1)$, softmax gives approximately $(0.6652,0.2447,0.0900)$.
The first class wins. The spelling of its label does not enter the
calculation. The head and decoder are fine-tuned together on labeled
pairs ([Radford et al., §3.2–3.3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3)).

In the [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-relation-prefix|GPT-2 example]],
the input ends at `Relation:`. After the colon, ` entailment` takes two
tokens, 39793 and 434. Give the first token probability 0.4 and the second
probability 0.5 given the first. The full string has probability
$0.4\cdot0.5=0.2$. If ` neutral` has probability 0.25 and
` contradiction` has probability 0.1, `neutral` wins. For
`entailment` to win, the second token's probability must exceed
$0.25/0.4=0.625$. At exactly 0.625 the two scores tie. Unlike a softmax
over three classes, the probabilities of these three strings need not
sum to one. We are scoring fixed candidates rather than generating freely.
See [the GPT-2 report, §§2–3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=2)
for the protocol without fine-tuning.

## What changes with demonstrations and tokenization

With [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|GPT-3]],
put the labeled example about the open and shut door before the test
pair. With two examples, the pair about the robin goes first. The
instruction and test pair stay the same, and the test label stays out
of the prompt. No weights change. A real model could assign different
probabilities under the new prefix. Here only the slider changes them,
so you can inspect the prompt and the scoring rule separately
([Brown et al., §2](https://arxiv.org/html/2005.14165v4#S2)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-continuity-2026-11-10/min-figure2-demonstrations.png|Three labeled rows and one test row with a blank go into the LM. Positive appears separately as its prediction.]]

*Sewon Min et al., [Rethinking the Role of Demonstrations, Figure 2, p. 11049](https://aclanthology.org/2022.emnlp-main.759.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure with its labels unchanged.*

The figure uses sentiment classification. Three labeled rows enter
the prompt and the fourth has a blank. `Positive`, below the LM,
is not in the prompt. In our task that position
corresponds to the label selected after the last `Relation:`.

In [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt3-to-llama1|the first LLaMA]],
the same label after `Relation:` takes three tokens. Set the third
conditional probability to 0.8 to isolate its contribution. At a
second-token probability of 0.65, two factors give $0.4\cdot0.65=0.26$;
three give $0.4\cdot0.65\cdot0.8=0.208$. Keep the `neutral` score at
0.25 and the winning candidate changes. Holding the other scores fixed
is a condition of the exercise. Real models have different vocabularies
and weights, so this does not establish which model is better at entailment.
See [LLaMA, §2.1](https://arxiv.org/html/2302.13971v1#S2.SS1) for the tokenizer.
