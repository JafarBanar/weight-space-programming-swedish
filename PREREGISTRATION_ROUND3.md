# Pre-registration, round 3: is there a fix that works?

Date: 2026-11-19. Written and committed before any round-3 development or test run.

## Why a third round

Rounds 1 and 2 ended with a negative result: weight edits found by screening the weights did not make the model
answer in Swedish, and they damaged Danish output. Round 2 also showed what the real failure is. On human-translated
questions `Qwen2.5-1.5B-Instruct` answers in **English** half the time when asked in Swedish, and about half the time
when asked in Danish. So the target changes. The goal of round 3 is:

> **Make the model answer in the language of the question (Swedish, Danish, Norwegian, English), with no added
> instruction at inference time, without lowering language quality in any of the four.**

This round asks whether any method reaches that goal, and how much model feedback each method needs. It does not
re-test the weight screen.

## What was already known when this was written

* Everything in rounds 1 and 2, including: prompt instruction "Svara alltid på svenska." gives 96% Swedish on the
  round-2 questions but is Swedish-only; "answer in the same language as the user's message" gives 79% Swedish and
  67% Danish; steering (S2) gives 60% Swedish and lowers Danish; the round-1 LoRA (80 synthetic prompts, Swedish only)
  removed drift on round-1 prompts at a 20% perplexity cost and was never tested on human-translated questions.
* **Not yet run:** anything on the fresh data below, and none of the new methods on any data.

## Fresh data

* **MKQA round-3 split** (`src/prompts3.py`, seed 3): questions from Apple MKQA that were **not** in the round-2
  dev or test split, selected by the same automatic rule (Swedish, Danish and English versions each identified as that
  language, at least five words). 300 train, 100 dev, 300 test questions, each in Swedish, Danish, Norwegian and
  English (professional translations). The Norwegian versions are new to this project.
* Reused, not fresh, and labelled so: the round-2 template prompts (T1), the native and synthetic grammar pairs, and
  the perplexity sentences (UD test sets). They are regression checks here.
* Regularisation and development text: UD **train** splits (Swedish Talbanken, Danish DDT, Norwegian Bokmål,
  English EWT), 200 sentences per language. The UD test sentences used for perplexity are not used in development.

## Methods

All methods are developed on the train and dev questions only, frozen in `results/r3/frozen.json`, committed, and then
run once on the test questions.

| Code | Method | Changes weights? | Model feedback used |
|---|---|---|---|
| B0 | Unedited model | no | none |
| P1 | "Svara alltid på svenska." (Swedish-only reference) | no | none |
| P2 | "Always answer in the same language as the user's message." (round 2) | no | none |
| P3 | Best of six request-aware system prompts, chosen on dev | no | dev answers for 6 prompts |
| W0a | Round-2 neuron edit (layer 23 neuron 981) | 1,536 weights | round 1 and 2 |
| S2 | Round-2 steering vector, unchanged | no (run-time) | round 2 |
| E1 | **Closed-form conditional edit**: in one MLP output matrix, add a rank-3 update `Σ_k α·v_k·p_kᵀ`. `p_k` is a ridge-regression probe that reads "the user wrote in language k" from the MLP's hidden activations; `v_k` is the mean difference in the residual stream between answers in language k and answers in English to prompts in language k. No gradient descent. | one matrix, rank 3 | activations on train questions; 9 dev runs to pick layer and strength |
| R3 | Control for E1: same matrix, same rank, same Frobenius norm, random directions | one matrix, rank 3 | none |
| F3 | **Learned edit of the same shape as E1**: rank-3 LoRA on the same single matrix, trained by gradient descent | one matrix, rank 3 | gradient descent on train questions |
| F2 | **Language-aware LoRA distillation**: rank 8 on the attention q/v projections of all layers | 1.1 M parameters | gradient descent on train questions |

Training targets for F2 and F3 ("prompt distillation"): the unedited model's own answers to the train questions when a
system instruction names the language ("Svara alltid på svenska." / "Svar altid på dansk." / "Svar alltid på norsk." /
none for English), kept only when the detector labels the answer as that language. The model is trained to give those
answers **without** the instruction. A second loss term keeps the tuned model's next-token distribution close to the
unedited model's on UD train sentences in the four languages (KL divergence, weight 1), to protect perplexity.

Fixed in advance: E1 ridge strength 0.01 × mean diagonal of the Gram matrix; E1 grid layers {6, 10, 14} × strength
{1, 2, 4}; F3 uses the layer chosen for E1, learning rate from {2e-4, 1e-3}, 2 epochs; F2 learning rate 2e-4, checkpoint
after epoch 1 or 2; batch size 8; seed 0.

**Dev selection rule (all methods):** on the 100 dev questions in four languages, take the mean of the four
right-language rates. Exclude a configuration if English→English is more than 2.5 points below the unedited model, if
the share of degenerate answers is more than 2.5 points above it, or if Swedish perplexity on 100 Talbanken *train*
sentences is more than 5% above it. Pick the highest mean; break ties towards the weaker intervention. If every
configuration of a method is excluded, the method is reported as failing in development and is still run on the test
set with its best non-degenerate configuration, labelled as such.

## What counts as "works"

A method works if all five hold on the test data (300 questions per language, greedy decoding, default system prompt):

* **G1 Swedish.** Swedish questions answered in Swedish at least 90% of the time, and the 95% interval of the gain
  over B0 (paired bootstrap over questions) excludes zero.
* **G2 No collateral.** Danish→Danish, Norwegian→Norwegian and English→English each no more than 2.5 points below B0.
* **G3 Language quality.** Perplexity on UD test text rises by at most 5% in each of the four languages; accuracy on
  native and synthetic grammar pairs moves by less than 3 points.
* **G4 Content.** On Swedish questions with a short gold answer, the share of answers containing it is no more than
  2 points below B0.
* **G5 Other prompts.** On the round-2 template prompts (T1), Swedish answers no more than 1 point below B0.

Danish and Norwegian are easily confused by the detector, so G2 is reported both strictly and with Danish and Norwegian
merged; the strict version decides.

Priors that the method passes all five: P3 25%, E1 20%, F3 35%, F2 55%, S2 5%, W0a 2%. Prior that at least one
weight-changing method passes: 60%.

## Secondary analyses (decided now, run after the main test)

* **Decoding.** For B0 and every passing method: sampling (temperature 0.7, top-p 0.9, three seeds) and greedy
  answers of 160 tokens, on the Swedish and Danish test questions.
* **Unseen system prompt.** For B0 and every passing weight method: the test questions under an English system prompt
  that was not used in training ("You are a concise assistant for a public library.").
* **Second model.** The best passing weight method is rerun on `Qwen2.5-0.5B-Instruct` with the same hyperparameters
  and no retuning (layer index scaled by depth), and judged by G1–G5 with the G1 level set to "at least halves the
  wrong-language answers" because that model is weaker. Prior that it passes: 45%.
* **Second language identifier.** Every language label in rounds 1–3 is recomputed with an independent identifier and
  the agreement with the primary detector is reported, with G1 and G2 recomputed under it.

## What would change the recommendation

If a weight-changing method passes G1–G5 and replicates on the second model, the recommendation gains a tested fix,
and the comparison E1 against F3 says how much gradient feedback buys over a closed-form construction of the same
size. If only prompts pass, the recommendation stays "prompt first". If nothing passes, that is reported as it is.
