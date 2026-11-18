# Pre-registration, round 2 (written and committed before any round-2 test data was run)

Date: 2026-11-19. Purpose: address six weaknesses of round 1 (see `RESEARCH_NOTE.md`, Limitations).

## What was already known when this was written

* All round-1 results, including the post-hoc finding that deleting layer 23 neuron 981 alone (W0a) removes
  most Danish/Norwegian drift on round-1 test prompts, while deleting layer 22 neuron 1821 alone makes it far worse.
* One dev-only probe on human-translated questions (`explore/06`, MKQA dev split, 60 questions per language,
  unedited `Qwen2.5-1.5B-Instruct`): Swedish questions were answered in English 32/60 times, in Swedish 27/60,
  in Norwegian 1/60. So on natural questions the dominant wrong language is English, not Danish.
* A literature survey (summarised in the report): language-selective neurons and wrong-language output are well
  replicated, but published neuron sets number hundreds to thousands, single-neuron effects shrink with model size,
  and nobody has measured Swedish→Danish drift or found language neurons from weights alone.
* **Not yet run, on any model:** the new template prompts (T1), the MKQA test split, the native grammar pairs,
  the three survey models (no forward pass of any kind), and the larger steering search.

## Fresh data (none of it used in round 1)

* **T1:** 320 synthetic Swedish prompts from 40 new templates × 8 new topics (`src/prompts2.py`), with English versions.
* **MKQA** (Apple, CC BY-SA 3.0): Natural Questions queries translated by professional translators.
  300 test questions in Swedish; the first 150 also in Danish and English. A disjoint 120-question dev split is used
  only for the sign step and for building steering directions. Selection of items is seeded and automatic.
* **Native grammar pairs:** 200 sentences from the UD Swedish-Talbanken treebank, each with one indefinite article
  swapped (en↔ett) against the treebank's gold gender annotation (`src/native_pairs.py`).

## A. Confirmation of the post-hoc single-neuron edit (Qwen2.5-1.5B-Instruct)

Edit fixed here: **W0a = zero the output column of layer 23 neuron 981** (1,536 weights). Also reported: W0 (both
neurons), two random single-neuron controls in layer 23, prompt instruction P1.

* **C1.** On T1, W0a cuts wrong-language answers by at least half relative to the unedited model, the 95% interval of
  the difference (bootstrap over templates and topics jointly) excludes zero, Swedish perplexity rises by at most 5%,
  and at most 2.5 points of English→English answers are lost. If fewer than 5% of unedited T1 answers are in the wrong
  language, the test is declared uninformative. *My prior: 65%.*
* **C2.** On MKQA Swedish questions the wrong answers are mostly English, which this edit should not touch: W0a changes
  the share of Swedish answers by less than 5 points in either direction. *Prior: 75%.* If C2 holds, the round-1 headline
  ("15% drift, mostly Danish") is a property of the round-1 prompt set, not of the model in general.
* **C3.** Capability removal, now on human-written Danish: under W0a the share of Danish answers to MKQA Danish
  questions falls below half of its unedited value. *Prior: 85%.*
* **C4.** No new capability: W0a moves accuracy on the native grammar pairs and on the synthetic pairs by less than
  3 points each. *Prior: 90%.*

## B. A frozen, mechanical recipe, and a survey across models

Recipe (identical for every model, no per-model choices):

1. *Weights only.* Build the Swedish-minus-Danish/Norwegian direction from the same 17 word pairs in the output
   embedding; z-score every MLP neuron's write along it within its layer. Candidates = all neurons with z < −6
   ("strong reading"). If there are none, candidates = the two most negative neurons ("weak reading").
2. *One batch of feedback for the sign.* Generate 32 tokens for 160 dev prompts (40 round-1 dev prompts, 60 MKQA-dev
   Swedish, 60 MKQA-dev Danish). For each candidate compare its mean activation on answers labelled Danish/Norwegian
   (a_da) and on answers labelled Swedish (a_sv). **Delete the neuron only if a_da > 0 and a_sv > −0.25·a_da**;
   otherwise keep it (it supports Swedish, or does not push towards Danish).
3. The edit is: zero the output columns of the neurons marked for deletion. If none is marked, there is no edit.

Models (all Apache-2.0, pinned revisions in `src/download_models.py`): `Qwen2.5-1.5B-Instruct` (does the recipe
reproduce W0a?), `Qwen2.5-0.5B-Instruct`, `EuroLLM-1.7B-Instruct`, `Qwen3-1.7B` (thinking disabled).

* **C5.** On Qwen2.5-1.5B the recipe selects exactly {L23/N981}. *Prior: 60%.*
* **S-read.** Each of the three other models has at least one strong outlier neuron (z < −6). *Prior: 50% per model.*
* **S-drift.** The unedited model answers at least 5% of T1 in the wrong language. *Priors: Qwen2.5-0.5B 70%,
  Qwen3-1.7B 50%, EuroLLM-1.7B 25%.*
* **S-fix.** Where the recipe yields an edit and the unedited model gives at least 10 Danish/Norwegian answers on T1,
  the edit at least halves that count and beats both random controls. *Prior that this happens in at least one of the
  three other models: 50%.* A model where the recipe finds nothing, or where its edit does nothing, counts against
  the generality of round 1.

Each model gets the same evaluation as in A: T1, MKQA sv/da/en, perplexity in four languages, both sets of grammar pairs.

## C. A fairer test of activation steering (Qwen2.5-1.5B)

54 configurations instead of 9: three ways to build the direction (A: instruction contrast, as in round 1;
B: mean answer state of Swedish answers minus Danish answers; C: Swedish answers minus drifting answers), layers
6/10/14/18/22/26, strengths 1/2/4. Selection on the round-1 dev-val prompts with the round-1 rule (best Swedish rate;
English and degeneracy within 2.5 points of the unedited model; ties to the weaker intervention). The winner (S2) is then
run once on the fresh test data.

* **C6.** S2 cuts T1 wrong-language answers by at least half. *Prior: 30%.* Either outcome is informative: a success
  means round 1 undersold steering; a failure after a six-fold larger search makes the round-1 conclusion firmer.

## Scoring and analysis

Answer language by the `lingua` detector as in round 1, with the detector-free function-word share as a robustness
check. Intervals: bootstrap over templates and topics jointly for T1 (2,000 resamples), over questions for MKQA.
Synthetic grammar pairs are reported in full and on a cleaned subset that drops any noun whose gender the Wiktionary
check (`results/language_checks.json`) does not confirm or lists as both genders.

## Not done in round 2

The LoRA baseline is not retrained (37 minutes on this laptop); its round-1 result stands. P1 is run only on the
Swedish prompt sets for the three survey models.
