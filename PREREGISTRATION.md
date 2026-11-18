# Pre-registration (written and committed before any weight edit was run)

Date: 2026-11-18. Model: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy decoding, seed 0.

## What was already known when this was written

Only the unedited model had been run, and only on development prompts (`explore/01`–`05`):

* The model answers 14–19% of plain Swedish prompts in Danish or Norwegian ("Scandinavian drift").
  Adding "Svara alltid på svenska." to the system prompt removed it on 48/48 dev prompts, so the
  model *can* answer in Swedish; it sometimes selects the wrong language.
* Reading the weights with **zero forward passes** (`explore/03`, `04`): a Swedish-minus-Danish/Norwegian
  direction built from 17 function-word translation pairs in the embedding matrix; the write strength of
  all 250,880 MLP neurons along it is indistinguishable from noise, except two outliers:
  layer 23 neuron 981 (z = −9.2) and layer 22 neuron 1821 (z = −7.9). Their output weights promote
  Danish tokens (" kvinder", " søger", "ø", " Danish", ".dk") and suppress Swedish ones ("Ö", "ä", " och", " ett").

No edit of any kind had been applied to the model at this point.

## Claim under test

> Scandinavian drift is an *obstacle* (a wrong-language selection failure hiding an existing ability),
> and a tiny targeted weight edit can remove it without adding or removing other capabilities.

## Conditions

| ID | Condition | Model feedback used to build it |
|---|---|---|
| B0 | Unedited model | – |
| P1 | Prompting: system prompt + "Svara alltid på svenska." | none |
| W0 | **Weights-only edit, fixed here in advance:** set the output columns of L23/N981 and L22/N1821 to zero (3,072 weights, 0.0002% of the model) | **zero forward passes** |
| W1 | Same edit family, tuned with feedback: top-K most Danish-writing neurons (K ∈ {2, 8, 32, 128}), output column multiplied by c ∈ {0, −1, −2, −4} (also −8 for K = 2) | dev-val generations, counted |
| R | Random-neuron control: same layers, same K, same c as W0 (and as W1 if different), 5 seeds | – |
| S1 | Activation steering (conventional alternative): difference of mean residuals between "answer in Swedish" and "answer in Danish" system prompts on dev-fit prompts, added at layer ∈ {8, 14, 20} with strength ∈ {1, 2, 4} | dev-fit + dev-val forward passes, counted |
| F1 | LoRA fine-tune (conventional alternative): rank 8 on q/v projections, 3 epochs, lr 2e-4, on 80 dev-fit prompts with the model's own Swedish answers (generated under P1) as targets. No tuning. | 80 generations + gradient steps |

Selection rule for W1 and S1 (dev-val only): highest Swedish-response rate on the 80 dev-val Swedish prompts,
subject to English-response rate on 40 dev-val English prompts ≥ baseline − 2.5 points and degenerate-output
rate ≤ baseline + 2.5 points. Ties go to the smaller intervention (smaller K, then smaller |c|; lower strength).

## Held-out evaluation (never used above)

* **Target:** share of responses in Swedish (lingua detector) on `sv_A` (dev templates × 20 unseen topics) and
  `sv_B` (8 unseen templates × 20 unseen topics), 320 prompts. Secondary: Swedish function-word share.
* **Regression checks:** English prompts → English (`en_B`); Danish prompts → Danish (`da_B`);
  perplexity on natural Swedish, English, Danish, Norwegian text (UD test sets); degenerate-output rate.
* **Capability check:** accuracy on 237 Swedish grammatical minimal pairs; Swedish perplexity.

## Predictions, and what would count against them

1. **H1 – the obstacle is edit-fixable.** Some edit in the W family cuts held-out drift by at least half,
   clearly more than the random-neuron control, with ≤ 5% relative increase in Swedish and English
   perplexity and ≤ 2.5 points lost on English-response rate.
   *Against:* no W edit beats random controls, or it only works with large collateral damage.
2. **H2 – reading the weights is enough.** W0, built with zero forward passes, already satisfies H1.
   My honest prior is about 30%: language choice is probably decided earlier and redundantly, so deleting
   two late neurons may do little. *Against:* W0 ≈ baseline while W1 or S1 work. That would mean feedback, not
   weight reading, does the work.
3. **H3 – no new capability.** No edit improves Swedish minimal-pair accuracy by more than 3 points or
   lowers Swedish perplexity by more than 2%. *Against:* a clear gain in either.
4. **Selection fix vs. capability removal (diagnostic, not a pass/fail prediction).** If an edit removes drift
   only by destroying Danish (Danish-response rate on Danish prompts collapses, Danish perplexity jumps),
   it removed a capability instead of fixing a selection error. P1 is the reference for a pure selection fix.

Uncertainty: bootstrap over test templates and topics jointly (they are crossed), 2,000 resamples.
