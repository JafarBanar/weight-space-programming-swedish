# Addendum to the round-2 pre-registration (written before the experiments below were run)

Date: 2026-11-19, after the round-2 confirmatory run on `Qwen2.5-1.5B-Instruct` had finished (results in
`results/r2/qwen2.5-1.5b/`) and after an external review of round 1. Everything here is exploratory unless stated.

## E1. Is layer 23 neuron 981 a language *selector* or a *continuation* unit? (reviewer's suggestion)

Round 1 only showed that deleting the neuron changes the language of whole generated answers. Under greedy decoding
that cannot separate "chooses the language" from "keeps Danish going once it has started".

Design (`src/run_extras.py`): 14 simple facts, each asked in Swedish and in Danish, each with a Swedish and a Danish
answer sentence of equal length whose words are aligned. Teacher-forced, no generation. For each item:

* **Neutral context:** prompt + the words the two answers share before they first differ. Margin = log P(Swedish word) − log P(Danish word) for the first differing word.
* **After a Danish prefix / after a Swedish prefix:** prompt + the Danish (or Swedish) answer up to its last differing word. Margin for that word.

The scored words are outside the 17 word pairs used to find the neuron. The neuron is zeroed **only at the positions
that predict the scored word** ("position-specific"), and separately at every position. Controls: layer 22 neuron 1821,
and the layer-23 neuron whose output-norm × RMS activation on these contexts is closest to neuron 981's.

Interpretation, fixed now. Let Δ be the change in margin when neuron 981 is zeroed position-specifically (positive = towards Swedish), averaged over facts, with a bootstrap interval over the 14 facts:

* **Selector:** Δ in the neutral context is clearly above zero and at least half of Δ after a Danish prefix.
* **Continuation unit:** Δ in the neutral context is below a quarter of Δ after a Danish prefix, or its interval includes zero.
* Anything in between is reported as mixed.

My expectation: continuation (about 65%), because the neuron sits in layer 23 of 28 and its output weights promote Danish word endings.

## E2. Request-aware prompt (reviewer's suggestion)

System prompt "You are a helpful assistant. Always answer in the same language as the user's message." on T1, and on
MKQA Swedish, Danish and English. Unlike "Svara alltid på svenska." this should not force Swedish onto Danish or English users.

## E3. Does Danish come back when asked for?

Under the single-neuron edit, Danish questions with the system prompt "You are a helpful assistant. Svar altid på dansk.",
compared with the unedited model under the same prompt. Tests whether the edit removes the ability to produce Danish or only
changes the default choice.

## E4. A calibrated null for the weight reading

Instead of calling the z-score distribution "noise": 200 control directions, each built like the real one but from 17
random token pairs drawn from the same 34 function-word tokens paired at random (so each direction mixes languages).
Report the distribution of the most negative z over all 250,880 neurons and where −9.2 and −7.9 fall.

## Analyses of saved outputs (no new model runs)

Per-template table of baseline failures, fixes and breaks; grammar pairs restricted to pairs whose two sentences have the
same number of tokens; MKQA answers scored for factual correctness against the dataset's gold answers, jointly with
answer language; round-1 Danish result recomputed without the three topic translations the reviewer flagged
(hjemmearbejde, genbrug, ligestilling); a blinded re-labelling of answers whose language label changed.
