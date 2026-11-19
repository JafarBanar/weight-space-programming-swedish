# Review of the AI-written Swedish and Danish material by a second language model

The synthetic prompts, topic words, Danish prompts, grammar pairs and word lists (everything in `NATIVE_REVIEW.md`) were
given to a different language model (Claude Opus, run as an independent reviewer with no access to the results) with the
instruction to judge every line strictly. **This is a model's judgement, not a native speaker's.** `NATIVE_REVIEW.md`
remains available for a human review.

## Outcome

| Section | Items reviewed | No issue | Errors | Notes on idiom or meaning |
|---|---|---|---|---|
| Swedish prompt templates | 56 | 47 | 0 | 9 |
| Swedish topic words | 48 | 44 | 0 | 4 |
| Danish templates and topics | 28 | 26 | 0 | 2 |
| Grammar pairs (sample of 40) | 40 | 32 | 1 | 7 |
| Word lists (genders, definite forms, adjectives) | 75 | 70 | 0 | 5 (homographs, three already dropped) |

**The one error.** In the verb-second pairs that begin with *Sedan* ("Sedan hittade hon en flaska." against "Sedan hon hittade
en flaska."), the supposedly wrong sentence is a well-formed subordinate clause, because *sedan* is also a conjunction
("after"). This affects 9 of the 50 verb-second pairs (237 pairs in all). The model scores 55.6% on those
pairs in every condition, close to chance, which is what an invalid pair would give.

**Effect on the results.** Grammar-pair accuracy with and without the *Sedan* pairs:

| Condition | All 237 pairs | Without the 9 *Sedan* pairs |
|---|---|---|
| R1 B0 unedited | 73.0% | 73.7% |
| R1 W0 both neurons | 72.6% | 73.2% |
| R1 W0a neuron 981 | 73.4% | 74.1% |
| R1 W0b neuron 1821 | 68.8% | 69.3% |
| R3 B0 unedited | 73.0% | 73.7% |
| R3 E1 closed-form | 75.1% | 75.9% |
| R3 F2 LoRA | 72.2% | 72.8% |

Every figure moves by less than one point and in the same direction, so no conclusion changes.

**Other points raised (no errors):**

* Templates: "Vilka är de största utmaningarna inom {t}?" fits fields but not topics such as *kaffe* or *sömn*; "Jämför {t} förr och nu." is
  terse; "missförstånd" would more usually be "missuppfattningar"; "Avsluta meningen" would more usually be "Fortsätt meningen".
* Topics: *evolution* and *global uppvärmning* usually take the definite form in running text; *träning*, *näringslära* and
  *trädgårdsodling* are correct but not the most everyday terms. Some template and topic combinations are odd in meaning
  ("Vad kostar det att ägna sig åt global uppvärmning?").
* Danish: *genbrug* for recycling is imprecise (*genanvendelse*), as an earlier reviewer also noted.
* Grammar pairs: several correct sentences are odd in meaning ("Vi har ett tungt flygplan hemma."), which could affect a
  language model's probabilities for both members of a pair equally.
* Word lists: every gender, definite form and adjective form was judged correct. The reviewer could not see why *hus* was
  dropped from the cleaned set; it was dropped because Swedish Wiktionary lists *hus* under both genders.
