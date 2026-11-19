# AI usage

This project was carried out with heavy use of AI tools, and this file says so plainly.

| Tool | What it did |
|---|---|
| **Claude Code** (Anthropic, model `claude-fable-5-2`), run locally on the author's laptop | Read the article, the authors' released Experiment 8 protocol and the SE.LLMA project page. Proposed the research question after probing the unedited model. Wrote all code in `src/` and `explore/`. Screened the model's weights and chose the neurons to edit (the role the article gives to "Astra"). Wrote both pre-registrations before the corresponding runs. Ran every experiment, produced the figures and tables, and drafted the research note and the PDF report. In round 2 it also ran the checks against online native sources, a literature survey (with a research sub-agent), and a blinded re-labelling of 48 answers. |
| **A second AI assistant** (a general chat model) | Reviewed the plan before round 1 and the results after it. From its first review: the factual-accuracy check with a correct code and a distractor (`src/notices.py`), fixed/broken counts, and checking the article authors' run logs. From its second review: the fixed-prefix test of whether the neuron selects or continues a language, the calibrated null for the weight screen, the per-template table, the request-aware prompt, the Danish-recovery test, the equal-token-length check on grammar pairs, dropping the item-level significance test, the chronology table, and several corrections of wording that had overstated the evidence. It also checked the Danish and Swedish material and flagged three imprecise Danish topic words. Its alternative experimental design (English source text, rank-one edit) was not adopted. |
| **lingua-language-detector**, **LanguageTool**, **Korp** (Språkbanken), **Wiktionary/Wikipedia APIs** (not LLMs) | Language labels for generated answers; spelling check; counts in native Swedish text; noun genders and inflected forms. |

No AI-generated text is used as *training* data except in the LoRA baseline, where the model under study is
fine-tuned on its own Swedish answers (self-distillation), as described in the note.

## What the AI agent did that a human should check

* The synthetic Swedish and English prompts, the Swedish grammatical minimal pairs and the round-1 Danish prompts were
  written by the AI agent. They pass the automatic checks in `results/language_checks.md`, and round 2 adds
  professionally translated questions (MKQA) and native treebank sentences, but **no native speaker has reviewed the
  synthetic material**.
* The audit of the language detector was done by the AI agent reading shuffled answers with the condition hidden. It is
  blinded, but it is not a human judgement.
* The literature survey was cut short by a usage limit; the papers cited in the report were opened and their abstracts
  checked, but body-level numbers quoted from them should be spot-checked before reuse.

## Author's role

In this project the author:

* chose the task and set its constraints (a small laptop experiment, private repository until submission);
* obtained and supplied the second assistant's two reviews, and decided that all of its points should be addressed;
* identified the six weaknesses of round 1 that round 2 had to answer (native-language check, more models and prompts,
  pre-registering the post-hoc result, a larger steering search, a full reproduction, and this file);
* arranged approved access to a university cluster in case it was needed (it was not used: all runs are on the laptop);
* did not write the code or the text by hand, and is responsible for checking the results and the write-up before submission.
