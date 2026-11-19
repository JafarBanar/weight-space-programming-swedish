# AI usage

This project was carried out with heavy use of AI tools, and this file says so plainly.

| Tool | What it did |
|---|---|
| **Claude Code** (Anthropic, model `claude-fable-5-2`), run locally on the author's laptop | Read the article, the authors' released Experiment 8 protocol and the SE.LLMA project page. Proposed the research question after probing the unedited model. Wrote all code in `src/` and `explore/`. Screened the model's weights and chose the neurons to edit (the role the article gives to "Astra"). Wrote both pre-registrations before the corresponding runs. Ran every experiment, produced the figures and tables, and drafted the research note and the PDF report. In round 2 it also ran the checks against online native sources, a literature survey (with a research sub-agent), and a blinded re-labelling of 48 answers. In round 3 it designed and pre-registered the search for a working fix, constructed the closed-form weight edit, trained the two LoRA variants, added the second language identifier, completed the literature survey from paper abstracts, and wrote the native-speaker review sheet. |
| **A second AI assistant** (a general chat model) | Reviewed the plan before round 1 and the results after it. From its first review: the factual-accuracy check with a correct code and a distractor (`src/notices.py`), fixed/broken counts, and checking the article authors' run logs. From its second review: the fixed-prefix test of whether the neuron selects or continues a language, the calibrated null for the weight screen, the per-template table, the request-aware prompt, the Danish-recovery test, the equal-token-length check on grammar pairs, dropping the item-level significance test, the chronology table, and several corrections of wording that had overstated the evidence. It also checked the Danish and Swedish material and flagged three imprecise Danish topic words. Its alternative experimental design (English source text, rank-one edit) was not adopted. |
| **A third AI assistant** (a coding assistant with access to the project folder) | Read the repository after round 2 and checked claims against the saved outputs. It found that the reproduction count (8,342) included language labels as well as the 7,160 generated answers, asked for a narrower recommendation, and pointed out that the 96% prompt result measures answer language and not correctness. All three were corrected. |
| **lingua-language-detector**, **langdetect**, **LanguageTool**, **Korp** (Språkbanken), **Wiktionary/Wikipedia APIs** (not LLMs) | Language labels for generated answers (primary and second identifier); spelling check; counts in native Swedish text; noun genders and inflected forms. |

No AI-generated text is used as *training* data except in the LoRA runs, where the model under study is fine-tuned on
its own answers (self-distillation): in round 1 on its Swedish answers to synthetic prompts, in round 3 on its answers
to human-translated questions when an instruction names the language.

## What the AI agent did that a human should check

* The synthetic Swedish and English prompts, the Swedish grammatical minimal pairs and the round-1 Danish prompts were
  written by the AI agent. They pass the automatic checks in `results/language_checks.md`, and rounds 2 and 3 rest on
  professionally translated questions (MKQA) and native treebank sentences, but **no native speaker has reviewed the
  synthetic material**. `NATIVE_REVIEW.md` lists all of it once so that a Swedish speaker can check it in about
  20 minutes; it has not been filled in.
* The audit of the language detector was done by the AI agent reading shuffled answers with the condition hidden. It is
  blinded, but it is not a human judgement.
* The literature survey: the round-2 part was cut short by a usage limit, and the round-3 part was done from arXiv
  search results and full abstracts only. Every cited paper's abstract was read, but the papers themselves were not,
  so claims about them are limited to what their abstracts state.

## Author's role

In this project the author:

* chose the task and set its constraints (a small laptop experiment, private repository until submission);
* obtained and supplied the second assistant's two reviews, and decided that all of its points should be addressed;
* identified the six weaknesses of round 1 that round 2 had to answer (native-language check, more models and prompts,
  pre-registering the post-hoc result, a larger steering search, a full reproduction, and this file);
* after round 2, asked for the remaining weak points to be fixed and for a method that works, which became round 3,
  and supplied the third assistant's check of the repository;
* arranged approved access to a university cluster in case it was needed (it was not used: all runs are on the laptop);
* did not write the code or the text by hand, and is responsible for checking the results and the write-up before submission.
