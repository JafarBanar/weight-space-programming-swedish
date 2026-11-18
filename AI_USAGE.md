# AI usage

This project was carried out with heavy use of an AI coding agent, and this file says so plainly.

| Tool | What it did |
|---|---|
| **Claude Code** (Anthropic, model `claude-fable-5-2`), run locally on the author's laptop | Read the article, the authors' released Experiment 8 protocol, and the SE.LLMA project page. Proposed the research question after probing the unedited model. Wrote all code in `src/` and `explore/`. Read the model's weights and chose the two neurons to edit (it played the role the article gives to "Astra"). Wrote the pre-registration before any edit was run. Ran every experiment, produced the figures and tables, and drafted the research note and the PDF report. |
| **A second AI assistant** (a general chat model) | Gave an independent review of the plan and proposed a different design (English source text, Swedish answer, rank-one edit). Its design was not adopted as the main experiment, but three of its suggestions were: a factual-accuracy check with a correct code and a distractor (`src/notices.py`), reporting fixed/broken counts, and checking the article authors' run logs for how much model feedback the headline edit used. |
| **lingua-language-detector** (not an LLM) | Labels the language of each generated answer. Validated on natural text in `results/detector_validation.json`; its errors on model-written text are discussed in the note. |

No AI-generated text is used as *training* data except in the LoRA baseline, where the model under study is
fine-tuned on its own Swedish answers (self-distillation), as described in the note.

**What the AI agent could not do and a human should check:** the Swedish, Danish and English prompts, the
Swedish grammatical minimal pairs and the Danish translations were written by the AI agent, not by a native
speaker. The audit of language-detector errors was also done by the AI agent reading the outputs.

**Author's role:** set the task and its constraints, supplied the second assistant's review, and is
responsible for checking the results and the write-up before submission.
