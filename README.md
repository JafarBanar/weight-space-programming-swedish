# Can reading a model's weights fix Swedish answers that drift into Danish?

A small, pre-registered test of one claim from
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, 2026), applied to a problem that matters for
[SE.LLMA](https://wasp-sweden.org/research/projects/sellma/): Swedish language models.

* **Research note (1–2 pages):** [RESEARCH_NOTE.md](RESEARCH_NOTE.md)
* **Report (PDF, longer):** [report/report.pdf](report/report.pdf)
* **Pre-registration, committed before any edit was run:** [PREREGISTRATION.md](PREREGISTRATION.md)
* **AI usage:** [AI_USAGE.md](AI_USAGE.md)

## The question in one paragraph

`Qwen2.5-1.5B-Instruct` answers about 15% of plain Swedish prompts in Danish, Norwegian or another
language, although it can answer the same prompts in Swedish when told to. The article argues that a
frontier LLM can read a network's weights, find the cause of such a failure, and remove it with a tiny edit.
I let an LLM agent read the weights **without running the model**, fixed the edit it proposed in
advance, and then measured it on held-out prompts against random edits, prompting, activation steering
and LoRA fine-tuning, with regression checks for English, Danish, natural-text perplexity, Swedish
grammar and factual accuracy.

## Reproduce

Requirements: Python 3.12, about 8 GB of free memory, 4 GB of disk. Runs on Apple Silicon (MPS), CUDA or CPU.
The full pipeline took about 2 hours on a MacBook Pro M3 Pro (18 GB).

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
./run_all.sh
```

`run_all.sh` downloads the model (pinned revision) and the Universal Dependencies text used for
perplexity, then runs every stage. Each stage caches its output in `results/`; the committed `results/`
therefore lets you rebuild tables and figures in seconds with `.venv/bin/python src/analyze.py`.
Delete `results/` to recompute everything from scratch.

A two-minute check that your setup reproduces the committed outputs (it recomputes two conditions on 160 prompts and compares answer by answer):

```bash
.venv/bin/python src/check_reproduction.py
```

To rebuild the PDF report (needs XeLaTeX):

```bash
cd report && xelatex report.tex && xelatex report.tex
```

Decoding is greedy and every random choice is seeded (seed 0), so a rerun on the same hardware and
library versions reproduces the committed outputs. Half-precision arithmetic differs slightly between
MPS, CUDA and CPU, so a few answers may differ on other hardware.

## Configuration

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0) |
| Precision / device | float16, Apple MPS (PyTorch 2.15.0, Transformers 5.23.0; all versions in `requirements.txt`) |
| Decoding | greedy, 48 new tokens (40 for notice questions), default Qwen system prompt unless stated |
| Seed | 0 for prompt shuffling, minimal pairs, notices; 0–4 for random-neuron controls |
| Weights-only edit (W0) | output columns of layer 23 neuron 981 and layer 22 neuron 1821 in `mlp.down_proj` set to zero: 3,072 of 1.54 billion weights |
| Language detector | `lingua-language-detector` 2.2.0, 11 candidate languages |

## Repository layout

```
PREREGISTRATION.md     hypotheses, conditions, selection rule, predictions (git history shows it predates the edits)
RESEARCH_NOTE.md       the 1–2 page note
report/                LaTeX source and PDF of the longer report
run_all.sh             one command to reproduce everything
src/
  prompts.py           synthetic Swedish/English/Danish prompts; dev and test use disjoint templates and topics
  minimal_pairs.py     237 synthetic Swedish grammatical minimal pairs
  notices.py           40 notice questions with a correct code and a distractor (post-hoc factual check)
  edits.py             weight reading, the neuron edit, random controls, activation steering
  evaluate.py          metrics: answer language, perplexity, minimal-pair accuracy
  run_experiment.py    main pipeline: read weights -> dev tuning -> held-out test
  run_notices.py       post-hoc factual-accuracy check
  run_posthoc.py       post-hoc: delete each of the two neurons separately
  run_lora.py          LoRA fine-tuning baseline
  robustness_weight_reading.py   does the weight reading depend on the word list?
  validate_detector.py language-detector accuracy on natural text
  check_reproduction.py  quick answer-by-answer comparison with the committed outputs
  analyze.py           tables, figures, examples
explore/               the exploratory scripts and outputs that led to the question (dev prompts only)
results/               every raw model output, per-item score and summary table
figures/               figures used in the note and report
logs/                  run logs
```

## Data and licences

* All prompts, minimal pairs and notices are synthetic and were written for this project (MIT, like the code).
* Natural text for perplexity comes from the Universal Dependencies r2.14 test sets for Swedish (Talbanken),
  English (EWT), Danish (DDT) and Norwegian Bokmål, all CC BY-SA 4.0. It is downloaded by
  `src/fetch_data.py` and not redistributed here.
* The model weights are downloaded from Hugging Face and not redistributed. The edit is fully specified by
  two neuron indices, so no edited checkpoint needs to be shared.
