# Can reading a model's weights fix Swedish answers that come out in the wrong language?

A pre-registered test, in three rounds, of one claim from
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, 2026), applied to a problem that matters for
[SE.LLMA](https://wasp-sweden.org/research/projects/sellma/): Swedish language models.

* **Research note (1–2 pages):** [RESEARCH_NOTE.md](RESEARCH_NOTE.md)
* **Report (PDF, longer):** [report/report.pdf](report/report.pdf)
* **Pre-registrations, each committed before the runs it covers:**
  [round 1](PREREGISTRATION.md), [round 2](PREREGISTRATION_ROUND2.md), [round-2 addendum](PREREGISTRATION_ROUND2_ADDENDUM.md),
  [round 3](PREREGISTRATION_ROUND3.md)
* **Review sheet for a native speaker (not yet filled in):** [NATIVE_REVIEW.md](NATIVE_REVIEW.md)
* **AI usage:** [AI_USAGE.md](AI_USAGE.md)

## The result in one paragraph

Screening the weights of `Qwen2.5-1.5B-Instruct` without running it nominates two neurons that control whether
answers come out in Swedish or Danish. The pre-registered edit (delete both) missed its criteria. Deleting only one
looked strong in round 1, but on fresh data in a pre-registered second round it barely changed how often the model
answers in Swedish, and it damaged Danish output in all four models tested. On human-translated questions the real
failure is answering in English. A third round, with its own pre-registration, asked whether any fix works: a tuned
request-aware prompt and a small LoRA that distils it into the weights passed five pre-set checks on fresh questions in
Swedish, Danish, Norwegian and English; a closed-form rank-3 weight edit got the language right most often but lost
answer content.

## Reproduce

Requirements: Python 3.12, about 8 GB of free memory, 15 GB of disk for four models. Runs on Apple Silicon (MPS),
CUDA or CPU. Rounds 1 and 2 take about 2 hours each and round 3 about 2.5 hours on a MacBook Pro M3 Pro (18 GB).
Keep the laptop awake and on mains power; a sleeping laptop stalls the runs.

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
./run_all.sh
```

```bash
./run_round2.sh
```

```bash
./run_round3.sh all qwen2.5-1.5b && ./run_round3.sh all qwen2.5-0.5b
```

`run_all.sh` (round 1), `run_round2.sh` (round 2) and `run_round3.sh` (round 3: `dev` tunes and freezes every method,
`test` runs the frozen methods once, `all` does both) download the pinned models and public data, then run every
stage. Each stage caches its output in `results/`; the committed `results/` therefore lets you rebuild all tables
and figures in seconds:

```bash
.venv/bin/python src/analyze.py && .venv/bin/python src/analyze_round2.py && .venv/bin/python src/analyze_extras.py && .venv/bin/python src/analyze_round3.py
```

Delete `results/` to recompute everything from scratch. Run the model scripts one at a time: on Apple hardware two
at once will exhaust memory.

A two-minute check that your setup reproduces the committed outputs:

```bash
.venv/bin/python src/check_reproduction.py
```

A full from-scratch rerun of round 1 (fresh clone, new environment, no cached results, same machine) was compared
with the committed results: all 7,160 generated answers are identical (8,342 stored strings in all, counting language
labels and settings), and 36,002 of 36,003 stored numbers are bit-identical; the one difference is the wall-clock time of LoRA training. See
[results/reproduction_check.md](results/reproduction_check.md). Round 2 was rerun in part (the main stages of all four
models; not the steering search or the exploratory extras): all 17,620 generated answers and 52,884 numbers are
identical, see [results/reproduction_check_round2.md](results/reproduction_check_round2.md). Round 3 was not rerun. To compare a rerun
of your own:

```bash
.venv/bin/python src/compare_runs.py <committed_results_dir> <your_results_dir>
```

To rebuild the PDF report (needs XeLaTeX):

```bash
cd report && xelatex report.tex && xelatex report.tex
```

Decoding is greedy and every random choice is seeded (seed 0). Half-precision arithmetic differs slightly between
MPS, CUDA and CPU, so some answers may differ on other hardware.

## Configuration

| Item | Value |
|---|---|
| Main model | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Survey models (round 2) | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760`, `utter-project/EuroLLM-1.7B-Instruct` @ `a25c7fa6`, `Qwen/Qwen3-1.7B` @ `70d244cc` (thinking disabled); full hashes in `src/download_models.py`. All Apache-2.0 |
| Precision / device | float16, Apple MPS (PyTorch 2.15.0, Transformers 5.23.0; all versions in `requirements.txt`) |
| Decoding | greedy, 48 new tokens (40 for notice questions), each model's default system prompt unless stated |
| Seeds | 0 for prompt shuffling, data selection and bootstrap; 0–4 and 200–201 for random-neuron controls |
| Round-1 edit (W0, pre-registered) | output columns of layer 23 neuron 981 and layer 22 neuron 1821 in `mlp.down_proj` set to zero |
| Round-2 edit (W0a, pre-registered) | output column of layer 23 neuron 981 set to zero |
| Language detector | `lingua-language-detector` 2.2.0, 11 candidate languages |
| Round-3 edits | every weight change of round 3 is stored: `results/r3/<model>/e1_edit.pt` (closed-form probes and write vectors) and `results/r3/<model>/adapters/` (LoRA factors); `frozen.json` names the ones tested |
| Second language identifier | `langdetect` 1.0.9, seed 0 (`src/second_detector.py`) |
| MKQA file | `mkqa.jsonl.gz`, SHA-256 `f77ac6bf6306e98816c5c047880db71a3c67d2f1491b169896d63e9bb14ef3d3` |

## Repository layout

```
PREREGISTRATION*.md    hypotheses, conditions and predictions; git history shows each predates its runs
NATIVE_REVIEW.md       checklist of all AI-written Swedish and Danish material, for a native speaker
RESEARCH_NOTE.md       the 1–2 page note
report/                LaTeX source and PDF of the longer report
run_all.sh             round 1, one command
run_round2.sh          round 2, one command
run_round3.sh          round 3 (dev, test or all; per model)
src/
  prompts.py, prompts2.py   synthetic prompts (round 1; round 2 with 40 new templates) and the MKQA loader
  minimal_pairs.py          237 synthetic Swedish grammatical minimal pairs
  native_pairs.py           200 pairs built from native treebank sentences
  notices.py                40 notice questions with a correct code and a distractor
  edits.py                  weight screen, neuron edit, random controls, activation steering
  evaluate.py               metrics: answer language, perplexity, minimal-pair accuracy
  run_experiment.py         round 1: screen -> dev tuning -> held-out test
  run_notices.py, run_posthoc.py, run_lora.py   round-1 additions
  run_round2.py             round 2: frozen recipe on four models, fresh data, larger steering search
  run_extras.py             round-2 exploratory: fixed-prefix test, request-aware prompt, Danish recovery
  prompts3.py               round 3: fresh MKQA split in four languages, UD train text for the regulariser
  run_round3.py             round 3: prompt choice, closed-form edit, LoRA distillation, frozen test, robustness
  second_detector.py        every language label recomputed with an independent identifier
  make_review_sheet.py      builds NATIVE_REVIEW.md
  check_language_sources.py checks against Wiktionary, Korp corpus counts, Wikipedia, LanguageTool
  analyze.py, analyze_round2.py, analyze_extras.py, analyze_round3.py   tables, figures, verdicts, saved-output analyses
  compare_runs.py, check_reproduction.py             reproduction checks
explore/               exploratory scripts and outputs that led to the questions (dev data only)
results/               every raw model output, per-item score and summary table (rounds 2 and 3 in results/r2/, results/r3/)
figures/               figures used in the note and report
logs/                  run logs
```

## Data and licences

* Synthetic prompts, minimal pairs and notices were written for this project (MIT, like the code).
* Downloaded at run time and not redistributed here: Universal Dependencies r2.14 treebanks for Swedish (Talbanken),
  English (EWT), Danish (DDT) and Norwegian Bokmål (CC BY-SA 4.0); Apple MKQA (CC BY-SA 3.0).
* An early commit in this repository's history contains a 200-sentence-per-language extract of the UD test sets
  (`data/ud_text.json`, CC BY-SA 4.0, attribution as above). It was removed from later commits and is regenerated by
  `src/fetch_data.py`.
* Model weights are downloaded from Hugging Face and not redistributed. Every edit is fully specified by neuron
  indices or by the small files listed under Configuration, so no edited checkpoint needs to be shared.
* `requirements.txt` gains `langdetect` for the second language identifier (round 3 only).
