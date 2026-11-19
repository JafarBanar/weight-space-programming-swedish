"""Build NATIVE_REVIEW.md: a 20-minute checklist that lets a Swedish (and Danish) speaker verify all
AI-written language material in this project. Nothing here is evaluated by a model.
  python src/make_review_sheet.py"""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import prompts, prompts2, minimal_pairs

L = ["# Review sheet for a native speaker", "",
     "The synthetic Swedish and Danish material in this project was written by an AI agent. It passed automatic checks",
     "(Swedish Wiktionary, Korp corpus counts, LanguageTool), but **no native speaker has read it**. This sheet lists",
     "all of it once. Mark each line: `ok`, or write the correction. About 20 minutes.", "",
     "Reviewer: ____________  Native language(s): ____________  Date: ____________", "",
     "## 1. Swedish prompt templates (all 56)", "",
     "`{t}` is replaced by a topic. Is the sentence natural, correct Swedish?", "",
     "| # | Round | Swedish template | English meaning | ok / correction |", "|---|---|---|---|---|"]
n = 0
for rnd, tpl in (("1", prompts.TEMPLATES), ("2", prompts2.TEMPLATES2)):
    for t in tpl:
        n += 1; L.append(f"| {n} | {rnd} | {t[0]} | {t[1]} | |")
L += ["", "## 2. Swedish topic words (all)", "", "Is each a correct, natural Swedish term for the English one?", "",
      "| # | Swedish | English | ok / correction |", "|---|---|---|---|"]
for i, t in enumerate(list(prompts.TOPICS) + list(prompts2.TOPICS2), 1):
    L.append(f"| {i} | {t[0]} | {t[1]} | |")
L += ["", "## 3. Danish templates and topics (round 1 only; a Danish speaker)", "", "| # | Danish | ok / correction |", "|---|---|---|"]
for i, t in enumerate(list(prompts.DA_TEST_TEMPLATES), 1):
    L.append(f"| {i} | {t if isinstance(t, str) else t[0]} | |")
L += ["", "Topics: " + ", ".join(prompts.DA_TEST_TOPICS), "",
      "## 4. Grammar pairs: 8 sampled per phenomenon (40 of 237)", "",
      "The first sentence should be correct Swedish and the second clearly wrong, for the stated reason only.", "",
      "| # | Phenomenon | Should be correct | Should be wrong | ok / comment |", "|---|---|---|---|---|"]
pairs = minimal_pairs.build(seed=0); rng = random.Random(7); k = 0
for ph in sorted({p["phenomenon"] for p in pairs}):
    for p in rng.sample([x for x in pairs if x["phenomenon"] == ph], 8):
        k += 1; L.append(f"| {k} | {ph.replace('_', ' ')} | {p['good']} | {p['bad']} | |")
L += ["", "## 5. Word lists behind the grammar pairs", "",
      "Common-gender nouns (take *en*): " + ", ".join(minimal_pairs.EN), "",
      "Neuter nouns (take *ett*): " + ", ".join(minimal_pairs.ETT), "",
      "Definite forms: " + ", ".join(f"{a} → {b}" for a, b in minimal_pairs.DEF.items()), "",
      "Adjectives (en-form, ett-form, plural/definite): " + "; ".join(" / ".join(a) for a in minimal_pairs.ADJ), "",
      "Known issues already handled in the analysis: *hus*, *tåg*, *fotografi*, *kort* are dropped in the cleaned set of 223 pairs.", ""]
open("NATIVE_REVIEW.md", "w").write("\n".join(L) + "\n")
print(len(L), "lines;", n, "templates;", k, "pairs")
